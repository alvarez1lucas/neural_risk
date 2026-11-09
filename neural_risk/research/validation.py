# neural_risk/research/validation.py
"""
Validación estadística de backtests cuando se prueban MUCHAS variantes.

Si se prueban N modelos/configuraciones, el mejor Sharpe va a ser alto
aunque ninguno tenga ventaja real. Estas funciones corrigen eso:

- probabilistic_sharpe_ratio (PSR): probabilidad de que el Sharpe real
  supere un umbral, dada la longitud de la serie y su asimetría/curtosis.
- deflated_sharpe_ratio (DSR): PSR usando como umbral el Sharpe máximo
  ESPERADO por azar después de N pruebas.
- probability_of_backtest_overfitting (PBO): con CSCV, probabilidad de que
  la variante elegida como mejor dentro de muestra quede por debajo de la
  mediana fuera de muestra.

Referencias: Bailey & López de Prado (2012, 2014); Bailey, Borwein,
López de Prado & Zhu (2017), "The Probability of Backtest Overfitting".

Todos los Sharpe acá son POR PERÍODO (sin anualizar), salvo que se diga lo
contrario.
"""

from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew

EULER_GAMMA = 0.5772156649015329


def sharpe_per_period(returns) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > 0 else 0.0


def probabilistic_sharpe_ratio(returns, sr_benchmark: float = 0.0) -> float:
    """P(SR verdadero > sr_benchmark). sr_benchmark por período."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    t = len(r)
    if t < 3:
        return float('nan')
    sr = sharpe_per_period(r)
    g3 = skew(r)
    g4 = kurtosis(r, fisher=False)  # curtosis "completa" (normal = 3)
    denom = 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2
    if denom <= 0:
        return float('nan')
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(t - 1) / np.sqrt(denom)))


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Sharpe máximo esperado (por período) entre n_trials pruebas SIN ventaja."""
    if n_trials < 2:
        return 0.0
    sd = np.sqrt(sr_variance)
    return float(sd * ((1 - EULER_GAMMA) * norm.ppf(1 - 1 / n_trials)
                       + EULER_GAMMA * norm.ppf(1 - 1 / (n_trials * np.e))))


def deflated_sharpe_ratio(returns, n_trials: int, sr_variance: Optional[float] = None,
                          trial_sharpes=None) -> dict:
    """
    returns:       retornos por período de la variante elegida.
    n_trials:      cuántas variantes se probaron EN TOTAL (incluidas las descartadas).
    sr_variance:   varianza de los Sharpe por período entre las pruebas. Si no
                   se pasa, se estima de `trial_sharpes`.
    Devuelve {'dsr', 'sr', 'sr0'}: DSR > 0.95 = el Sharpe sobrevive a la
    corrección por pruebas múltiples con 95% de confianza.
    """
    if sr_variance is None:
        if trial_sharpes is None or len(trial_sharpes) < 2:
            raise ValueError("Pasá sr_variance o al menos 2 trial_sharpes")
        sr_variance = float(np.var(np.asarray(trial_sharpes, dtype=float), ddof=1))
    sr0 = expected_max_sharpe(n_trials, sr_variance)
    return {'dsr': probabilistic_sharpe_ratio(returns, sr0),
            'sr': sharpe_per_period(returns), 'sr0': sr0}


def probability_of_backtest_overfitting(returns: pd.DataFrame, n_splits: int = 16) -> dict:
    """
    CSCV. returns: DataFrame (períodos x variantes) de retornos por período,
    TODAS las variantes probadas, alineadas en el tiempo.

    Parte la muestra en `n_splits` bloques contiguos; para cada combinación
    de la mitad de los bloques como "dentro de muestra", elige la variante
    con mejor Sharpe ahí y mira su ranking en el resto.
    Devuelve {'pbo', 'logits', 'n_combinations'}. PBO cerca de 0 = la
    selección generaliza; >= 0.5 = elegir "el mejor" no sirve.
    """
    if n_splits % 2:
        raise ValueError("n_splits tiene que ser par")
    M = returns.dropna(how='all').fillna(0.0).values
    t, n = M.shape
    if n < 2:
        raise ValueError("Hacen falta al menos 2 variantes")
    blocks = np.array_split(np.arange(t), n_splits)

    def _sr(mat):
        sd = mat.std(axis=0, ddof=1)
        return np.where(sd > 0, mat.mean(axis=0) / np.where(sd > 0, sd, 1), 0.0)

    logits = []
    for is_blocks in combinations(range(n_splits), n_splits // 2):
        is_idx = np.concatenate([blocks[b] for b in is_blocks])
        oos_idx = np.concatenate([blocks[b] for b in range(n_splits) if b not in is_blocks])
        best = int(np.argmax(_sr(M[is_idx])))
        oos_sr = _sr(M[oos_idx])
        # ranking relativo (1..n) de la elegida fuera de muestra
        rank = (oos_sr < oos_sr[best]).sum() + 1
        w = rank / (n + 1)
        logits.append(np.log(w / (1 - w)))
    logits = np.array(logits)
    return {'pbo': float((logits <= 0).mean()), 'logits': logits, 'n_combinations': len(logits)}
