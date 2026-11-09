# neural_risk/research/signals.py
"""Transformaciones causales de predicciones en señales y ensembles."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def extreme_signal(x: pd.Series, window: int, q: float = 0.1) -> pd.Series:
    """+1 / -1 cuando x supera el cuantil móvil (pasado) 1-q / q; 0 si no."""
    hi = x.rolling(window, min_periods=window // 2).quantile(1 - q)
    lo = x.rolling(window, min_periods=window // 2).quantile(q)
    return pd.Series(np.where(x > hi, 1.0, np.where(x < lo, -1.0, 0.0)), index=x.index).where(hi.notna())


def rolling_z(pred: pd.Series, window: int) -> pd.Series:
    """Z-score con media y desvío móviles (sin look-ahead)."""
    mp = window // 2
    return (pred - pred.rolling(window, min_periods=mp).mean()) / (pred.rolling(window, min_periods=mp).std() + 1e-12)


def weighted_ensemble(preds: dict, weights: dict, window: int) -> pd.Series:
    """Promedio ponderado FIJO de las predicciones normalizadas."""
    return sum(w * rolling_z(preds[name], window) for name, w in weights.items()) / sum(weights.values())


def adaptive_ensemble(preds: dict, y: pd.Series, h: int, window: int, step: int,
                      lookback: int) -> tuple:
    """
    Pesos adaptativos y causales: cada `step` barras, el peso de cada modelo
    es su IC (Spearman) en las `lookback` barras anteriores, usando solo
    targets ya conocidos (se corta h barras antes). IC negativo -> peso 0.
    Devuelve (predicción combinada, historial de pesos).
    """
    names = list(preds)
    z = pd.DataFrame({n: rolling_z(preds[n], window) for n in names})
    out = pd.Series(np.nan, index=z.index)
    history = []
    for start in range(0, len(z), step):
        end_known = start - h
        w = {n: 0.0 for n in names}
        if end_known > lookback // 2:
            sl = slice(max(0, end_known - lookback), end_known)
            for n in names:
                m = z[n].iloc[sl].notna() & y.iloc[sl].notna()
                if m.sum() > 100:
                    w[n] = max(spearmanr(z[n].iloc[sl][m], y.iloc[sl][m])[0], 0.0)
        total = sum(w.values())
        if total == 0:
            w = {n: 1.0 / len(names) for n in names}
        else:
            w = {n: v / total for n, v in w.items()}
        te = slice(start, min(start + step, len(z)))
        out.iloc[te] = sum(w[n] * z[n].iloc[te] for n in names)
        history.append({'desde': str(z.index[start]), **{n: round(v, 3) for n, v in w.items()}})
    return out, pd.DataFrame(history)
