"""Tests del módulo neural_risk.research (solo dependencias del core)."""

import numpy as np
import pandas as pd
import pytest

from neural_risk.research import (ExperimentRegistry, deflated_sharpe_ratio, expected_max_sharpe,
                                  extreme_signal, forward_return, probabilistic_sharpe_ratio,
                                  probability_of_backtest_overfitting, ridge_walk_forward,
                                  simulate, stats, walk_forward)


def _bars(n=2000, seed=0, drift=0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(drift, 0.01, n))), index=idx)
    open_ = close.shift(1).fillna(close.iloc[0])
    f = pd.DataFrame({'x': rng.normal(size=n)}, index=idx)
    f.attrs = {'close': close, 'open': open_, 'high': np.maximum(open_, close), 'low': np.minimum(open_, close)}
    return f


# ---------------------------------------------------------------- walk-forward
class _Spy:
    """Modelo que registra cuántas filas vio en cada fit."""
    seen = []

    def fit(self, X, y):
        _Spy.seen.append(len(X))
        return self

    def predict(self, X):
        return np.zeros(len(X))


def test_walk_forward_never_trains_on_unknown_targets():
    f = _bars(3000)
    h, every, min_train = 24, 240, 1000
    y = forward_return(f, h)
    _Spy.seen = []
    walk_forward(_Spy, f[['x']], y, h, every, min_train)
    starts = list(range(min_train, len(f), every))
    # cada fit usa a lo sumo las filas anteriores a (inicio del bloque - h)
    assert _Spy.seen and all(n <= s - h for n, s in zip(_Spy.seen, starts))


def test_walk_forward_no_predictions_before_min_train():
    f = _bars(3000)
    y = forward_return(f, 12)
    pred = ridge_walk_forward(f[['x']], y, 12, 240, 1000)
    assert pred.iloc[:1000].isna().all()
    assert pred.iloc[1000:].notna().mean() > 0.99


def test_forward_return_uses_next_open():
    f = _bars(50)
    y = forward_return(f, 3)
    i = 10
    expected = f.attrs['close'].iloc[i + 3] / f.attrs['open'].iloc[i + 1] - 1
    assert y.iloc[i] == pytest.approx(expected)


# ---------------------------------------------------------------- backtest
def test_simulate_flat_price_only_pays_costs():
    f = _bars(500)
    flat = pd.Series(100.0, index=f.index)
    f.attrs = {'close': flat, 'open': flat, 'high': flat, 'low': flat}
    pred = pd.Series(np.random.default_rng(1).normal(size=len(f)), index=f.index)
    no_funding = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    res = simulate(f, pred, hold=5, q=0.1, cost=0.001, funding=no_funding, per_day=24)
    n_sides = 2 * len(res['trades'])
    assert len(res['trades']) > 0
    # cada lado cuesta 0.1%; el precio no se mueve -> solo costos (±1 lado abierto al final)
    assert res['equity'].iloc[-1] <= (1 - 0.001) ** n_sides + 1e-9
    assert res['equity'].iloc[-1] >= (1 - 0.001) ** (n_sides + 1) - 1e-9


def test_simulate_liquidates_with_high_leverage_on_crash():
    f = _bars(600)
    close = f.attrs['close'].copy()
    # caída de 50% después de que el umbral móvil (min 360 barras) ya da señales
    close.iloc[450:] = close.iloc[449] * 0.5
    f.attrs = {'close': close, 'open': close.shift(1).fillna(close.iloc[0]),
               'high': close, 'low': close}
    pred = pd.Series(1.0, index=f.index)
    pred.iloc[::7] = 5.0  # señales long recurrentes
    no_funding = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    res = simulate(f, pred, hold=500, q=0.2, cost=0.0, funding=no_funding, per_day=24, leverage=5)
    assert res['liquidated_at'] is not None
    assert res['equity'].iloc[-1] == 0.0


def test_stats_on_known_curve():
    eq = pd.Series([1.0, 1.1, 0.99, 1.2])
    st = stats(eq, per_year=3)
    assert st['total_return'] == pytest.approx(0.2)
    assert st['max_drawdown'] == pytest.approx(0.99 / 1.1 - 1)


def test_extreme_signal_is_causal():
    x = pd.Series(np.random.default_rng(2).normal(size=500))
    a = extreme_signal(x, 100)
    x2 = x.copy()
    x2.iloc[400:] = 1e6  # cambiar el futuro no puede alterar el pasado
    b = extreme_signal(x2, 100)
    pd.testing.assert_series_equal(a.iloc[:400], b.iloc[:400])


# ---------------------------------------------------------------- validación
def test_psr_high_for_strong_edge_and_half_for_none():
    rng = np.random.default_rng(3)
    assert probabilistic_sharpe_ratio(rng.normal(0.1, 1, 5000)) > 0.99
    assert 0.2 < probabilistic_sharpe_ratio(rng.normal(0.0, 1, 5000)) < 0.8


def test_expected_max_sharpe_grows_with_trials():
    assert expected_max_sharpe(1, 0.01) == 0.0
    assert expected_max_sharpe(10, 0.01) < expected_max_sharpe(100, 0.01) < expected_max_sharpe(1000, 0.01)


def test_dsr_penalizes_lucky_best_of_many():
    rng = np.random.default_rng(4)
    trials = rng.normal(0, 1, (1000, 200))  # 200 estrategias sin ventaja
    srs = trials.mean(0) / trials.std(0, ddof=1)
    best = trials[:, srs.argmax()]
    assert probabilistic_sharpe_ratio(best) > 0.95          # parece buena sola...
    # ...pero corregida por 200 pruebas no es significativa (en ruido el DSR
    # del mejor ronda 0.5: es una moneda al aire)
    assert deflated_sharpe_ratio(best, 200, trial_sharpes=srs)['dsr'] < 0.95


def test_pbo_high_for_noise_low_for_real_edge():
    rng = np.random.default_rng(5)
    noise = pd.DataFrame(rng.normal(0, 1, (2000, 30)))
    assert probability_of_backtest_overfitting(noise, 10)['pbo'] > 0.3
    edge = noise.copy()
    edge[0] += 0.15  # una variante con ventaja real y persistente
    assert probability_of_backtest_overfitting(edge, 10)['pbo'] < 0.05


# ---------------------------------------------------------------- registro
def test_registry_counts_distinct_configs(tmp_path):
    reg = ExperimentRegistry(str(tmp_path / 'exp.jsonl'))
    reg.log('estudio', {'model': 'ridge', 'alpha': 10}, {'sharpe': 1.6})
    reg.log('estudio', {'model': 'ridge', 'alpha': 10}, {'sharpe': 1.6})  # repetida
    reg.log('estudio', {'model': 'xgb'}, {'sharpe': 0.5})
    reg.log('otro', {'model': 'xgb'}, {'sharpe': 0.1})
    assert reg.n_trials('estudio') == 2
    assert len(reg.load('estudio')) == 3
