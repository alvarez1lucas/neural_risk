# scripts/compare_models.py
"""
Compara modelos sobre las MISMAS features de flujo/derivados, el mismo
target y el mismo walk-forward (reentreno mensual, sin ver el futuro), y
backtestea cada uno con comisiones de Binance Futures y apalancamiento.

Uso:
    python scripts/compare_models.py --symbol BTCUSDT --timeframe 30m --hold-hours 24 --q 0.05
"""
import argparse
import json
import os
import sys
import time

from _pathutils import project_root

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from neural_risk.data.derivatives_features import build_derivatives_features
from feature_diagnostics import forward_return
from backtest_intraday import load, simulate, stats

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Comisiones Binance (VIP 0). Futuros USDⓈ-M: maker 0.02% / taker 0.05%;
# con descuento BNB (10%): 0.018% / 0.045%. Spot: 0.10% (0.075% con BNB).
# + slippage estimado por lado (BTC perp es muy líquido).
BINANCE_FEES = {
    'spot_taker_vip0':        0.0010 + 0.0005,
    'futures_taker_vip0':     0.0005 + 0.0001,
    'futures_taker_bnb':      0.00045 + 0.0001,
    'futures_maker_vip0':     0.0002,
    'futures_maker_bnb':      0.00018,
}

MODELS = {
    'ridge': lambda: Ridge(alpha=10.0),
    'xgboost': lambda: xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.03,
                                        subsample=0.7, colsample_bytree=0.7, min_child_weight=50,
                                        reg_lambda=10.0, n_jobs=4, random_state=42),
    'mlp': lambda: MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3,
                                max_iter=50, early_stopping=True, random_state=42),
}


def walk_forward(model_name, X, y, h, retrain_every, min_train):
    pred = pd.Series(np.nan, index=X.index)
    Xv, yv = X.values, y.values
    for start in range(min_train, len(X), retrain_every):
        tr_end = start - h
        Xt, yt = Xv[:tr_end], yv[:tr_end]
        ok = ~np.isnan(Xt).any(axis=1) & ~np.isnan(yt)
        if ok.sum() < 500:
            continue
        sc = StandardScaler().fit(Xt[ok])
        yt_ok = yt[ok]
        y_scale = yt_ok.std()
        model = MODELS[model_name]()
        model.fit(np.clip(sc.transform(Xt[ok]), -5, 5), yt_ok / y_scale)
        te = slice(start, min(start + retrain_every, len(X)))
        Xe = Xv[te]
        okp = ~np.isnan(Xe).any(axis=1)
        out = np.full(len(Xe), np.nan)
        if okp.any():
            out[okp] = model.predict(np.clip(sc.transform(Xe[okp]), -5, 5))
        pred.iloc[te] = out
    return pred


def rolling_z(pred: pd.Series, window: int) -> pd.Series:
    """Normalización CAUSAL (antes: rank(pct=True) sobre toda la serie = look-ahead)."""
    mp = window // 2
    return (pred - pred.rolling(window, min_periods=mp).mean()) / (pred.rolling(window, min_periods=mp).std() + 1e-12)


def weighted_ensemble(preds: dict, weights: dict, window: int) -> pd.Series:
    """Promedio ponderado FIJO de las predicciones normalizadas."""
    return sum(w * rolling_z(preds[name], window) for name, w in weights.items()) / sum(weights.values())


def adaptive_ensemble(preds: dict, y: pd.Series, h: int, window: int, step: int,
                      lookback: int) -> tuple:
    """
    Pesos ADAPTATIVOS y causales: cada `step` barras, el peso de cada modelo
    es su IC (Spearman) en las `lookback` barras anteriores, usando solo
    targets ya conocidos (se corta h barras antes). IC negativo -> peso 0.
    Es lo que el agente debería hacer con Thompson Sampling: premiar al
    modelo que viene acertando, pero midiéndolo barra a barra en vez de
    esperar trades cerrados.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--symbol', default='BTCUSDT')
    ap.add_argument('--timeframe', default='30m')
    ap.add_argument('--hold-hours', type=float, default=24)
    ap.add_argument('--q', type=float, default=0.05)
    ap.add_argument('--leverages', nargs='+', type=float, default=[1, 2, 3, 5])
    args = ap.parse_args()

    k5, prem, fund, met = load(args.symbol)
    f = build_derivatives_features(k5, prem, fund, met, args.timeframe)
    per_day = int(round(pd.Timedelta(days=1) / pd.Series(f.index).diff().median()))
    hold = int(round(args.hold_hours * per_day / 24))
    y = forward_return(f, hold)
    X = f[[c for c in f.columns]]

    cache = os.path.join(project_root(), 'backtest_results',
                         f'preds_{args.symbol}_{args.timeframe}_h{args.hold_hours:g}.pkl')
    if os.path.exists(cache):
        preds = dict(pd.read_pickle(cache))
    else:
        preds = {}
        for name in MODELS:
            t0 = time.time()
            preds[name] = walk_forward(name, X, y, hold, per_day * 30, per_day * 180)
            print(f"{name}: {time.time() - t0:.0f}s", flush=True)
        pd.to_pickle(preds, cache)

    base_models = {n: preds[n] for n in MODELS}
    zw = per_day * 30
    preds['ensemble_igual'] = weighted_ensemble(base_models, {'ridge': 1, 'xgboost': 1, 'mlp': 1}, zw)
    preds['ensemble_70_20_10'] = weighted_ensemble(base_models, {'ridge': .7, 'xgboost': .2, 'mlp': .1}, zw)
    preds['ensemble_60_30_10'] = weighted_ensemble(base_models, {'ridge': .6, 'xgboost': .3, 'mlp': .1}, zw)
    preds['ensemble_80_20_0'] = weighted_ensemble(base_models, {'ridge': .8, 'xgboost': .2}, zw)
    preds['ensemble_adaptativo_ic'], weight_hist = adaptive_ensemble(
        base_models, y, hold, zw, step=per_day * 30, lookback=per_day * 180)
    weight_hist.to_csv(os.path.join(project_root(), 'backtest_results',
                                    f'pesos_adaptativos_{args.symbol}_{args.timeframe}.csv'), index=False)
    for single in ['funding_z', 'top_ls_acct', 'crowd_ls']:
        # una sola feature, con el signo que tenía en los primeros 180 días
        first = f.index < f.index[0] + pd.Timedelta(days=180)
        m = f[single].notna() & y.notna() & first
        sign = np.sign(spearmanr(f.loc[m, single], y[m])[0])
        preds[f'solo_{single}'] = f[single] * sign

    start = preds['ridge'].first_valid_index()
    fb = f.loc[start:]
    fb.attrs = {k: v.loc[start:] for k, v in f.attrs.items()}
    years = (fb.index[-1] - fb.index[0]).days / 365.25
    bh = fb.attrs['close'] / fb.attrs['close'].iloc[0]
    bh_st = stats(bh, per_day * 365)

    rows = []
    for name, pred in preds.items():
        pb = pred.loc[start:]
        m = pb.notna() & y.loc[start:].notna()
        ic = spearmanr(pb[m], y.loc[start:][m])[0]
        for fee_name, cost in BINANCE_FEES.items():
            for L in args.leverages:
                res = simulate(fb, pb, hold, args.q, cost, fund['funding_rate'], per_day, leverage=L)
                st = stats(res['equity'], per_day * 365)
                tr = pd.DataFrame(res['trades'])
                rows.append({'modelo': name, 'ic_oos': ic, 'comision': fee_name, 'apalancamiento': L,
                             'retorno_total': st['total_return'], 'cagr': st['cagr'],
                             'sharpe': st['sharpe'], 'max_dd': st['max_drawdown'],
                             'liquidado': str(res['liquidated_at']) if res['liquidated_at'] is not None else '',
                             'trades': len(tr),
                             'win_rate': float((tr['side'] * (tr['exit'] / tr['entry'] - 1) > 0).mean()) if len(tr) else np.nan})
    df = pd.DataFrame(rows)
    out_dir = os.path.join(project_root(), 'backtest_results')
    df.to_csv(os.path.join(out_dir, f'compare_models_{args.symbol}_{args.timeframe}_h{args.hold_hours:g}.csv'), index=False)

    pd.set_option('display.width', 220)
    print(f"\n{args.symbol} {args.timeframe} | tenencia {args.hold_hours:g}h | extremo {args.q:.0%} | "
          f"{fb.index[0].date()} a {fb.index[-1].date()} ({years:.1f} años)")
    print(f"Buy & Hold: retorno {bh_st['total_return']:+.0%}, CAGR {bh_st['cagr']:+.1%}, Sharpe {bh_st['sharpe']:.2f}, maxDD {bh_st['max_drawdown']:.0%}")

    base = df[(df.comision == 'futures_taker_vip0') & (df.apalancamiento == 1)].sort_values('sharpe', ascending=False)
    print("\n1) MODELOS -- futuros taker VIP0, 1x")
    print(base[['modelo', 'ic_oos', 'retorno_total', 'cagr', 'sharpe', 'max_dd', 'trades', 'win_rate']].round(3).to_string(index=False))

    print("\nPesos promedio del ensemble adaptativo (por IC de los 180 días previos):")
    print(weight_hist[list(MODELS)].mean().round(3).to_string())

    best = base.iloc[0]['modelo']
    print(f"\n2) COMISIONES BINANCE -- {best}, 1x")
    t = df[(df.modelo == best) & (df.apalancamiento == 1)]
    print(t[['comision', 'retorno_total', 'cagr', 'sharpe', 'max_dd']].round(3).to_string(index=False))

    print(f"\n3) APALANCAMIENTO -- {best}")
    t = df[(df.modelo == best) & df.comision.isin(['futures_taker_vip0', 'futures_maker_vip0'])]
    print(t[['comision', 'apalancamiento', 'retorno_total', 'cagr', 'sharpe', 'max_dd', 'liquidado']].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
