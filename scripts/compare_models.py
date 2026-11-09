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
import xgboost as xgb

from neural_risk.data.derivatives_features import build_derivatives_features
from neural_risk.research import (BINANCE_FEES, adaptive_ensemble, forward_return, simulate,
                                  stats, walk_forward, weighted_ensemble)
from backtest_intraday import load

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

MODELS = {
    'ridge': lambda: Ridge(alpha=10.0),
    'xgboost': lambda: xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.03,
                                        subsample=0.7, colsample_bytree=0.7, min_child_weight=50,
                                        reg_lambda=10.0, n_jobs=4, random_state=42),
    'mlp': lambda: MLPRegressor(hidden_layer_sizes=(32, 16), alpha=1e-3, learning_rate_init=1e-3,
                                max_iter=50, early_stopping=True, random_state=42),
}


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
            preds[name] = walk_forward(MODELS[name], X, y, hold, per_day * 30, per_day * 180, scale_y=True)
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
