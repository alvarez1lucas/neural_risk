# scripts/intraday_grid.py
"""
Grilla de robustez para backtest_intraday: timeframe x horizonte/hold x
umbral de señal, con costos de futuros (taker y maker) y funding.
Un resultado que solo funciona en UNA combinación es sobreajuste; uno real
debería verse en una zona amplia de la grilla.

Uso:
    python scripts/intraday_grid.py --symbol BTCUSDT --timeframes 5m 15m 30m 1h 4h
"""
import argparse
import json
import os
import sys
import time

from _pathutils import project_root

import numpy as np
import pandas as pd

from neural_risk.data.derivatives_features import build_derivatives_features
from feature_diagnostics import ridge_walk_forward, forward_return
from backtest_intraday import load, simulate, stats, COST_PER_SIDE

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

HOLD_HOURS = [1, 4, 12, 24]   # tenencia en HORAS, se traduce a barras por timeframe
QS = [0.01, 0.02, 0.05, 0.10]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='BTCUSDT')
    p.add_argument('--timeframes', nargs='+', default=['15m', '30m', '1h', '4h'])
    args = p.parse_args()

    k5, prem, fund, met = load(args.symbol)
    rows = []
    for tf in args.timeframes:
        t0 = time.time()
        f = build_derivatives_features(k5, prem, fund, met, tf)
        per_day = int(round(pd.Timedelta(days=1) / pd.Series(f.index).diff().median()))
        bars_per_hour = per_day / 24
        for hh in HOLD_HOURS:
            hold = int(round(hh * bars_per_hour))
            if hold < 1:
                continue
            y = forward_return(f, hold)
            pred = ridge_walk_forward(f, y, hold, retrain_every=per_day * 30, min_train=per_day * 180)
            start = pred.first_valid_index()
            fb = f.loc[start:]
            fb.attrs = {'close': f.attrs['close'].loc[start:], 'open': f.attrs['open'].loc[start:]}
            pb = pred.loc[start:]
            for q in QS:
                for cname in ['sin_costos', 'futures_taker', 'futures_maker']:
                    cost = 0.0 if cname == 'sin_costos' else COST_PER_SIDE[cname]
                    res = simulate(fb, pb, hold, q, cost, fund['funding_rate'], per_day)
                    st = stats(res['equity'], per_day * 365)
                    tr = pd.DataFrame(res['trades'])
                    gross = float((tr['side'] * (tr['exit'] / tr['entry'] - 1)).mean() * 1e4) if len(tr) else np.nan
                    long_share = float((tr['side'] > 0).mean()) if len(tr) else np.nan
                    yearly = res['equity'].groupby(res['equity'].index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
                    rows.append({'tf': tf, 'hold_h': hh, 'hold_bars': hold, 'q': q, 'costs': cname,
                                 'total_return': st['total_return'], 'sharpe': st['sharpe'],
                                 'max_dd': st['max_drawdown'], 'n_trades': len(tr),
                                 'trades_per_month': len(tr) / (len(fb) / per_day / 30),
                                 'gross_bps_trade': gross, 'long_share': long_share,
                                 'years_positive': int((yearly > 0).sum()), 'years': int(len(yearly))})
        print(f"{tf}: listo en {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = os.path.join(project_root(), 'backtest_results', f'intraday_grid_{args.symbol}.csv')
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    for cname in ['sin_costos', 'futures_taker', 'futures_maker']:
        piv = df[df.costs == cname].pivot_table(index=['tf', 'hold_h'], columns='q', values='sharpe')
        print(f"\nSharpe -- {cname}  (filas: timeframe, tenencia en horas | columnas: umbral)")
        print(piv.round(2).to_string())
    piv = df[df.costs == 'futures_taker'].pivot_table(index=['tf', 'hold_h'], columns='q', values='gross_bps_trade')
    print("\nVentaja bruta por trade (bps) -- costo ida y vuelta futures taker = 12 bps")
    print(piv.round(1).to_string())
    piv = df[df.costs == 'futures_taker'].pivot_table(index=['tf', 'hold_h'], columns='q', values='trades_per_month')
    print("\nTrades por mes")
    print(piv.round(0).to_string())
    print(f"\nGrilla completa en {out}")


if __name__ == '__main__':
    main()
