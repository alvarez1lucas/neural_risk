# scripts/backtest_intraday.py
"""
Backtest intradía sobre futuros perpetuos (Binance USDⓈ-M) con las features
de flujo/derivados (neural_risk/data/derivatives_features.py).

- Señal: ridge walk-forward sobre todas las features (reentreno mensual,
  solo con datos cuyo target ya se conocía).
- Regla: si la predicción cae en el extremo `q` (umbral = cuantil móvil de
  los últimos 30 días de predicciones) se abre LONG/SHORT al OPEN de la
  barra siguiente y se mantiene `hold` barras. Una señal en la misma
  dirección extiende la tenencia; una opuesta da vuelta la posición.
- Costos: comisión + slippage por lado según escenario, y FUNDING real cada
  8h (los longs pagan funding positivo, los shorts lo cobran).
- Exposición fija de 1x el capital (sin apalancamiento) para que las
  métricas sean comparables con Buy & Hold.

Uso:
    python scripts/backtest_intraday.py --symbol BTCUSDT --timeframe 1h --hold 12 --q 0.02
"""

import argparse
import json
import os
import sys

from _pathutils import project_root

import numpy as np
import pandas as pd

from neural_risk.data.binance_vision import BinanceVisionLoader
from neural_risk.data.derivatives_features import build_derivatives_features
from neural_risk.research import COST_PER_SIDE, forward_return, ridge_walk_forward, simulate, stats

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass


def load(symbol: str):
    """(klines_5m, premium_5m, funding, metrics_5m) desde data/binance_vision."""
    return BinanceVisionLoader(os.path.join(project_root(), 'data', 'binance_vision')).load_symbol(symbol)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='BTCUSDT')
    p.add_argument('--timeframe', default='1h')
    p.add_argument('--hold', type=int, default=12)
    p.add_argument('--q', type=float, default=0.02)
    p.add_argument('--horizon', type=int, default=None, help='Horizonte del target (default = hold)')
    args = p.parse_args()
    h = args.horizon or args.hold

    k5, prem, fund, met = load(args.symbol)
    f = build_derivatives_features(k5, prem, fund, met, args.timeframe)
    per_day = int(round(pd.Timedelta(days=1) / pd.Series(f.index).diff().median()))
    per_year = per_day * 365
    feats = list(f.columns)
    y = forward_return(f, h)
    pred = ridge_walk_forward(f[feats], y, h, retrain_every=per_day * 30, min_train=per_day * 180)

    start = pred.first_valid_index()
    f_bt = f.loc[start:]
    f_bt.attrs = {k: v.loc[start:] for k, v in f.attrs.items()}
    pred_bt = pred.loc[start:]
    bh = f_bt.attrs['close'] / f_bt.attrs['close'].iloc[0]

    summary = {'symbol': args.symbol, 'timeframe': args.timeframe, 'hold': args.hold, 'q': args.q,
               'horizon': h, 'period': [str(f_bt.index[0]), str(f_bt.index[-1])],
               'buy_and_hold': stats(bh, per_year), 'scenarios': {}}
    print(f"\n{args.symbol} {args.timeframe} | hold {args.hold} | extremo {args.q:.0%} | "
          f"{f_bt.index[0].date()} a {f_bt.index[-1].date()}")
    print(f"Buy & Hold: {summary['buy_and_hold']}")

    out_dir = os.path.join(project_root(), 'backtest_results', f'intraday_{args.symbol}_{args.timeframe}_h{args.hold}_q{args.q}')
    os.makedirs(out_dir, exist_ok=True)
    curves = {'buy_and_hold': bh}
    for name, cost in [('sin_costos', 0.0)] + list(COST_PER_SIDE.items()):
        res = simulate(f_bt, pred_bt, args.hold, args.q, cost, fund['funding_rate'], per_day)
        eq, trades = res['equity'], res['trades']
        st = stats(eq, per_year)
        tr = pd.DataFrame(trades)
        if len(tr):
            tr['ret'] = tr['side'] * (tr['exit'] / tr['entry'] - 1)
        yearly = eq.groupby(eq.index.year).apply(lambda s: s.iloc[-1] / s.iloc[0] - 1)
        st.update({'n_trades': int(len(tr)),
                   'trades_per_month': float(len(tr) / (len(eq) / per_day / 30)),
                   'win_rate': float((tr['ret'] > 0).mean()) if len(tr) else 0.0,
                   'avg_trade_gross_bps': float(tr['ret'].mean() * 1e4) if len(tr) else 0.0,
                   'funding_total': float(tr['funding'].sum()) if len(tr) else 0.0,
                   'exposure': None,
                   'yearly': {int(k): round(float(v), 4) for k, v in yearly.items()}})
        summary['scenarios'][name] = st
        curves[name] = eq
        print(f"  {name:14s} ret {st['total_return']:+8.1%} | sharpe {st['sharpe']:5.2f} | maxDD {st['max_drawdown']:6.1%} "
              f"| trades {st['n_trades']} ({st['trades_per_month']:.1f}/mes) | gross/trade {st['avg_trade_gross_bps']:.1f} bps "
              f"| funding {st['funding_total']:+.3f} | por año {st['yearly']}")
        if name == 'futures_taker' and len(tr):
            tr.to_csv(os.path.join(out_dir, 'trades_futures_taker.csv'), index=False)

    pd.DataFrame(curves).to_csv(os.path.join(out_dir, 'equity.csv'))
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=1, default=str)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, eq in curves.items():
        ax.plot(eq.index, eq.values, label=name, lw=1.3 if name != 'buy_and_hold' else 1, ls='--' if name == 'buy_and_hold' else '-')
    ax.set_yscale('log')
    ax.set_title(f'{args.symbol} {args.timeframe} hold={args.hold} q={args.q}')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'equity.png'), dpi=110)
    print(f"Resultados en {out_dir}")


if __name__ == '__main__':
    main()
