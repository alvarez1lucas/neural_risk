# scripts/feature_diagnostics.py
"""
¿Hay señal, y alcanza para pagar los costos?

Para cada timeframe y horizonte:
1. IC (Spearman) de cada feature contra el retorno futuro, en ambas mitades.
2. Ventaja BRUTA fuera de muestra de operar los extremos de cada feature
   (decil superior/inferior, umbrales con ventana móvil pasada; el SIGNO se
   decide en la 1a mitad y se mide en la 2a).
3. Modelo lineal (ridge) walk-forward con todas las features: techo de lo
   que se puede extraer combinándolas.
Todo comparado contra costos ida y vuelta: spot taker, futuros taker,
futuros maker.

Entrada en el OPEN de la barra siguiente, salida al CLOSE de t+h.

Uso:
    python scripts/feature_diagnostics.py --symbol BTCUSDT --timeframes 5m 15m 1h 4h
"""

import argparse
import json
import os
import sys
import warnings

from _pathutils import project_root

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from neural_risk.data.derivatives_features import build_derivatives_features
from neural_risk.research import extreme_signal, forward_return, ridge_walk_forward
from backtest_intraday import load

warnings.filterwarnings('ignore')
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Costo ida y vuelta (2 lados), en bps
COSTS_BPS = {
    'spot_taker': 2 * (10 + 5),      # 0.10% comisión + 0.05% slippage por lado
    'futures_taker': 2 * (5 + 1),    # 0.05% + ~1bp slippage en BTC perp
    'futures_maker': 2 * 2,          # 0.02% maker, sin slippage (pero con riesgo de no-fill)
}


def analyze(f: pd.DataFrame, tf: str, horizons, rng) -> dict:
    per_day = int(round(pd.Timedelta(days=1) / pd.Series(f.index).diff().median()))
    window = per_day * 30
    feats = [c for c in f.columns if not c.startswith('hour_')] + ['hour_sin', 'hour_cos']
    noise_cols = []
    for k in range(10):
        f[f'NOISE_{k}'] = rng.normal(size=len(f))
        noise_cols.append(f'NOISE_{k}')
    half = f.index[len(f) // 2]
    first, second = f.index < half, f.index >= half
    out = {'timeframe': tf, 'bars': len(f), 'horizons': {}}

    for h in horizons:
        y = forward_return(f, h)
        n_eff = y.notna().sum() / h
        thr = 2 / np.sqrt(n_eff)
        rows = []
        for c in feats + noise_cols:
            x = f[c]
            m = x.notna() & y.notna()
            if m.sum() < 1000 or x[m].std() == 0:
                continue
            ic1 = spearmanr(x[m & first], y[m & first])[0]
            ic2 = spearmanr(x[m & second], y[m & second])[0]
            sig = extreme_signal(x, window) * np.sign(ic1)
            s2 = sig[second & y.notna()]
            trades = s2[s2 != 0]
            gross = (trades * y.loc[trades.index]).mean() * 1e4 if len(trades) else np.nan
            rows.append({'feature': c, 'IC_1a': ic1, 'IC_2a': ic2,
                         'gross_bps_2a': gross, 'n_signals_2a': len(trades)})
        r = pd.DataFrame(rows).set_index('feature')
        noise = r.loc[r.index.str.startswith('NOISE')]
        real = r.loc[~r.index.str.startswith('NOISE')]

        # Modelo combinado
        pred = ridge_walk_forward(f[feats], y, h, retrain_every=per_day * 30, min_train=per_day * 180)
        m = pred.notna() & y.notna()
        ic_model = spearmanr(pred[m], y[m])[0] if m.sum() > 1000 else np.nan
        sig = extreme_signal(pred, window)
        st = sig[m & (sig != 0)]
        model_gross = (st * y.loc[st.index]).mean() * 1e4 if len(st) else np.nan
        # Ventaja bruta promedio del mejor decil en función de lo extremo del umbral
        model_by_q = {}
        for q in (0.1, 0.05, 0.02, 0.01):
            sq = extreme_signal(pred, window, q)
            tq = sq[m & (sq != 0)]
            model_by_q[q] = {'gross_bps': float((tq * y.loc[tq.index]).mean() * 1e4) if len(tq) else None,
                             'n': int(len(tq))}

        out['horizons'][h] = {
            'n_eff': float(n_eff), 'ic_noise_threshold': float(thr),
            'noise_max_abs_ic_2a': float(noise['IC_2a'].abs().max()),
            'noise_max_abs_gross_bps': float(noise['gross_bps_2a'].abs().max()),
            'avg_abs_move_bps': float(y.abs().mean() * 1e4),
            'features': real.sort_values('IC_2a', key=np.abs, ascending=False).round(4).to_dict('index'),
            'model': {'ic_oos': float(ic_model), 'gross_bps_top_decile': float(model_gross),
                      'by_quantile': model_by_q},
        }

        print(f"\n=== {tf} | horizonte {h} barras | n_eff ~{n_eff:.0f} | umbral IC azar {thr:.3f} "
              f"| movimiento medio {y.abs().mean() * 1e4:.0f} bps ===")
        print(f"Ruido: |IC 2a mitad| max {noise['IC_2a'].abs().max():.3f}, "
              f"|ventaja| max {noise['gross_bps_2a'].abs().max():.1f} bps")
        show = real.assign(consistente=np.sign(real['IC_1a']) == np.sign(real['IC_2a']))
        show = show.sort_values('gross_bps_2a', ascending=False).head(8)
        print(show.round(3).to_string())
        print(f"MODELO ridge walk-forward: IC OOS {ic_model:.3f} | ventaja bruta decil extremo {model_gross:.1f} bps")
        for q, v in model_by_q.items():
            print(f"   extremo {int(q * 100)}%: {v['gross_bps'] if v['gross_bps'] is None else round(v['gross_bps'], 1)} bps "
                  f"({v['n']} barras con señal)")
        print("   costos ida y vuelta: " + ", ".join(f"{k} {v} bps" for k, v in COSTS_BPS.items()))

    for c in noise_cols:
        del f[c]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symbol', default='BTCUSDT')
    p.add_argument('--timeframes', nargs='+', default=['5m', '15m', '1h', '4h'])
    p.add_argument('--horizons', nargs='+', type=int, default=[1, 4, 12])
    args = p.parse_args()

    k5, prem, fund, met = load(args.symbol)

    rng = np.random.default_rng(0)
    results = {'symbol': args.symbol, 'costs_bps': COSTS_BPS, 'timeframes': {}}
    for tf in args.timeframes:
        f = build_derivatives_features(k5, prem, fund, met, tf)
        results['timeframes'][tf] = analyze(f, tf, args.horizons, rng)

    out_dir = os.path.join(project_root(), 'backtest_results')
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'feature_diagnostics_{args.symbol}.json')
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=1, default=str)
    print(f"\nResultados completos en {path}")


if __name__ == '__main__':
    main()
