# neural_risk/research/backtest.py
"""
Backtest de futuros perpetuos (Binance USDⓈ-M) a partir de una predicción.

- Si la predicción cae en el extremo `q` (umbral = cuantil móvil de los
  últimos 30 días) se abre LONG/SHORT al OPEN de la barra siguiente y se
  mantiene `hold` barras. Una señal en la misma dirección extiende la
  tenencia; una opuesta da vuelta la posición.
- Costos: comisión + slippage por lado, y FUNDING real liquidado en cada
  barra (los longs pagan funding positivo, los shorts lo cobran).
- Apalancamiento con margen cruzado y liquidación intrabarra.
"""

import numpy as np
import pandas as pd

# Costo por lado (comisión + slippage), escenarios simplificados.
COST_PER_SIDE = {'spot_taker': 0.0015, 'futures_taker': 0.0006, 'futures_maker': 0.0002}

# Comisiones Binance (VIP 0). Futuros USDⓈ-M: maker 0.02% / taker 0.05%;
# con descuento BNB (10%): 0.018% / 0.045%. Spot: 0.10%.
# + slippage estimado por lado (BTC perp es muy líquido).
BINANCE_FEES = {
    'spot_taker_vip0':        0.0010 + 0.0005,
    'futures_taker_vip0':     0.0005 + 0.0001,
    'futures_taker_bnb':      0.00045 + 0.0001,
    'futures_maker_vip0':     0.0002,
    'futures_maker_bnb':      0.00018,
}


def simulate(f: pd.DataFrame, pred: pd.Series, hold: int, q: float, cost: float,
             funding: pd.Series, per_day: int, leverage: float = 1.0,
             maint_margin: float = 0.005) -> dict:
    """
    f:        features con attrs 'open', 'close' (y opcional 'high', 'low').
    pred:     predicción alineada con f.index.
    funding:  serie de funding rate indexada por hora de liquidación.
    leverage: exposición = leverage x equity (margen cruzado sobre toda la
              cuenta). Si dentro de una barra el movimiento adverso (High/Low)
              deja el equity por debajo del margen de mantenimiento, la cuenta
              se liquida (equity -> 0) y la simulación termina.

    Devuelve {'equity': Series, 'trades': list[dict], 'liquidated_at': ts|None}.
    """
    close, open_ = f.attrs['close'], f.attrs['open']
    window = per_day * 30
    hi = pred.rolling(window, min_periods=window // 2).quantile(1 - q)
    lo = pred.rolling(window, min_periods=window // 2).quantile(q)
    signal = np.where(pred > hi, 1, np.where(pred < lo, -1, 0))

    idx = f.index
    n = len(idx)
    # funding liquidado dentro de cada barra (calc_time en (open, open+delta])
    fund_per_bar = funding.groupby(np.searchsorted(idx.values, funding.index.values, side='right') - 1).sum()
    fund_arr = np.zeros(n)
    valid = (fund_per_bar.index >= 0) & (fund_per_bar.index < n)
    fund_arr[fund_per_bar.index[valid]] = fund_per_bar.values[valid]

    o, c = open_.values, close.values
    hi_px = f.attrs.get('high', close).values
    lo_px = f.attrs.get('low', close).values
    L = leverage
    liquidated_at = None
    pos = 0            # posición vigente durante la barra i
    bars_left = 0
    pending = 0        # señal del cierre i-1, se ejecuta al open de i
    equity = np.ones(n)
    e = 1.0
    trades, cur = [], None
    for i in range(n):
        # 1) ejecutar al open de i
        if pending != 0:
            if pos == pending:
                bars_left = hold  # extiende
            else:
                if pos != 0:
                    e *= 1 + L * pos * (o[i] / c[i - 1] - 1)  # gap cierre anterior -> open
                    e *= (1 - L * cost)
                    cur['exit'], cur['exit_time'] = o[i], idx[i]
                    trades.append(cur)
                e *= (1 - L * cost)
                pos, bars_left = pending, hold
                cur = {'side': pos, 'entry': o[i], 'entry_time': idx[i], 'funding': 0.0}
            pending = 0
        # 2) PnL de la barra i
        if pos != 0:
            ref = o[i] if (cur is not None and cur['entry_time'] == idx[i]) else c[i - 1]
            worst = lo_px[i] if pos > 0 else hi_px[i]
            if 1 + L * pos * (worst / ref - 1) <= maint_margin * L:
                e = 0.0
                liquidated_at = idx[i]
                equity[i:] = 0.0
                break
            e *= 1 + L * pos * (c[i] / ref - 1)
            if fund_arr[i]:
                e *= 1 - L * pos * fund_arr[i]
                cur['funding'] -= pos * fund_arr[i]
            bars_left -= 1
            pending_exit = bars_left <= 0
        else:
            pending_exit = False
        # 3) señal al cierre de i
        s = signal[i]
        if s != 0 and not np.isnan(pred.iloc[i]):
            pending = s
        elif pending_exit:
            # salida por tiempo al cierre de i
            e *= (1 - L * cost)
            cur['exit'], cur['exit_time'] = c[i], idx[i]
            trades.append(cur)
            pos, cur = 0, None
        equity[i] = e
    eq = pd.Series(equity, index=idx)
    return {'equity': eq, 'trades': trades, 'liquidated_at': liquidated_at}


def stats(eq: pd.Series, per_year: float) -> dict:
    """Retorno total, CAGR, Sharpe anualizado y máximo drawdown de una curva de equity."""
    r = eq.pct_change().dropna()
    years = len(r) / per_year
    tot = eq.iloc[-1] / eq.iloc[0] - 1
    dd = (eq / eq.cummax() - 1).min()
    return {'total_return': float(tot), 'cagr': float((1 + tot) ** (1 / years) - 1) if years > 0 and tot > -1 else -1.0,
            'sharpe': float(r.mean() / r.std() * np.sqrt(per_year)) if r.std() > 0 else 0.0,
            'max_drawdown': float(dd)}


def trades_frame(trades: list) -> pd.DataFrame:
    """Lista de trades -> DataFrame con el retorno bruto de cada uno ('ret')."""
    tr = pd.DataFrame(trades)
    if len(tr):
        tr['ret'] = tr['side'] * (tr['exit'] / tr['entry'] - 1)
    return tr
