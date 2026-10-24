# neural_risk/data/derivatives_features.py
"""
Features de flujo y derivados a partir de los datos de Binance Vision
(ver binance_vision.py). Pensadas para timeframes intradía (5m a 4h).

Todas son CAUSALES: el valor de la barra que abre en t usa solo datos
disponibles al CIERRE de esa barra (t + intervalo). Premium index y métricas
de OI/ratios se toman como disponibles 5 min después de su timestamp
(cierre de su vela de 5 min); el funding, a su hora de liquidación.
"""

import numpy as np
import pandas as pd


def resample_klines(k5: pd.DataFrame, rule: str) -> pd.DataFrame:
    if rule in ('5m', '5min'):
        return k5.copy()
    agg = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
           'quote_volume': 'sum', 'count': 'sum', 'taker_buy_volume': 'sum',
           'taker_buy_quote_volume': 'sum'}
    return k5.resample(rule.replace('m', 'min') if rule.endswith('m') else rule).agg(agg).dropna(subset=['close'])


def _asof_close(series_or_df, bar_index: pd.DatetimeIndex, bar_delta: pd.Timedelta):
    """Último valor disponible al cierre de cada barra (sin look-ahead)."""
    src = series_or_df.sort_index()
    aligned = src.reindex(bar_index + bar_delta, method='ffill')
    aligned.index = bar_index
    return aligned


def _z(x: pd.Series, window: int) -> pd.Series:
    return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-12)


def build_derivatives_features(k5: pd.DataFrame, premium5: pd.DataFrame, funding: pd.DataFrame,
                               metrics5: pd.DataFrame, rule: str = '1h') -> pd.DataFrame:
    bars = resample_klines(k5, rule)
    delta = pd.Series(bars.index).diff().median()
    per_day = int(round(pd.Timedelta(days=1) / delta))
    close = bars['close']
    ret = np.log(close).diff()
    f = pd.DataFrame(index=bars.index)

    # --- Precio (controles) ---
    for k in (1, 3, 12, 48):
        f[f'mom_{k}'] = close.pct_change(k)
    f['vol_1d'] = ret.rolling(per_day).std()
    f['range_z'] = _z(np.log(bars['high'] / bars['low']), per_day * 7)

    # --- Flujo agresor REAL (taker buy vs sell) ---
    buy = bars['taker_buy_volume']
    sell = bars['volume'] - buy
    imb = (buy - sell) / bars['volume'].replace(0, np.nan)
    f['taker_imb'] = imb
    for k in (3, 12):
        f[f'taker_imb_{k}'] = (buy - sell).rolling(k).sum() / bars['volume'].rolling(k).sum()
    f['taker_imb_z'] = _z(imb, per_day * 7)
    f['volume_z'] = _z(np.log(bars['quote_volume'].replace(0, np.nan)), per_day * 7)
    f['trade_size_z'] = _z(np.log(bars['quote_volume'] / bars['count'].replace(0, np.nan)), per_day * 7)
    # Flujo que NO movió el precio (absorción): compra agresiva con retorno negativo, etc.
    f['flow_vs_ret'] = f['taker_imb_12'] - _z(close.pct_change(12), per_day * 7) * f['taker_imb_12'].rolling(per_day * 7).std()

    # --- Basis perp vs spot (premium index) ---
    # FIX look-ahead: las velas del premium index están fechadas por su
    # APERTURA; el valor recién se conoce al cierre (+5 min).
    prem_src = premium5['premium'].copy()
    prem_src.index = prem_src.index + pd.Timedelta(minutes=5)
    prem = _asof_close(prem_src, bars.index, delta)
    f['premium'] = prem
    f['premium_z'] = _z(prem, per_day * 7)
    f['premium_chg'] = prem.diff(max(1, per_day // 6))

    # --- Funding ---
    fr = _asof_close(funding['funding_rate'], bars.index, delta)
    f['funding'] = fr
    f['funding_z'] = _z(fr, per_day * 30)

    # --- Open interest y posicionamiento ---
    m = metrics5.copy()
    m['sum_open_interest'] = m['sum_open_interest'].replace(0, np.nan)
    # Conservador: no está documentado si create_time marca el inicio o el
    # fin de la ventana de 5 min -> se asume disponible 5 min después.
    m.index = m.index + pd.Timedelta(minutes=5)
    m = _asof_close(m, bars.index, delta)
    oi = m['sum_open_interest']
    for k in (max(1, per_day // 24), max(1, per_day // 4), per_day):
        f[f'oi_chg_{k}'] = np.log(oi).diff(k)
    k4 = max(1, per_day // 4)
    # OI sube con precio subiendo = longs nuevos; OI baja con precio subiendo = cierre de shorts
    f['oi_x_price'] = np.sign(close.pct_change(k4)) * f[f'oi_chg_{k4}']
    f['top_ls_acct'] = np.log(m['count_toptrader_long_short_ratio'])
    f['top_ls_pos'] = np.log(m['sum_toptrader_long_short_ratio'])
    f['crowd_ls'] = np.log(m['count_long_short_ratio'])
    f['crowd_ls_z'] = _z(f['crowd_ls'], per_day * 7)
    f['smart_vs_crowd'] = f['top_ls_pos'] - f['crowd_ls']
    f['taker_ls_ratio'] = np.log(m['sum_taker_long_short_vol_ratio'].clip(lower=1e-3))

    # --- Estacionalidad intradía ---
    f['hour_sin'] = np.sin(2 * np.pi * bars.index.hour / 24)
    f['hour_cos'] = np.cos(2 * np.pi * bars.index.hour / 24)

    f = f.replace([np.inf, -np.inf], np.nan)
    f.attrs['close'] = close
    f.attrs['open'] = bars['open']
    f.attrs['high'] = bars['high']
    f.attrs['low'] = bars['low']
    return f
