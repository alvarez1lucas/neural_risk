# neural_risk/data/binance_vision.py
"""
Descarga de datos históricos de Binance USDⓈ-M Futures desde
data.binance.vision (gratis, sin API key), con caché local.

Por qué: las features del pipeline salen todas de velas OHLCV -- casi sin
información predictiva (ver diagnóstico de IC). Estos datos agregan
información que NO está en el precio:
- klines de futuros con taker_buy_volume: flujo agresor REAL (compras vs
  ventas a mercado), no el proxy "color de la vela".
- funding rate (cada 8h): costo de mantener longs/shorts apalancados.
- premium index klines: basis perp vs índice spot.
- metrics (cada 5 min): open interest, ratio long/short de top traders y
  de cuentas, ratio de volumen taker long/short.

La API REST pública solo da 30 días de open interest/ratios; el
histórico largo existe únicamente en estos archivos.

Todos los timestamps se devuelven en UTC naive, al INICIO de la barra.
"""

import io
import os
import zipfile
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import pandas as pd

BASE_URL = "https://data.binance.vision/data/futures/um"
KLINE_COLS = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
              'quote_volume', 'count', 'taker_buy_volume', 'taker_buy_quote_volume', 'ignore']


def _months(start: str, end: str) -> List[str]:
    return [p.strftime('%Y-%m') for p in pd.period_range(start, end, freq='M')]


def _days(start: str, end: str) -> List[str]:
    return [d.strftime('%Y-%m-%d') for d in pd.date_range(start, end, freq='D')]


class BinanceVisionLoader:
    def __init__(self, cache_dir: str = "./data/binance_vision", max_workers: int = 8):
        self.cache_dir = cache_dir
        self.max_workers = max_workers

    # ------------------------------------------------------------------
    def _fetch(self, rel_path: str) -> Optional[bytes]:
        """Descarga (o lee de caché) un zip. None si no existe en el servidor."""
        local = os.path.join(self.cache_dir, rel_path)
        missing_marker = local + '.missing'
        if os.path.exists(local):
            with open(local, 'rb') as f:
                return f.read()
        if os.path.exists(missing_marker):
            return None
        try:
            with urllib.request.urlopen(f"{BASE_URL}/{rel_path}", timeout=60) as r:
                data = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                os.makedirs(os.path.dirname(local), exist_ok=True)
                open(missing_marker, 'w').close()
                return None
            raise
        os.makedirs(os.path.dirname(local), exist_ok=True)
        with open(local, 'wb') as f:
            f.write(data)
        return data

    @staticmethod
    def _read_zip_csv(data: bytes, header_names: Optional[List[str]] = None) -> pd.DataFrame:
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            raw = z.read(z.namelist()[0])
        first = raw.split(b'\n', 1)[0]
        has_header = not first[:1].isdigit()  # archivos viejos vienen sin header
        if has_header:
            return pd.read_csv(io.BytesIO(raw))
        return pd.read_csv(io.BytesIO(raw), header=None, names=header_names)

    def _fetch_many(self, rel_paths: List[str], header_names=None) -> pd.DataFrame:
        with ThreadPoolExecutor(self.max_workers) as ex:
            blobs = list(ex.map(self._fetch, rel_paths))
        frames = [self._read_zip_csv(b, header_names) for b in blobs if b is not None]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    # ------------------------------------------------------------------
    def klines(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """Velas de futuros perp con taker_buy_volume. start/end: 'YYYY-MM'."""
        paths = [f"monthly/klines/{symbol}/{interval}/{symbol}-{interval}-{m}.zip" for m in _months(start, end)]
        df = self._fetch_many(paths, KLINE_COLS)
        if df.empty:
            return df
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
        num = ['open', 'high', 'low', 'close', 'volume', 'quote_volume', 'count',
               'taker_buy_volume', 'taker_buy_quote_volume']
        return df[num].astype(float)

    def premium_index(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        """Premium index (perp vs índice spot) -- 'close' es el basis relativo."""
        paths = [f"monthly/premiumIndexKlines/{symbol}/{interval}/{symbol}-{interval}-{m}.zip"
                 for m in _months(start, end)]
        df = self._fetch_many(paths, KLINE_COLS)
        if df.empty:
            return df
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
        return df[['close']].astype(float).rename(columns={'close': 'premium'})

    def funding_rate(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        paths = [f"monthly/fundingRate/{symbol}/{symbol}-fundingRate-{m}.zip" for m in _months(start, end)]
        df = self._fetch_many(paths, ['calc_time', 'funding_interval_hours', 'last_funding_rate'])
        if df.empty:
            return df
        df['timestamp'] = pd.to_datetime(df['calc_time'], unit='ms').dt.floor('min')
        df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
        return df[['last_funding_rate']].astype(float).rename(columns={'last_funding_rate': 'funding_rate'})

    def metrics(self, symbol: str, start: str, end: str) -> pd.DataFrame:
        """OI y ratios long/short cada 5 min. start/end: 'YYYY-MM-DD'."""
        paths = [f"daily/metrics/{symbol}/{symbol}-metrics-{d}.zip" for d in _days(start, end)]
        df = self._fetch_many(paths)
        if df.empty:
            return df
        # create_time marca el FIN del intervalo de 5 min -> se usa tal cual
        # como momento en que el dato está disponible.
        df['timestamp'] = pd.to_datetime(df['create_time'])
        df = df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
        cols = ['sum_open_interest', 'sum_open_interest_value', 'count_toptrader_long_short_ratio',
                'sum_toptrader_long_short_ratio', 'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']
        return df[cols].apply(pd.to_numeric, errors='coerce')
