"""Conectores a Yahoo Finance y Binance para descargar datos OHLCV."""
import pandas as pd
import numpy as np
from typing import Optional

try:
    import ccxt
except ImportError:
    ccxt = None

try:
    import yfinance as yf
except ImportError:
    yf = None


class BinanceLoader:
    """
    Conector real a Binance via ccxt. Reemplaza el MOCK (np.random) que
    usa run_data_fetcher.py hoy. Formato de salida: DataFrame con
    columnas Open/High/Low/Close/Volume (capitalizadas), indexado por
    fecha -- el MISMO formato que AutomatedRiskEngine.prepare_asset_features
    espera.
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None):
        if ccxt is None:
            raise ImportError("Install ccxt: pip install ccxt")
        self.client = ccxt.binance({
            'apiKey': api_key, 'secret': api_secret, 'enableRateLimit': True
        })

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1h',
                   limit: int = 1000, since: Optional[int] = None) -> pd.DataFrame:
        """symbol: ej. 'BTC/USDT'. since: timestamp en ms (para paginar)."""
        raw = self.client.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
        df = pd.DataFrame(raw, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        return df

    def fetch_current_price(self, symbol: str) -> float:
        ticker = self.client.fetch_ticker(symbol)
        return float(ticker['last'])

    def fetch_full_history(self, symbol: str, timeframe: str = '1d',
                          total_candles: int = 2000, batch_size: int = 1000) -> pd.DataFrame:
        """
        Pagina hacia atrás para juntar más historico del que Binance
        devuelve en una sola llamada. Util para train_models.py.
        """
        all_frames = []
        since = None
        remaining = total_candles

        while remaining > 0:
            limit = min(batch_size, remaining)
            df_batch = self.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit, since=since)
            if df_batch.empty:
                break
            all_frames.append(df_batch)
            since = int(df_batch.index[0].timestamp() * 1000) - 1
            remaining -= len(df_batch)
            if len(df_batch) < limit:
                break

        if not all_frames:
            return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        combined = pd.concat(all_frames[::-1])
        combined = combined[~combined.index.duplicated(keep='first')]
        return combined.sort_index()


class YahooFinanceLoader:
    """Conector a Yahoo Finance via yfinance (benchmarks, activos tradicionales)."""

    def __init__(self):
        if yf is None:
            raise ImportError("Install yfinance: pip install yfinance")

    def fetch_ohlcv(self, ticker: str, period: str = '6mo', interval: str = '1d') -> pd.DataFrame:
        """ticker: ej. 'BTC-USD', 'SPY', 'GC=F'"""
        df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]