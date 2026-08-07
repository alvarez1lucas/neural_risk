# scripts/run_data_fetcher.py
"""
LAYER 1: DATA FETCHER
ESTADO: guarda columna 'open' (faltaba). Intervalo lee de config.yaml.
"""

import time
import logging
import pandas as pd
import sqlite3
import yaml
import os
from datetime import datetime
from typing import Dict, List
import numpy as np

from _pathutils import resolve_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [DATA_FETCHER] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def _resolve_env(value):
    """Resuelve placeholders '${VAR}' del config.yaml contra variables de entorno."""
    if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
        return os.environ.get(value[2:-1])
    return value


class DataFetcher:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_path = resolve_path(config_path)  # FIX (#3): robusto al CWD
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.exchange = self.config['exchanges']['primary']
        self.assets = self.config['exchanges']['assets']
        self.quote = self.config['exchanges']['quote_currency']
        self.db_path = resolve_path(self.config['database']['path'])  # FIX (#3)

        # NUEVO: conexion real a Binance via ccxt (BinanceLoader), detras
        # de un flag explicito en config.yaml (exchanges.use_mock). Por
        # defecto sigue en True -- nadie pierde el modo mock sin pedirlo.
        self.use_mock = self.config['exchanges'].get('use_mock', True)
        self.timeframe = self.config['exchanges'].get('timeframe', '1m')
        self.binance_loader = None

        if not self.use_mock:
            try:
                from neural_risk.data.loaders import BinanceLoader
                creds = self.config['exchanges'].get('credentials', {})
                api_key = _resolve_env(creds.get('api_key'))
                api_secret = _resolve_env(creds.get('api_secret'))
                self.binance_loader = BinanceLoader(api_key=api_key, api_secret=api_secret)
                logger.info("BinanceLoader real inicializado (use_mock=false)")
            except Exception as e:
                logger.error(f"No se pudo inicializar BinanceLoader real, cayendo a MOCK: {e}")
                self.use_mock = True

        self._init_database()
        self._init_cache()
        logger.info(f"DataFetcher initialized: {self.exchange}, assets={self.assets}, use_mock={self.use_mock}")

    def _init_database(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY, timestamp DATETIME, asset TEXT,
                open REAL, price REAL, volume REAL, high REAL, low REAL,
                UNIQUE(timestamp, asset)
            )
        ''')
        try:
            cursor.execute('ALTER TABLE market_data ADD COLUMN open REAL')
        except sqlite3.OperationalError:
            pass
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_asset_timestamp ON market_data(asset, timestamp)')
        conn.commit()
        conn.close()

    def _init_cache(self):
        self.cache = {}
        self.cache_timestamps = {}

    def fetch_from_api(self) -> Dict[str, Dict]:
        if self.use_mock:
            return self._fetch_mock()
        return self._fetch_real()

    def _fetch_mock(self) -> Dict[str, Dict]:
        try:
            prices = {}
            for asset in self.assets:
                base_price = 45000 if asset == "BTC" else 2500 if asset == "ETH" else 100
                open_price = base_price * (1 + np.random.randn() * 0.005)
                close_price = open_price * (1 + np.random.randn() * 0.01)
                prices[asset] = {
                    'open': open_price, 'price': close_price,
                    'volume': np.random.uniform(1e6, 1e7),
                    'high': max(open_price, close_price) * 1.005,
                    'low': min(open_price, close_price) * 0.995
                }
            logger.info(f"[MOCK] Fetched prices: {list(prices.keys())}")
            return prices
        except Exception as e:
            logger.error(f"API fetch error (mock): {e}")
            return {}

    def _fetch_real(self) -> Dict[str, Dict]:
        """
        NUEVO: usa BinanceLoader (ccxt) para traer la última vela real
        de cada activo. Si un activo puntual falla (símbolo no listado,
        rate limit, etc.), se loguea y se sigue con el resto -- no se
        aborta todo el ciclo por un solo activo.
        """
        prices = {}
        for asset in self.assets:
            symbol = f"{asset}/{self.quote}"
            try:
                df = self.binance_loader.fetch_ohlcv(symbol, timeframe=self.timeframe, limit=1)
                if df.empty:
                    logger.warning(f"Sin datos de Binance para {symbol}")
                    continue
                last = df.iloc[-1]
                prices[asset] = {
                    'open': float(last['Open']), 'price': float(last['Close']),
                    'volume': float(last['Volume']), 'high': float(last['High']),
                    'low': float(last['Low'])
                }
            except Exception as e:
                logger.error(f"Error real fetch {symbol}: {e}")
                continue

        logger.info(f"[REAL] Fetched prices: {list(prices.keys())}")
        return prices

    def cache_prices(self, prices: Dict[str, Dict]):
        for asset, data in prices.items():
            self.cache[asset] = data
            self.cache_timestamps[asset] = datetime.now()

    def save_to_db(self, prices: Dict[str, Dict]):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            timestamp = datetime.now()
            for asset, data in prices.items():
                try:
                    cursor.execute('''
                        INSERT INTO market_data (timestamp, asset, open, price, volume, high, low)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (timestamp, asset, data['open'], data['price'], data['volume'], data['high'], data['low']))
                except sqlite3.IntegrityError:
                    pass
            conn.commit()
            conn.close()
            logger.info(f"Saved to DB: {len(prices)} assets")
        except Exception as e:
            logger.error(f"DB save error: {e}")

    def run(self, interval_seconds: int = None):
        if interval_seconds is None:
            interval_seconds = self.config.get('cycle_timing', {}).get('data_fetcher_interval', 60)
        logger.info(f"Starting data fetcher loop (interval={interval_seconds}s)")
        while True:
            try:
                prices = self.fetch_from_api()
                if prices:
                    self.cache_prices(prices)
                    self.save_to_db(prices)
                    logger.info(f"Cycle complete: {len(prices)} assets cached")
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(interval_seconds)

    def get_latest_price(self, asset: str) -> float:
        return self.cache.get(asset, {}).get('price', 0)

    def get_last_n_prices(self, asset: str, n: int = 100) -> List[float]:
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT price FROM market_data WHERE asset = ? ORDER BY timestamp DESC LIMIT ?"
            df = pd.read_sql(query, conn, params=(asset, n))
            conn.close()
            return df['price'].values[::-1]
        except:
            return []


if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.run()