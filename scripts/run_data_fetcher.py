# scripts/run_data_fetcher.py
"""
LAYER 1: DATA FETCHER
Ejecutar en background: python scripts/run_data_fetcher.py

Responsabilidades:
- Obtener precio actual cada minuto
- Guardar en cache (Redis/memory) para acceso rápido
- Mantener histórico en SQLite
- No bloquea otros servicios
"""

import time
import logging
import pandas as pd
import sqlite3
import yaml
import os
from datetime import datetime, timedelta
from typing import Dict, List
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [DATA_FETCHER] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataFetcher:
    """Obtiene y cachea datos de mercado"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.exchange = self.config['exchanges']['primary']
        self.assets = self.config['exchanges']['assets']
        self.quote = self.config['exchanges']['quote_currency']
        self.db_path = self.config['database']['path']
        
        self._init_database()
        self._init_cache()
        
        logger.info(f"DataFetcher initialized: {self.exchange}, assets={self.assets}")
    
    def _init_database(self):
        """Crea tablas si no existen"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                asset TEXT,
                price REAL,
                volume REAL,
                high REAL,
                low REAL,
                UNIQUE(timestamp, asset)
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_asset_timestamp 
            ON market_data(asset, timestamp)
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_cache(self):
        """Inicializa cache en-memory"""
        self.cache = {}
        self.cache_timestamps = {}
    
    def fetch_from_api(self) -> Dict[str, Dict]:
        """
        Obtiene precios del exchange.
        
        IMPORTANTE: Implementar según tu exchange
        - Binance: usar ccxt o requests directo
        - Kraken: usar API REST
        - Coinbase: usar python3-coinbasepro
        
        Retorna: {asset: {price, volume, high, low}}
        """
        
        try:
            # MOCK para pruebas (reemplaza con real API)
            prices = {}
            for asset in self.assets:
                # Simulamos datos
                base_price = 45000 if asset == "BTC" else 2500 if asset == "ETH" else 100
                price = base_price * (1 + np.random.randn() * 0.01)
                
                prices[asset] = {
                    'price': price,
                    'volume': np.random.uniform(1e6, 1e7),
                    'high': price * 1.01,
                    'low': price * 0.99
                }
            
            logger.info(f"Fetched prices: {list(prices.keys())}")
            return prices
        
        except Exception as e:
            logger.error(f"API fetch error: {e}")
            return {}
    
    def cache_prices(self, prices: Dict[str, Dict]):
        """Guarda en cache en-memory"""
        for asset, data in prices.items():
            self.cache[asset] = data
            self.cache_timestamps[asset] = datetime.now()
    
    def save_to_db(self, prices: Dict[str, Dict]):
        """Persiste en SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now()
            for asset, data in prices.items():
                try:
                    cursor.execute('''
                        INSERT INTO market_data 
                        (timestamp, asset, price, volume, high, low)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        timestamp, asset,
                        data['price'], data['volume'],
                        data['high'], data['low']
                    ))
                except sqlite3.IntegrityError:
                    # Mismo timestamp/asset ya existe, skip
                    pass
            
            conn.commit()
            conn.close()
            logger.info(f"Saved to DB: {len(prices)} assets")
        
        except Exception as e:
            logger.error(f"DB save error: {e}")
    
    def run(self, interval_seconds: int = 60):
        """Loop principal: ejecuta cada N segundos"""
        
        logger.info(f"Starting data fetcher loop (interval={interval_seconds}s)")
        
        while True:
            try:
                # 1. Fetch
                prices = self.fetch_from_api()
                
                if prices:
                    # 2. Cache
                    self.cache_prices(prices)
                    
                    # 3. DB
                    self.save_to_db(prices)
                    
                    logger.info(f"Cycle complete: {len(prices)} assets cached")
                
                # Espera siguiente ciclo
                time.sleep(interval_seconds)
            
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(interval_seconds)
    
    def get_latest_price(self, asset: str) -> float:
        """Obtiene precio del cache (muy rápido)"""
        return self.cache.get(asset, {}).get('price', 0)
    
    def get_last_n_prices(self, asset: str, n: int = 100) -> List[float]:
        """Obtiene últimos N precios del DB"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT price FROM market_data 
                WHERE asset = ? 
                ORDER BY timestamp DESC 
                LIMIT ?
            '''
            df = pd.read_sql(query, conn, params=(asset, n))
            conn.close()
            return df['price'].values[::-1]  # Reverse para orden cronológico
        except:
            return []


if __name__ == "__main__":
    fetcher = DataFetcher()
    fetcher.run(interval_seconds=60)  # Cada minuto
