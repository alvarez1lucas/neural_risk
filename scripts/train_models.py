# scripts/train_models.py
"""
LAYER 2: MODEL TRAINING (Offline, cron job @ 00:00 UTC)
Ejecutar 1x/día: python scripts/train_models.py

Responsabilidades:
- Cargar histórico completo (últimos 6 meses)
- Entrenar 9 expertos en paralelo
- Guardar modelos en pickle cache
- run_engine.py reutiliza sin reentrenar
"""

import os
import sys
import pickle
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np
import yaml
import sqlite3

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [TRAIN] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Entrena todos los modelos offline"""
    
    def __init__(self, config_path: str = None):
        # Si no se especifica config_path, busca desde el directorio del script
        if config_path is None:
            # Busca en: ./config (si corro desde raíz) o ../config (si corro desde scripts/)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            root_dir = os.path.dirname(script_dir)
            
            potential_paths = [
                os.path.join(root_dir, 'config', 'config.yaml'),  # ../config/config.yaml
                'config/config.yaml',                               # ./config/config.yaml
                './config/config.yaml',
            ]
            
            config_path = None
            for path in potential_paths:
                if os.path.exists(path):
                    config_path = path
                    break
            
            if config_path is None:
                raise FileNotFoundError(f"config.yaml not found. Tried: {potential_paths}")
        
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.db_path = self.config['database']['path']
        self.assets = self.config['exchanges']['assets']
        self.window_sizes = self.config['timeframes']
        self.model_cache_path = "./data/trained_models.pkl"
        
        os.makedirs(os.path.dirname(self.model_cache_path), exist_ok=True)
        logger.info(f"ModelTrainer initialized: {len(self.assets)} assets")
    
    def load_asset_data(self, asset: str, months: int = 6) -> pd.DataFrame:
        """Carga últimos N meses del asset"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            cutoff_date = datetime.now() - timedelta(days=30*months)
            
            query = '''
                SELECT timestamp, price, volume, high, low 
                FROM market_data 
                WHERE asset = ? AND timestamp >= ?
                ORDER BY timestamp ASC
            '''
            
            df = pd.read_sql(query, conn, params=(asset, cutoff_date.isoformat()))
            conn.close()
            
            if len(df) == 0:
                logger.warning(f"No data for {asset}")
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            
            # Calcular features
            df['returns'] = df['price'].pct_change()
            df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
            df['volatility'] = df['returns'].rolling(20).std()
            df['ma_20'] = df['price'].rolling(20).mean()
            df['ma_50'] = df['price'].rolling(50).mean()
            df['rsi'] = self._calculate_rsi(df['price'])
            
            logger.info(f"Loaded {asset}: {len(df)} candles, {df.index[0].date()} to {df.index[-1].date()}")
            
            return df.dropna()
        
        except Exception as e:
            logger.error(f"Error loading data for {asset}: {e}")
            return None
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train_hmm(self, df: pd.DataFrame) -> dict:
        """Entrena HMM para detección de régimen"""
        try:
            from hmmlearn.hmm import GaussianHMM
            
            features = df[['returns', 'volatility']].dropna().values
            
            if len(features) < 100:
                logger.warning("Insufficient data for HMM")
                return None
            
            model = GaussianHMM(n_components=3, covariance_type="full", n_iter=1000)
            model.fit(features)
            
            logger.info(f"✅ HMM trained: {len(features)} data points")
            return model
        
        except Exception as e:
            logger.error(f"HMM training error: {e}")
            return None
    
    def train_xgboost(self, df: pd.DataFrame) -> dict:
        """Entrena XGBoost para signals rápidas"""
        try:
            import xgboost as xgb
            
            X = df[['returns', 'volatility', 'rsi', 'ma_20', 'ma_50']].dropna()
            y = (X['returns'].shift(-1) > 0).astype(int)  # Target: sube o baja
            
            if len(X) < 100:
                logger.warning("Insufficient data for XGBoost")
                return None
            
            X = X[:-1]
            y = y[:-1]
            
            model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            model.fit(X, y)
            
            logger.info(f"✅ XGBoost trained: {len(X)} data points")
            return model
        
        except Exception as e:
            logger.error(f"XGBoost training error: {e}")
            return None
    
    def train_garch(self, df: pd.DataFrame) -> dict:
        """Entrena GARCH para volatilidad condicional"""
        try:
            from arch import arch_model
            
            returns = df['log_returns'].dropna() * 100  # En %
            
            if len(returns) < 100:
                logger.warning("Insufficient data for GARCH")
                return None
            
            model = arch_model(returns, vol='Garch', p=1, q=1)
            results = model.fit(disp='off')
            
            logger.info(f"✅ GARCH trained: {len(returns)} data points")
            return results
        
        except Exception as e:
            logger.error(f"GARCH training error: {e}")
            return None
    
    def train_all_models(self) -> dict:
        """Entrena todos los modelos para todos los assets"""
        
        trained_models = {}
        
        for asset in self.assets:
            logger.info(f"\n{'='*60}")
            logger.info(f"Training models for {asset}")
            logger.info(f"{'='*60}")
            
            # Cargar datos
            df = self.load_asset_data(asset, months=6)
            if df is None:
                logger.warning(f"Skipping {asset}")
                continue
            
            # Entrenar modelos
            asset_models = {
                'hmm': self.train_hmm(df),
                'xgboost': self.train_xgboost(df),
                'garch': self.train_garch(df),
                'last_update': datetime.now().isoformat()
            }
            
            trained_models[asset] = asset_models
        
        return trained_models
    
    def save_models(self, models: dict):
        """Guarda modelos en pickle cache"""
        try:
            with open(self.model_cache_path, 'wb') as f:
                pickle.dump(models, f)
            
            logger.info(f"✅ Models saved to {self.model_cache_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def run(self):
        """Ejecuta entrenamiento completo"""
        
        logger.info("🚀 Starting daily model training...")
        start_time = datetime.now()
        
        try:
            # Entrenar todos
            models = self.train_all_models()
            
            # Guardar cache
            if models:
                self.save_models(models)
                
                elapsed = (datetime.now() - start_time).total_seconds()
                logger.info(f"✅ Training complete in {elapsed:.1f}s")
            else:
                logger.warning("No models trained")
        
        except Exception as e:
            logger.error(f"Training failed: {e}")


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
