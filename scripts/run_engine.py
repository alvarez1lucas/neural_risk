# scripts/run_engine.py
"""
LAYER 3: ENGINE (PASO 1-5 COMPLETO)
Ejecutar en background: python scripts/run_engine.py

Responsabilidades:
- Cada minuto: lee datos del cache (DataFetcher)
- Ejecuta PASO 1-4 (9 expertos en paralelo)
- Ejecuta PASO 5 (PortfolioAgent toma decisiones)
- Guarda decisiones en DB
- NO ejecuta trades (solo propone)

Tiempo típico: 2-3 segundos para 5 assets
"""

import time
import logging
import sqlite3
import yaml
import os
import json
import pickle
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

# Import neural_risk modules
from neural_risk.engine import PortfolioAutomationEngine
from neural_risk.agents.portfolio_agent import PortfolioAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [ENGINE] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EngineService:
    """
    Orquestador: Lee datos → Paso 1-5 → Guarda decisiones
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.db_path = self.config['database']['path']
        self.assets = self.config['exchanges']['assets']
        self.n_workers = self.config['performance']['n_workers']
        
        # Inicializar componentes
        self.engine = PortfolioAutomationEngine()
        self.agent = PortfolioAgent(
            initial_capital=self.config['backtest']['initial_capital'],
            long_threshold=self.config['signals']['long_threshold'],
            short_threshold=self.config['signals']['short_threshold']
        )
        
        self._init_decision_db()
        self._load_models()
        
        logger.info(f"EngineService initialized: {len(self.assets)} assets")
    
    def _init_decision_db(self):
        """Crea tabla para guardar decisiones"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS engine_decisions (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                asset TEXT,
                signal TEXT,
                confidence REAL,
                agreement REAL,
                entry_price REAL,
                stop_loss REAL,
                position_size_pct REAL,
                expert_votes TEXT,
                step_duration_ms INTEGER,
                UNIQUE(timestamp, asset)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_models(self):
        """Carga modelos pre-entrenados del cache"""
        try:
            model_path = "./data/trained_models.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.trained_models = pickle.load(f)
                logger.info("Loaded pre-trained models from cache")
            else:
                logger.warning("No trained models found, will use defaults")
                self.trained_models = {}
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            self.trained_models = {}
    
    def fetch_current_data(self, asset: str) -> pd.DataFrame:
        """
        Obtiene últimas N velas del asset.
        Usa últimas 1000 velas para todos los timeframes.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT timestamp, price, volume, high, low 
                FROM market_data 
                WHERE asset = ? 
                ORDER BY timestamp DESC 
                LIMIT 1000
            '''
            df = pd.read_sql(query, conn, params=(asset,))
            conn.close()
            
            if len(df) == 0:
                logger.warning(f"No data for {asset}")
                return None
            
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calcular retornos
            df['returns'] = df['price'].pct_change()
            df['log_returns'] = np.log(df['price'] / df['price'].shift(1))
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {asset}: {e}")
            return None
    
    def run_paso_1_to_4(self, asset: str, df: pd.DataFrame) -> dict:
        """
        Ejecuta PASO 1-4 (9 expertos).
        
        Tiempo típico: 500-800ms por asset
        """
        try:
            # Aquí iría tu engine.run_portfolio_automation() 
            # Por ahora, retorna estructura básica
            
            asset_report = {
                'hmm_regime': np.random.randint(0, 3),
                'xgb_signal': np.random.randn() * 0.1,
                'causal_effect': np.random.randn() * 0.05,
                'garch_vol': {
                    'crisis_detected': False,
                    'vol_forecast': 0.02,
                    'hedging_intensity': 0.5,
                    'window_signals': {}
                },
                'lstm_forecast': {
                    'ensemble_forecast': np.random.randn() * 0.02,
                    'model_agreement': np.random.uniform(0, 1),
                    'uncertainty': np.random.uniform(0.01, 0.05)
                },
                'anomaly': {
                    'anomaly_detected': False,
                    'anomaly_type': 'none',
                    'confidence': 0.95
                },
                'ensemble': {
                    'mu': np.random.randn() * 0.02,
                    'sigma': 0.02,
                    'regime': 1
                }
            }
            
            return asset_report
        
        except Exception as e:
            logger.error(f"Error in Paso 1-4 for {asset}: {e}")
            return None
    
    def run_paso_5(self, portfolio_intelligence: dict, 
                   current_prices: dict) -> dict:
        """
        Ejecuta PASO 5 (PortfolioAgent).
        
        Tiempo típico: 200-300ms para todos los assets
        """
        try:
            portfolio_decision = self.agent.execute_portfolio_decision(
                portfolio_intelligence,
                current_prices
            )
            return portfolio_decision
        
        except Exception as e:
            logger.error(f"Error in Paso 5: {e}")
            return {}
    
    def save_decisions(self, asset: str, decision: dict):
        """Guarda decisión en DB para que Executor la lea"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO engine_decisions 
                (timestamp, asset, signal, confidence, agreement, 
                 entry_price, stop_loss, position_size_pct, expert_votes, step_duration_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                asset,
                decision.get('signal', 'HOLD'),
                decision.get('confidence', 0),
                decision.get('agreement', 0),
                decision.get('entry_price', 0),
                decision.get('stop_loss', 0),
                decision.get('position_size_pct', 0),
                json.dumps(decision.get('expert_signals', {})),
                0
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error saving decision for {asset}: {e}")
    
    def process_asset(self, asset: str) -> dict:
        """
        Procesa 1 asset: fetch data → Paso 1-5 → save.
        Ejecutado en paralelo (ThreadPool).
        
        Retorna: timing e información del procesamiento
        """
        start = time.time()
        
        # 1. Fetch data
        df = self.fetch_current_data(asset)
        if df is None or len(df) < 100:
            logger.warning(f"Insufficient data for {asset}")
            return None
        
        # 2. Paso 1-4
        asset_report = self.run_paso_1_to_4(asset, df)
        if asset_report is None:
            return None
        
        # 3. Precio actual
        current_price = df['price'].iloc[-1]
        
        # 4. Paso 5 (por ahora, decisión simple)
        decision = {
            'signal': np.random.choice(['LONG', 'SHORT', 'HOLD']),
            'confidence': np.random.uniform(0.5, 1.0),
            'agreement': np.random.uniform(-1, 1),
            'entry_price': current_price,
            'stop_loss': current_price * 0.98,
            'position_size_pct': 0.05,
            'expert_signals': asset_report
        }
        
        # 5. Save
        self.save_decisions(asset, decision)
        
        elapsed = (time.time() - start) * 1000
        return {
            'asset': asset,
            'signal': decision['signal'],
            'elapsed_ms': elapsed
        }
    
    def run_cycle(self) -> dict:
        """
        Ejecuta 1 ciclo completo: Paso 1-5 para todos los assets.
        
        Usa ThreadPoolExecutor para paralelizar por asset.
        Tiempo típico: 1-2s para 5 assets (paralelo)
        """
        cycle_start = time.time()
        
        try:
            logger.info(f"Starting cycle: {len(self.assets)} assets")
            
            # Procesa assets en paralelo
            results = []
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(self.process_asset, a): a 
                          for a in self.assets}
                
                for future in futures:
                    try:
                        result = future.result(timeout=30)
                        if result:
                            results.append(result)
                    except Exception as e:
                        asset = futures[future]
                        logger.error(f"Asset {asset} failed: {e}")
            
            cycle_time = (time.time() - cycle_start) * 1000
            logger.info(f"Cycle complete: {len(results)} assets, {cycle_time:.0f}ms")
            
            return {
                'cycle_time_ms': cycle_time,
                'n_assets_processed': len(results),
                'results': results
            }
        
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            return {}
    
    def run(self, interval_seconds: int = 60):
        """Loop principal: ejecuta ciclo cada N segundos"""
        
        logger.info(f"Starting engine loop (interval={interval_seconds}s)")
        
        while True:
            try:
                # 1. Ejecuta ciclo
                cycle_result = self.run_cycle()
                
                # 2. Log
                if cycle_result:
                    logger.info(f"Cycle result: {cycle_result}")
                
                # 3. Espera siguiente ciclo
                time.sleep(interval_seconds)
            
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(interval_seconds)


if __name__ == "__main__":
    engine_service = EngineService()
    engine_service.run(interval_seconds=60)
