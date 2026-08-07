# scripts/run_engine.py
"""
LAYER 3: ENGINE (ciclo LIVE, 60-300s)

ESTADO: SOLO predice (nunca fitea) usando data/trained_models.pkl,
generado por train_models.py. PortfolioAgent recibe fractional_kelly y
max_position_size desde config (fix aplicado por el usuario).

PENDIENTE: StrategyRouter no se invoca aca (solo alcanzable via
engine.run_portfolio_automation, que este script no llama).
prepare_asset_features corre el feature engineering pesado en CADA
ciclo (no cacheado) -- diferido a revision de Paso 2.
"""

import time
import logging
import sqlite3
import yaml
import os
import json
import pickle
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import numpy as np

from _pathutils import resolve_path

from neural_risk.engine import AutomatedRiskEngine
from neural_risk.data.data_processor import DataProcessor
from neural_risk.data.feature_engineering import RiskFeaturePipeline
from neural_risk.data.labeling import RiskLabeler
from neural_risk.cortex.feature_jury import FeatureJury
from neural_risk.agents.strategy_router import StrategyRouter
from neural_risk.agents.portfolio_agent import PortfolioAgent
from neural_risk.data.macro_context import MacroContextBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ENGINE] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class EngineService:
    def __init__(self, config_path: str = "config/config.yaml"):
        # FIX (#3): resuelto de forma robusta al CWD -- antes fallaba
        # con FileNotFoundError si se corría desde otro directorio que
        # no fuera la raíz del proyecto.
        config_path = resolve_path(config_path)
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.db_path = resolve_path(self.config['database']['path'])
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)  # FIX: garantiza que exista la carpeta antes de conectar
        self.assets = self.config['exchanges']['assets']
        self.n_workers = self.config['performance']['n_workers']
        self.model_cache_path = resolve_path("./data/trained_models.pkl")

        processor = DataProcessor()
        pipeline = RiskFeaturePipeline()
        labeler = RiskLabeler()
        jury = FeatureJury()
        router = StrategyRouter(risk_appetite=self.config.get('risk_appetite', 0.7))

        self.engine = AutomatedRiskEngine(
            processor=processor, pipeline=pipeline, labeler=labeler,
            jury=jury, trainer_class=None, router=router
        )

        # NUEVO (mejora #3 de 3): contexto macro, cacheado (mismo
        # archivo de cache que usa train_models.py -- si ese script ya
        # corrió, este simplemente reutiliza el cache reciente en vez
        # de volver a descargar).
        self.macro_builder = MacroContextBuilder()
        self.macro_features = pd.DataFrame()  # se puebla en el primer run_cycle()

        self.agent = PortfolioAgent(
            initial_capital=self.config['backtest']['initial_capital'],
            long_threshold=self.config['signals']['long_threshold'],
            short_threshold=self.config['signals']['short_threshold'],
            adaptive_sl_threshold=self.config['signals']['sl_adaptive_threshold'],
            base_staking_apy=self.config['signals']['base_staking_apy'],
            fractional_kelly=self.config['signals']['fractional_kelly'],
            max_position_size=self.config['signals']['max_position_size']
        )

        self.model_cache = {}
        self.model_cache_mtime = None
        self._load_model_cache()

        self._init_decision_db()
        self._init_feedback_tracking()

        logger.info(f"EngineService initialized: {len(self.assets)} assets")

    def _load_model_cache(self):
        try:
            mtime = os.path.getmtime(self.model_cache_path)
        except OSError:
            logger.warning(f"No existe {self.model_cache_path} todavia -- correr train_models.py primero.")
            self.model_cache = {}
            self.model_cache_mtime = None
            return

        if self.model_cache_mtime is not None and mtime == self.model_cache_mtime:
            return

        try:
            with open(self.model_cache_path, 'rb') as f:
                self.model_cache = pickle.load(f)
            self.model_cache_mtime = mtime
            logger.info(f"Model cache recargado ({len(self.model_cache)} activos con modelos)")
        except Exception as e:
            logger.error(f"Error cargando model cache: {e}")

    def _init_decision_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS engine_decisions (
                id INTEGER PRIMARY KEY, timestamp DATETIME, asset TEXT, signal TEXT,
                confidence REAL, agreement REAL, entry_price REAL, stop_loss REAL,
                position_size_pct REAL, expert_votes TEXT, step_duration_ms INTEGER,
                regime_key TEXT,
                UNIQUE(timestamp, asset)
            )
        ''')
        # Por si la tabla ya existia de una corrida previa sin regime_key.
        try:
            cursor.execute('ALTER TABLE engine_decisions ADD COLUMN regime_key TEXT')
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()

    def _init_feedback_tracking(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            try:
                cursor.execute('ALTER TABLE fills ADD COLUMN feedback_synced INTEGER DEFAULT 0')
            except sqlite3.OperationalError:
                pass
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"No se pudo inicializar feedback tracking (¿corrio ya el executor?): {e}")

    def fetch_current_data(self, asset: str) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT timestamp, open, high, low, price as close, volume
                FROM market_data WHERE asset = ? ORDER BY timestamp DESC LIMIT 1000
            '''
            df = pd.read_sql(query, conn, params=(asset,))
            conn.close()

            if len(df) < 200:
                logger.warning(f"Datos insuficientes para {asset}: {len(df)} filas (minimo ~200)")
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.set_index('timestamp')
            df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logger.error(f"Error fetching data for {asset}: {e}")
            return None

    def sync_feedback_from_fills(self):
        try:
            conn = sqlite3.connect(self.db_path)
            fills_query = '''
                SELECT f.id as fill_id, f.order_id, f.asset, f.pnl, f.fill_price, f.quantity, o.timestamp as order_ts
                FROM fills f JOIN orders o ON f.order_id = o.order_id
                WHERE COALESCE(f.feedback_synced, 0) = 0
            '''
            fills_df = pd.read_sql(fills_query, conn)
            if len(fills_df) == 0:
                conn.close()
                return

            cursor = conn.cursor()
            for _, fill in fills_df.iterrows():
                dec_query = '''
                    SELECT expert_votes, entry_price, regime_key FROM engine_decisions
                    WHERE asset = ? AND timestamp <= ? ORDER BY timestamp DESC LIMIT 1
                '''
                dec_df = pd.read_sql(dec_query, conn, params=(fill['asset'], fill['order_ts']))
                if len(dec_df) == 0:
                    continue
                try:
                    expert_votes = json.loads(dec_df['expert_votes'].iloc[0])
                except (TypeError, json.JSONDecodeError):
                    expert_votes = {}

                regime_key = dec_df['regime_key'].iloc[0] if 'regime_key' in dec_df.columns else None

                entry_price = dec_df['entry_price'].iloc[0] or fill['fill_price']
                notional = entry_price * fill['quantity'] if entry_price and fill['quantity'] else 0
                pnl_pct = (fill['pnl'] / notional) if notional else 0.0

                # price_return = movimiento real de precio (signo puro,
                # independiente de LONG/SHORT) -- se lo pasamos al
                # agente para que cada experto reciba crédito según si
                # SU voto coincidió con lo que realmente pasó, no según
                # si el trade agregado ganó (ver fix en portfolio_agent.py).
                price_return = None
                if entry_price and entry_price > 0:
                    price_return = (fill['fill_price'] - entry_price) / entry_price

                if expert_votes:
                    self.agent.record_expert_feedback(
                        expert_votes, pnl_pct, price_return=price_return, regime_key=regime_key
                    )

                cursor.execute("UPDATE fills SET feedback_synced = 1 WHERE id = ?", (int(fill['fill_id']),))

            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error sincronizando feedback de fills: {e}")

    def process_asset(self, asset: str) -> dict:
        start = time.time()

        cache_entry = self.model_cache.get(asset)
        if cache_entry is None or 'best_feats' not in cache_entry:
            logger.warning(f"Sin modelos entrenados para {asset} -- correr train_models.py primero")
            return None

        raw_df = self.fetch_current_data(asset)
        if raw_df is None:
            return None

        X_filtered, _, best_feats, returns, df_features = self.engine.prepare_asset_features(
            asset, raw_df, cached_best_feats=cache_entry['best_feats'],
            macro_context=self.macro_features
        )
        if X_filtered is None:
            logger.warning(f"Features insuficientes para {asset} en este ciclo")
            return None

        fast_models = cache_entry.get('fast', {}).get('models', {})
        slow_models = cache_entry.get('slow', {}).get('models', {})
        anomaly_detector = cache_entry.get('anomaly_detector')

        try:
            intelligence = self.engine.predict_with_cached_experts(
                asset, X_filtered, returns, df_features, fast_models, slow_models, anomaly_detector
            )
        except Exception as e:
            logger.error(f"Prediccion fallo para {asset}: {e}")
            return None

        close_col = f'{asset}_Close'
        current_price = float(df_features[close_col].iloc[-1])
        current_prices = {asset: current_price}

        try:
            portfolio_decision = self.agent.execute_portfolio_decision({asset: intelligence}, current_prices)
        except Exception as e:
            logger.error(f"PortfolioAgent fallo para {asset}: {e}")
            return None

        decision = portfolio_decision['decisions'].get(asset)
        if decision is None:
            return None

        self.save_decisions(asset, decision)
        elapsed = (time.time() - start) * 1000
        return {'asset': asset, 'signal': decision['signal'], 'elapsed_ms': elapsed}

    def save_decisions(self, asset: str, decision: dict):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO engine_decisions
                (timestamp, asset, signal, confidence, agreement, entry_price, stop_loss,
                 position_size_pct, expert_votes, step_duration_ms, regime_key)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(), asset, decision.get('signal', 'HOLD'), decision.get('confidence', 0),
                decision.get('agreement', 0), decision.get('entry_price', 0), decision.get('stop_loss', 0),
                decision.get('position_size_pct', 0), json.dumps(decision.get('expert_signals', {})), 0,
                decision.get('regime_key')
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving decision for {asset}: {e}")

    def run_cycle(self) -> dict:
        cycle_start = time.time()
        try:
            self._load_model_cache()

            # NUEVO: se refresca (o se lee del cache en disco si sigue
            # fresco -- get_macro_features respeta max_age_hours) una
            # sola vez por ciclo, compartido entre todos los activos.
            self.macro_features = self.macro_builder.get_macro_features()

            logger.info(f"Starting cycle: {len(self.assets)} assets")
            self.sync_feedback_from_fills()

            results = []
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {executor.submit(self.process_asset, a): a for a in self.assets}
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
            return {'cycle_time_ms': cycle_time, 'n_assets_processed': len(results), 'results': results}
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            return {}

    def run(self, interval_seconds: int = None):
        if interval_seconds is None:
            interval_seconds = self.config.get('cycle_timing', {}).get('engine_interval', 60)
        logger.info(f"Starting engine loop (interval={interval_seconds}s)")
        while True:
            try:
                cycle_result = self.run_cycle()
                if cycle_result:
                    logger.info(f"Cycle result: {cycle_result}")
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(interval_seconds)


if __name__ == "__main__":
    engine_service = EngineService()
    engine_service.run()