# scripts/train_models.py
"""
LAYER 2: MODEL TRAINING (Offline, scheduling dinamico por tier)
ESTADO: reescrito completo. Usa el mismo pipeline real que engine.py.
"""

import os
import time
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict
import pandas as pd
import numpy as np
import yaml
import sqlite3

from _pathutils import resolve_path

from neural_risk.engine import AutomatedRiskEngine
from neural_risk.data.data_processor import DataProcessor
from neural_risk.data.feature_engineering import RiskFeaturePipeline
from neural_risk.data.labeling import RiskLabeler
from neural_risk.cortex.feature_jury import FeatureJury
from neural_risk.agents.strategy_router import StrategyRouter
from neural_risk.data.macro_context import MacroContextBuilder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [TRAIN] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    def __init__(self, config_path: str = "config/config.yaml",
                 schedule_path: str = "config/model_schedule.yaml"):
        # FIX (#3): las 4 rutas de este constructor eran relativas al
        # CWD -- fallaban (o, peor, creaban archivos nuevos en el lugar
        # equivocado sin avisar) si no se corria desde la raiz del
        # proyecto. Ahora se resuelven de forma robusta.
        config_path = resolve_path(config_path)
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.schedule_path = resolve_path(schedule_path)
        self.db_path = resolve_path(self.config['database']['path'])
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)  # FIX: garantiza que exista la carpeta antes de conectar
        self.assets = self.config['exchanges']['assets']
        self.model_cache_path = resolve_path("./data/trained_models.pkl")

        os.makedirs(os.path.dirname(self.model_cache_path), exist_ok=True)

        self.engine = AutomatedRiskEngine(
            processor=DataProcessor(), pipeline=RiskFeaturePipeline(), labeler=RiskLabeler(),
            jury=FeatureJury(), trainer_class=None,
            router=StrategyRouter(risk_appetite=self.config.get('risk_appetite', 0.7))
        )

        # NUEVO (mejora #3 de 3): contexto macro (VIX/DXY/SPX/Oro),
        # cacheado en disco (max_age_hours=6 por defecto -- no tiene
        # sentido re-descargarlo en cada corrida de train_models.py).
        # Si falla la descarga (sin internet, Yahoo caído), devuelve
        # DataFrame vacío y el pipeline sigue funcionando sin contexto
        # macro -- no es una dependencia dura.
        self.macro_builder = MacroContextBuilder()

        logger.info(f"ModelTrainer initialized: {len(self.assets)} assets")

    def _load_schedule(self) -> Dict:
        with open(self.schedule_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)['model_schedule']

    def _load_cache(self) -> Dict:
        if os.path.exists(self.model_cache_path):
            try:
                with open(self.model_cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.error(f"Error cargando cache existente, se arranca de cero: {e}")
        return {}

    def _save_cache(self, cache: Dict):
        try:
            with open(self.model_cache_path, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            logger.error(f"Error guardando cache: {e}")

    def _is_due(self, last_trained_iso, interval_days: int) -> bool:
        if last_trained_iso is None:
            return True
        last = datetime.fromisoformat(last_trained_iso)
        return (datetime.now() - last) >= timedelta(days=interval_days)

    def load_asset_history(self, asset: str, months: int = 6) -> pd.DataFrame:
        try:
            conn = sqlite3.connect(self.db_path)
            cutoff_date = datetime.now() - timedelta(days=30 * months)
            query = '''
                SELECT timestamp, open, high, low, price as close, volume
                FROM market_data WHERE asset = ? AND timestamp >= ? ORDER BY timestamp ASC
            '''
            df = pd.read_sql(query, conn, params=(asset, cutoff_date.isoformat()))
            conn.close()

            if len(df) < 200:
                logger.warning(f"Historico insuficiente para {asset}: {len(df)} filas")
                return None

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.set_index('timestamp')
            df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})

            logger.info(f"Cargado {asset}: {len(df)} velas, {df.index[0].date()} a {df.index[-1].date()}")
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except Exception as e:
            logger.error(f"Error cargando historico de {asset}: {e}")
            return None

    def run_cycle(self):
        schedule = self._load_schedule()
        cache = self._load_cache()

        fast_interval = schedule['fast_tier']['retrain_interval_days']
        slow_interval = schedule['slow_tier']['retrain_interval_days']

        # NUEVO: se construye UNA sola vez por ciclo (compartido entre
        # todos los activos) -- el contexto macro no depende del activo,
        # y así se evita pegarle a Yahoo Finance una vez por activo.
        macro_features = self.macro_builder.get_macro_features()
        if not macro_features.empty:
            logger.info(f"Contexto macro cargado: {list(macro_features.columns)}")
        else:
            logger.info("Sin contexto macro disponible este ciclo (se sigue sin él)")

        for asset in self.assets:
            entry = cache.get(asset, {})
            fast_due = self._is_due(entry.get('fast', {}).get('last_trained'), fast_interval)
            slow_due = self._is_due(entry.get('slow', {}).get('last_trained'), slow_interval)

            if not fast_due and not slow_due:
                logger.info(f"{asset}: nada vencido hoy, se salta.")
                continue

            raw_df = self.load_asset_history(asset)
            if raw_df is None:
                continue

            X_filtered, y, best_feats, returns, df_features = self.engine.prepare_asset_features(
                asset, raw_df, macro_context=macro_features
            )
            if X_filtered is None:
                logger.warning(f"{asset}: datos insuficientes tras feature engineering, se salta.")
                continue

            entry['best_feats'] = best_feats
            entry['features_computed_at'] = datetime.now().isoformat()

            anomaly_detector = entry.get('anomaly_detector')

            if fast_due:
                logger.info(f"{asset}: reentrenando tier RAPIDO...")
                fast_models, anomaly_detector = self.engine.fit_fast_tier_experts(
                    X_filtered, y, returns, existing_anomaly=anomaly_detector
                )
                entry['fast'] = {'models': fast_models, 'last_trained': datetime.now().isoformat()}
                logger.info(f"{asset}: tier rapido OK.")

            if slow_due:
                logger.info(f"{asset}: reentrenando tier LENTO (puede tardar)...")
                slow_models, anomaly_detector = self.engine.fit_slow_tier_experts(
                    X_filtered, y, len(best_feats), anomaly_detector
                )
                entry['slow'] = {'models': slow_models, 'last_trained': datetime.now().isoformat()}
                logger.info(f"{asset}: tier lento OK.")

            entry['anomaly_detector'] = anomaly_detector
            cache[asset] = entry
            self._save_cache(cache)

        logger.info("Ciclo de entrenamiento completo.")

    def run(self, check_interval_hours: int = 6):
        logger.info(f"Iniciando loop de entrenamiento (chequeo cada {check_interval_hours}h)")
        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"Error en ciclo de entrenamiento: {e}")
            time.sleep(check_interval_hours * 3600)


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()