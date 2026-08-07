# scripts/run_portfolio_hedge.py
"""
HEDGING A NIVEL PORTAFOLIO (asesor -- este script NO ejecuta ordenes)
Ejecutar como servicio: python scripts/run_portfolio_hedge.py

Calcula la asignación de capital óptima entre TODOS los activos del
portafolio (via PortfolioHedgeOptimizer, Differential Evolution sobre
Expected Shortfall conjunto) y la compara contra lo que el executor
tiene REALMENTE abierto hoy. Guarda la recomendación en la DB.

DECISIÓN DE DISEÑO: este script en si NO coloca ni modifica ordenes --
solo calcula y guarda la recomendacion. La EJECUCION real (parcial:
aumentar/reducir posiciones existentes hacia el peso objetivo) vive en
run_executor.py (metodo execute_rebalance(), corre cada ciclo del
executor) -- separado a proposito, mismo criterio que separa
"calcular" de "ejecutar" en el resto del sistema (train_models.py vs
run_engine.py).

Cadencia: corre cada N horas (config['cycle_timing']['portfolio_hedge_interval_hours'],
default 24) -- es una optimización costosa (Differential Evolution),
no tiene sentido correrla cada 60-300s como el engine.
"""

import time
import logging
import sqlite3
import yaml
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np

from _pathutils import resolve_path

from neural_risk.optimization.hedging import PortfolioHedgeOptimizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [HEDGE] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class PortfolioHedgeService:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_path = resolve_path(config_path)  # FIX (#3): robusto al CWD
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.db_path = resolve_path(self.config['database']['path'])  # FIX (#3)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)  # FIX: garantiza que exista la carpeta antes de conectar
        self.assets = self.config['exchanges']['assets']
        self.optimizer = PortfolioHedgeOptimizer(confidence=0.95, max_iter=200)

        self._init_db()
        logger.info(f"PortfolioHedgeService initialized: {len(self.assets)} assets")

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS portfolio_hedge_recommendations (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                optimal_weights TEXT,
                current_weights TEXT,
                expected_shortfall REAL,
                max_delta REAL,
                rebalance_recommended INTEGER,
                n_assets_used INTEGER
            )
        ''')
        conn.commit()
        conn.close()

    def _build_aligned_returns(self, min_rows: int = 100) -> dict:
        """
        Arma retornos ALINEADOS por timestamp entre todos los activos
        configurados. Usa 'price' (Close) de market_data. Se descartan
        filas donde no TODOS los activos tienen dato (pivot + dropna),
        para que la matriz de retornos sea consistente.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            placeholders = ','.join(['?'] * len(self.assets))
            query = f'''
                SELECT timestamp, asset, price FROM market_data
                WHERE asset IN ({placeholders})
                ORDER BY timestamp ASC
            '''
            df = pd.read_sql(query, conn, params=self.assets)
            conn.close()

            if df.empty:
                return {}

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            wide = df.pivot_table(index='timestamp', columns='asset', values='price')
            wide = wide.dropna()  # solo filas donde TODOS los activos tienen precio

            if len(wide) < min_rows + 1:
                logger.warning(
                    f"Historico alineado insuficiente: {len(wide)} filas "
                    f"(minimo {min_rows + 1}). ¿Corrieron ya varios ciclos de todos los activos?"
                )
                return {}

            returns = wide.pct_change().dropna()
            return {asset: returns[asset].values for asset in returns.columns}

        except Exception as e:
            logger.error(f"Error armando retornos alineados: {e}")
            return {}

    def _get_current_weights(self) -> dict:
        """
        Pesos ACTUALES reales, leídos de las posiciones abiertas del
        executor (orders.status='FILLED'), no del libro interno de
        PortfolioAgent (que en producción no se usa -- ver decisión de
        arquitectura documentada en portfolio_agent.py).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT asset, quantity, price FROM orders WHERE status = 'FILLED'"
            df = pd.read_sql(query, conn)
            conn.close()

            if df.empty:
                return {}

            df['notional'] = df['quantity'] * df['price']
            by_asset = df.groupby('asset')['notional'].sum()
            total = by_asset.sum()

            if total <= 0:
                return {}

            return (by_asset / total).to_dict()

        except Exception as e:
            logger.error(f"Error leyendo posiciones actuales: {e}")
            return {}

    def _save_recommendation(self, optimal_weights: dict, current_weights: dict,
                            expected_shortfall, rebalance_info: dict, n_assets: int):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO portfolio_hedge_recommendations
                (timestamp, optimal_weights, current_weights, expected_shortfall,
                 max_delta, rebalance_recommended, n_assets_used)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                json.dumps(optimal_weights),
                json.dumps(current_weights),
                expected_shortfall,
                rebalance_info.get('max_delta', 0.0),
                int(rebalance_info.get('rebalance_recommended', False)),
                n_assets
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error guardando recomendacion: {e}")

    def run_cycle(self):
        logger.info("Calculando hedging optimo a nivel portafolio...")

        returns_by_asset = self._build_aligned_returns()
        if len(returns_by_asset) < 2:
            logger.warning(
                "Menos de 2 activos con historico alineado suficiente -- "
                "se salta este ciclo (PortfolioHedgeOptimizer necesita >=2)."
            )
            return

        result = self.optimizer.optimize(returns_by_asset)

        if not result.get('success', False) and 'reason' in result:
            logger.warning(f"Optimizacion no ejecutada: {result['reason']}")
            return

        current_weights = self._get_current_weights()
        rebalance_info = self.optimizer.suggest_rebalance(
            current_weights, result['weights'], threshold=0.05
        )

        self._save_recommendation(
            optimal_weights=result['weights'],
            current_weights=current_weights,
            expected_shortfall=result['expected_shortfall'],
            rebalance_info=rebalance_info,
            n_assets_used=len(returns_by_asset)
        )

        logger.info(f"Pesos optimos: {result['weights']}")
        logger.info(f"Pesos actuales: {current_weights}")
        logger.info(f"Expected Shortfall (95%): {result['expected_shortfall']:.4f}")
        if rebalance_info['rebalance_recommended']:
            logger.warning(
                f"REBALANCEO SUGERIDO -- delta maximo: {rebalance_info['max_delta']:.2%}"
            )
        else:
            logger.info("Sin necesidad de rebalancear (dentro del umbral)")

    def run(self, interval_hours: int = None):
        if interval_hours is None:
            interval_hours = self.config.get('cycle_timing', {}).get('portfolio_hedge_interval_hours', 24)

        logger.info(f"Starting portfolio hedge loop (interval={interval_hours}h)")

        while True:
            try:
                self.run_cycle()
            except Exception as e:
                logger.error(f"Cycle error: {e}")

            time.sleep(interval_hours * 3600)


if __name__ == "__main__":
    service = PortfolioHedgeService()
    service.run()