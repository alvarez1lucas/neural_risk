# scripts/run_executor.py
"""
LAYER 4: EXECUTOR (Coloca trades reales)

ESTADO: intervalo lee de config.yaml (cycle_timing.executor_interval).
get_pending_decisions dedup por status='FILLED' real, no por fecha.
update_fills/check_and_close_stop_losses implementado. get_current_equity
usa equity real (init + pnl realizado), no initial_capital fijo.
"""

import time
import logging
import sqlite3
import yaml
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Optional
import pandas as pd
import numpy as np

from _pathutils import resolve_path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [EXECUTOR] - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class OrderExecutor:
    def __init__(self, config_path: str = "config/config.yaml"):
        config_path = resolve_path(config_path)  # FIX (#3): robusto al CWD
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.db_path = resolve_path(self.config['database']['path'])  # FIX (#3)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)  # FIX: garantiza que exista la carpeta antes de conectar
        self.exchange = self.config['exchanges']['primary']
        self.max_retries = self.config['execution']['max_retries']
        self.retry_delay = self.config['execution']['retry_delay_ms'] / 1000
        self.max_orders_per_minute = self.config['execution']['max_orders_per_minute']
        self.max_daily_loss = self.config['execution']['max_daily_loss_pct']

        self._init_orders_db()
        self._init_exchange_client()
        self._init_hedge_tracking()

        logger.info(f"OrderExecutor initialized: {self.exchange}")

    def _init_orders_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY, timestamp DATETIME, order_id TEXT UNIQUE,
                asset TEXT, side TEXT, quantity REAL, price REAL, order_type TEXT,
                status TEXT, filled_price REAL, filled_quantity REAL, stop_loss REAL,
                commission REAL, error_message TEXT, attempts INTEGER DEFAULT 1
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY, timestamp DATETIME, order_id TEXT,
                asset TEXT, quantity REAL, fill_price REAL, pnl REAL
            )
        ''')
        conn.commit()
        conn.close()

    def _init_exchange_client(self):
        logger.info("Exchange client ready (MOCK)")

    def _init_hedge_tracking(self):
        """
        Agrega columna 'executed_at' a portfolio_hedge_recommendations
        (creada por run_portfolio_hedge.py) para no re-aplicar la misma
        recomendación en cada ciclo del executor.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            try:
                cursor.execute('ALTER TABLE portfolio_hedge_recommendations ADD COLUMN executed_at DATETIME')
            except sqlite3.OperationalError:
                pass
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"No se pudo inicializar hedge tracking (¿corrió ya run_portfolio_hedge.py?): {e}")

    def get_pending_decisions(self) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT d.*, o.order_id
                FROM engine_decisions d
                LEFT JOIN orders o ON d.asset = o.asset AND o.status = 'FILLED'
                WHERE d.signal IN ('LONG', 'SHORT')
                AND o.order_id IS NULL
                AND d.timestamp > datetime('now', '-5 minutes')
                ORDER BY d.confidence DESC
                LIMIT ?
            '''
            df = pd.read_sql(query, conn, params=(self.max_orders_per_minute,))
            conn.close()
            return df.to_dict('records') if len(df) > 0 else []
        except Exception as e:
            logger.error(f"Error getting pending decisions: {e}")
            return []

    def get_current_equity(self) -> float:
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT COALESCE(SUM(pnl), 0) as total_pnl FROM fills"
            df = pd.read_sql(query, conn)
            conn.close()
            realized_pnl = df['total_pnl'].iloc[0]
        except Exception as e:
            logger.error(f"Error calculating equity: {e}")
            realized_pnl = 0.0
        initial_capital = self.config['backtest']['initial_capital']
        return initial_capital + realized_pnl

    def check_daily_loss(self) -> bool:
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT COALESCE(SUM(pnl), 0) as total_pnl FROM fills WHERE date(timestamp) = date('now')"
            df = pd.read_sql(query, conn)
            conn.close()
            daily_pnl = df['total_pnl'].iloc[0]
            initial_capital = self.config['backtest']['initial_capital']
            daily_loss_pct = daily_pnl / initial_capital
            if daily_loss_pct < -self.max_daily_loss:
                logger.warning(f"Daily loss limit hit: {daily_loss_pct:.2%}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error checking daily loss: {e}")
            return True

    def validate_order(self, decision: Dict) -> bool:
        if not self.check_daily_loss():
            logger.warning("Order rejected: daily loss limit")
            return False
        if decision.get('confidence', 0) < self.config['signals']['min_confidence']:
            logger.warning(f"Order rejected: low confidence ({decision.get('confidence')})")
            return False
        if decision.get('position_size_pct', 0) < 0.001:
            logger.warning("Order rejected: position too small")
            return False
        return True

    def place_order(self, decision: Dict) -> Optional[str]:
        if not self.validate_order(decision):
            return None
        asset = decision['asset']
        signal = decision['signal']
        price = decision['entry_price']
        position_size_pct = decision['position_size_pct']
        stop_loss = decision.get('stop_loss')

        current_equity = self.get_current_equity()
        quantity = (current_equity * position_size_pct) / price
        side = 'BUY' if signal == 'LONG' else 'SELL'

        for attempt in range(self.max_retries):
            try:
                order_id = f"{asset}_{signal}_{int(time.time())}"
                logger.info(f"Order placed: {side} {quantity:.4f} {asset} @ {price} (equity={current_equity:.2f}, SL={stop_loss})")
                self._save_order_to_db({
                    'order_id': order_id, 'asset': asset, 'side': side, 'quantity': quantity,
                    'price': price, 'filled_price': price, 'filled_quantity': quantity,
                    'stop_loss': stop_loss, 'status': 'FILLED', 'attempts': attempt + 1
                })
                return order_id
            except Exception as e:
                logger.warning(f"Order attempt {attempt+1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    self._save_order_to_db({
                        'order_id': f"{asset}_{signal}_{int(time.time())}_ERROR", 'asset': asset,
                        'side': side, 'quantity': quantity, 'price': price, 'stop_loss': stop_loss,
                        'status': 'FAILED', 'error_message': str(e), 'attempts': self.max_retries
                    })
                    return None
        return None

    def _save_order_to_db(self, order_info: Dict):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO orders
                (timestamp, order_id, asset, side, quantity, price, status,
                 filled_price, filled_quantity, stop_loss, error_message, attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(), order_info.get('order_id'), order_info.get('asset'),
                order_info.get('side'), order_info.get('quantity', 0), order_info.get('price', 0),
                order_info.get('status'), order_info.get('filled_price'), order_info.get('filled_quantity'),
                order_info.get('stop_loss'), order_info.get('error_message'), order_info.get('attempts', 1)
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error saving order: {e}")

    def get_latest_price(self, asset: str) -> Optional[float]:
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT price FROM market_data WHERE asset = ? ORDER BY timestamp DESC LIMIT 1"
            df = pd.read_sql(query, conn, params=(asset,))
            conn.close()
            if len(df) == 0:
                return None
            return float(df['price'].iloc[0])
        except Exception as e:
            logger.error(f"Error fetching latest price for {asset}: {e}")
            return None

    def get_open_positions(self) -> list:
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM orders WHERE status = 'FILLED'"
            df = pd.read_sql(query, conn)
            conn.close()
            return df.to_dict('records') if len(df) > 0 else []
        except Exception as e:
            logger.error(f"Error getting open positions: {e}")
            return []

    def _close_position(self, order: Dict, exit_price: float, reason: str):
        try:
            asset = order['asset']
            side = order['side']
            quantity = order['quantity']
            entry_price = order.get('filled_price') or order['price']
            order_id = order['order_id']

            if side == 'BUY':
                pnl = (exit_price - entry_price) * quantity
            else:
                pnl = (entry_price - exit_price) * quantity

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO fills (timestamp, order_id, asset, quantity, fill_price, pnl)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), order_id, asset, quantity, exit_price, pnl))
            cursor.execute("UPDATE orders SET status = ? WHERE order_id = ?", (f'CLOSED_{reason}', order_id))
            conn.commit()
            conn.close()

            logger.info(f"Position closed [{reason}]: {asset} {side} qty={quantity:.4f} entry={entry_price:.2f} exit={exit_price:.2f} pnl={pnl:.2f}")
        except Exception as e:
            logger.error(f"Error closing position {order.get('order_id')}: {e}")

    def check_and_close_stop_losses(self):
        open_positions = self.get_open_positions()
        for order in open_positions:
            asset = order['asset']
            stop_loss = order.get('stop_loss')
            if stop_loss is None:
                continue
            current_price = self.get_latest_price(asset)
            if current_price is None:
                logger.warning(f"No hay precio actual para {asset}, no se puede chequear SL")
                continue
            side = order['side']
            breached = (
                (side == 'BUY' and current_price <= stop_loss) or
                (side == 'SELL' and current_price >= stop_loss)
            )
            if breached:
                logger.warning(f"STOP LOSS activado: {asset} {side} price={current_price:.2f} SL={stop_loss:.2f}")
                self._close_position(order, current_price, reason='SL')

    def update_fills(self):
        try:
            self.check_and_close_stop_losses()
        except Exception as e:
            logger.error(f"Error updating fills: {e}")

    def get_current_notional_by_asset(self) -> Dict[str, float]:
        """Notional real (quantity*price) por activo, sumando todas las órdenes FILLED."""
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT asset, quantity, price FROM orders WHERE status = 'FILLED'"
            df = pd.read_sql(query, conn)
            conn.close()
            if df.empty:
                return {}
            df['notional'] = df['quantity'] * df['price']
            return df.groupby('asset')['notional'].sum().to_dict()
        except Exception as e:
            logger.error(f"Error calculando notional actual: {e}")
            return {}

    def get_latest_hedge_recommendation(self) -> Optional[Dict]:
        """
        Lee la recomendación de PortfolioHedgeOptimizer más reciente que
        (a) sugiere rebalancear y (b) todavía no fue aplicada
        (executed_at IS NULL). Generada por scripts/run_portfolio_hedge.py.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT * FROM portfolio_hedge_recommendations
                WHERE rebalance_recommended = 1 AND executed_at IS NULL
                ORDER BY timestamp DESC LIMIT 1
            '''
            df = pd.read_sql(query, conn)
            conn.close()
            if len(df) == 0:
                return None
            row = df.iloc[0]
            return {
                'id': int(row['id']),
                'optimal_weights': json.loads(row['optimal_weights']),
                'timestamp': row['timestamp']
            }
        except Exception as e:
            logger.error(f"Error leyendo recomendación de hedging: {e}")
            return None

    def _mark_hedge_executed(self, rec_id: int):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE portfolio_hedge_recommendations SET executed_at = ? WHERE id = ?",
                (datetime.now(), rec_id)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error marcando recomendación como ejecutada: {e}")

    def _place_rebalance_order(self, asset: str, side: str, quantity: float, price: float) -> Optional[str]:
        """Aumenta exposición: orden nueva, misma mecánica que place_order pero sin
        pasar por validate_order (esas validaciones son para señales de trading,
        no para ajustes de hedging a nivel portafolio)."""
        try:
            order_id = f"{asset}_REBALANCE_{side}_{int(time.time())}"
            self._save_order_to_db({
                'order_id': order_id, 'asset': asset, 'side': side,
                'quantity': quantity, 'price': price, 'filled_price': price,
                'filled_quantity': quantity, 'stop_loss': None,
                'status': 'FILLED', 'attempts': 1
            })
            logger.info(f"Rebalance: {side} {quantity:.4f} {asset} @ {price:.2f} (ajuste de hedging)")
            return order_id
        except Exception as e:
            logger.error(f"Error colocando orden de rebalanceo para {asset}: {e}")
            return None

    def _reduce_position(self, asset: str, reduce_notional: float, current_price: float) -> Dict:
        """
        Reduce exposición: cierra PARCIALMENTE una o más órdenes FILLED
        de este activo hasta cubrir 'reduce_notional', empezando por las
        más antiguas (FIFO). Registra el PnL realizado de la porción
        cerrada en 'fills' y reduce 'quantity' en 'orders' (o la cierra
        del todo si la reducción consume la orden entera).
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = "SELECT * FROM orders WHERE asset = ? AND status = 'FILLED' ORDER BY timestamp ASC"
            df = pd.read_sql(query, conn, params=(asset,))

            if df.empty:
                conn.close()
                return {'reduced_notional': 0.0, 'note': 'sin posición abierta para reducir'}

            cursor = conn.cursor()
            remaining = reduce_notional
            total_reduced = 0.0

            for _, order in df.iterrows():
                if remaining <= 0:
                    break

                entry_price = order['filled_price'] or order['price']
                order_notional = order['quantity'] * entry_price
                reduce_from_this = min(remaining, order_notional)
                reduce_qty = reduce_from_this / entry_price

                side = order['side']
                if side == 'BUY':
                    pnl = (current_price - entry_price) * reduce_qty
                else:
                    pnl = (entry_price - current_price) * reduce_qty

                cursor.execute('''
                    INSERT INTO fills (timestamp, order_id, asset, quantity, fill_price, pnl)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (datetime.now(), order['order_id'], asset, reduce_qty, current_price, pnl))

                new_qty = order['quantity'] - reduce_qty
                if new_qty <= 1e-9:
                    cursor.execute(
                        "UPDATE orders SET status = 'CLOSED_REBALANCE' WHERE order_id = ?",
                        (order['order_id'],)
                    )
                else:
                    cursor.execute(
                        "UPDATE orders SET quantity = ? WHERE order_id = ?",
                        (new_qty, order['order_id'])
                    )

                total_reduced += reduce_from_this
                remaining -= reduce_from_this

            conn.commit()
            conn.close()

            logger.info(f"Rebalance: reducida posición {asset} en notional={total_reduced:.2f} (ajuste de hedging)")
            return {'reduced_notional': total_reduced}

        except Exception as e:
            logger.error(f"Error reduciendo posición de {asset}: {e}")
            return {'reduced_notional': 0.0, 'note': f'error: {e}'}

    def execute_rebalance(self, min_delta_pct: float = 0.05) -> Dict:
        """
        Aplica (parcialmente) la última recomendación de
        PortfolioHedgeOptimizer que todavía no se ejecutó. Ajusta
        posiciones EXISTENTES hacia el peso objetivo: compra más si
        falta exposición, vende una FRACCIÓN de lo abierto si sobra.

        NOTA: aumentar exposición respeta el circuit breaker de daily
        loss (check_daily_loss). Reducir exposición NUNCA se bloquea
        por ese chequeo -- bajar riesgo no debería impedirse jamás por
        el mismo control pensado para frenar NUEVO riesgo.

        NOTA: si un activo tiene posición abierta pero el optimizador
        no devolvió peso para él (ej. le faltaba histórico alineado),
        se deja esa posición intacta -- no se liquida a ciegas por
        falta de dato, es una decisión conservadora deliberada.
        """
        rec = self.get_latest_hedge_recommendation()
        if rec is None:
            return {'rebalanced': False, 'reason': 'sin recomendación pendiente'}

        equity = self.get_current_equity()
        if equity <= 0:
            return {'rebalanced': False, 'reason': 'equity inválido'}

        current_notional = self.get_current_notional_by_asset()
        actions = []

        for asset, target_weight in rec['optimal_weights'].items():
            target_notional = target_weight * equity
            current = current_notional.get(asset, 0.0)
            delta_notional = target_notional - current
            delta_pct = abs(delta_notional) / equity

            if delta_pct < min_delta_pct:
                continue

            current_price = self.get_latest_price(asset)
            if current_price is None or current_price <= 0:
                logger.warning(f"Sin precio actual para {asset}, se salta ese ajuste")
                continue

            if delta_notional > 0:
                if not self.check_daily_loss():
                    logger.warning(f"Rebalance BUY rechazado para {asset}: daily loss limit activo")
                    continue
                qty = delta_notional / current_price
                order_id = self._place_rebalance_order(asset, 'BUY', qty, current_price)
                actions.append({
                    'asset': asset, 'action': 'INCREASE',
                    'notional': round(delta_notional, 2), 'order_id': order_id
                })
            else:
                result = self._reduce_position(asset, abs(delta_notional), current_price)
                actions.append({
                    'asset': asset, 'action': 'DECREASE',
                    'notional': round(abs(delta_notional), 2), **result
                })

        self._mark_hedge_executed(rec['id'])

        return {'rebalanced': True, 'recommendation_id': rec['id'], 'actions': actions}

    def run_cycle(self) -> Dict:
        cycle_start = time.time()
        try:
            self.update_fills()

            # NUEVO: aplica (si hay) la ultima recomendacion de
            # PortfolioHedgeOptimizer que run_portfolio_hedge.py dejo
            # pendiente. Se hace ANTES de procesar nuevas señales
            # LONG/SHORT, para que el sizing de las señales nuevas
            # parta de un equity/posiciones ya rebalanceadas.
            rebalance_result = self.execute_rebalance()
            if rebalance_result.get('rebalanced'):
                logger.info(f"Rebalance ejecutado: {rebalance_result}")

            pending = self.get_pending_decisions()
            logger.info(f"Pending decisions: {len(pending)}")
            orders_placed = 0
            for decision in pending:
                order_id = self.place_order(decision)
                if order_id:
                    orders_placed += 1
            cycle_time = (time.time() - cycle_start) * 1000
            logger.info(f"Cycle complete: {orders_placed} orders placed, {cycle_time:.0f}ms")
            return {
                'cycle_time_ms': cycle_time, 'orders_placed': orders_placed,
                'rebalance': rebalance_result
            }
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            return {}

    def run(self, interval_seconds: int = None):
        if interval_seconds is None:
            interval_seconds = self.config.get('cycle_timing', {}).get('executor_interval', 60)
        logger.info(f"Starting executor loop (interval={interval_seconds}s)")
        while True:
            try:
                cycle_result = self.run_cycle()
                logger.info(f"Executor cycle: {cycle_result}")
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Loop error: {e}")
                time.sleep(interval_seconds)


if __name__ == "__main__":
    executor = OrderExecutor()
    executor.run()