# scripts/run_executor.py
"""
LAYER 4: EXECUTOR (Coloca trades reales)
Ejecutar en background: python scripts/run_executor.py

Responsabilidades:
- Cada minuto: lee decisiones del Engine
- Ejecuta trades en exchange (Binance, Kraken, etc.)
- Maneja orden placement (market, limit, with retry)
- Log de órdenes en DB
- Manejo robusto de errores (network, rate limit, etc.)

IMPORTANTE: Separado del Engine para que fallos de exchange 
no afecten la generación de signals.
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [EXECUTOR] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OrderExecutor:
    """Ejecuta órdenes en exchange de forma robusta"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.db_path = self.config['database']['path']
        self.exchange = self.config['exchanges']['primary']
        self.max_retries = self.config['execution']['max_retries']
        self.retry_delay = self.config['execution']['retry_delay_ms'] / 1000
        self.max_orders_per_minute = self.config['execution']['max_orders_per_minute']
        self.max_daily_loss = self.config['execution']['max_daily_loss_pct']
        
        self._init_orders_db()
        self._init_exchange_client()
        
        logger.info(f"OrderExecutor initialized: {self.exchange}")
    
    def _init_orders_db(self):
        """Crea tabla para histórico de órdenes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                order_id TEXT UNIQUE,
                asset TEXT,
                side TEXT,
                quantity REAL,
                price REAL,
                order_type TEXT,
                status TEXT,
                filled_price REAL,
                filled_quantity REAL,
                commission REAL,
                error_message TEXT,
                attempts INTEGER DEFAULT 1
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS fills (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                order_id TEXT,
                asset TEXT,
                quantity REAL,
                fill_price REAL,
                pnl REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _init_exchange_client(self):
        """Inicializa cliente del exchange (MOCK para ahora)"""
        # Aquí iría: 
        # import ccxt
        # self.exchange_client = ccxt.binance({
        #     'apiKey': os.environ['BINANCE_API_KEY'],
        #     'secret': os.environ['BINANCE_API_SECRET']
        # })
        
        logger.info("Exchange client ready (MOCK)")
    
    def get_pending_decisions(self) -> list:
        """Obtiene decisiones pendientes del Engine que no fueron ejecutadas"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT d.*, o.order_id
                FROM engine_decisions d
                LEFT JOIN orders o ON d.asset = o.asset 
                    AND date(d.timestamp) = date(o.timestamp)
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
    
    def check_daily_loss(self) -> bool:
        """Verifica si ya alcanzamos max daily loss → cancela ejecuciones"""
        try:
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT COALESCE(SUM(pnl), 0) as total_pnl
                FROM fills
                WHERE date(timestamp) = date('now')
            '''
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            daily_pnl = df['total_pnl'].iloc[0]
            
            # Obtener initial capital
            initial_capital = self.config['backtest']['initial_capital']
            daily_loss_pct = daily_pnl / initial_capital
            
            if daily_loss_pct < -self.max_daily_loss:
                logger.warning(f"Daily loss limit hit: {daily_loss_pct:.2%}")
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Error checking daily loss: {e}")
            return True  # Default: permitir si error
    
    def validate_order(self, decision: Dict) -> bool:
        """Valida que la orden sea ejecutable"""
        
        # 1. Check daily loss
        if not self.check_daily_loss():
            logger.warning(f"Order rejected: daily loss limit")
            return False
        
        # 2. Check confidence
        if decision.get('confidence', 0) < self.config['signals']['min_confidence']:
            logger.warning(f"Order rejected: low confidence ({decision.get('confidence')})")
            return False
        
        # 3. Check position size
        if decision.get('position_size_pct', 0) < 0.001:
            logger.warning(f"Order rejected: position too small")
            return False
        
        return True
    
    def place_order(self, decision: Dict) -> Optional[str]:
        """
        Coloca orden en exchange.
        
        Retorna: order_id si success, None si error
        """
        
        # Validación previa
        if not self.validate_order(decision):
            return None
        
        asset = decision['asset']
        signal = decision['signal']
        price = decision['entry_price']
        position_size_pct = decision['position_size_pct']
        
        # Calcular cantidad
        initial_capital = self.config['backtest']['initial_capital']
        quantity = (initial_capital * position_size_pct) / price
        
        # Lado (BUY/SELL)
        side = 'BUY' if signal == 'LONG' else 'SELL'
        
        # Intentos con retry logic
        for attempt in range(self.max_retries):
            try:
                # Aquí iría: order = self.exchange_client.create_market_order(...)
                # Por ahora, MOCK
                
                order_id = f"{asset}_{signal}_{int(time.time())}"
                filled_price = price
                filled_quantity = quantity
                
                logger.info(f"Order placed: {side} {quantity:.4f} {asset} @ {price}")
                
                # Guardar en DB
                self._save_order_to_db({
                    'order_id': order_id,
                    'asset': asset,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'filled_price': filled_price,
                    'filled_quantity': filled_quantity,
                    'status': 'FILLED',
                    'attempts': attempt + 1
                })
                
                return order_id
            
            except Exception as e:
                logger.warning(f"Order attempt {attempt+1}/{self.max_retries} failed: {e}")
                
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    # Guardar error en DB
                    self._save_order_to_db({
                        'order_id': f"{asset}_{signal}_{int(time.time())}_ERROR",
                        'asset': asset,
                        'side': side,
                        'quantity': quantity,
                        'price': price,
                        'status': 'FAILED',
                        'error_message': str(e),
                        'attempts': self.max_retries
                    })
                    
                    return None
        
        return None
    
    def _save_order_to_db(self, order_info: Dict):
        """Guarda orden en DB"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO orders 
                (timestamp, order_id, asset, side, quantity, price, 
                 status, filled_price, filled_quantity, error_message, attempts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                order_info.get('order_id'),
                order_info.get('asset'),
                order_info.get('side'),
                order_info.get('quantity', 0),
                order_info.get('price', 0),
                order_info.get('status'),
                order_info.get('filled_price'),
                order_info.get('filled_quantity'),
                order_info.get('error_message'),
                order_info.get('attempts', 1)
            ))
            
            conn.commit()
            conn.close()
        
        except Exception as e:
            logger.error(f"Error saving order: {e}")
    
    def update_fills(self):
        """Actualiza fills desde exchange (sincronización)"""
        try:
            # Aquí iría: obtener fills del exchange y actualizar DB
            # Por ahora, MOCK
            pass
        
        except Exception as e:
            logger.error(f"Error updating fills: {e}")
    
    def run_cycle(self) -> Dict:
        """
        Ejecuta 1 ciclo: obtiene decisiones → coloca órdenes → log
        """
        cycle_start = time.time()
        
        try:
            # 1. Obtener decisiones pendientes
            pending = self.get_pending_decisions()
            logger.info(f"Pending decisions: {len(pending)}")
            
            # 2. Colocar órdenes
            orders_placed = 0
            for decision in pending:
                order_id = self.place_order(decision)
                if order_id:
                    orders_placed += 1
            
            # 3. Sincronizar fills
            self.update_fills()
            
            cycle_time = (time.time() - cycle_start) * 1000
            
            logger.info(f"Cycle complete: {orders_placed} orders placed, {cycle_time:.0f}ms")
            
            return {
                'cycle_time_ms': cycle_time,
                'orders_placed': orders_placed
            }
        
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            return {}
    
    def run(self, interval_seconds: int = 60):
        """Loop principal: ejecuta ciclo cada N segundos"""
        
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
    executor.run(interval_seconds=60)
