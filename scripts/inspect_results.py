# scripts/inspect_results.py
"""
Muestra un resumen rápido del estado del sistema: última data cacheada,
últimas decisiones del engine, órdenes/fills del executor, y
recomendaciones de hedging. Pensado para correr mientras los 4
servicios están activos en background, para ver qué está pasando sin
tener que escribir SQL a mano.

Uso: python scripts/inspect_results.py
"""
import sqlite3
import yaml
import pandas as pd
import pickle
import os
from datetime import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 140)


def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)
    db_path = config['database']['path']

    conn = sqlite3.connect(db_path)

    print("=" * 70)
    print(f"RESUMEN DEL SISTEMA -- {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- 1. Market data ---
    try:
        df = pd.read_sql("SELECT asset, COUNT(*) as filas, MAX(timestamp) as ultimo_dato FROM market_data GROUP BY asset", conn)
        print("\n[1] MARKET DATA (histórico disponible por activo)")
        print(df.to_string(index=False) if len(df) else "  (vacío -- ¿corriste seed_market_data.py o run_data_fetcher.py?)")
    except Exception as e:
        print(f"\n[1] MARKET DATA: error -- {e}")

    # --- 2. Modelos entrenados (pickle) ---
    print("\n[2] MODELOS ENTRENADOS (data/trained_models.pkl)")
    cache_path = "./data/trained_models.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        for asset, entry in cache.items():
            fast_ts = entry.get('fast', {}).get('last_trained', 'nunca')
            slow_ts = entry.get('slow', {}).get('last_trained', 'nunca')
            n_feats = len(entry.get('best_feats', []))
            print(f"  {asset}: {n_feats} features | fast_tier={fast_ts} | slow_tier={slow_ts}")
    else:
        print("  (no existe todavía -- ¿corriste train_models.py?)")

    # --- 3. Últimas decisiones del engine ---
    try:
        df = pd.read_sql(
            "SELECT timestamp, asset, signal, confidence, agreement, entry_price, stop_loss "
            "FROM engine_decisions ORDER BY timestamp DESC LIMIT 15", conn
        )
        print(f"\n[3] ÚLTIMAS DECISIONES DEL ENGINE ({len(df)} de las más recientes)")
        print(df.to_string(index=False) if len(df) else "  (vacío -- ¿corriste run_engine.py?)")
    except Exception as e:
        print(f"\n[3] ENGINE_DECISIONS: error -- {e}")

    # --- 4. Órdenes abiertas ---
    try:
        df = pd.read_sql(
            "SELECT timestamp, asset, side, quantity, price, status, stop_loss "
            "FROM orders WHERE status = 'FILLED' ORDER BY timestamp DESC", conn
        )
        print(f"\n[4] POSICIONES ABIERTAS ({len(df)})")
        print(df.to_string(index=False) if len(df) else "  (ninguna abierta)")
    except Exception as e:
        print(f"\n[4] ORDERS: error -- {e}")

    # --- 5. Fills / PnL realizado ---
    try:
        df = pd.read_sql("SELECT COUNT(*) as n_fills, COALESCE(SUM(pnl),0) as pnl_total FROM fills", conn)
        n_fills = df['n_fills'].iloc[0]
        pnl_total = df['pnl_total'].iloc[0]
        print(f"\n[5] FILLS / PnL REALIZADO")
        print(f"  Trades cerrados: {n_fills} | PnL total: {pnl_total:.2f}")
        if n_fills > 0:
            last_fills = pd.read_sql(
                "SELECT timestamp, asset, quantity, fill_price, pnl FROM fills ORDER BY timestamp DESC LIMIT 10", conn
            )
            print(last_fills.to_string(index=False))
    except Exception as e:
        print(f"\n[5] FILLS: error -- {e}")

    # --- 6. Recomendaciones de hedging ---
    try:
        df = pd.read_sql(
            "SELECT timestamp, expected_shortfall, max_delta, rebalance_recommended, executed_at "
            "FROM portfolio_hedge_recommendations ORDER BY timestamp DESC LIMIT 5", conn
        )
        print(f"\n[6] RECOMENDACIONES DE HEDGING ({len(df)} de las más recientes)")
        print(df.to_string(index=False) if len(df) else "  (vacío -- ¿corriste run_portfolio_hedge.py?)")
    except Exception as e:
        print(f"\n[6] HEDGE RECOMMENDATIONS: error -- {e} (tabla puede no existir todavía)")

    conn.close()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()