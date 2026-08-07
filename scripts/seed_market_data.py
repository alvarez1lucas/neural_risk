# scripts/seed_market_data.py
"""
Siembra histórico sintético en market_data para poder probar el sistema
completo SIN esperar ~17hs a que run_data_fetcher.py acumule 200 filas
a 300s por ciclo. Genera velas DIARIAS sintéticas (random walk realista)
para los últimos ~250 días, por cada activo de config.yaml.

Uso: python scripts/seed_market_data.py
Después de correr esto, train_models.py y run_engine.py ya tienen
suficiente historia para trabajar desde el primer ciclo.

NOTA: esto es solo para PROBAR el sistema rápido. Para un uso real,
dejá que run_data_fetcher.py (con datos reales, use_mock=false) acumule
histórico genuino con el tiempo -- esta siembra es 100% sintética.
"""
import sqlite3
import yaml
import os
import numpy as np
from datetime import datetime, timedelta

np.random.seed(123)


def generate_daily_ohlcv(n_days, start_price):
    returns = np.random.randn(n_days) * 0.02
    close = start_price * np.cumprod(1 + returns)
    open_ = np.roll(close, 1)
    open_[0] = start_price
    daily_range = np.abs(np.random.randn(n_days)) * 0.02 * close
    high = np.maximum(open_, close) + daily_range * 0.5
    low = np.minimum(open_, close) - daily_range * 0.5
    volume = np.random.uniform(1e6, 1e7, n_days)
    return open_, high, low, close, volume


def main():
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    db_path = config['database']['path']
    assets = config['exchanges']['assets']
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY, timestamp DATETIME, asset TEXT,
            open REAL, price REAL, volume REAL, high REAL, low REAL,
            UNIQUE(timestamp, asset)
        )
    ''')
    conn.commit()

    n_days = 250
    base_prices = {"BTC": 45000, "ETH": 2500, "SOL": 100, "AVAX": 35, "ARB": 1.2}
    start_date = datetime.now() - timedelta(days=n_days)

    total_inserted = 0
    for asset in assets:
        base_price = base_prices.get(asset, 50)
        open_, high, low, close, volume = generate_daily_ohlcv(n_days, base_price)

        rows_inserted = 0
        for i in range(n_days):
            ts = start_date + timedelta(days=i)
            try:
                cursor.execute('''
                    INSERT INTO market_data (timestamp, asset, open, price, volume, high, low)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (ts.isoformat(), asset, float(open_[i]), float(close[i]),
                      float(volume[i]), float(high[i]), float(low[i])))
                rows_inserted += 1
            except sqlite3.IntegrityError:
                pass

        print(f"  {asset}: {rows_inserted} filas sembradas ({start_date.date()} a {datetime.now().date()})")
        total_inserted += rows_inserted

    conn.commit()
    conn.close()
    print(f"\nTotal: {total_inserted} filas sembradas en {db_path}")
    print("Ya podés correr train_models.py sin esperar acumulación real.")


if __name__ == "__main__":
    main()