#!/usr/bin/env python3
"""
QUICK TRAIN: Entrena modelos desde CSV (para testing/demo)
Alternativa rápida a train_models.py cuando no hay datos en DB
"""

import os
import sys
import pickle
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [QUICK_TRAIN] - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_dummy_models():
    """
    Entrena modelos simplificados y los guarda en pickle.
    Usado para demo/testing cuando no hay datos en DB.
    """
    
    logger.info("=" * 60)
    logger.info("Quick training from CSV files")
    logger.info("=" * 60)
    
    # Carga datos desde CSV
    btc_data = None
    eth_data = None
    
    for csv_file in ['data/BTC_USD_data.csv', 'data/ETH_USD_data.csv']:
        if Path(csv_file).exists():
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"✅ Loaded {csv_file} ({len(df)} rows)")
                if 'BTC' in csv_file:
                    btc_data = df
                else:
                    eth_data = df
            except Exception as e:
                logger.error(f"Failed to load {csv_file}: {e}")
    
    # Modelos dummy (para demo)
    models = {}
    
    for asset, data in [('BTC', btc_data), ('ETH', eth_data)]:
        if data is None:
            logger.warning(f"No data for {asset}, skipping")
            continue
        
        logger.info(f"\nTraining models for {asset}...")
        
        # Modelo 1: Simple MA (Moving Average)
        closes = data['Close'].values if 'Close' in data.columns else data['High'].values
        
        # Store training info (simplified)
        models[f'{asset}_ma'] = {
            'type': 'moving_average',
            'mean': float(closes.mean()),
            'std': float(closes.std()),
            'last_price': float(closes[-1]),
        }
        
        # Modelo 2: Volatility (simplified GARCH-like)
        returns = np.diff(closes) / closes[:-1]
        models[f'{asset}_volatility'] = {
            'type': 'volatility',
            'mean_return': float(returns.mean()),
            'volatility': float(returns.std()),
        }
        
        logger.info(f"  ✅ MA: mean=${closes.mean():.2f}, std=${closes.std():.2f}")
        logger.info(f"  ✅ Vol: return={returns.mean():.4f}, volatility={returns.std():.4f}")
    
    # Guardar en pickle
    try:
        output_path = Path('data/trained_models.pkl')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(models, f)
        
        size_mb = output_path.stat().st_size / (1024 * 1024)
        logger.info(f"\n✅ Models saved to {output_path} ({size_mb:.2f} MB)")
        
    except Exception as e:
        logger.error(f"Failed to save models: {e}")
        return False
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Quick training complete!")
    logger.info("=" * 60)
    logger.info("\nNext: python scripts/pre_deploy_check.py")
    
    return True


if __name__ == "__main__":
    success = train_dummy_models()
    sys.exit(0 if success else 1)
