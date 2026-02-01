# ============================================================================
# SCRIPT INTEGRADOR: Ejecuta el Engine Completo
# ============================================================================
"""
Este script demuestra cómo usar el AutomatedRiskEngine con todos los componentes integrados.

Flujo:
1. Cargar datos de múltiples activos
2. Instanciar todos los componentes del pipeline
3. Ejecutar el engine de punta a punta
4. Analizar resultados
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Imports del proyecto
from neural_risk.data.data_processor import DataProcessor
from neural_risk.data.feature_engineering import RiskFeaturePipeline
from neural_risk.data.labeling import RiskLabeler
from neural_risk.cortex.feature_jury import FeatureJury
from neural_risk.models.risk_model import NeuralRiskModel
from neural_risk.models.trainer import RiskTrainer
from neural_risk.agents.strategy_router import StrategyRouter
from neural_risk.engine import AutomatedRiskEngine


def load_data():
    """Carga los datos de activos disponibles en el workspace."""
    data = {}
    
    base_path = Path(__file__).parent
    
    # Buscar archivos CSV
    csv_files = {
        'BTC': base_path / 'BTC_USD_data.csv',
        'ETH': base_path / 'ETH_USD_data.csv',
    }
    
    for ticker, file_path in csv_files.items():
        if file_path.exists():
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                df.columns = [c.replace('Date', 'Timestamp') for c in df.columns]
                data[ticker] = df
                print(f"✓ Cargado: {ticker} ({len(df)} filas)")
            except Exception as e:
                print(f"✗ Error cargando {ticker}: {e}")
    
    # Si no hay archivos reales, crear datos sintéticos para demo
    if len(data) == 0:
        print("\n⚠️  No se encontraron CSV. Generando datos sintéticos para DEMO...")
        np.random.seed(42)
        
        for ticker in ['BTC', 'ETH', 'SPY']:
            dates = pd.date_range('2023-01-01', periods=500, freq='D')
            prices = 100 * np.exp(np.cumsum(np.random.randn(500) * 0.02))
            
            data[ticker] = pd.DataFrame({
                'Open': prices * (1 + np.random.randn(500) * 0.01),
                'High': prices * (1 + np.abs(np.random.randn(500) * 0.01)),
                'Low': prices * (1 - np.abs(np.random.randn(500) * 0.01)),
                'Close': prices,
                'Volume': np.random.randint(1e6, 1e9, 500)
            }, index=dates)
            print(f"✓ Generado: {ticker} (datos sintéticos, {len(data[ticker])} filas)")
    
    return data


def initialize_components(num_features: int = 60):
    """Inicializa todos los componentes del pipeline."""
    print("\n" + "="*60)
    print("INICIALIZANDO COMPONENTES")
    print("="*60 + "\n")
    
    # 1. Data Processor
    processor = DataProcessor()
    print("✓ DataProcessor inicializado")
    
    # 2. Feature Pipeline
    pipeline = RiskFeaturePipeline()
    print("✓ RiskFeaturePipeline inicializado")
    
    # 3. Labeler
    labeler = RiskLabeler()
    print("✓ RiskLabeler inicializado")
    
    # 4. Feature Jury
    jury = FeatureJury(significance=0.05, max_lag=3, mi_threshold=0.01)
    print("✓ FeatureJury inicializado")
    
    # 5. Neural Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✓ Device: {device}")
    
    model = NeuralRiskModel(
        num_features=num_features,
        hidden_size=64,
        num_heads=4,
        dropout=0.1
    )
    print(f"✓ NeuralRiskModel inicializado ({num_features} features)")
    
    # 6. Trainer
    trainer = RiskTrainer(model, lr=1e-3, device=device)
    print("✓ RiskTrainer inicializado")
    
    # 7. Strategy Router
    router = StrategyRouter(risk_appetite=0.7)
    print("✓ StrategyRouter inicializado")
    
    return processor, pipeline, labeler, jury, model, trainer, router


def main():
    """Ejecuta el pipeline completo."""
    
    print("\n" + "╔" + "="*58 + "╗")
    print("║" + " "*10 + "NEURAL RISK ENGINE - DEMO INTEGRACIÓN" + " "*11 + "║")
    print("╚" + "="*58 + "╝" + "\n")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("📥 PASO 1: Cargando Datos")
    print("-" * 60)
    assets_data = load_data()
    
    if len(assets_data) == 0:
        print("❌ No hay datos disponibles. Abortando.")
        return
    
    # ========================================================================
    # STEP 2: Initialize Components
    # ========================================================================
    print("\n📦 PASO 2: Inicializando Componentes")
    print("-" * 60)
    processor, pipeline, labeler, jury, model, trainer, router = initialize_components(
        num_features=60
    )
    
    # ========================================================================
    # STEP 3: Create Engine
    # ========================================================================
    print("\n⚙️  PASO 3: Creando Engine")
    print("-" * 60)
    engine = AutomatedRiskEngine(
        processor=processor,
        pipeline=pipeline,
        labeler=labeler,
        jury=jury,
        model=model,
        trainer=trainer,
        router=router
    )
    print("✓ AutomatedRiskEngine creado")
    
    # ========================================================================
    # STEP 4: Run Engine
    # ========================================================================
    print("\n🚀 PASO 4: Ejecutando Engine")
    print("-" * 60)
    
    try:
        result = engine.run_portfolio_automation(
            assets_data=assets_data,
            benchmark_name='SPY' if 'SPY' in assets_data else list(assets_data.keys())[0],
            train=True,
            train_split=0.8
        )
        
        # ====================================================================
        # STEP 5: Analyze Results
        # ====================================================================
        print("\n📊 PASO 5: Resultados")
        print("="*60)
        
        print("\n✅ Portfolio Weights:")
        for ticker, weight in result['portfolio_weights'].items():
            print(f"   {ticker}: {weight:.2%}")
        
        print("\n✅ Market State:")
        for key, value in result['market_state'].items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")
        
        print("\n✅ Selected Features per Asset:")
        for ticker, features in result['selected_features'].items():
            print(f"   {ticker}: {len(features)} features")
            if len(features) > 0:
                print(f"      Primeras 5: {', '.join(features[:5])}")
        
        print("\n" + "="*60)
        print("✅ ENGINE EXECUTION COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")
        
        return result
        
    except Exception as e:
        print(f"\n❌ Error durante ejecución: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    result = main()
