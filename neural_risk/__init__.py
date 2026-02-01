# neural_risk/__init__.py
"""
Neural Risk Engine: Sistema Integrado de Gestión de Riesgo Cuantitativo

Módulos principales:
- data: Procesamiento y ingeniería de features
- cortex: Selección de features y lógica causal
- models: Modelos neuronales y tradicionales
- agents: Router de estrategias y toma de decisiones
- metrics: Análisis de riesgo y performance
- optimization: Optimización de portafolios
- engine: Orquestador central
"""

# Data Pipeline
from .data.data_processor import DataProcessor
from .data.feature_engineering import RiskFeaturePipeline
from .data.labeling import RiskLabeler

# Cortex (Feature Selection)
from .cortex.feature_jury import FeatureJury
from .cortex.causal_selector import CausalSelector

# Models (Clásicos)
from .models.risk_model import NeuralRiskModel
from .models.trainer import RiskTrainer
from .models.base import RiskModel
from .models.ensemble_trainer import EnsembleTrainer, KalmanFilterRegime
from .models.temporal_cv import TemporalCrossValidator, WalkForwardValidator

# Models (Nuevos 5 expertos para cripto)
from .models.garch_volatility import GARCHVolatilityExpert, MultiWindowGARCH
from .models.lstm_transformer import LSTMForecastingExpert, TransformerForecastingExpert, SequentialForecastingEnsemble
from .models.reinforcement_learning import RLAllocationExpert, MultiArmedBanditExpert, MarketEnvironment
from .models.copula_expert import CopulaExpert, MultiAssetCopulaExpert
from .models.anomaly_detection import AnomalyDetector, AnomalyDetectionAutoencoder, DynamicAnomalyThreshold

# Agents
from .agents.strategy_router import StrategyRouter

# Engine
from .engine import AutomatedRiskEngine

__version__ = "0.2.0"  # Actualizado: nuevos expertos integrados

__all__ = [
    # Data Pipeline
    'DataProcessor',
    'RiskFeaturePipeline',
    'RiskLabeler',
    
    # Cortex
    'FeatureJury',
    'CausalSelector',
    
    # Modelos Clásicos
    'NeuralRiskModel',
    'RiskTrainer',
    'RiskModel',
    
    # Ensemble & Validation
    'EnsembleTrainer',
    'KalmanFilterRegime',
    'TemporalCrossValidator',
    'WalkForwardValidator',
    
    # Nuevos Expertos Cripto
    'GARCHVolatilityExpert',
    'MultiWindowGARCH',
    'LSTMForecastingExpert',
    'TransformerForecastingExpert',
    'SequentialForecastingEnsemble',
    'RLAllocationExpert',
    'MultiArmedBanditExpert',
    'MarketEnvironment',
    'CopulaExpert',
    'MultiAssetCopulaExpert',
    'AnomalyDetector',
    'AnomalyDetectionAutoencoder',
    'DynamicAnomalyThreshold',
    
    # Agentes
    'StrategyRouter',
    
    # Engine
    'AutomatedRiskEngine',
]