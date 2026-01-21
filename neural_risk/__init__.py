# neural_risk/__init__.py

# 1. Importamos de la carpeta 'data' y el archivo 'data_processor'
from .data.data_processor import DataProcessor

# 2. Importamos de la carpeta 'cortex' y el archivo 'causal_selector'
from .cortex.causal_selector import CausalFeatureSelector

# 3. Importamos de la carpeta 'agents' y el archivo 'strategy_router'
from .agents.strategy_router import StrategyRouter

# 4. Importamos modelos (si ya tienen código adentro)
from .models.risk_model import NeuralRiskModel
from .models.trainer import RiskTrainer