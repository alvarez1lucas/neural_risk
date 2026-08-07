# neural_risk/__init__.py
"""
Neural Risk Engine: Sistema Integrado de Gestion de Riesgo Cuantitativo

ESTADO: Reescrito para importar de forma DEFENSIVA. Antes, este archivo
importaba TODO el arbol de forma ansiosa -- lo que significaba que
`import neural_risk.agents.portfolio_agent` (que solo necesita numpy/
pandas/scipy) fallaba con ImportError si no tenias instalado torch,
xgboost, hmmlearn, arch y statsmodels, aunque PortfolioAgent no use
ninguna de esas librerias. Esto iba directamente en contra del objetivo
de escalabilidad de la libreria: cualquiera que quisiera usar una sola
pieza liviana estaba obligado a instalar el peso completo.

Ahora cada import esta envuelto en try/except. Si una dependencia
pesada falta, ese simbolo queda como None en el namespace (no rompe el
import del paquete completo) y __all__ solo lista lo que efectivamente
cargo. Simbolos en None dan un error claro y especifico recien cuando
se INTENTAN USAR, no al importar el paquete.
"""

import warnings

__version__ = "0.2.0"
__all__ = []


def _safe_import(module_path, names, extra_note=""):
    """
    Importa 'names' desde 'module_path'. Si falla (dependencia externa
    faltante), deja esos simbolos en None y avisa con un warning en vez
    de tirar abajo la carga de TODO el paquete.
    """
    import importlib
    try:
        mod = importlib.import_module(module_path, package="neural_risk")
        result = {}
        for name in names:
            obj = getattr(mod, name)
            result[name] = obj
            __all__.append(name)
        return result
    except ImportError as e:
        warnings.warn(
            f"neural_risk: no se pudo cargar {module_path} ({e}). "
            f"Los simbolos {names} no estaran disponibles hasta instalar "
            f"la dependencia faltante.{(' ' + extra_note) if extra_note else ''}",
            ImportWarning
        )
        return {name: None for name in names}


# ---- Data Pipeline ----
_r = _safe_import('.data.data_processor', ['DataProcessor'])
DataProcessor = _r['DataProcessor']

_r = _safe_import('.data.feature_engineering', ['RiskFeaturePipeline'],
                   extra_note="Requiere statsmodels, hmmlearn, networkx, torch.")
RiskFeaturePipeline = _r['RiskFeaturePipeline']

_r = _safe_import('.data.labeling', ['RiskLabeler'])
RiskLabeler = _r['RiskLabeler']

# ---- Cortex (Feature Selection) ----
_r = _safe_import('.cortex.feature_jury', ['FeatureJury'],
                   extra_note="Requiere statsmodels.")
FeatureJury = _r['FeatureJury']

_r = _safe_import('.cortex.causal_selector', ['CausalSelector'],
                   extra_note="Requiere statsmodels.")
CausalSelector = _r['CausalSelector']

# ---- Models (nucleo neuronal) ----
_r = _safe_import('.models.risk_model', ['NeuralRiskModel'], extra_note="Requiere torch.")
NeuralRiskModel = _r['NeuralRiskModel']

_r = _safe_import('.models.trainer', ['RiskTrainer'], extra_note="Requiere torch.")
RiskTrainer = _r['RiskTrainer']

_r = _safe_import('.models.base', ['RiskModel'])
RiskModel = _r['RiskModel']

_r = _safe_import('.models.ensemble_trainer', ['EnsembleTrainer', 'KalmanFilterRegime'],
                   extra_note="Requiere torch, xgboost.")
EnsembleTrainer = _r['EnsembleTrainer']
KalmanFilterRegime = _r['KalmanFilterRegime']

_r = _safe_import('.models.temporal_cv', ['TemporalCrossValidator', 'WalkForwardValidator'])
TemporalCrossValidator = _r['TemporalCrossValidator']
WalkForwardValidator = _r['WalkForwardValidator']

# ---- Models (9 expertos) ----
_r = _safe_import('.models.garch_volatility', ['GARCHVolatilityExpert', 'MultiWindowGARCH'],
                   extra_note="Requiere arch.")
GARCHVolatilityExpert = _r['GARCHVolatilityExpert']
MultiWindowGARCH = _r['MultiWindowGARCH']

_r = _safe_import('.models.lstm_transformer',
                   ['LSTMForecastingExpert', 'TransformerForecastingExpert', 'SequentialForecastingEnsemble'],
                   extra_note="Requiere torch.")
LSTMForecastingExpert = _r['LSTMForecastingExpert']
TransformerForecastingExpert = _r['TransformerForecastingExpert']
SequentialForecastingEnsemble = _r['SequentialForecastingEnsemble']

_r = _safe_import('.models.reinforcement_learning',
                   ['RLAllocationExpert', 'MultiArmedBanditExpert', 'MarketEnvironment'],
                   extra_note="RLAllocationExpert requiere stable-baselines3 (alcance futuro de la libreria).")
RLAllocationExpert = _r['RLAllocationExpert']
MultiArmedBanditExpert = _r['MultiArmedBanditExpert']
MarketEnvironment = _r['MarketEnvironment']

_r = _safe_import('.models.copula_expert', ['CopulaExpert', 'MultiAssetCopulaExpert'],
                   extra_note="CopulaExpert requiere copulae (alcance futuro de la libreria).")
CopulaExpert = _r['CopulaExpert']
MultiAssetCopulaExpert = _r['MultiAssetCopulaExpert']

_r = _safe_import('.models.anomaly_detection',
                   ['AnomalyDetector', 'AnomalyDetectionAutoencoder', 'DynamicAnomalyThreshold'],
                   extra_note="Requiere torch.")
AnomalyDetector = _r['AnomalyDetector']
AnomalyDetectionAutoencoder = _r['AnomalyDetectionAutoencoder']
DynamicAnomalyThreshold = _r['DynamicAnomalyThreshold']

# ---- Agents ----
_r = _safe_import('.agents.strategy_router', ['StrategyRouter'])
StrategyRouter = _r['StrategyRouter']

_r = _safe_import('.agents.portfolio_agent', ['PortfolioAgent'])
PortfolioAgent = _r['PortfolioAgent']

# ---- Metrics ----
_r = _safe_import('.metrics.performance', ['PerformanceMetrics'])
PerformanceMetrics = _r['PerformanceMetrics']

_r = _safe_import('.metrics.risk_analytics', ['RiskAnalytics'])
RiskAnalytics = _r['RiskAnalytics']

_r = _safe_import('.metrics.technical', ['MarketIndicators'])
MarketIndicators = _r['MarketIndicators']

# ---- Optimization ----
_r = _safe_import('.optimization.hedging', ['PortfolioHedgeOptimizer'])
PortfolioHedgeOptimizer = _r['PortfolioHedgeOptimizer']

# ---- Engine (requiere TODO lo de arriba para funcionar de verdad,
# pero el import en si mismo no falla si algo falta -- fallara recien
# si se intenta instanciar AutomatedRiskEngine sin las dependencias
# de los expertos que efectivamente use) ----
_r = _safe_import('.engine', ['AutomatedRiskEngine'],
                   extra_note="Requiere TODAS las dependencias de los 9 expertos para operar de verdad.")
AutomatedRiskEngine = _r['AutomatedRiskEngine']