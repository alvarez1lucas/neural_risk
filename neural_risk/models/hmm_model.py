# neural_risk/models/hmm_model.py
from .base import RiskModel
from hmmlearn.hmm import GaussianHMM
import numpy as np

class RegimeHMMModel(RiskModel):
    """
    Clasifica el mercado en regímenes (ej: Tendencia, Rango, Caos)
    usando métricas de Hurst y GARCH para identificar la microestructura.
    """
    def __init__(self, n_components=3):
        super().__init__(model_name="HMM_Regime_Detector")
        self.model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000)

    def fit(self, X, y=None):
        # Seleccionamos features clave de tu pipeline (Hurst, Volatilidad, etc.)
        context_features = [c for c in X.columns if 'hurst' in c.lower() or 'garch' in c.lower()]
        data = X[context_features].values
        print(f"Entrenando {self.model_name} con estados de mercado...")
        self.model.fit(data)
        self.is_trained = True

    def predict(self, X):
        context_features = [c for c in X.columns if 'hurst' in c.lower() or 'garch' in c.lower()]
        # Devuelve el estado actual (0, 1 o 2) para que el Agente ajuste el riesgo
        return self.model.predict(X[context_features].values)