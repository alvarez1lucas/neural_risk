# neural_risk/models/hmm_model.py
"""
ESTADO: fallback agregado si el jurado no aprueba columnas hurst/garch.
"""
from neural_risk.models.base import RiskModel
from hmmlearn.hmm import GaussianHMM
import numpy as np

class RegimeHMMModel(RiskModel):
    def __init__(self, n_components=3):
        super().__init__(model_name="HMM_Regime_Detector")
        self.model = GaussianHMM(n_components=n_components, covariance_type="full", n_iter=1000)
        self._context_features = None

    def _get_context_features(self, X):
        context_features = [c for c in X.columns if 'hurst' in c.lower() or 'garch' in c.lower()]
        if not context_features:
            context_features = X.select_dtypes(include=[np.number]).columns[:3].tolist()
        return context_features

    def fit(self, X, y=None):
        context_features = self._get_context_features(X)
        self._context_features = context_features
        data = X[context_features].values
        print(f"Entrenando {self.model_name} con estados de mercado... features usadas: {context_features}")
        self.model.fit(data)
        self.is_trained = True

    def predict(self, X):
        context_features = self._context_features or self._get_context_features(X)
        return self.model.predict(X[context_features].values)