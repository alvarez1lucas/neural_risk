# neural_risk/models/causal_strategy.py
"""
ESTADO: Parcheado -- FIX CRITICO. CausalInferenceModel no implementaba
predict() (metodo abstracto de RiskModel), lo cual la hacia LITERALMENTE
NO INSTANCIABLE (TypeError: Can't instantiate abstract class...). Como
fit_fast_tier_experts() envuelve la creacion en try/except, este error
quedaba silenciosamente atrapado en TODOS los ciclos, sin excepcion --
el experto CAUSAL nunca funciono, ni una sola vez, desde el principio
del proyecto. Confirmado con test real y arreglado.
"""
from neural_risk.models.base import RiskModel
import pandas as pd
import numpy as np
import xgboost as xgb

class CausalInferenceModel(RiskModel):
    def __init__(self):
        super().__init__(model_name="Causal_Effect_Model")
        self.model = xgb.XGBRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        """
        Predice señal de trading binaria (1 = comprar, 0 = vender).
        Usa la predicción cruda y aplica umbral 0.5.
        """
        if not self.is_trained:
            raise Exception("El modelo debe ser entrenado antes de predecir.")
        raw_pred = self.model.predict(X)
        return (raw_pred > 0.5).astype(int)

    def estimate_effect(self, X):
        """Calcula el impacto causado por un cambio en las variables de control."""
        base_pred = self.model.predict(X)
        X_shock = X * 1.01
        shock_pred = self.model.predict(X_shock)
        return shock_pred - base_pred