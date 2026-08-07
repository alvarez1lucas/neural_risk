# neural_risk/models/causal_strategy.py
# NOTA (sin fix aplicado, menor): shockea todas las columnas por igual
# al 1%, incluidas posibles binarias/normalizadas -- metodologicamente
# flojo pero no rompe nada. Pendiente si se quiere mas precision.
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

    def estimate_effect(self, X):
        base_pred = self.model.predict(X)
        X_shock = X * 1.01
        shock_pred = self.model.predict(X_shock)
        return shock_pred - base_pred