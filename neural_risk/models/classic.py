from .base import RiskModel
import xgboost as xgb
import numpy as np

class XGBoostVolModel(RiskModel):
    """
    Modelo de XGBoost optimizado para la predicción de volatilidad realizada.
    """
    def __init__(self, params=None):
        super().__init__(model_name="XGBoost_Volatility")
        self.params = params if params else {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 6
        }
        self.model = xgb.XGBRegressor(**self.params)

    def fit(self, X, y):
        print(f"Entrenando {self.model_name}...")
        self.model.fit(X, y)
        self.is_trained = True
        
    def predict(self, X):
        if not self.is_trained:
            raise Exception("El modelo debe ser entrenado antes de predecir.")
        return self.model.predict(X)