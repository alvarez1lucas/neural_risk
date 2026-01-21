import pandas as pd
import numpy as np

class RiskLabeler:
    """
    Genera etiquetas (targets) para entrenar modelos de Machine Learning
    enfocados en la predicción de eventos de riesgo.
    """
    
    @staticmethod
    def label_volatility_jump(series: pd.Series, window: int = 5, threshold_std: float = 1.5):
        """
        Target: 1 si la volatilidad futura es significativamente mayor a la actual.
        """
        # Volatilidad realizada móvil (pasada)
        past_vol = series.pct_change().rolling(window=20).std()
        
        # Volatilidad futura (lo que queremos predecir)
        # Shift(-window) mueve los datos del futuro al presente para el entrenamiento
        future_vol = series.pct_change().rolling(window=window).std().shift(-window)
        
        # 1 si hay salto, 0 si no
        return (future_vol > (past_vol * threshold_std)).astype(int)

    @staticmethod
    def label_returns_direction(series: pd.Series, horizon: int = 1):
        """Target simple de dirección para modelos Challenger."""
        return (series.shift(-horizon) > series).astype(int)