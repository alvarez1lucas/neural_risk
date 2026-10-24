import pandas as pd
import numpy as np

class RiskLabeler:
    """
    Evolución del Labeler para un balance 70% Retorno / 30% Risk.
    Diseñado para ser agnóstico al activo y temporalidad.
    """

    @staticmethod
    def triple_barrier_label(series: pd.Series, t_events: int = 5, pt_sl: list = [2, 1]):
        """
        Versión simplificada del Triple-Barrier Method.
        - 70% enfoque en capturar tendencias (Profit Taking).
        - 30% enfoque en evitar colas (Stop Loss).
        
        Args:
            series: Precios de cierre.
            t_events: Horizonte temporal (barrera vertical).
            pt_sl: Multiplicadores para [Profit Taking, Stop Loss].
        """
        # 1. Calculamos Volatilidad Diaria para que las barreras sean adaptativas
        # Esto hace que el código funcione igual para BTC (volatilidad alta) o Oro (baja)
        returns = series.pct_change()
        vol = returns.rolling(window=20).std()
        
        # 2. Definir barreras dinámicas
        upper_barrier = vol * pt_sl[0]
        lower_barrier = vol * pt_sl[1]
        
        prices = series.values
        upper = upper_barrier.values
        lower = lower_barrier.values

        labels = []
        for i in range(len(series) - t_events):
            # Retorno acumulado en la ventana futura
            cum_returns = (prices[i+1 : i+t_events+1] / prices[i]) - 1

            # FIX: antes se chequeaba any(>upper) antes que any(<lower), sin
            # mirar el orden temporal -- un camino que tocaba el stop y
            # después el profit quedaba etiquetado +1. Ahora gana la
            # barrera que se toca PRIMERO. Con vol NaN (warm-up) -> 0.
            hit_up = np.flatnonzero(cum_returns > upper[i])
            hit_dn = np.flatnonzero(cum_returns < -lower[i])
            first_up = hit_up[0] if len(hit_up) else np.inf
            first_dn = hit_dn[0] if len(hit_dn) else np.inf

            if first_up < first_dn:
                labels.append(1)   # Éxito (Alpha)
            elif first_dn < first_up:
                labels.append(-1)  # Riesgo (RM)
            else:
                labels.append(0)   # Neutral/Ruido

        return pd.Series(labels, index=series.index[:-t_events])

    @staticmethod
    def label_risk_adjusted_returns(series: pd.Series, window: int = 20, threshold: float = 0.0):
        """
        Target basado en Ratio de Sharpe local. 
        Busca momentos donde el retorno justifica el riesgo.
        """
        returns = series.pct_change()
        rolling_sharpe = (returns.rolling(window).mean() / returns.rolling(window).std()) * np.sqrt(252)
        
        # Shift para predecir el Sharpe del futuro
        future_sharpe = rolling_sharpe.shift(-window)
        return (future_sharpe > threshold).astype(int)