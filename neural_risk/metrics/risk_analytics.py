# neural_risk/metrics/risk_analytics.py
import numpy as np
import pandas as pd
from scipy.stats import norm

class RiskAnalytics:
    @staticmethod
    def calculate_var(returns, confidence=0.95):
        """Value at Risk Paramétrico."""
        mu = np.mean(returns)
        sigma = np.std(returns)
        return norm.ppf(1 - confidence, mu, sigma)

    @staticmethod
    def calculate_expected_shortfall(returns, confidence=0.95):
        """Expected Shortfall (CVaR): Promedio de pérdidas más allá del VaR."""
        var = RiskAnalytics.calculate_var(returns, confidence)
        tail_loss = returns[returns <= var]
        return tail_loss.mean() if len(tail_loss) > 0 else var

    @staticmethod
    def tail_dependence_copula(x, y):
        """
        Métrica simplificada de dependencia de colas.
        Útil para saber si una feature y el precio 'caen juntos' en extremos.
        """
        # Transformar a rangos uniformes para la cópula
        u = pd.Series(x).rank(pct=True)
        v = pd.Series(y).rank(pct=True)
        
        # Medir correlación en el percentil 5% (cola inferior)
        lower_tail = ((u < 0.05) & (v < 0.05)).sum() / (0.05 * len(u))
        return lower_tail
