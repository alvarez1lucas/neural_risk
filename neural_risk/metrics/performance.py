# neural_risk/metrics/performance.py
import numpy as np

class PerformanceMetrics:
    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0, periods=252):
        """Ratio de Sharpe anualizado."""
        adj_returns = returns - (risk_free_rate / periods)
        if np.std(adj_returns) == 0: return 0
        return np.mean(adj_returns) / np.std(adj_returns) * np.sqrt(periods)

    @staticmethod
    def sortino_ratio(returns, risk_free_rate=0, periods=252):
        """Ratio de Sortino: Solo penaliza la volatilidad negativa."""
        adj_returns = returns - (risk_free_rate / periods)
        downside_std = np.std(returns[returns < 0])
        if downside_std == 0: return 0
        return np.mean(adj_returns) / downside_std * np.sqrt(periods)

    @staticmethod
    def max_drawdown(equity_curve):
        """La caída más grande desde un pico."""
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        return drawdown.min()