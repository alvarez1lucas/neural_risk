"""Optimización de cobertura con Differential Evolution para hedging eficiente."""
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from typing import Dict, Optional

from neural_risk.metrics.risk_analytics import RiskAnalytics


class PortfolioHedgeOptimizer:
    """
    Optimiza los PESOS de asignación de capital a nivel PORTAFOLIO
    (entre varios activos) usando Differential Evolution para minimizar
    el Expected Shortfall (CVaR) conjunto.

    NO es redundante con PortfolioAgent (verificado antes de implementar):
    - PortfolioAgent.PositionSizer dimensiona CADA activo de forma
      INDEPENDIENTE (Kelly fraccionado por activo, sin considerar cómo
      se mueven juntos los activos entre sí).
    - Este optimizador busca la asignación CONJUNTA de capital entre
      activos que minimice el riesgo de cola del PORTAFOLIO como un
      todo, usando la matriz de retornos conjunta (correlación incluida).
    Son complementarios.
    """

    def __init__(self, confidence: float = 0.95, max_iter: int = 200, seed: int = 42):
        self.confidence = confidence
        self.max_iter = max_iter
        self.seed = seed
        self.last_result = None

    def _portfolio_returns(self, weights: np.ndarray, returns_matrix: np.ndarray) -> np.ndarray:
        return returns_matrix @ weights

    def _objective(self, weights: np.ndarray, returns_matrix: np.ndarray) -> float:
        w = np.abs(weights)
        total = w.sum()
        if total == 0:
            return 1e6
        w = w / total
        port_returns = self._portfolio_returns(w, returns_matrix)
        es = RiskAnalytics.calculate_expected_shortfall(port_returns, confidence=self.confidence)
        return -es

    def optimize(self, returns_by_asset: Dict[str, np.ndarray],
                min_weight: float = 0.0, max_weight: float = 1.0) -> Dict:
        """
        returns_by_asset: {'BTC': array_de_retornos, 'ETH': array_de_retornos, ...}
        Todos con la misma longitud (alineados en el tiempo).
        """
        asset_names = list(returns_by_asset.keys())
        n_assets = len(asset_names)

        if n_assets < 2:
            return {
                'weights': {asset_names[0]: 1.0} if asset_names else {},
                'expected_shortfall': None, 'success': False,
                'reason': 'Se necesitan >=2 activos para optimizar hedging conjunto'
            }

        lengths = [len(v) for v in returns_by_asset.values()]
        if len(set(lengths)) > 1:
            raise ValueError(
                f"Los retornos de cada activo deben tener la misma longitud, "
                f"recibido: {dict(zip(asset_names, lengths))}"
            )

        returns_matrix = np.column_stack([returns_by_asset[a] for a in asset_names])
        bounds = [(min_weight, max_weight)] * n_assets

        result = differential_evolution(
            self._objective, bounds, args=(returns_matrix,),
            maxiter=self.max_iter, seed=self.seed, polish=True
        )

        w = np.abs(result.x)
        w = w / w.sum()

        final_es = RiskAnalytics.calculate_expected_shortfall(
            self._portfolio_returns(w, returns_matrix), confidence=self.confidence
        )

        self.last_result = {
            'weights': {a: float(wi) for a, wi in zip(asset_names, w)},
            'expected_shortfall': float(final_es),
            'success': bool(result.success), 'iterations': int(result.nit)
        }
        return self.last_result

    def suggest_rebalance(self, current_weights: Dict[str, float],
                         optimal_weights: Dict[str, float], threshold: float = 0.05) -> Dict:
        """Compara pesos actuales vs. óptimos, sugiere si rebalancear (evita costos por diffs chicas)."""
        all_assets = set(list(current_weights.keys()) + list(optimal_weights.keys()))
        deltas = {a: optimal_weights.get(a, 0) - current_weights.get(a, 0) for a in all_assets}
        max_delta = max((abs(d) for d in deltas.values()), default=0.0)
        return {
            'deltas': deltas, 'max_delta': max_delta,
            'rebalance_recommended': max_delta > threshold
        }