# neural_risk/research/__init__.py
"""
Infraestructura de investigación: walk-forward, backtest de perpetuos,
señales/ensembles, validación estadística (Deflated Sharpe, PBO) y
registro de experimentos. Solo depende del core (numpy/pandas/scipy/sklearn).
"""

from .backtest import BINANCE_FEES, COST_PER_SIDE, simulate, stats, trades_frame
from .registry import ExperimentRegistry
from .signals import adaptive_ensemble, extreme_signal, rolling_z, weighted_ensemble
from .validation import (deflated_sharpe_ratio, expected_max_sharpe,
                         probabilistic_sharpe_ratio, probability_of_backtest_overfitting,
                         sharpe_per_period)
from .walk_forward import forward_return, ridge_walk_forward, walk_forward

__all__ = [
    'walk_forward', 'ridge_walk_forward', 'forward_return',
    'simulate', 'stats', 'trades_frame', 'COST_PER_SIDE', 'BINANCE_FEES',
    'extreme_signal', 'rolling_z', 'weighted_ensemble', 'adaptive_ensemble',
    'probabilistic_sharpe_ratio', 'deflated_sharpe_ratio', 'expected_max_sharpe',
    'probability_of_backtest_overfitting', 'sharpe_per_period',
    'ExperimentRegistry',
]
