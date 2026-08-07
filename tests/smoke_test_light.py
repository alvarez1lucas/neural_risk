"""
SMOKE TEST -- capas que corren sin torch/xgboost/hmmlearn/arch/statsmodels
(PortfolioAgent, StrategyRouter, PortfolioHedgeOptimizer, metrics/).

Simula lo que engine.py PRODUCIRIA (asset_report con las 9 señales) sin
necesitar los modelos pesados reales -- así se valida el circuito
completo de decisión + hedging con datos sintéticos, de punta a punta,
de verdad ejecutando código (no solo revisándolo).
"""
import numpy as np
import pandas as pd
import sys
import os

# FIX: antes tenía hardcodeado el path del sandbox de desarrollo
# ('/home/claude/neural_risk_project'), que no existe en ninguna otra
# máquina -- rompía en Windows/cualquier otro entorno. Ahora se resuelve
# de forma portable, relativa a la ubicación real de este archivo
# (funciona sin importar desde qué directorio se lo invoque).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neural_risk.agents.portfolio_agent import PortfolioAgent
from neural_risk.agents.strategy_router import StrategyRouter
from neural_risk.optimization.hedging import PortfolioHedgeOptimizer

np.random.seed(42)


def fake_asset_report(bias=0.0):
    """Simula lo que predict_with_cached_experts() devolvería."""
    return {
        'hmm_regime': np.random.choice([0, 1, 2]),
        'xgb_signal': np.random.randn() * 0.03 + bias,
        'causal_effect': np.random.randn() * 0.02 + bias,
        'garch_vol': {
            'crisis_detected': np.random.random() < 0.1,
            'vol_forecast': abs(np.random.randn() * 0.02),
        },
        'lstm_forecast': {'ensemble_forecast': np.random.randn() * 0.03 + bias},
        'anomaly': {'anomaly_detected': np.random.random() < 0.05},
        'ensemble': {'mu': np.random.randn() * 0.02 + bias, 'sigma': abs(np.random.randn() * 0.01) + 0.005},
        'hurst': np.clip(np.random.randn() * 0.1 + 0.5, 0, 1),
    }


print("=" * 60)
print("PASO 1: PortfolioAgent.execute_portfolio_decision")
print("=" * 60)

agent = PortfolioAgent(initial_capital=100000)
portfolio_intelligence = {
    'BTC': fake_asset_report(bias=0.03),
    'ETH': fake_asset_report(bias=-0.02),
    'SOL': fake_asset_report(bias=0.01),
}
current_prices = {'BTC': 45000, 'ETH': 2500, 'SOL': 100}

decision_result = agent.execute_portfolio_decision(portfolio_intelligence, current_prices)
for ticker, d in decision_result['decisions'].items():
    print(f"  {ticker}: signal={d['signal']:5s} conf={d['confidence']:.3f} "
          f"SL={d['stop_loss']:.2f} size={d['position_size_pct']:.4f} "
          f"garch_trust={d['garch_trust']:.2f} anomaly_trust={d['anomaly_trust']:.2f}")

print("\nSimulando 30 trades cerrados para poblar el circuito de feedback...")
for i in range(30):
    ticker = np.random.choice(['BTC', 'ETH', 'SOL'])
    entry = current_prices[ticker]
    signal = np.random.choice(['LONG', 'SHORT'])
    exit_price = entry * (1 + np.random.randn() * 0.02)
    agent.update_positions(
        ticker=ticker, signal=signal, entry_price=entry, stop_loss=entry * 0.98,
        position_size=1.0, confidence=0.7,
        expert_votes={'HMM': 0.1, 'XGB': 0.2, 'GARCH': -0.1, 'ANOMALY': 0.0, 'ENSEMBLE': 0.15}
    )
    agent.close_position(ticker, exit_price)

metrics = agent.get_portfolio_metrics()
print("\nMétricas del portafolio tras 30 trades (incluye max_drawdown/VaR/CVaR nuevos):")
for k, v in metrics.items():
    print(f"  {k}: {v}")

assert metrics['portfolio_value'] != agent.initial_capital, \
    "FALLO: portfolio_value sigue congelado (el fix no funcionó)"
print("\n  OK: portfolio_value SE ACTUALIZA correctamente tras cerrar trades (fix verificado)")

weights_after = agent.weighting.get_weights()
print(f"\n  Pesos de Thompson Sampling tras feedback: {weights_after}")
uniform = 1.0 / len(weights_after)
assert any(abs(w - uniform) > 1e-6 for w in weights_after.values()), \
    "FALLO: los pesos siguen uniformes, Thompson Sampling no se movió"
print("  OK: los pesos DEJARON de ser uniformes (Thompson Sampling está aprendiendo)")


print("\n" + "=" * 60)
print("PASO 2: StrategyRouter.allocate_capital")
print("=" * 60)

router = StrategyRouter(risk_appetite=0.7)
signals = {'BTC': (0.02, 0.01), 'ETH': (-0.015, 0.02), 'SOL': (0.01, 0.03)}
df_master = pd.DataFrame({t: {'mu': s[0], 'sigma': s[1]} for t, s in signals.items()}).T
market_state = {'hurst': 0.6, 'volatility': 0.02, 'crash_prob': 0.1}

weights = router.allocate_capital(signals, df_master, market_state)
print(f"  Pesos asignados: {weights}")
assert abs(sum(weights.values()) - 1.0) < 1e-6, "FALLO: los pesos no suman 1.0"
print(f"  OK: suma = {sum(weights.values()):.6f} (=1.0)")

# Verificar el fix del bug de Kelly (MEAN_REVERSION no debe colapsar a 0.05 siempre)
kelly_neg = router._calculate_kelly(mu=-0.05, sigma=0.02)
kelly_pos = router._calculate_kelly(mu=0.05, sigma=0.02)
print(f"  Kelly(mu=-0.05)={kelly_neg:.4f} vs Kelly(mu=+0.05)={kelly_pos:.4f}")
assert abs(kelly_neg - kelly_pos) < 1e-9, \
    "FALLO: Kelly da magnitudes distintas para +mu y -mu simétricos (bug de signo no arreglado)"
print("  OK: Kelly da la MISMA magnitud para mu negativo y positivo simétricos (fix verificado)")


print("\n" + "=" * 60)
print("PASO 3: PortfolioHedgeOptimizer.optimize (Differential Evolution)")
print("=" * 60)

n_days = 200
returns_by_asset = {
    'BTC': np.random.randn(n_days) * 0.02,
    'ETH': np.random.randn(n_days) * 0.03,
    'SOL': np.random.randn(n_days) * 0.04,
}

optimizer = PortfolioHedgeOptimizer(confidence=0.95, max_iter=50)  # max_iter bajo para que el smoke test sea rapido
result = optimizer.optimize(returns_by_asset)
print(f"  Pesos óptimos: { {k: round(v,4) for k,v in result['weights'].items()} }")
print(f"  Expected Shortfall: {result['expected_shortfall']:.4f}")
print(f"  Convergió: {result['success']} en {result['iterations']} iteraciones")

assert abs(sum(result['weights'].values()) - 1.0) < 1e-6, "FALLO: pesos del hedge optimizer no suman 1.0"
print("  OK: pesos suman 1.0")

current_weights = {'BTC': 0.8, 'ETH': 0.1, 'SOL': 0.1}
rebalance = optimizer.suggest_rebalance(current_weights, result['weights'])
print(f"  Rebalanceo sugerido: {rebalance['rebalance_recommended']} (max_delta={rebalance['max_delta']:.2%})")


print("\n" + "=" * 60)
print("TODOS LOS TESTS PASARON")
print("=" * 60)