# neural_risk/agents/strategy_router.py
import pandas as pd
import numpy as np
from typing import Dict, Tuple

class StrategyRouter:
    """
    Toma los outputs probabilísticos del NeuralRiskModel y 
    decide la estrategia de ejecución y el sizing.
    
    Integra:
    - Risk Management (30%): Detección de crashes y alta incertidumbre
    - Alpha Generation (70%): Trend-following y Mean-Reversion
    """
    
    def __init__(self, risk_appetite: float = 0.7):
        """
        risk_appetite: 0.7 significa que toleramos cierta incertidumbre.
                      Valores altos → más agresivo
                      Valores bajos → más conservador
        """
        self.risk_appetite = risk_appetite
        self.strategy_history = []

    def allocate_capital(self, signals: Dict[str, Tuple[float, float]], 
                        df_master: pd.DataFrame, 
                        market_state: Dict = None) -> Dict[str, float]:
        """
        Toma todas las señales (mu, sigma) por activo y retorna weights normalizados.
        
        Args:
            signals: {'BTC': (mu, sigma), 'ETH': (mu, sigma), ...}
            df_master: DataFrame con todas las features
            market_state: Estado actual del mercado {'hurst': 0.6, 'volatility': 0.02, ...}
            
        Returns:
            {'BTC': 0.4, 'ETH': 0.3, 'SPY': 0.3}  (suma = 1.0)
        """
        if market_state is None:
            market_state = {'hurst': 0.5, 'volatility': 0.02, 'crash_prob': 0.1}
        
        weights = {}
        strategy_decisions = {}
        
        for ticker, (mu, sigma) in signals.items():
            # --- STEP 1: Evaluar Confianza de la Señal ---
            signal_to_noise = abs(mu) / (sigma + 1e-6)
            confidence = min(signal_to_noise / 2.0, 1.0)  # Normalize to [0, 1]
            
            # --- STEP 2: Risk Management Filter (30%) ---
            if market_state.get('crash_prob', 0) > self.risk_appetite:
                strategy_decisions[ticker] = {
                    "action": "PROTECT_CAPITAL",
                    "leverage": 0.0,
                    "weight": 0.0,
                    "reason": "Crash regime detected"
                }
                weights[ticker] = 0.0
                continue
            
            if signal_to_noise < 0.5:
                strategy_decisions[ticker] = {
                    "action": "REDUCE_EXPOSURE",
                    "leverage": 0.3,
                    "weight": 0.05,
                    "reason": "High uncertainty"
                }
                weights[ticker] = 0.05
                continue
            
            # --- STEP 3: Alpha Generation (70%) ---
            hurst = market_state.get('hurst', 0.5)
            
            # Caso A: Tendencia Confirmada (Hurst > 0.55 indica persistencia)
            if hurst > 0.55 and mu > 0:
                kelly = self._calculate_kelly(mu, sigma)
                strategy_decisions[ticker] = {
                    "action": "AGGRESSIVE_LONG",
                    "leverage": kelly,
                    "weight": kelly * confidence,
                    "model": "Trend_Following",
                    "reason": f"Trend persistence (Hurst={hurst:.2f})"
                }
                weights[ticker] = kelly * confidence
            
            # Caso B: Reversión a la media (Hurst < 0.45)
            elif hurst < 0.45 and mu < 0:
                kelly = self._calculate_kelly(mu, sigma) * 0.5
                strategy_decisions[ticker] = {
                    "action": "MEAN_REVERSION",
                    "leverage": kelly,
                    "weight": kelly * confidence,
                    "model": "Oscillator",
                    "reason": f"Mean reversion regime (Hurst={hurst:.2f})"
                }
                weights[ticker] = kelly * confidence
            
            # Caso C: Señal positiva pero neutral (0.45 < Hurst < 0.55)
            elif mu > 0:
                kelly = self._calculate_kelly(mu, sigma) * 0.3
                strategy_decisions[ticker] = {
                    "action": "LONG_NEUTRAL",
                    "leverage": kelly,
                    "weight": kelly * confidence,
                    "reason": "Positive signal in neutral regime"
                }
                weights[ticker] = kelly * confidence
            
            else:
                strategy_decisions[ticker] = {
                    "action": "STAY_FLAT",
                    "leverage": 0,
                    "weight": 0,
                    "reason": "No edge detected"
                }
                weights[ticker] = 0.0
        
        # --- STEP 4: Normalization ---
        total_weight = sum(weights.values())
        if total_weight > 0:
            normalized_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            # Si no hay señales, distribución uniforme de cash
            n_assets = len(signals)
            normalized_weights = {k: 1.0 / n_assets for k in signals.keys()}
        
        # Guardar en historial
        self.strategy_history.append({
            'timestamp': pd.Timestamp.now(),
            'weights': normalized_weights,
            'decisions': strategy_decisions,
            'market_state': market_state
        })
        
        return normalized_weights

    def _calculate_kelly(self, mu: float, sigma: float) -> float:
        """
        Criterio de Kelly Simplificado para Sizing Agnóstico.
        f* = mu / sigma^2
        
        Limitamos el apalancamiento para evitar 'Ruina del Apostador'.
        """
        kelly = mu / (sigma**2 + 1e-6)
        # Cap máximo de 2x, mínimo de 0.05x
        kelly_capped = min(max(abs(kelly) * np.sign(mu), 0.05), 2.0)
        return kelly_capped

    def get_strategy_report(self, last_n: int = 10) -> pd.DataFrame:
        """Retorna un reporte de las últimas N decisiones."""
        if len(self.strategy_history) == 0:
            return pd.DataFrame()
        
        recent = self.strategy_history[-last_n:]
        report_data = []
        
        for entry in recent:
            row = {
                'timestamp': entry['timestamp'],
                'market_hurst': entry['market_state'].get('hurst', np.nan),
                'market_vol': entry['market_state'].get('volatility', np.nan),
                **{f"{ticker}_weight": w for ticker, w in entry['weights'].items()}
            }
            report_data.append(row)
        
        return pd.DataFrame(report_data)