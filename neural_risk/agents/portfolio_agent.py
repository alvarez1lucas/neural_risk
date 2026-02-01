# neural_risk/agents/portfolio_agent.py
"""
PASO 5: AGENTE DE PORTAFOLIO INTELIGENTE
Orquesta decisiones de trading basadas en 9 expertos con:
- Evaluación dinámica de performance
- Ponderación adaptativa (Thompson sampling)
- Short selling support
- Stop loss dinámico
- Kelly Criterion positioning
- Staking/Lending cuando HOLD
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TradeRecord:
    """Registro de trade para histórico y análisis"""
    ticker: str
    entry_price: float
    entry_time: pd.Timestamp
    signal_type: str  # 'LONG', 'SHORT', 'HOLD'
    position_size: float
    stop_loss: float
    take_profit: Optional[float]
    confidence: float
    expert_votes: Dict
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    
    def close_trade(self, exit_price: float, exit_time: pd.Timestamp):
        """Cierra el trade y calcula PnL"""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.pnl = (exit_price - self.entry_price) * self.position_size
        self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        

class ExpertEvaluator:
    """
    Evalúa performance de cada experto usando métricas de riesgo-ajustadas.
    
    Métricas:
    - Sharpe Ratio: return / volatility
    - Sortino Ratio: return / downside_volatility (MEJOR para crashes)
    - Calmar Ratio: return / max_drawdown (penaliza DD severamente)
    - Win Rate: % trades positivos
    - Profit Factor: gross_profit / gross_loss
    """
    
    def __init__(self, lookback_window: int = 100, risk_free_rate: float = 0.05):
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.expert_history = {}  # {expert_name: [trades]}
        
    def add_trade_result(self, expert_name: str, pnl_pct: float, 
                        is_winning: bool, max_dd: float):
        """Registra resultado de trade"""
        if expert_name not in self.expert_history:
            self.expert_history[expert_name] = []
        
        self.expert_history[expert_name].append({
            'pnl_pct': pnl_pct,
            'is_winning': is_winning,
            'max_dd': max_dd
        })
    
    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Sharpe = (mean_return - rf_rate) / std_return"""
        if len(returns) < 2:
            return 0.0
        
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        if std_ret == 0:
            return 0.0
        
        return (mean_ret - self.risk_free_rate / 252) / std_ret
    
    def calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """
        Sortino = (mean_return - rf_rate) / downside_deviation
        Mejor que Sharpe porque solo penaliza downside
        """
        if len(returns) < 2:
            return 0.0
        
        mean_ret = np.mean(returns)
        
        # Downside deviation: solo volatilidad negativa
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            downside_dev = 0.0
        else:
            downside_dev = np.std(downside_returns)
        
        if downside_dev == 0:
            return mean_ret * 100  # Si no hay downside, return puro
        
        return (mean_ret - self.risk_free_rate / 252) / downside_dev
    
    def calculate_calmar_ratio(self, returns: np.ndarray) -> float:
        """Calmar = annual_return / max_drawdown"""
        if len(returns) < 2:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_dd = np.min(drawdown)
        
        annual_return = np.sum(returns)
        
        if max_dd == 0:
            return 0.0
        
        return annual_return / abs(max_dd)
    
    def calculate_win_rate(self, returns: np.ndarray) -> float:
        """% de trades positivos"""
        if len(returns) == 0:
            return 0.5
        return np.sum(returns > 0) / len(returns)
    
    def calculate_profit_factor(self, returns: np.ndarray) -> float:
        """Profit Factor = gross_profit / abs(gross_loss)"""
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        
        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = np.sum(np.abs(losses)) if len(losses) > 0 else 1
        
        if gross_loss == 0:
            return 10.0 if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def evaluate_expert(self, expert_name: str) -> Dict:
        """
        Calcula todas las métricas para un experto.
        
        Returns:
            {
                'sharpe': float,
                'sortino': float,
                'calmar': float,
                'win_rate': float,
                'profit_factor': float,
                'composite_score': float (0-1)
            }
        """
        if expert_name not in self.expert_history:
            return {
                'sharpe': 0.0,
                'sortino': 0.0,
                'calmar': 0.0,
                'win_rate': 0.5,
                'profit_factor': 1.0,
                'composite_score': 0.5,
                'n_trades': 0
            }
        
        history = self.expert_history[expert_name]
        if len(history) == 0:
            return self._default_scores(0)
        
        # Últimos N trades
        recent = history[-self.lookback_window:]
        returns = np.array([t['pnl_pct'] for t in recent])
        
        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns)
        win_rate = self.calculate_win_rate(returns)
        profit_factor = self.calculate_profit_factor(returns)
        
        # Composite score: ponderado hacia Sortino (mejor para crashes)
        # Pesos: Sortino 40%, Win Rate 30%, Profit Factor 20%, Sharpe 10%
        composite = (
            0.40 * self._normalize_metric(sortino, 2.0) +  # Sortino ideal ~2.0
            0.30 * win_rate +                               # Win Rate es 0-1
            0.20 * self._normalize_metric(profit_factor, 2.0) +  # PF ideal ~2.0
            0.10 * self._normalize_metric(sharpe, 1.5)      # Sharpe ideal ~1.5
        )
        
        return {
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'calmar': float(calmar),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'composite_score': float(np.clip(composite, 0, 1)),
            'n_trades': len(returns)
        }
    
    def _normalize_metric(self, value: float, ideal: float) -> float:
        """Normaliza métrica a 0-1 range"""
        if ideal == 0:
            return 0.5
        normalized = value / ideal
        return np.clip(normalized, 0, 1)
    
    def _default_scores(self, n_trades: int) -> Dict:
        """Scores por defecto si no hay data"""
        return {
            'sharpe': 0.0,
            'sortino': 0.0,
            'calmar': 0.0,
            'win_rate': 0.5,
            'profit_factor': 1.0,
            'composite_score': 0.5,
            'n_trades': n_trades
        }


class DynamicWeighting:
    """
    Pondera múltiples expertos usando Thompson Sampling (multi-armed bandit).
    
    Cada experto es un "arm" (brazo de tragamonedas).
    Usa distribuciones Beta para exploración/explotación adaptativa.
    """
    
    def __init__(self, n_experts: int, initial_alpha: float = 1.0, 
                 initial_beta: float = 1.0):
        """
        n_experts: número de expertos
        alpha, beta: parámetros Beta (1, 1 = uniforme inicialmente)
        """
        self.n_experts = n_experts
        self.expert_names = []
        
        # Beta distribution parameters per expert
        self.alpha = np.ones(n_experts) * initial_alpha  # Successes
        self.beta = np.ones(n_experts) * initial_beta     # Failures
        self.weights = np.ones(n_experts) / n_experts
        
        self.trade_history = []
    
    def set_expert_names(self, names: List[str]):
        """Set expert names for tracking"""
        self.expert_names = names[:self.n_experts]
    
    def select_arm(self) -> int:
        """
        Selecciona experto usando Thompson Sampling.
        
        Thompson = sample from Beta(alpha, beta) para cada experto,
                   return argmax
        """
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)
    
    def update_arm(self, arm_idx: int, reward: float):
        """
        Actualiza arm después de observar reward.
        
        reward: 0-1 (donde 1 es excelente, 0 es terrible)
        """
        if reward > 0.5:
            self.alpha[arm_idx] += 1  # Success
        else:
            self.beta[arm_idx] += 1   # Failure
        
        self.recalculate_weights()
    
    def recalculate_weights(self):
        """
        Recalcula pesos basados en distribuciones Beta.
        
        Weight = E[Beta(alpha, beta)] = alpha / (alpha + beta)
        """
        self.weights = self.alpha / (self.alpha + self.beta)
        self.weights = self.weights / np.sum(self.weights)
    
    def get_weights(self) -> Dict[str, float]:
        """Retorna pesos por nombre de experto"""
        if len(self.expert_names) == 0:
            return {f"expert_{i}": w for i, w in enumerate(self.weights)}
        return {name: w for name, w in zip(self.expert_names, self.weights)}


class SignalGenerator:
    """
    Genera señales LONG/SHORT/HOLD combinando múltiples expertos.
    
    Lógica:
    - LONG: agreement > threshold_long
    - SHORT: agreement < -threshold_short
    - HOLD: else
    """
    
    def __init__(self, long_threshold: float = 0.60, 
                 short_threshold: float = -0.60,
                 min_confidence: float = 0.50):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.min_confidence = min_confidence
    
    def generate_signal(self, asset_report: Dict, 
                       expert_weights: Dict) -> Dict:
        """
        Genera señal ponderada por expertis.
        
        Args:
            asset_report: de engine.run_portfolio_automation
            expert_weights: {expert_name: weight}
        
        Returns:
            {
                'signal': 'LONG' | 'SHORT' | 'HOLD',
                'agreement': float (-1 a +1),
                'confidence': float (0-1),
                'expert_breakdown': Dict
            }
        """
        
        # Extraer señales de cada experto
        expert_signals = {}
        
        # HMM: 0=bear, 1=normal, 2=bull → map to -0.5, 0, 0.5
        if 'hmm_regime' in asset_report:
            regime = asset_report['hmm_regime']
            expert_signals['HMM'] = (regime - 1) * 0.5  # -0.5 a 0.5
        
        # XGBoost: signal directo
        if 'xgb_signal' in asset_report:
            expert_signals['XGB'] = np.clip(asset_report['xgb_signal'], -1, 1)
        
        # Causal: effect directo
        if 'causal_effect' in asset_report:
            expert_signals['CAUSAL'] = np.clip(asset_report['causal_effect'] / 0.05, -1, 1)
        
        # GARCH: si crisis → SHORT, si calm → LONG
        if 'garch_vol' in asset_report:
            garch_vol = asset_report['garch_vol']
            if garch_vol.get('crisis_detected', False):
                expert_signals['GARCH'] = -0.7  # SHORT crisis
            else:
                signal_val = garch_vol.get('hedging_intensity', 0.5)
                expert_signals['GARCH'] = 2 * signal_val - 1  # -1 a 1
        
        # LSTM: ensemble forecast directo
        if 'lstm_forecast' in asset_report:
            lstm_forecast = asset_report['lstm_forecast'].get('ensemble_forecast', 0)
            expert_signals['LSTM'] = np.clip(lstm_forecast / 0.05, -1, 1)
        
        # Ensemble: μ directo
        if 'ensemble' in asset_report:
            mu = asset_report['ensemble'].get('mu', 0)
            expert_signals['ENSEMBLE'] = np.clip(mu / 0.05, -1, 1)
        
        # Anomaly: si detectada → HOLD (señal 0)
        if 'anomaly' in asset_report:
            if asset_report['anomaly'].get('anomaly_detected', False):
                expert_signals['ANOMALY'] = 0  # NEUTRAL/HOLD
            else:
                expert_signals['ANOMALY'] = 0  # Normal, no signal
        
        # Calcular acuerdo ponderado
        total_weight = 0
        weighted_sum = 0
        
        for expert_name, signal in expert_signals.items():
            weight = expert_weights.get(expert_name, 1.0 / len(expert_signals))
            weighted_sum += signal * weight
            total_weight += weight
        
        if total_weight > 0:
            agreement = weighted_sum / total_weight
        else:
            agreement = 0
        
        # Calcular confianza como acuerdo entre expertos
        if len(expert_signals) > 1:
            signals_array = np.array(list(expert_signals.values()))
            disagreement = np.std(signals_array)
            confidence = 1.0 - np.clip(disagreement / 2, 0, 1)
        else:
            confidence = 0.5
        
        # Generar señal
        if confidence < self.min_confidence:
            signal_type = 'HOLD'
        elif agreement > self.long_threshold:
            signal_type = 'LONG'
        elif agreement < self.short_threshold:
            signal_type = 'SHORT'
        else:
            signal_type = 'HOLD'
        
        return {
            'signal': signal_type,
            'agreement': float(agreement),
            'confidence': float(confidence),
            'expert_breakdown': expert_signals
        }


class DynamicStopLoss:
    """
    Calcula stop loss dinámico basado en:
    - Volatilidad del modelo (σ)
    - Confianza de la señal
    - Volatilidad de mercado (GARCH)
    - Adaptable (threshold configurable)
    """
    
    def __init__(self, adaptive_threshold: float = 0.70):
        """
        adaptive_threshold: 0-1, qué tan strict es el stop loss
        0.50 = 2x σ (loose)
        0.70 = 1.5x σ (moderate)
        0.90 = 1x σ (tight)
        """
        self.adaptive_threshold = adaptive_threshold
    
    def calculate_stop_loss(self, entry_price: float, 
                           asset_report: Dict,
                           signal_type: str,
                           confidence: float) -> float:
        """
        Calcula nivel de stop loss.
        
        Fórmula:
        SL = entry ± (σ_ensemble * confidence * vol_factor * threshold)
        
        Donde:
        - σ_ensemble: uncertainty from NeuralRiskModel
        - confidence: from SignalGenerator (0-1)
        - vol_factor: from GARCH (1-2x)
        - threshold: adaptive (0.5-0.9)
        """
        
        # Base: ensemble uncertainty
        sigma = asset_report.get('ensemble', {}).get('sigma', 0.01)
        
        # GARCH factor (1-2x)
        garch_vol = asset_report.get('garch_vol', {}).get('vol_forecast', 0.02)
        vol_factor = 1 + garch_vol / 0.02  # Normalized
        vol_factor = np.clip(vol_factor, 1.0, 2.0)
        
        # Calcular distancia
        distance = entry_price * sigma * confidence * vol_factor * self.adaptive_threshold
        
        # LONG: SL por debajo
        if signal_type == 'LONG':
            return entry_price - distance
        # SHORT: SL por encima
        elif signal_type == 'SHORT':
            return entry_price + distance
        else:  # HOLD
            return entry_price


class PositionSizer:
    """
    Calcula tamaño de posición usando Kelly Criterion.
    
    Kelly Criterion: f = (p*b - q) / b
    Donde:
    - p: win rate
    - q: loss rate (1-p)
    - b: ratio win/loss
    
    Para seguridad: aplicamos fractional Kelly (0.25x)
    """
    
    def __init__(self, fractional_kelly: float = 0.25,
                 max_position_size: float = 0.10):
        """
        fractional_kelly: 0.25 = 25% de Kelly recomendado
        max_position_size: máximo % del portfolio en 1 trade
        """
        self.fractional_kelly = fractional_kelly
        self.max_position_size = max_position_size
    
    def calculate_position_size(self, 
                               win_rate: float,
                               avg_win: float,
                               avg_loss: float,
                               portfolio_value: float,
                               confidence: float) -> float:
        """
        Calcula cantidad de capital a arriesgar.
        
        Returns:
            Tamaño de posición en $
        """
        
        if avg_loss == 0 or win_rate <= 0 or avg_win <= 0:
            return portfolio_value * 0.05  # Default: 5%
        
        # Kelly Criterion
        loss_rate = 1 - win_rate
        b = avg_win / avg_loss
        
        kelly_fraction = (win_rate * b - loss_rate) / b
        kelly_fraction = np.clip(kelly_fraction, 0, 0.5)  # Evitar >50%
        
        # Fractional Kelly para seguridad
        fractional = kelly_fraction * self.fractional_kelly
        
        # Ajustar por confianza
        risk_adjusted = fractional * confidence
        
        # Limitar a máximo
        final_fraction = np.clip(risk_adjusted, 0, self.max_position_size)
        
        return portfolio_value * final_fraction


class StakingAllocator:
    """
    Si señal = HOLD, asigna capital a lending pools.
    
    APY dinámico según:
    - Oportunidad de mercado (MktCap, liquidity)
    - Risk profil (vaults vs staking)
    """
    
    def __init__(self, base_apy: float = 0.05):
        """base_apy: APY base (default 5% anual)"""
        self.base_apy = base_apy
        self.apy_history = []
    
    def get_staking_apy(self, ticker: str, 
                        asset_report: Dict) -> float:
        """
        Retorna APY recomendado según condiciones.
        
        Lógica:
        - Si GARCH crisis: APY bajo (conservador)
        - Si anomaly detected: APY bajo
        - Si calm market: APY normal
        """
        
        # Factores de riesgo
        crisis_risk = 0.0
        anomaly_risk = 0.0
        
        if asset_report.get('garch_vol', {}).get('crisis_detected', False):
            crisis_risk = 0.3  # Reduce APY 30%
        
        if asset_report.get('anomaly', {}).get('anomaly_detected', False):
            anomaly_risk = 0.2  # Reduce APY 20%
        
        # APY final
        total_risk = min(crisis_risk + anomaly_risk, 0.5)
        apy = self.base_apy * (1 - total_risk)
        
        # Por ticker (cripto típico)
        if ticker in ['BTC', 'ETH']:
            apy *= 1.0  # Standard
        elif ticker in ['SOL', 'AVAX']:
            apy *= 1.2  # Slightly higher
        else:
            apy *= 0.9  # Lower for unknown
        
        return float(np.clip(apy, 0.01, 0.15))
    
    def calculate_daily_interest(self, 
                                amount: float,
                                apy: float) -> float:
        """Calcula interés diario"""
        daily_rate = apy / 365
        return amount * daily_rate


class PortfolioAgent:
    """
    AGENTE PASO 5: Orquestador inteligente.
    
    Coordina:
    1. Evaluación de expertos
    2. Ponderación dinámica
    3. Generación de señales
    4. Dimensionamiento de posiciones
    5. Stop loss dinámico
    6. Asignación de staking
    7. Ejecución y tracking
    """
    
    def __init__(self, initial_capital: float = 100000,
                 expert_names: List[str] = None,
                 long_threshold: float = 0.60,
                 short_threshold: float = -0.60,
                 adaptive_sl_threshold: float = 0.70,
                 base_staking_apy: float = 0.05):
        
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}  # {ticker: TradeRecord}
        self.closed_trades = []
        
        # Componentes
        self.expert_names = expert_names or [
            'HMM', 'XGB', 'CAUSAL', 'DEEP_TFT',
            'GARCH', 'LSTM', 'ANOMALY', 'ENSEMBLE', 'COPULA'
        ]
        
        self.evaluator = ExpertEvaluator()
        self.weighting = DynamicWeighting(len(self.expert_names))
        self.weighting.set_expert_names(self.expert_names)
        
        self.signal_generator = SignalGenerator(
            long_threshold=long_threshold,
            short_threshold=short_threshold
        )
        
        self.stop_loss_calc = DynamicStopLoss(
            adaptive_threshold=adaptive_sl_threshold
        )
        
        self.position_sizer = PositionSizer()
        self.staking_allocator = StakingAllocator(base_apy=base_staking_apy)
        
        self.trading_log = []
    
    def execute_portfolio_decision(self, 
                                  portfolio_intelligence: Dict,
                                  current_prices: Dict) -> Dict:
        """
        Executa decisión de portafolio completa.
        
        Args:
            portfolio_intelligence: {ticker: asset_report} de Paso 4
            current_prices: {ticker: price}
        
        Returns:
            {
                'decisions': {ticker: decision_dict},
                'portfolio_value': float,
                'pnl': float,
                'positions': Dict
            }
        """
        
        decisions = {}
        total_allocated = 0
        
        for ticker, asset_report in portfolio_intelligence.items():
            current_price = current_prices.get(ticker, 0)
            if current_price <= 0:
                continue
            
            # 1. Generar señal
            expert_weights = self.weighting.get_weights()
            signal_result = self.signal_generator.generate_signal(
                asset_report, expert_weights
            )
            
            signal_type = signal_result['signal']
            agreement = signal_result['agreement']
            confidence = signal_result['confidence']
            
            # 2. Calcular stop loss
            stop_loss = self.stop_loss_calc.calculate_stop_loss(
                current_price, asset_report, signal_type, confidence
            )
            
            # 3. Dimensionar posición
            # (Usando histórico de evaluador)
            evaluations = {
                name: self.evaluator.evaluate_expert(name)
                for name in self.expert_names
            }
            avg_win_rate = np.mean([e['win_rate'] for e in evaluations.values()])
            avg_pf = np.mean([e['profit_factor'] for e in evaluations.values()])
            
            position_size_pct = self.position_sizer.calculate_position_size(
                win_rate=avg_win_rate,
                avg_win=0.02,  # Default 2% win
                avg_loss=0.01,  # Default 1% loss
                portfolio_value=self.portfolio_value,
                confidence=confidence
            )
            
            # 4. Decisión final
            decision = {
                'ticker': ticker,
                'signal': signal_type,
                'agreement': agreement,
                'confidence': confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'position_size_pct': position_size_pct / self.portfolio_value,
                'expert_weights': expert_weights,
                'expert_signals': signal_result['expert_breakdown']
            }
            
            # 5. Si HOLD, asignar staking
            if signal_type == 'HOLD':
                apy = self.staking_allocator.get_staking_apy(ticker, asset_report)
                daily_interest = self.staking_allocator.calculate_daily_interest(
                    position_size_pct, apy
                )
                decision['staking_apy'] = apy
                decision['daily_interest'] = daily_interest
            
            decisions[ticker] = decision
            total_allocated += position_size_pct
        
        # Log trading
        self.trading_log.append({
            'timestamp': pd.Timestamp.now(),
            'decisions': decisions,
            'total_allocated': total_allocated,
            'portfolio_value': self.portfolio_value
        })
        
        return {
            'decisions': decisions,
            'portfolio_value': self.portfolio_value,
            'cash': self.cash,
            'total_allocated': total_allocated,
            'expert_evaluations': {
                name: self.evaluator.evaluate_expert(name)
                for name in self.expert_names
            }
        }
    
    def update_positions(self, ticker: str, 
                        signal: str,
                        entry_price: float,
                        stop_loss: float,
                        position_size: float):
        """Abre nueva posición"""
        
        trade = TradeRecord(
            ticker=ticker,
            entry_price=entry_price,
            entry_time=pd.Timestamp.now(),
            signal_type=signal,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=None,  # Calculado dinámicamente
            confidence=0.0,
            expert_votes={}
        )
        
        self.positions[ticker] = trade
        self.cash -= position_size * entry_price
    
    def close_position(self, ticker: str, 
                      exit_price: float):
        """Cierra posición existente"""
        
        if ticker not in self.positions:
            return
        
        trade = self.positions[ticker]
        trade.close_trade(exit_price, pd.Timestamp.now())
        
        self.cash += exit_price * trade.position_size
        self.closed_trades.append(trade)
        del self.positions[ticker]
    
    def get_portfolio_metrics(self) -> Dict:
        """Calcula métricas del portafolio"""
        
        total_pnl = sum(t.pnl or 0 for t in self.closed_trades)
        total_pnl_pct = total_pnl / self.initial_capital if self.initial_capital > 0 else 0
        
        if len(self.closed_trades) > 0:
            win_rate = sum(1 for t in self.closed_trades if (t.pnl or 0) > 0) / len(self.closed_trades)
            gross_profit = sum(t.pnl for t in self.closed_trades if (t.pnl or 0) > 0)
            gross_loss = abs(sum(t.pnl for t in self.closed_trades if (t.pnl or 0) < 0))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        else:
            win_rate = 0
            profit_factor = 0
        
        returns = np.array([t.pnl_pct or 0 for t in self.closed_trades])
        if len(returns) > 0:
            sortino = self.evaluator.calculate_sortino_ratio(returns)
            sharpe = self.evaluator.calculate_sharpe_ratio(returns)
        else:
            sortino = 0
            sharpe = 0
        
        return {
            'total_pnl': float(total_pnl),
            'total_pnl_pct': float(total_pnl_pct),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'sortino_ratio': float(sortino),
            'sharpe_ratio': float(sharpe),
            'n_trades': len(self.closed_trades),
            'open_positions': len(self.positions),
            'portfolio_value': self.portfolio_value
        }
