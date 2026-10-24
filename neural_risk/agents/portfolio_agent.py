# neural_risk/agents/portfolio_agent.py
"""
PASO 5: AGENTE DE PORTAFOLIO INTELIGENTE

ESTADO: incluye extension de hedging ponderado por trust ratio (GARCH/
Anomaly) y fractional_kelly/max_position_size configurables desde
PortfolioAgent.__init__ (antes ignorados, PositionSizer() se
instanciaba sin argumentos).
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass, field
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

from neural_risk.metrics.performance import PerformanceMetrics
from neural_risk.metrics.risk_analytics import RiskAnalytics

# NOTA sobre por qué NO se reemplaza ExpertEvaluator.calculate_sharpe_ratio/
# calculate_sortino_ratio por PerformanceMetrics.sharpe_ratio/sortino_ratio:
# PerformanceMetrics anualiza multiplicando por sqrt(periods=252), asumiendo
# retornos PERIÓDICOS regulares (ej. diarios). ExpertEvaluator opera sobre
# pnl_pct POR TRADE, que no ocurren a intervalos regulares -- anualizar eso
# con sqrt(252) sería matemáticamente incorrecto (mezclaría trade-level con
# time-series-level). Por eso conviven ambas implementaciones a propósito,
# no es una duplicación sin arreglar.


@dataclass
class TradeRecord:
    ticker: str
    entry_price: float
    entry_time: pd.Timestamp
    signal_type: str
    position_size: float
    stop_loss: float
    take_profit: Optional[float]
    confidence: float
    expert_votes: Dict = field(default_factory=dict)
    regime_key: Optional[str] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

    def close_trade(self, exit_price: float, exit_time: pd.Timestamp, fee_rate: float = 0.0):
        """
        FIX: antes esta fórmula asumía siempre LONG -- para un SHORT,
        una baja de precio (ganancia real) se calculaba como pérdida y
        viceversa, porque nunca se chequeaba self.signal_type. Esto
        afectaba a backtest.py (que usa TradeRecord/close_position) --
        run_executor.py ya calculaba esto bien por separado
        (_close_position sí distingue side='BUY'/'SELL').
        """
        self.exit_price = exit_price
        self.exit_time = exit_time
        if self.signal_type == 'SHORT':
            self.pnl = (self.entry_price - exit_price) * self.position_size
            self.pnl_pct = (self.entry_price - exit_price) / self.entry_price
        else:  # 'LONG' (default)
            self.pnl = (exit_price - self.entry_price) * self.position_size
            self.pnl_pct = (exit_price - self.entry_price) / self.entry_price
        # NUEVO: costos de transacción (comisión + slippage) por lado,
        # sobre el notional de entrada y de salida.
        if fee_rate:
            fees = fee_rate * (self.entry_price + exit_price) * self.position_size
            self.pnl -= fees
            self.pnl_pct -= fee_rate * (self.entry_price + exit_price) / self.entry_price


class ExpertEvaluator:
    def __init__(self, lookback_window: int = 100, risk_free_rate: float = 0.05):
        self.lookback_window = lookback_window
        self.risk_free_rate = risk_free_rate
        self.expert_history = {}

    def add_trade_result(self, expert_name: str, pnl_pct: float,
                        is_winning: bool, max_dd: float):
        if expert_name not in self.expert_history:
            self.expert_history[expert_name] = []
        self.expert_history[expert_name].append({
            'pnl_pct': pnl_pct, 'is_winning': is_winning, 'max_dd': max_dd
        })

    def calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        if std_ret == 0:
            return 0.0
        return (mean_ret - self.risk_free_rate / 252) / std_ret

    def calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        downside_returns = returns[returns < 0]
        downside_dev = np.std(downside_returns) if len(downside_returns) > 0 else 0.0
        if downside_dev == 0:
            return mean_ret * 100
        return (mean_ret - self.risk_free_rate / 252) / downside_dev

    def calculate_calmar_ratio(self, returns: np.ndarray) -> float:
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
        if len(returns) == 0:
            return 0.5
        return np.sum(returns > 0) / len(returns)

    def calculate_profit_factor(self, returns: np.ndarray) -> float:
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        gross_profit = np.sum(wins) if len(wins) > 0 else 0
        gross_loss = np.sum(np.abs(losses)) if len(losses) > 0 else 1
        if gross_loss == 0:
            return 10.0 if gross_profit > 0 else 0.0
        return gross_profit / gross_loss

    def evaluate_expert(self, expert_name: str) -> Dict:
        if expert_name not in self.expert_history:
            return self._default_scores(0)
        history = self.expert_history[expert_name]
        if len(history) == 0:
            return self._default_scores(0)

        recent = history[-self.lookback_window:]
        returns = np.array([t['pnl_pct'] for t in recent])

        sharpe = self.calculate_sharpe_ratio(returns)
        sortino = self.calculate_sortino_ratio(returns)
        calmar = self.calculate_calmar_ratio(returns)
        win_rate = self.calculate_win_rate(returns)
        profit_factor = self.calculate_profit_factor(returns)

        composite = (
            0.40 * self._normalize_metric(sortino, 2.0) +
            0.30 * win_rate +
            0.20 * self._normalize_metric(profit_factor, 2.0) +
            0.10 * self._normalize_metric(sharpe, 1.5)
        )

        return {
            'sharpe': float(sharpe), 'sortino': float(sortino), 'calmar': float(calmar),
            'win_rate': float(win_rate), 'profit_factor': float(profit_factor),
            'composite_score': float(np.clip(composite, 0, 1)), 'n_trades': len(returns)
        }

    def _normalize_metric(self, value: float, ideal: float) -> float:
        if ideal == 0:
            return 0.5
        return np.clip(value / ideal, 0, 1)

    def _default_scores(self, n_trades: int) -> Dict:
        return {
            'sharpe': 0.0, 'sortino': 0.0, 'calmar': 0.0,
            'win_rate': 0.5, 'profit_factor': 1.0,
            'composite_score': 0.5, 'n_trades': n_trades
        }


class DynamicWeighting:
    def __init__(self, n_experts: int, initial_alpha: float = 1.0,
                 initial_beta: float = 1.0):
        self.n_experts = n_experts
        self.expert_names = []
        self.alpha = np.ones(n_experts) * initial_alpha
        self.beta = np.ones(n_experts) * initial_beta
        self.weights = np.ones(n_experts) / n_experts
        self.trade_history = []

    def set_expert_names(self, names: List[str]):
        self.expert_names = names[:self.n_experts]

    def select_arm(self) -> int:
        samples = np.random.beta(self.alpha, self.beta)
        return np.argmax(samples)

    def update_arm(self, arm_idx: int, reward: float):
        if reward > 0.5:
            self.alpha[arm_idx] += 1
        else:
            self.beta[arm_idx] += 1
        self.recalculate_weights()

    def recalculate_weights(self):
        self.weights = self.alpha / (self.alpha + self.beta)
        self.weights = self.weights / np.sum(self.weights)

    def get_weights(self) -> Dict[str, float]:
        if len(self.expert_names) == 0:
            return {f"expert_{i}": w for i, w in enumerate(self.weights)}
        return {name: w for name, w in zip(self.expert_names, self.weights)}

    def get_index(self, expert_name: str) -> Optional[int]:
        if expert_name in self.expert_names:
            return self.expert_names.index(expert_name)
        return None


def get_regime_key(asset_report: Dict) -> str:
    """
    NUEVO (mejora #2 de 3 pedida por el usuario): deriva un identificador
    de régimen de mercado a partir de las señales que ya calculan los
    expertos (hmm_regime, garch_vol.crisis_detected). No agrega ningún
    cálculo nuevo -- solo etiqueta lo que ya existe.
    """
    if asset_report.get('garch_vol', {}).get('crisis_detected', False):
        return 'CRISIS'
    regime = asset_report.get('hmm_regime', 1)
    return f'HMM_{regime}'


class RegimeConditionedWeighting:
    """
    NUEVO (mejora #2 de 3): mantiene un DynamicWeighting SEPARADO por
    régimen de mercado (CRISIS / HMM_0 / HMM_1 / HMM_2), en vez de un
    único set de pesos global que deriva lento con todo el historial
    mezclado.

    Por qué: Thompson Sampling global es REACTIVO -- tarda en
    "reaprender" cada vez que el mercado cambia de régimen, porque
    promedia trades de contextos distintos en el mismo posterior. Con
    pesos por régimen, si GARCH+Anomaly históricamente rinden mejor en
    crisis, el sistema empieza a confiar más en ellos apenas DETECTA la
    crisis (vía HMM/GARCH), en vez de esperar a acumular varios trades
    perdedores en el régimen equivocado para corregirse. Esto es lo que
    lo acerca a "anticipatorio" en vez de puramente reactivo.

    Cold start: un régimen con pocas observaciones (< min_observations)
    devuelve los pesos GLOBALES en vez de los propios -- evita
    decisiones ruidosas basadas en 2-3 trades de un régimen recién
    visto. El DynamicWeighting global se sigue actualizando SIEMPRE (con
    TODOS los trades, de cualquier régimen), como fallback robusto y
    como comportamiento por defecto mientras no hay suficiente historia
    -- equivalente al comportamiento de antes de esta mejora.
    """

    def __init__(self, expert_names: List[str], min_observations: int = 10):
        self.expert_names = expert_names
        self.n_experts = len(expert_names)
        self.min_observations = min_observations

        self.global_weighting = DynamicWeighting(self.n_experts)
        self.global_weighting.set_expert_names(expert_names)

        self.regime_weightings: Dict[str, DynamicWeighting] = {}

    def _get_or_create(self, regime_key: str) -> DynamicWeighting:
        if regime_key not in self.regime_weightings:
            dw = DynamicWeighting(self.n_experts)
            dw.set_expert_names(self.expert_names)
            self.regime_weightings[regime_key] = dw
        return self.regime_weightings[regime_key]

    def get_weights(self, regime_key: Optional[str] = None) -> Dict[str, float]:
        if regime_key is None:
            return self.global_weighting.get_weights()

        regime_dw = self.regime_weightings.get(regime_key)
        if regime_dw is None:
            return self.global_weighting.get_weights()

        # alpha/beta arrancan en 1.0 cada uno (prior uniforme) -> restamos
        # ese prior para contar solo observaciones REALES acumuladas.
        n_observations = float(np.sum(regime_dw.alpha + regime_dw.beta) - 2 * self.n_experts)
        if n_observations < self.min_observations:
            return self.global_weighting.get_weights()

        return regime_dw.get_weights()

    def update(self, expert_name: str, reward: float, regime_key: Optional[str] = None):
        """Actualiza SIEMPRE el global (fallback robusto), y además el
        régimen específico si se proporciona uno."""
        idx = self.global_weighting.get_index(expert_name)
        if idx is not None:
            self.global_weighting.update_arm(idx, reward)

        if regime_key is not None:
            regime_dw = self._get_or_create(regime_key)
            idx2 = regime_dw.get_index(expert_name)
            if idx2 is not None:
                regime_dw.update_arm(idx2, reward)

    def get_index(self, expert_name: str) -> Optional[int]:
        """Todos los sub-weightings comparten el mismo orden de expert_names."""
        return self.global_weighting.get_index(expert_name)

    def get_regime_summary(self) -> Dict[str, Dict]:
        """Utilidad de diagnóstico: pesos y n° de observaciones por régimen visto."""
        summary = {'GLOBAL': self.global_weighting.get_weights()}
        for regime_key, dw in self.regime_weightings.items():
            n_obs = float(np.sum(dw.alpha + dw.beta) - 2 * self.n_experts)
            summary[regime_key] = {'weights': dw.get_weights(), 'n_observations': n_obs}
        return summary


def _trust_ratio(expert_weights: Dict, expert_name: str) -> float:
    """trust=1.0 -> neutro. <1.0 -> viene fallando. >1.0 -> viene acertando."""
    n = len(expert_weights) if expert_weights else 9
    if n == 0:
        return 1.0
    uniform = 1.0 / n
    weight = expert_weights.get(expert_name, uniform)
    if uniform == 0:
        return 1.0
    return float(np.clip(weight / uniform, 0.0, 2.0))


class SignalGenerator:
    def __init__(self, long_threshold: float = 0.60,
                 short_threshold: float = -0.60,
                 min_confidence: float = 0.50,
                 anomaly_veto_trust_threshold: float = 0.5):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.min_confidence = min_confidence
        self.anomaly_veto_trust_threshold = anomaly_veto_trust_threshold

    def generate_signal(self, asset_report: Dict,
                       expert_weights: Dict) -> Dict:

        # FIX: solo votan DIRECCIÓN los expertos que predicen el target
        # (triple barrier, escala -1..1): XGB, CAUSAL, LSTM, ENSEMBLE.
        # - HMM votaba (regime-1)*0.5, tratando el número de estado como
        #   ordinal alcista/bajista -- un estado oculto no tiene dirección.
        #   Se usa solo como contexto (regime_key).
        # - GARCH votaba 2*hedging_intensity-1: vol ALTA = señal ALCISTA.
        #   La volatilidad no es dirección; ahora modula la confianza.
        # - LSTM/ENSEMBLE se dividían por 0.05 (como si fueran retornos),
        #   lo que saturaba en ±1 cualquier predicción > 0.05.
        # Expertos que fallaron en este ciclo no votan (antes votaban 0 y
        # diluían el resto).
        failed = set(asset_report.get('failures', []))
        expert_signals = {}

        if 'xgb_signal' in asset_report and 'XGB' not in failed:
            expert_signals['XGB'] = float(np.clip(asset_report['xgb_signal'], -1, 1))

        if 'causal_effect' in asset_report and 'CAUSAL' not in failed:
            expert_signals['CAUSAL'] = float(np.clip(asset_report['causal_effect'] / 0.05, -1, 1))

        if 'lstm_forecast' in asset_report and 'LSTM' not in failed:
            lstm_forecast = asset_report['lstm_forecast'].get('ensemble_forecast', 0)
            expert_signals['LSTM'] = float(np.clip(lstm_forecast, -1, 1))

        if 'ensemble' in asset_report and 'ENSEMBLE' not in failed:
            mu = asset_report['ensemble'].get('mu', 0)
            expert_signals['ENSEMBLE'] = float(np.clip(mu, -1, 1))

        anomaly_detected = asset_report.get('anomaly', {}).get('anomaly_detected', False)

        total_weight = 0
        weighted_sum = 0

        for expert_name, signal in expert_signals.items():
            weight = expert_weights.get(expert_name, 1.0 / len(expert_signals))
            weighted_sum += signal * weight
            total_weight += weight

        agreement = weighted_sum / total_weight if total_weight > 0 else 0

        if len(expert_signals) > 1:
            signals_array = np.array(list(expert_signals.values()))
            disagreement = np.std(signals_array)
            confidence = 1.0 - np.clip(disagreement / 2, 0, 1)
        else:
            confidence = 0.5

        if asset_report.get('garch_vol', {}).get('crisis_detected', False):
            confidence *= 0.5

        if confidence < self.min_confidence:
            signal_type = 'HOLD'
        elif agreement > self.long_threshold:
            signal_type = 'LONG'
        elif agreement < self.short_threshold:
            signal_type = 'SHORT'
        else:
            signal_type = 'HOLD'

        anomaly_trust = _trust_ratio(expert_weights, 'ANOMALY')
        anomaly_override_applied = False

        if anomaly_detected:
            if anomaly_trust >= self.anomaly_veto_trust_threshold:
                signal_type = 'HOLD'
                anomaly_override_applied = True
            else:
                confidence = confidence * anomaly_trust
                if confidence < self.min_confidence:
                    signal_type = 'HOLD'
                    anomaly_override_applied = True

        return {
            'signal': signal_type,
            'agreement': float(agreement),
            'confidence': float(confidence),
            'expert_breakdown': expert_signals,
            'anomaly_override': anomaly_override_applied,
            'anomaly_trust': anomaly_trust
        }


class DynamicStopLoss:
    """
    FIX: antes la distancia del stop era entry * ensemble_sigma * ... --
    pero ensemble_sigma está en escala de ETIQUETAS (y valía 1.0 por
    defecto cuando el ensemble fallaba, o sea siempre): stops a 50-70%
    del precio. Ahora stop y take-profit se miden en volatilidad de
    RETORNOS, con los mismos múltiplos que las barreras del triple-barrier
    labeling (pt_sl=[2, 1]) -- la ejecución replica lo que el modelo
    aprendió a predecir.
    """

    def __init__(self, sl_vol_mult: float = 1.0, tp_vol_mult: float = 2.0):
        self.sl_vol_mult = sl_vol_mult
        self.tp_vol_mult = tp_vol_mult

    @staticmethod
    def _vol(asset_report: Dict) -> float:
        vol = asset_report.get('realized_vol')
        if vol is None or not np.isfinite(vol) or vol <= 0:
            vol = asset_report.get('garch_vol', {}).get('vol_forecast', np.nan)
        if vol is None or not np.isfinite(vol) or vol <= 0:
            vol = 0.02
        return float(vol)

    def calculate_stop_loss(self, entry_price: float, asset_report: Dict,
                           signal_type: str, confidence: float = 1.0,
                           garch_trust: float = 1.0) -> float:
        distance = entry_price * self._vol(asset_report) * self.sl_vol_mult
        if signal_type == 'LONG':
            return entry_price - distance
        elif signal_type == 'SHORT':
            return entry_price + distance
        return entry_price

    def calculate_take_profit(self, entry_price: float, asset_report: Dict,
                              signal_type: str) -> Optional[float]:
        distance = entry_price * self._vol(asset_report) * self.tp_vol_mult
        if signal_type == 'LONG':
            return entry_price + distance
        elif signal_type == 'SHORT':
            return entry_price - distance
        return None


class PositionSizer:
    """
    FIX: con pocos trades, win_rate/avg_win/avg_loss observados son ruido
    puro -- un solo trade perdedor dejaba Kelly en 0, y con sizing 0 no hay
    trades nuevos ni feedback: el sistema quedaba bloqueado para siempre.
    Ahora las estadísticas se encogen hacia un prior (shrinkage, peso
    `prior_trades`), y una señal válida abre al menos `min_position_size`
    para que el agente siga aprendiendo.
    """

    def __init__(self, fractional_kelly: float = 0.25,
                 max_position_size: float = 0.10,
                 min_position_size: float = 0.01,
                 prior_trades: int = 30,
                 prior_win_rate: float = 0.5,
                 prior_payoff: float = 1.5):
        self.fractional_kelly = fractional_kelly
        self.max_position_size = max_position_size
        self.min_position_size = min_position_size
        self.prior_trades = prior_trades
        self.prior_win_rate = prior_win_rate
        self.prior_payoff = prior_payoff

    def calculate_position_size(self, win_rate: float, avg_win: float, avg_loss: float,
                               portfolio_value: float, confidence: float,
                               n_trades: int = 0) -> float:
        w = n_trades / (n_trades + self.prior_trades)
        payoff = avg_win / avg_loss if avg_loss > 0 and avg_win > 0 else self.prior_payoff
        p = w * win_rate + (1 - w) * self.prior_win_rate
        b = w * payoff + (1 - w) * self.prior_payoff
        kelly_fraction = np.clip((p * b - (1 - p)) / b, 0, 0.5)
        risk_adjusted = kelly_fraction * self.fractional_kelly * confidence
        final_fraction = np.clip(risk_adjusted, self.min_position_size, self.max_position_size)
        return portfolio_value * final_fraction


class StakingAllocator:
    def __init__(self, base_apy: float = 0.05):
        self.base_apy = base_apy
        self.apy_history = []

    def get_staking_apy(self, ticker: str, asset_report: Dict) -> float:
        crisis_risk = 0.3 if asset_report.get('garch_vol', {}).get('crisis_detected', False) else 0.0
        anomaly_risk = 0.2 if asset_report.get('anomaly', {}).get('anomaly_detected', False) else 0.0
        total_risk = min(crisis_risk + anomaly_risk, 0.5)
        apy = self.base_apy * (1 - total_risk)
        if ticker in ['BTC', 'ETH']:
            apy *= 1.0
        elif ticker in ['SOL', 'AVAX']:
            apy *= 1.2
        else:
            apy *= 0.9
        return float(np.clip(apy, 0.01, 0.15))

    def calculate_daily_interest(self, amount: float, apy: float) -> float:
        return amount * (apy / 365)


class PortfolioAgent:
    def __init__(self, initial_capital: float = 100000,
                 expert_names: List[str] = None,
                 long_threshold: float = 0.60,
                 short_threshold: float = -0.60,
                 min_confidence: float = 0.50,
                 sl_vol_mult: float = 1.0,
                 tp_vol_mult: float = 2.0,
                 base_staking_apy: float = 0.05,
                 fractional_kelly: float = 0.25,
                 max_position_size: float = 0.10):
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.closed_trades = []

        # Solo los expertos que votan dirección (ver SignalGenerator). Antes
        # incluía DEEP_TFT/COPULA/ANOMALY, que nunca votaban: diluían los
        # pesos y su "trust" nunca aprendía nada.
        self.expert_names = expert_names or ['XGB', 'CAUSAL', 'LSTM', 'ENSEMBLE']

        self.evaluator = ExpertEvaluator()
        # FIX (mejora #2 de 3): antes un unico DynamicWeighting global.
        # Ahora RegimeConditionedWeighting mantiene pesos separados por
        # regimen (con fallback automatico a global mientras hay pocas
        # observaciones de ese regimen) -- ver docstring de la clase.
        self.weighting = RegimeConditionedWeighting(self.expert_names, min_observations=10)

        self.signal_generator = SignalGenerator(
            long_threshold=long_threshold, short_threshold=short_threshold,
            min_confidence=min_confidence
        )
        self.stop_loss_calc = DynamicStopLoss(sl_vol_mult=sl_vol_mult, tp_vol_mult=tp_vol_mult)
        self.position_sizer = PositionSizer(
            fractional_kelly=fractional_kelly,
            max_position_size=max_position_size
        )
        self.staking_allocator = StakingAllocator(base_apy=base_staking_apy)

        self.trading_log = []

    def _get_position_sizing_inputs(self) -> Tuple[float, float]:
        if len(self.closed_trades) == 0:
            return 0.02, 0.01
        returns = np.array([t.pnl_pct for t in self.closed_trades if t.pnl_pct is not None])
        if len(returns) == 0:
            return 0.02, 0.01
        wins = returns[returns > 0]
        losses = returns[returns < 0]
        avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.02
        avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.01
        return avg_win, avg_loss

    def execute_portfolio_decision(self,
                                  portfolio_intelligence: Dict,
                                  current_prices: Dict) -> Dict:

        decisions = {}
        total_allocated = 0
        avg_win, avg_loss = self._get_position_sizing_inputs()

        for ticker, asset_report in portfolio_intelligence.items():
            current_price = current_prices.get(ticker, 0)
            if current_price <= 0:
                continue

            # NUEVO (mejora #2 de 3): pesos CONDICIONADOS al régimen
            # detectado para este activo en este ciclo, en vez de un
            # único set global -- ver RegimeConditionedWeighting.
            regime_key = get_regime_key(asset_report)
            expert_weights = self.weighting.get_weights(regime_key=regime_key)
            signal_result = self.signal_generator.generate_signal(asset_report, expert_weights)

            signal_type = signal_result['signal']
            agreement = signal_result['agreement']
            confidence = signal_result['confidence']

            garch_trust = _trust_ratio(expert_weights, 'GARCH')

            stop_loss = self.stop_loss_calc.calculate_stop_loss(
                current_price, asset_report, signal_type, confidence,
                garch_trust=garch_trust
            )
            take_profit = self.stop_loss_calc.calculate_take_profit(
                current_price, asset_report, signal_type
            )

            # Win rate REAL de los trades del agente (antes: promedio del
            # win rate por experto, que con expertos sin historia valía 0.5
            # aunque el agente viniera perdiendo).
            closed = [t for t in self.closed_trades if t.pnl is not None]
            agent_win_rate = (sum(t.pnl > 0 for t in closed) / len(closed)) if closed else 0.5

            position_size_pct = self.position_sizer.calculate_position_size(
                win_rate=agent_win_rate, avg_win=avg_win, avg_loss=avg_loss,
                portfolio_value=self.portfolio_value, confidence=confidence,
                n_trades=len(closed)
            )

            decision = {
                'ticker': ticker,
                'signal': signal_type,
                'agreement': agreement,
                'confidence': confidence,
                'entry_price': current_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'position_size_pct': position_size_pct / self.portfolio_value,
                'expert_weights': expert_weights,
                'expert_signals': signal_result['expert_breakdown'],
                'regime_key': regime_key,
                'anomaly_override': signal_result.get('anomaly_override', False),
                'garch_trust': garch_trust,
                'anomaly_trust': signal_result.get('anomaly_trust', 1.0)
            }

            if signal_type == 'HOLD':
                apy = self.staking_allocator.get_staking_apy(ticker, asset_report)
                decision['staking_apy'] = apy
                decision['daily_interest'] = self.staking_allocator.calculate_daily_interest(
                    position_size_pct, apy
                )

            decisions[ticker] = decision
            total_allocated += position_size_pct

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
            'expert_evaluations': {name: self.evaluator.evaluate_expert(name) for name in self.expert_names}
        }

    def update_positions(self, ticker: str, signal: str, entry_price: float,
                        stop_loss: float, position_size: float,
                        confidence: float = 0.0,
                        expert_votes: Optional[Dict] = None,
                        regime_key: Optional[str] = None,
                        take_profit: Optional[float] = None,
                        timestamp: Optional[pd.Timestamp] = None):
        trade = TradeRecord(
            ticker=ticker, entry_price=entry_price,
            entry_time=timestamp if timestamp is not None else pd.Timestamp.now(),
            signal_type=signal, position_size=position_size, stop_loss=stop_loss,
            take_profit=take_profit, confidence=confidence, expert_votes=expert_votes or {},
            regime_key=regime_key
        )
        self.positions[ticker] = trade
        self.cash -= position_size * entry_price

    def close_position(self, ticker: str, exit_price: float,
                       timestamp: Optional[pd.Timestamp] = None, fee_rate: float = 0.0):
        if ticker not in self.positions:
            return
        trade = self.positions[ticker]
        trade.close_trade(exit_price, timestamp if timestamp is not None else pd.Timestamp.now(),
                          fee_rate=fee_rate)
        self.cash += exit_price * trade.position_size
        self.closed_trades.append(trade)
        del self.positions[ticker]

        # FIX: portfolio_value nunca se actualizaba tras cerrar una
        # posición -- quedaba congelado en initial_capital para
        # siempre. Como PositionSizer usa self.portfolio_value como
        # base para el sizing (Kelly), esto significaba que el
        # dimensionamiento de nuevas posiciones NUNCA reaccionaba a
        # ganancias/pérdidas reales dentro de un backtest -- mismo tipo
        # de bug que ya se arregló en run_executor.get_current_equity(),
        # pero acá adentro de PortfolioAgent. Se sigue el mismo criterio:
        # equity = capital inicial + PnL REALIZADO (no se marca a
        # mercado el valor de posiciones todavía abiertas).
        self.portfolio_value = self.initial_capital + sum(t.pnl or 0 for t in self.closed_trades)

        self._record_trade_feedback(trade)

    def _record_trade_feedback(self, trade: TradeRecord):
        """
        NUEVO: calcula price_return -- el movimiento REAL de precio
        (positivo si subió, negativo si bajó), independiente de si la
        posición era LONG o SHORT. Se pasa a record_expert_feedback para
        que cada experto reciba crédito según si SU voto coincidió con
        lo que realmente pasó, no según si el trade agregado ganó.
        También propaga trade.regime_key (el régimen bajo el cual se
        tomó la decisión), para que el feedback alimente el
        RegimeConditionedWeighting específico de ese régimen.
        """
        if trade.pnl_pct is None or not trade.expert_votes or trade.exit_price is None:
            return
        price_return = (trade.exit_price - trade.entry_price) / trade.entry_price
        self.record_expert_feedback(
            trade.expert_votes, trade.pnl_pct,
            price_return=price_return, regime_key=trade.regime_key
        )

    def record_expert_feedback(self, expert_votes: Dict, pnl_pct: float,
                              price_return: Optional[float] = None,
                              regime_key: Optional[str] = None):
        """
        FIX (precisión del aprendizaje adaptativo): antes, TODOS los
        expertos que participaron en una señal recibían el MISMO reward
        (1.0 si el trade agregado ganó, 0.0 si perdió) -- sin importar
        si el voto INDIVIDUAL de cada uno acertó la dirección real del
        precio. Ejemplo del problema: si HMM votaba alcista pero el
        agregado terminaba en SHORT (porque los otros 8 expertos pesaban
        más) y ese SHORT ganaba, HMM igual recibía crédito aunque su
        voto estuvo mal. Esto diluía la señal de Thompson Sampling y
        retrasaba el aprendizaje real de cuáles expertos confiar.

        Ahora, si se provee 'price_return' (movimiento de precio real,
        signo puro, INDEPENDIENTE de si la posición fue LONG o SHORT):
        cada experto recibe reward según si el signo de SU voto
        coincidió con el signo de price_return. Sin price_return
        (compatibilidad hacia atrás), cae al comportamiento anterior.

        NUEVO (mejora #2 de 3): si se provee 'regime_key', el feedback
        también alimenta el RegimeConditionedWeighting de ESE régimen
        específico, además del global -- así el sistema aprende
        "quién confiar" de forma separada según el contexto de mercado.

        NOTA: is_winning que alimenta a ExpertEvaluator (Sharpe/Sortino/
        Win Rate por experto) sigue atado al resultado real del TRADE
        (pnl_pct > 0) a propósito -- eso mide "calidad de las métricas
        de riesgo cuando este experto participó", que es una pregunta
        distinta de "¿acertó la dirección?". No se deben confundir.
        """
        is_winning = pnl_pct > 0

        for expert_name, vote in expert_votes.items():
            self.evaluator.add_trade_result(
                expert_name=expert_name, pnl_pct=pnl_pct,
                is_winning=is_winning, max_dd=0.0
            )

            if price_return is not None:
                agreed = (vote > 0 and price_return > 0) or (vote < 0 and price_return < 0)
                reward = 1.0 if agreed else 0.0
            else:
                reward = 1.0 if is_winning else 0.0

            self.weighting.update(expert_name, reward, regime_key=regime_key)

    def _build_equity_curve(self) -> pd.Series:
        """
        NUEVO: construye la curva de equity real a partir de los trades
        cerrados, ordenados por fecha de cierre. Necesaria para calcular
        max_drawdown (PerformanceMetrics.max_drawdown espera una serie
        temporal de equity, no un valor final).
        """
        if not self.closed_trades:
            return pd.Series([self.initial_capital])

        sorted_trades = sorted(self.closed_trades, key=lambda t: t.exit_time)
        equity = [self.initial_capital]
        index = [sorted_trades[0].entry_time]
        running = self.initial_capital

        for t in sorted_trades:
            running += (t.pnl or 0)
            equity.append(running)
            index.append(t.exit_time)

        return pd.Series(equity, index=index)

    def get_portfolio_metrics(self) -> Dict:
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

        # NUEVO: métricas de riesgo del portafolio via neural_risk.metrics
        # (antes completamente huérfano -- nada lo usaba). Distinto de
        # sharpe/sortino de arriba: acá es riesgo de COLA de la
        # distribución de retornos por trade (VaR/CVaR), y drawdown real
        # sobre una curva de equity temporal (que antes no existía).
        if len(returns) > 1:
            var_95 = float(RiskAnalytics.calculate_var(returns, confidence=0.95))
            cvar_95 = float(RiskAnalytics.calculate_expected_shortfall(returns, confidence=0.95))
        else:
            var_95 = 0.0
            cvar_95 = 0.0

        equity_curve = self._build_equity_curve()
        max_dd = float(PerformanceMetrics.max_drawdown(equity_curve)) if len(equity_curve) > 1 else 0.0

        return {
            'total_pnl': float(total_pnl), 'total_pnl_pct': float(total_pnl_pct),
            'win_rate': float(win_rate), 'profit_factor': float(profit_factor),
            'sortino_ratio': float(sortino), 'sharpe_ratio': float(sharpe),
            'var_95': var_95, 'cvar_95': cvar_95, 'max_drawdown': max_dd,
            'n_trades': len(self.closed_trades), 'open_positions': len(self.positions),
            'portfolio_value': self.portfolio_value
        }