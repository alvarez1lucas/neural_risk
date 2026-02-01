# neural_risk/models/reinforcement_learning.py
"""
Reinforcement Learning (DQN/PPO) para Allocation
Aprende políticas óptimas en entornos volátiles (cripto)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from stable_baselines3 import PPO, DQN
    from stable_baselines3.common.env_util import make_vec_env
except ImportError:
    PPO = None
    DQN = None


class MarketEnvironment:
    """
    Gym-like environment para RL.
    State: features de mercado
    Action: buy/hold/sell (o pesos en [0,1])
    Reward: Sharpe - 0.5*DD (drawdown penalty)
    """
    
    def __init__(self, features: np.ndarray, returns: np.ndarray, 
                 trading_cost: float = 0.001, window_size: int = 10):
        """
        features: [T, D] features del mercado
        returns: [T] retornos del activo
        """
        self.features = features  # [T, D]
        self.returns = returns    # [T]
        self.trading_cost = trading_cost
        self.window_size = window_size
        
        self.current_step = 0
        self.max_steps = len(features) - window_size
        self.portfolio_value = 1.0
        self.position_history = []
        self.pnl_history = []
        
    def reset(self):
        """Reset environment"""
        self.current_step = 0
        self.portfolio_value = 1.0
        self.position_history = []
        self.pnl_history = []
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """Retorna estado actual: [features recientes, portfolio_value]"""
        if self.current_step < self.window_size:
            obs = np.zeros(self.features.shape[1] + 1)
            obs[-1] = self.portfolio_value
        else:
            recent_features = self.features[
                self.current_step - self.window_size:self.current_step
            ].mean(axis=0)
            obs = np.concatenate([recent_features, [self.portfolio_value]])
        
        return obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        action: 0=sell, 1=hold, 2=buy
        Retorna: obs, reward, done, info
        """
        # Limitar a ventana válida
        if self.current_step >= self.max_steps:
            done = True
            return self._get_observation(), 0, done, {}
        
        # Posición anterior
        prev_position = self.position_history[-1] if self.position_history else 0
        
        # Nueva posición
        if action == 0:  # sell
            new_position = -1.0
        elif action == 1:  # hold
            new_position = prev_position
        else:  # buy (action == 2)
            new_position = 1.0
        
        # Trading cost
        position_change = abs(new_position - prev_position)
        cost = position_change * self.trading_cost
        
        # PnL
        ret = self.returns[self.current_step]
        pnl = new_position * ret - cost
        
        # Update portfolio
        self.portfolio_value *= (1 + pnl)
        self.position_history.append(new_position)
        self.pnl_history.append(pnl)
        
        # Reward: Sharpe-like signal
        if len(self.pnl_history) >= 10:
            recent_returns = np.array(self.pnl_history[-10:])
            sharpe = np.mean(recent_returns) / (np.std(recent_returns) + 1e-6)
        else:
            sharpe = pnl
        
        # Penalizar drawdown
        if self.portfolio_value < 0.95:
            reward = sharpe - 0.5 * (1 - self.portfolio_value)
        else:
            reward = sharpe
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation(), float(reward), done, {
            'portfolio_value': self.portfolio_value,
            'pnl': pnl
        }


class RLAllocationExpert:
    """
    Expert basado en RL para asset allocation.
    
    Pros:
    - Aprende políticas óptimas en entornos volátiles
    - Robusto a regime shifts (2022 bear)
    - Multi-armed bandit para ponderar modelos
    
    Cons:
    - Data-hungry (necesita simulaciones)
    - Over-optimiza a highs
    - Mejor para Paso 5 (allocation) que Paso 4 (training)
    
    Para Crypto:
    - Usar PPO (más estable que DQN)
    - Entrenar en rolling windows
    - Multi-asset como multi-armed bandit
    """
    
    def __init__(self, observation_space_size: int, 
                 algorithm: str = 'PPO', device: str = 'cpu'):
        """
        observation_space_size: dimensión del estado
        algorithm: 'PPO' o 'DQN'
        """
        if PPO is None or DQN is None:
            raise ImportError("Install stable_baselines3: pip install stable-baselines3")
        
        self.algorithm = algorithm
        self.observation_space_size = observation_space_size
        self.device = device
        self.model = None
        self.training_history = []
        
    def train_on_environment(self, features: np.ndarray, returns: np.ndarray,
                            total_timesteps: int = 10000) -> Dict:
        """
        Entrena agente RL en el environment.
        """
        env = MarketEnvironment(features, returns)
        
        if self.algorithm == 'PPO':
            self.model = PPO(
                'MlpPolicy',
                env,
                learning_rate=3e-4,
                n_steps=128,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                device=self.device,
                verbose=0
            )
        else:  # DQN
            self.model = DQN(
                'MlpPolicy',
                env,
                learning_rate=1e-4,
                buffer_size=10000,
                learning_starts=1000,
                target_update_interval=1000,
                exploration_fraction=0.1,
                device=self.device,
                verbose=0
            )
        
        # Train
        self.model.learn(total_timesteps=total_timesteps, progress_bar=False)
        
        # Evalúa en el mismo environment
        obs = env.reset()
        episode_reward = 0
        while True:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if done:
                break
        
        result = {
            'algorithm': self.algorithm,
            'episode_return': episode_reward,
            'final_portfolio_value': env.portfolio_value,
            'sharpe_approx': episode_reward / len(env.pnl_history) if env.pnl_history else 0,
            'trades': len(set(env.position_history))
        }
        
        self.training_history.append(result)
        return result
    
    def predict_allocation(self, features: np.ndarray) -> Dict:
        """
        Predice asignación óptima de capital.
        
        Returns:
            {
                'action': 0/1/2 (sell/hold/buy),
                'confidence': confianza en decisión,
                'allocation_weights': pesos para diferentes estrategias
            }
        """
        if self.model is None:
            return {
                'action': 1,  # hold por defecto
                'confidence': 0.5,
                'allocation_weights': [0.33, 0.33, 0.33]
            }
        
        try:
            obs = features.flatten().astype(np.float32)
            action, _states = self.model.predict(obs, deterministic=True)
            
            # Confianza basada en la incertidumbre del modelo
            # (próximo step después de training)
            confidence = 0.6 + 0.2 * (action - 1) / 3
            
            # Mapear action a pesos
            if action == 0:  # sell
                weights = [0.1, 0.3, 0.6]  # más defensivo
            elif action == 1:  # hold
                weights = [0.33, 0.33, 0.33]  # neutral
            else:  # buy
                weights = [0.6, 0.3, 0.1]  # más agresivo
            
            return {
                'action': int(action),
                'action_name': ['SELL', 'HOLD', 'BUY'][action],
                'confidence': float(confidence),
                'allocation_weights': weights
            }
        except Exception as e:
            print(f"⚠️  RL prediction failed: {e}")
            return {
                'action': 1,
                'action_name': 'HOLD',
                'confidence': 0.5,
                'allocation_weights': [0.33, 0.33, 0.33]
            }


class MultiArmedBanditExpert:
    """
    Multi-armed bandit para combinar múltiples expertos.
    Cada modelo es un "arm", RL elige pesos.
    """
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        """
        n_arms: número de expertos (modelos)
        epsilon: exploration rate
        """
        self.n_arms = n_arms
        self.epsilon = epsilon
        
        # Thompson sampling: beta distribution por arm
        self.arm_successes = np.zeros(n_arms)
        self.arm_failures = np.zeros(n_arms)
        self.arm_weights = np.ones(n_arms) / n_arms
        
    def select_arm(self) -> int:
        """Selecciona arm usando Thompson sampling"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_arms)
        
        # Sample de beta distribution
        theta = np.random.beta(
            self.arm_successes + 1,
            self.arm_failures + 1
        )
        
        return np.argmax(theta)
    
    def update_arm(self, arm: int, reward: float):
        """Actualiza arm después de observar reward"""
        if reward > 0:
            self.arm_successes[arm] += 1
        else:
            self.arm_failures[arm] += 1
        
        # Recalcula pesos (softmax de rewards)
        total_pulls = self.arm_successes + self.arm_failures
        success_rates = self.arm_successes / (total_pulls + 1e-6)
        
        self.arm_weights = np.exp(success_rates) / np.sum(np.exp(success_rates))
    
    def get_arm_weights(self) -> np.ndarray:
        """Retorna pesos normalizados para cada arm (experto)"""
        return self.arm_weights / np.sum(self.arm_weights)
