# neural_risk/models/copula_expert.py
"""
Copula-based Expert para Dependencia Multivariada
Perfecto para cripto multi-asset (contagio en crashes)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from copulae import GaussianCopula, TCopula, Clayton
except ImportError:
    GaussianCopula = None


class CopulaExpert:
    """
    Expert en dependencia multivariada via cópulas.
    
    Pros:
    - Modela joint tails (contagio 2022)
    - Complementa causal inference
    - Detects changes en correlación
    
    Cons:
    - Computacionalmente intenso para >10 assets
    - Asume distribuciones marginales
    
    Para Crypto:
    - Detecta cuando ETH-SOL correl >> 0.8
    - Señal: diversificar si tail risk alto
    - Uso en rolling windows
    """
    
    def __init__(self, copula_type: str = 'gaussian', n_assets: int = 5):
        """
        copula_type: 'gaussian', 'student', 'clayton'
        """
        if GaussianCopula is None:
            raise ImportError("Install copulae: pip install copulae")
        
        self.copula_type = copula_type
        self.n_assets = n_assets
        self.copula = None
        self.asset_names = []
        self.marginal_params = {}
        self.correlation_matrix = None
        
    def fit(self, X: pd.DataFrame) -> None:
        """
        Ajusta cópula a datos multivariados.
        
        Args:
            X: [T, N] datos (returns o features normalizados)
        """
        X_clean = X.fillna(0).values
        
        if len(X_clean) < 50:
            print("⚠️  Copula: Datos insuficientes (<50)")
            return
        
        try:
            # Fit marginals (asumimos Normal, pero podría ser Student)
            self.asset_names = X.columns.tolist()
            
            for col in X.columns:
                data = X[col].values
                mu = np.mean(data)
                sigma = np.std(data)
                self.marginal_params[col] = {'mu': mu, 'sigma': sigma}
            
            # Rank data para cópula (uniforme)
            ranks = X_clean.argsort(axis=0).argsort(axis=0) / len(X_clean)
            
            # Fit copula
            if self.copula_type == 'gaussian':
                self.copula = GaussianCopula(dim=X_clean.shape[1])
            elif self.copula_type == 'student':
                self.copula = TCopula(dim=X_clean.shape[1])
            else:  # clayton
                self.copula = Clayton(dim=X_clean.shape[1])
            
            self.copula.fit(ranks)
            
            # Almacena correlation
            self.correlation_matrix = np.corrcoef(X_clean.T)
            
        except Exception as e:
            print(f"⚠️  Copula fit failed: {e}")
            self.copula = None
    
    def get_tail_dependence(self, u: float = 0.05) -> Dict:
        """
        Calcula dependencia en colas.
        
        Args:
            u: quantile (0.05 = tail 5% inferior)
        
        Returns:
            {
                'tail_prob': probabilidad conjunta en colas,
                'tail_dependence_coeff': coeficiente de dependencia,
                'contagion_risk': riesgo de contagio,
                'diversification_benefit': beneficio de diversificación
            }
        """
        if self.copula is None or self.correlation_matrix is None:
            return {
                'tail_prob': np.nan,
                'tail_dependence_coeff': np.nan,
                'contagion_risk': 'UNKNOWN',
                'diversification_benefit': 0.5
            }
        
        try:
            # CDF en cola (todos los assets en cuantil bajo)
            tail_point = np.array([u] * self.n_assets)
            tail_prob = self.copula.cdf(tail_point)
            
            # Tail dependence coefficient
            # λ = P(X1 ≤ x1 | X2 ≤ x2) con x1,x2 en cola
            if tail_prob > 0:
                independence_prob = u ** self.n_assets  # Si independientes
                tdc = tail_prob / independence_prob
            else:
                tdc = 0.0
            
            # Contagion risk: correlación promedio en tail
            mean_corr = np.mean(self.correlation_matrix[
                np.triu_indices_from(self.correlation_matrix, k=1)
            ])
            
            contagion_risk = 'HIGH' if tdc > 1.5 else ('MID' if tdc > 0.8 else 'LOW')
            
            # Diversification benefit (1 - correlation)
            div_benefit = 1.0 - mean_corr if mean_corr > 0 else 0.5
            
            return {
                'tail_prob': float(tail_prob),
                'tail_dependence_coeff': float(tdc),
                'contagion_risk': contagion_risk,
                'diversification_benefit': float(div_benefit),
                'mean_correlation': float(mean_corr)
            }
        except Exception as e:
            print(f"⚠️  Tail dependence calc failed: {e}")
            return {
                'tail_prob': np.nan,
                'tail_dependence_coeff': np.nan,
                'contagion_risk': 'UNKNOWN',
                'diversification_benefit': 0.5
            }
    
    def get_dynamic_weights(self) -> Dict:
        """
        Propone pesos dinámicos basados en dependencia.
        Si correlación alta en colas → diversifica.
        """
        tail_info = self.get_tail_dependence()
        
        if tail_info['contagion_risk'] == 'HIGH':
            # Diversifica: pesos iguales entre activos
            weights = np.ones(self.n_assets) / self.n_assets
            signal = 'DIVERSIFY_URGENTLY'
        elif tail_info['contagion_risk'] == 'MID':
            # Pesos parcialmente diversificados
            weights = np.ones(self.n_assets) / self.n_assets * 0.8 + 0.05
            weights = weights / np.sum(weights)
            signal = 'BALANCE_PORTFOLIO'
        else:
            # Baja dependencia: puede concentrarse si quiere
            # Pero mantenemos diversificación prudente
            weights = np.ones(self.n_assets) / self.n_assets
            signal = 'CONCENTRATE_OK'
        
        return {
            'weights': weights.tolist(),
            'signal': signal,
            'confidence': 1.0 - tail_info['diversification_benefit']
        }
    
    def get_correlation_change(self, X_prev: pd.DataFrame, X_curr: pd.DataFrame) -> Dict:
        """
        Detecta cambios significativos en correlación.
        Indicador de regime shift.
        """
        corr_prev = X_prev.corr().values
        corr_curr = X_curr.corr().values
        
        # Norma de Frobenius: cambio total en correlación
        corr_change = np.linalg.norm(corr_curr - corr_prev, 'fro')
        
        # Threshold: si cambio > 0.5, significa algo nuevo
        regime_shift = corr_change > 0.5
        
        return {
            'correlation_change': float(corr_change),
            'regime_shift_detected': regime_shift,
            'prev_correlation': float(np.mean(corr_prev)),
            'curr_correlation': float(np.mean(corr_curr)),
            'change_direction': 'INCREASING' if corr_curr.mean() > corr_prev.mean() else 'DECREASING'
        }


class MultiAssetCopulaExpert:
    """
    Cópula para portafolio multi-asset.
    Combina información de dependencia para decisiones de asignación.
    """
    
    def __init__(self, asset_names: list):
        self.asset_names = asset_names
        self.copulas = {
            asset: CopulaExpert('gaussian', len(asset_names))
            for asset in asset_names
        }
        self.portfolio_tail_risk = 0.5
        
    def fit_all(self, X_dict: Dict[str, pd.DataFrame]) -> None:
        """
        Ajusta cópula para cada activo.
        X_dict: {'BTC': df_btc, 'ETH': df_eth, ...}
        """
        for asset, X in X_dict.items():
            if asset in self.copulas:
                self.copulas[asset].fit(X)
    
    def get_portfolio_tail_risk(self) -> Dict:
        """
        Calcula tail risk del portafolio completo.
        """
        tail_risks = []
        contagion_signals = []
        
        for asset, copula in self.copulas.items():
            tail_info = copula.get_tail_dependence()
            tail_risks.append(tail_info['tail_dependence_coeff'])
            contagion_signals.append(tail_info['contagion_risk'])
        
        # Agregado
        avg_tail_risk = np.nanmean(tail_risks)
        high_risk_count = sum(1 for s in contagion_signals if s == 'HIGH')
        
        portfolio_risk = 'CRISIS' if high_risk_count >= 2 else (
            'HIGH' if avg_tail_risk > 1.2 else (
            'MID' if avg_tail_risk > 0.8 else 'LOW'
            )
        )
        
        return {
            'portfolio_tail_risk': portfolio_risk,
            'avg_tail_dependence': float(avg_tail_risk),
            'assets_in_high_contagion': high_risk_count,
            'recommendation': 'REDUCE_LEVERAGE' if portfolio_risk in ['CRISIS', 'HIGH'] else 'NORMAL'
        }
    
    def get_optimal_rebalancing(self) -> Dict:
        """
        Propone rebalancing basado en dependencia.
        """
        portfolio_risk = self.get_portfolio_tail_risk()
        
        rebalance_weights = {}
        total_weight = 0
        
        for asset, copula in self.copulas.items():
            tail_info = copula.get_tail_dependence()
            
            # Peso inverso al tail risk
            if tail_info['tail_dependence_coeff'] > 0:
                weight = 1.0 / (1.0 + tail_info['tail_dependence_coeff'])
            else:
                weight = 1.0 / len(self.copulas)
            
            rebalance_weights[asset] = weight
            total_weight += weight
        
        # Normaliza
        rebalance_weights = {
            asset: w / total_weight for asset, w in rebalance_weights.items()
        }
        
        return {
            'rebalancing_weights': rebalance_weights,
            'portfolio_risk': portfolio_risk['portfolio_tail_risk'],
            'urgency': 'IMMEDIATE' if portfolio_risk['portfolio_tail_risk'] == 'CRISIS' else 'GRADUAL'
        }
