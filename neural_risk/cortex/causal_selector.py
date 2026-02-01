# neural_risk/cortex/causal_selector.py
"""
Módulo de selección causal y análisis de causalidad de Granger.
Complementa el trabajo del FeatureJury con análisis más profundos de causalidad.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
import warnings


class CausalSelector:
    """
    Evalúa causalidad bidireccional entre features y el target
    usando tests de Granger y análisis de retroalimentación.
    """
    
    def __init__(self, max_lag: int = 5, significance: float = 0.05):
        self.max_lag = max_lag
        self.significance = significance
        self.causal_matrix = None
    
    def evaluate_causality(self, df: pd.DataFrame, target: pd.Series) -> dict:
        """
        Evalúa relaciones causales entre features y target.
        
        Returns:
            {
                'direct_causes': features que causan target,
                'feedback_loops': features causadas por target,
                'neutral': features sin relación causal,
                'causal_matrix': matriz de p-values
            }
        """
        warnings.filterwarnings('ignore')
        
        common_idx = df.index.intersection(target.index)
        X = df.loc[common_idx].select_dtypes(include=[np.number]).fillna(0)
        y = target.loc[common_idx].values
        
        direct_causes = []
        feedback_loops = []
        neutral = []
        
        p_value_matrix = {}
        
        for col in X.columns:
            try:
                # Test: Feature → Target
                data_forward = np.column_stack([y, X[col].values])
                
                if X[col].std() == 0:
                    neutral.append(col)
                    continue
                
                res_forward = grangercausalitytests(data_forward, maxlag=self.max_lag, verbose=False)
                p_forward = min([res_forward[lag][0]['ssr_ftest'][1] for lag in res_forward])
                
                # Test: Target → Feature (feedback)
                data_backward = np.column_stack([X[col].values, y])
                res_backward = grangercausalitytests(data_backward, maxlag=self.max_lag, verbose=False)
                p_backward = min([res_backward[lag][0]['ssr_ftest'][1] for lag in res_backward])
                
                p_value_matrix[col] = {'forward': p_forward, 'backward': p_backward}
                
                # Clasificar relación
                forward_sig = p_forward < self.significance
                backward_sig = p_backward < self.significance
                
                if forward_sig and not backward_sig:
                    direct_causes.append(col)
                elif backward_sig and not forward_sig:
                    feedback_loops.append(col)
                elif forward_sig and backward_sig:
                    # Causalidad bidireccional, pero consideramos como causa directa
                    direct_causes.append(col)
                else:
                    neutral.append(col)
                    
            except:
                neutral.append(col)
        
        self.causal_matrix = pd.DataFrame(p_value_matrix).T
        
        return {
            'direct_causes': direct_causes,
            'feedback_loops': feedback_loops,
            'neutral': neutral,
            'causal_matrix': self.causal_matrix,
            'p_values': p_value_matrix
        }
    
    def get_causal_graph(self) -> pd.DataFrame:
        """Retorna la matriz de causalidad para visualización."""
        if self.causal_matrix is not None:
            return self.causal_matrix
        return pd.DataFrame()
