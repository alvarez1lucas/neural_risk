# neural_risk/cortex/causal_selector.py
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

class CausalFeatureSelector:
    def __init__(self, max_lag=5):
        self.max_lag = max_lag

    def filter_by_causality(self, df: pd.DataFrame, target: str, significance: float = 0.05) -> list:
        """
        Recorre todas las features y elimina las que no causan (Granger) al target.
        """
        valid_features = []
        features = df.columns.drop(target)
        
        print(f"🔬 Iniciando análisis causal en {len(features)} variables...")
        
        for feat in features:
            try:
                # Test de Granger: ¿La feature en t-lag predice el Target en t?
                test_result = grangercausalitytests(df[[target, feat]], maxlag=self.max_lag, verbose=False)
                
                # Chequeamos si algun lag tiene p-value < 0.05
                is_causal = any([res[0]['ssr_ftest'][1] < significance for lag, res in test_result.items()])
                
                if is_causal:
                    valid_features.append(feat)
            except:
                continue
                
        print(f" Features causales seleccionadas: {len(valid_features)}/{len(features)}")
        return valid_features