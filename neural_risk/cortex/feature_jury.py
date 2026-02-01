# neural_risk/cortex/feature_jury.py
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LassoCV
from statsmodels.tsa.stattools import grangercausalitytests

class FeatureJury:
    """
    Sistema de Consenso Estadístico para Selección de Features.
    Filtra features redundantes, no-causales o puramente aleatorias.
    """
    def __init__(self, significance=0.05, max_lag=5, mi_threshold=0.01):
        self.significance = significance
        self.max_lag = max_lag
        self.mi_threshold = mi_threshold

    def evaluate(self, df: pd.DataFrame, target: pd.Series) -> list:
        """
        Ejecuta el juicio y retorna la lista de features aprobadas por consenso.
        """
        warnings.filterwarnings('ignore')
        df_clean = df.select_dtypes(include=[np.number]).fillna(0)
        
        # Sincronización de índices
        common_idx = df_clean.index.intersection(target.index)
        X = df_clean.loc[common_idx]
        y = target.loc[common_idx]

        print(f"⚖️ Iniciando Juicio para {len(X.columns)} features...")

        # 1. Juez de Causalidad Temporal (Granger)
        granger_approved = self._run_granger(X, y)
        
        # 2. Juez de Información Mutua (No-linealidad)
        mi_approved = self._run_mutual_info(X, y)
        
        # 3. Juez de Importancia Estructural (Lasso)
        lasso_approved = self._run_lasso(X, y)

        # --- SISTEMA DE VOTACIÓN ---
        votes = {}
        for feat in X.columns:
            score = 0
            if feat in granger_approved: score += 1
            if feat in mi_approved: score += 1
            if feat in lasso_approved: score += 1
            votes[feat] = score

        # Consenso: Al menos 2 de 3 jueces deben aprobar la feature
        final_features = [f for f, v in votes.items() if v >= 2]
        
        print(f"✅ Veredicto: {len(final_features)} features aprobadas por mayoría.")
        return final_features

    def _run_granger(self, X, y):
        approved = []
        for col in X.columns:
            try:
                # Granger necesita: [Target, Feature]
                test_df = pd.concat([y, X[col]], axis=1)
                if X[col].std() == 0: continue
                
                res = grangercausalitytests(test_df, maxlag=self.max_lag, verbose=False)
                p_values = [res[lag][0]['ssr_ftest'][1] for lag in res]
                if min(p_values) < self.significance:
                    approved.append(col)
            except: continue
        return approved

    def _run_mutual_info(self, X, y):
        """
        Mide la reducción de incertidumbre. Es excelente para relaciones no lineales.
        """
        # I(X;Y) = H(Y) - H(Y|X)
        mi_scores = mutual_info_regression(X, y)
        mi_series = pd.Series(mi_scores, index=X.columns)
        # Filtramos por un umbral de importancia relativa
        return mi_series[mi_series > self.mi_threshold].index.tolist()

    def _run_lasso(self, X, y):
        """
        Lasso aplica penalización L1, llevando a cero las features redundantes.
        """
        # Estandarización interna para Lasso
        X_std = (X - X.mean()) / X.std()
        model = LassoCV(cv=5, random_state=42).fit(X_std.fillna(0), y)
        # Solo features con coeficiente distinto de cero
        return X.columns[np.abs(model.coef_) > 1e-5].tolist()