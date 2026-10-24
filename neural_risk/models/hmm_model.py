# neural_risk/models/hmm_model.py
"""
ESTADO: Parcheado (2 fixes de robustez numerica).
"""
from neural_risk.models.base import RiskModel
from hmmlearn.hmm import GaussianHMM
import numpy as np

class RegimeHMMModel(RiskModel):
    def __init__(self, n_components=3):
        super().__init__(model_name="HMM_Regime_Detector")
        # FIX ('covars must be symmetric, positive-definite'):
        # covariance_type cambiado de 'full' a 'diag'. 'full' estima una
        # matriz de covarianza COMPLETA entre las features de contexto
        # usadas (hurst/garch, o el fallback de hasta 3 columnas
        # numericas) -- si esas columnas estan correlacionadas entre si
        # (comun: 'hurst' y 'garch_vol' suelen moverse juntas) o hay
        # pocos datos distintos, la matriz se vuelve singular/no-
        # positiva-definida y hmmlearn crashea. 'diag' asume covarianza
        # diagonal (independencia entre features) -- mucho mas robusto,
        # y coincide con lo que YA se usaba en las HMM internas de
        # feature_engineering.py (gaussian_hmm_regimes,
        # hmm_log_likelihood_anomaly), que siempre usaron 'diag'. Esta
        # era la unica inconsistencia entre ambos usos de HMM.
        self.model = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000)
        self._context_features = None

    def _get_context_features(self, X):
        context_features = [c for c in X.columns if 'hurst' in c.lower() or 'garch' in c.lower()]
        if not context_features:
            context_features = X.select_dtypes(include=[np.number]).columns[:3].tolist()
        return context_features

    def _sanitize(self, data: np.ndarray) -> np.ndarray:
        """
        NUEVO: sanitiza NaN/inf antes de pasarle datos a hmmlearn.
        RiskFeaturePipeline ya deberia dejar los datos limpios rio
        arriba, pero un residual (ej. una columna del fallback que
        resulte constante y produzca inf al normalizar en otro lado)
        hacia crashear el fit/predict de forma dificil de diagnosticar.
        Defensa en profundidad, no reemplaza la limpieza rio arriba.
        """
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    def fit(self, X, y=None):
        context_features = self._get_context_features(X)
        self._context_features = context_features
        data = self._sanitize(X[context_features].values)
        print(f"Entrenando {self.model_name} con estados de mercado... features usadas: {context_features}")
        self.model.fit(data)
        # FIX: los índices de estado de hmmlearn son arbitrarios y cambian
        # entre reentrenamientos (el mismo punto podía salir 0 en un fit y
        # 1 en el siguiente). Se reordenan por la media de la primera
        # feature de contexto, así 'estado 0' significa siempre lo mismo.
        order = np.argsort(self.model.means_[:, 0])
        self._state_map = np.empty_like(order)
        self._state_map[order] = np.arange(len(order))
        self.is_trained = True

    def predict(self, X):
        # FIX: usa las MISMAS columnas fijadas en fit(), no recalcula
        # el filtro (evita mismatch de dimensiones entre fit y predict
        # si el fallback llegara a elegir columnas distintas).
        context_features = self._context_features or self._get_context_features(X)
        data = self._sanitize(X[context_features].values)
        raw_states = self.model.predict(data)
        state_map = getattr(self, '_state_map', None)
        return state_map[raw_states] if state_map is not None else raw_states