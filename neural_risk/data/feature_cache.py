# neural_risk/data/feature_cache.py
"""
Cache de features (Paso 2) para iterar modelos rápido sin recalcular el
feature engineering pesado (RiskFeaturePipeline.transform + FeatureJury)
en cada corrida.

MOTIVACION (validada por el uso real del usuario en notebooks): las
features tardan mucho en calcularse (HMM en loop, IsolationForest,
KMeans, Granger/MI/Lasso del jurado); los MODELOS se iteran rápido. Sin
esta capa, cada vez que se prueba un cambio en un modelo, se vuelve a
pagar el costo completo de Paso 1-3 -- exactamente el patrón que el
usuario venía haciendo a mano exportando X_features_full.csv,
X_features_selected.csv, y_target.csv desde el notebook.

Esto formaliza ese mismo patrón como parte de la librería: un .pkl por
activo (rápido de cargar, preserva dtypes/índice exactos) + exportación
opcional a los mismos CSVs que el usuario ya venía generando a mano
(para seguir pudiendo abrirlos en el notebook si lo prefiere).

Invalidación automática: la "huella" del cache depende del ticker + el
rango de fechas + la cantidad de filas del raw_df de entrada. Si
cambia cualquiera de esos tres (datos nuevos, distinto rango), el
cache se invalida solo -- no hace falta borrarlo a mano ni acordarse
de versionarlo.
"""
import os
import pickle
import hashlib
import pandas as pd
from typing import Optional, Dict


class FeatureCache:
    """
    Cachea el df_features (Paso 1-2) por activo. La selección de features
    (Paso 3: target + jurado) se vuelve a correr encima, así el mismo cache
    sirve en modo entrenamiento y en modo predicción -- antes se cacheaba
    la salida de prepare_asset_features entera, y un hit en modo predicción
    devolvía el y/best_feats del entrenamiento.
    """

    def __init__(self, cache_dir: str = "./data/feature_cache"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _fingerprint(self, ticker: str, raw_df: pd.DataFrame, variant: str = '') -> str:
        # Incluye el último Close: si el proveedor revisa la vela final, la
        # huella cambia aunque el rango de fechas sea el mismo.
        last_close = raw_df.select_dtypes('number').iloc[-1].sum() if len(raw_df) else 0
        key = f"{ticker}_{raw_df.index.min()}_{raw_df.index.max()}_{len(raw_df)}_{last_close:.8f}_{variant}"
        return hashlib.md5(key.encode()).hexdigest()[:12]

    def _cache_path(self, ticker: str, fingerprint: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker}_{fingerprint}.pkl")

    def load(self, ticker: str, raw_df: pd.DataFrame, variant: str = '') -> Optional[pd.DataFrame]:
        """Devuelve el df_features cacheado, o None si no hay cache válido."""
        path = self._cache_path(ticker, self._fingerprint(ticker, raw_df, variant))
        if not os.path.exists(path):
            return None
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"⚠️  FeatureCache: cache corrupto para {ticker}, se recalcula ({e})")
            return None

    def save(self, ticker: str, raw_df: pd.DataFrame, df_features: pd.DataFrame,
             variant: str = '') -> str:
        path = self._cache_path(ticker, self._fingerprint(ticker, raw_df, variant))
        with open(path, 'wb') as f:
            pickle.dump(df_features, f)
        return path

    def export_to_csv(self, ticker: str, raw_df: pd.DataFrame, out_dir: str = ".",
                      variant: str = '') -> Dict[str, str]:
        """Exporta el df_features cacheado a CSV (portabilidad/inspección)."""
        cached = self.load(ticker, raw_df, variant)
        if cached is None:
            raise ValueError(
                f"No hay cache para {ticker} con este dataset -- "
                f"correr prepare_asset_features_cached() primero"
            )
        os.makedirs(out_dir, exist_ok=True)
        full_path = os.path.join(out_dir, f"{ticker}_X_features_full.csv")
        cached.select_dtypes(include='number').to_csv(full_path)
        return {'X_features_full': full_path}
