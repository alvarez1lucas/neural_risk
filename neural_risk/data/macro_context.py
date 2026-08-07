# neural_risk/data/macro_context.py
"""
Contexto MACRO externo (fuera del universo cripto) como features de
apoyo -- NO como activos a tradear. Motivación (pedido explícito del
usuario): cripto correlaciona con regímenes de risk-on/risk-off del
mercado tradicional, y el sistema hoy no ve nada fuera de OHLCV cripto.

DISEÑO: en vez de mezclar VIX/DXY/SPX/Oro con RiskFeaturePipeline
(que asume activos comparables entre sí -- cointegración, centralidad
de grafos, etc., no tiene sentido entre BTC y el VIX), se calculan
features SIMPLES y derivadas (retorno, z-score rodante, volatilidad
rodante) por cada serie macro, con prefijo 'MACRO_', y se agregan como
columnas de CONTEXTO al feature set de cada activo antes de que el
jurado (FeatureJury) decida si aportan señal real.

Sin look-ahead: el alineamiento usa ffill() por fecha -- el valor
"conocido" de VIX un sábado/domingo es el cierre del viernes, igual que
en la vida real (los mercados tradicionales no operan fin de semana,
cripto sí).
"""

import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime, timedelta
from typing import Dict, Optional

from neural_risk.data.loaders import YahooFinanceLoader


DEFAULT_MACRO_TICKERS = {
    'VIX': '^VIX',        # Volatilidad implícita S&P500 -- proxy de miedo/risk-off
    'DXY': 'DX-Y.NYB',    # Índice dólar -- fuerte anti-correlación histórica con cripto
    'SPX': '^GSPC',       # S&P 500 -- correlación risk-on con cripto en años recientes
    'GOLD': 'GC=F',       # Oro -- refugio tradicional, contraste útil vs. "oro digital"
}


class MacroContextBuilder:
    """Descarga y deriva features de contexto macro, con cache en disco."""

    def __init__(self, tickers: Optional[Dict[str, str]] = None,
                 cache_path: str = "./data/macro_context_cache.pkl",
                 max_age_hours: float = 6.0):
        self.tickers = tickers or DEFAULT_MACRO_TICKERS
        self.cache_path = cache_path
        self.max_age_hours = max_age_hours

    def fetch_raw(self, period: str = '1y') -> pd.DataFrame:
        """
        Descarga el Close diario de cada ticker macro y arma un único
        DataFrame alineado por fecha. Si algún ticker falla, se omite
        (no aborta el resto) -- el macro context es un "nice to have",
        no debe tumbar el pipeline principal si Yahoo Finance falla.
        """
        loader = YahooFinanceLoader()
        series_list = []

        for name, ticker in self.tickers.items():
            try:
                df = loader.fetch_ohlcv(ticker, period=period, interval='1d')
                if df.empty:
                    continue
                series_list.append(df['Close'].rename(f'MACRO_{name}_Close'))
            except Exception as e:
                print(f"⚠️  MacroContext: no se pudo descargar {name} ({ticker}): {e}")
                continue

        if not series_list:
            return pd.DataFrame()

        combined = pd.concat(series_list, axis=1)
        combined.index = pd.to_datetime(combined.index).tz_localize(None) if combined.index.tz is not None else combined.index
        return combined.sort_index()

    def compute_features(self, raw_df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Deriva, por cada serie macro cruda: retorno logarítmico,
        z-score rodante del nivel, y volatilidad rodante del retorno.
        Todas con prefijo MACRO_ para que sea evidente en el jurado/
        modelos de dónde vienen.
        """
        if raw_df.empty:
            return pd.DataFrame()

        out = pd.DataFrame(index=raw_df.index)

        for col in raw_df.columns:
            series = raw_df[col]
            log_ret = np.log(series / series.shift(1))

            roll_mean = series.rolling(window=window).mean()
            roll_std = series.rolling(window=window).std()
            zscore = (series - roll_mean) / (roll_std + 1e-9)

            ret_vol = log_ret.rolling(window=window).std()

            base = col.replace('_Close', '')
            out[f'{base}_ret'] = log_ret
            out[f'{base}_zscore'] = zscore
            out[f'{base}_vol'] = ret_vol

        return out

    def _load_cache(self) -> Optional[Dict]:
        if not os.path.exists(self.cache_path):
            return None
        try:
            with open(self.cache_path, 'rb') as f:
                cache = pickle.load(f)
            age_hours = (datetime.now() - cache['fetched_at']).total_seconds() / 3600
            if age_hours > self.max_age_hours:
                return None
            return cache
        except Exception:
            return None

    def _save_cache(self, features_df: pd.DataFrame):
        try:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, 'wb') as f:
                pickle.dump({'fetched_at': datetime.now(), 'features': features_df}, f)
        except Exception as e:
            print(f"⚠️  MacroContext: no se pudo guardar cache: {e}")

    def get_macro_features(self, period: str = '1y', force_refresh: bool = False) -> pd.DataFrame:
        """
        Punto de entrada principal: devuelve features macro listas para
        mergear, usando cache en disco (evita pegarle a Yahoo Finance en
        cada ciclo de 15-300s del engine live -- el macro context cambia
        lento, no hace falta refrescarlo tan seguido).
        """
        if not force_refresh:
            cached = self._load_cache()
            if cached is not None:
                return cached['features']

        raw_df = self.fetch_raw(period=period)
        if raw_df.empty:
            print("⚠️  MacroContext: sin datos macro disponibles, devolviendo DataFrame vacío")
            return pd.DataFrame()

        features_df = self.compute_features(raw_df)
        self._save_cache(features_df)
        return features_df


def merge_macro_context(df_features: pd.DataFrame, macro_features: pd.DataFrame) -> pd.DataFrame:
    """
    Alinea el contexto macro al índice del activo (ffill -- ver nota de
    look-ahead en el docstring del módulo) y lo concatena. Si
    macro_features está vacío (sin internet, o falló la descarga), no
    hace nada -- el pipeline sigue funcionando sin contexto macro.
    """
    if macro_features is None or macro_features.empty:
        return df_features

    macro_aligned = macro_features.reindex(df_features.index, method='ffill')
    return pd.concat([df_features, macro_aligned], axis=1)