# neural_risk/data/data_processor.py
"""
Módulo de limpieza e ingeniería de características para series temporales financieras.

ESTADO: Parcheado -- se eliminaron 2 definiciones duplicadas de
_optimize_types y 2 de get_returns (en Python, cuando un método se
define varias veces en la misma clase, gana la última silenciosamente;
las anteriores quedaban como código muerto invisible, confuso para
cualquiera que leyera el archivo de arriba hacia abajo). Se dejó UNA
sola versión de cada uno, consolidada.

FIX escalabilidad: 'from statsmodels.tsa.stattools import adfuller' se
movió DENTRO de check_stationarity() (import local) -- era el ÚNICO
método de esta clase que necesita statsmodels, pero el import estaba a
nivel de módulo, lo que obligaba a instalar statsmodels para usar
CUALQUIER método de DataProcessor (auto_clean, rename_columns,
get_returns...), aunque no lo necesiten.
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, List
from scipy.stats.mstats import winsorize

class DataProcessor:
    """
    Módulo de limpieza e ingeniería de características para series temporales financieras.
    Incluye técnicas avanzadas de ML (Fractional Differentiation) para preservar memoria
    estadística mientras se logra estacionariedad.
    """

    def __init__(self):
        pass

    def auto_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline de limpieza optimizado."""
        df = df.copy()
        df = self._optimize_types(df)
        if not isinstance(df.index, pd.DatetimeIndex) and 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        
        df = df.sort_index()
        vol_cols = [c for c in df.columns if 'vol' in c.lower()]
        price_cols = [c for c in df.columns if c not in vol_cols]
        
        if price_cols: df[price_cols] = df[price_cols].ffill()
        if vol_cols: df[vol_cols] = df[vol_cols].fillna(0)
        
        return df.dropna()

    def _optimize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sub-rutina interna para inferencia de tipos. Intenta pasar
        columnas tipo 'object' a numérico (limpiando comas de miles) o,
        si falla, a fecha. Si ninguna conversión aplica, se mantiene
        como objeto (ej. tickers, categorías).
        """
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='raise')
                except:
                    try:
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass
        return df

    def prepare_portfolio_df(self, assets_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Toma un diccionario de activos {'BTC': df1, 'SPY': df2}
        y devuelve un único DF alineado y listo para el Jurado.
        """
        processed_dfs = []
        
        for ticker, df in assets_data.items():
            df_clean = self.auto_clean(df)
            df_renamed = df_clean.add_prefix(f"{ticker}_")
            processed_dfs.append(df_renamed)
            
        combined = pd.concat(processed_dfs, axis=1, join='outer')
        combined = combined.ffill().dropna()
        
        print(f"📊 Portfolio alineado: {len(combined.columns)} columnas para {list(assets_data.keys())}")
        return combined

    def get_returns(self, df: pd.DataFrame, columns: list = None,
                   method: str = 'log', dropna: bool = True) -> pd.DataFrame:
        """
        Calcula retornos para las columnas indicadas (o todas si no se
        especifican). Agrega columnas nuevas con sufijo '_ret', sin
        pisar las originales.
        
        FIX: antes existían 3 versiones de este método en el archivo.
        La que quedaba activa (la última) NO hacía dropna() al final --
        quien llamara a get_returns() esperando datos limpios (como
        sugería el comportamiento de las 2 versiones anteriores)
        recibía un NaN silencioso en la primera fila. Ahora dropna es
        un parámetro explícito, default True (comportamiento seguro).
        """
        df_ret = pd.DataFrame(index=df.index)
        cols = columns if columns else df.columns
        
        for col in cols:
            if method == 'log':
                df_ret[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))
            else:
                df_ret[f"{col}_ret"] = df[col].pct_change()
        
        return df_ret.dropna() if dropna else df_ret

    def rename_columns(self, df: pd.DataFrame, asset_name: str) -> pd.DataFrame:
        """
        Agrega el nombre del activo como prefijo a todas las columnas.
        Ejemplo: 'Close' -> 'AAPL_Close'
        """
        df = df.copy()
        df.columns = [f"{asset_name}_{col}" for col in df.columns]
        return df

    def check_stationarity(self, series: pd.Series, threshold: float = 0.05) -> Dict:
        """
        Ejecuta el Test de Dickey-Fuller Aumentado (ADF) para verificar si una serie
        es apta para Machine Learning.
        """
        from statsmodels.tsa.stattools import adfuller  # import local: único método que lo necesita

        clean_series = series.dropna()
        if len(clean_series) < 20:
            return {'error': 'Insuficientes datos para test ADF'}
            
        result = adfuller(clean_series, maxlag=None, autolag='AIC')
        
        p_value = result[1]
        is_stationary = p_value < threshold
        
        return {
            'is_stationary': is_stationary,
            'p_value': round(p_value, 6),
            'test_stat': round(result[0], 4),
            'n_lags': result[2],
            'recommendation': "Ready for ML models" if is_stationary else "Must differentiate (Try FracDiff)"
        }

    def get_weights_ffd(self, d: float, thres: float, lim: int) -> np.ndarray:
        """
        Calcula los pesos para la Diferenciación Fraccionaria (Fixed Window).
        Matemática basada en Marcos López de Prado (Advances in Financial ML).
        """
        w, k = [1.], 1
        while True:
            w_k = -w[-1] / k * (d - k + 1)
            if abs(w_k) < thres:
                break
            w.append(w_k)
            k += 1
            if k >= lim:
                break
        return np.array(w[::-1]).reshape(-1, 1)

    def fractional_diff(self, series: pd.Series, d: float = 0.4, thres: float = 1e-5) -> pd.Series:
        w = self.get_weights_ffd(d, thres, len(series))
        width = len(w) - 1
        
        column_name = series.name if series.name else 'close'
        
        series_values = series.dropna().values
        if len(series_values) < width:
            return pd.Series(index=series.index, data=np.nan)

        transformed = []
        for i in range(width, len(series_values)):
            window0 = series_values[i-width : i+1]
            dot_prod = np.dot(w.T, window0)[0]
            transformed.append(dot_prod)
            
        new_index = series.dropna().index[width:]
        return pd.Series(data=transformed, index=new_index, name=f"{column_name}_frac_d{d}")

    @staticmethod
    def merge_datasets(dfs: list, ffill: bool = True) -> pd.DataFrame:
        """
        Une múltiples DataFrames por fecha (Index) y aplica relleno total.
        """
        combined_df = pd.concat(dfs, axis=1, join='outer')
        if ffill:
            combined_df = combined_df.ffill()
        return combined_df.dropna()

    def handle_outliers(self, df: pd.DataFrame, limits: list = [0.01, 0.01]) -> pd.DataFrame:
        """
        Aplica Winsorization para limitar valores extremos sin eliminarlos.
        """
        df_clean = df.copy()
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            df_clean[col] = winsorize(df_clean[col], limits=limits)
        return df_clean

    def standardize_timezone(self, df: pd.DataFrame, tz: str = 'UTC') -> pd.DataFrame:
        """
        Asegura que todos los activos compartan la misma zona horaria antes del merge.
        """
        df = df.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize(tz)
        else:
            df.index = df.index.tz_convert(tz)
        return df