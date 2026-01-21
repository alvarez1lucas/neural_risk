import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
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
        """
        Pipeline maestro de limpieza automática para DataFrames financieros.
        
        Realiza:
        1. Inferencia y conversión inteligente de tipos de datos.
        2. Detección de fechas.
        3. Relleno financiero lógico (Forward Fill para precios, 0 para volumen).
        4. Eliminación de filas vacías residuales.
        
        Args:
            df (pd.DataFrame): DataFrame crudo (OHLCV, Tasas, etc).
            
        Returns:
            pd.DataFrame: DataFrame limpio y listo para cálculo.
        """
        df = df.copy()
        
        # 1. Optimización de Tipos (Object -> Float/Datetime)
        df = self._optimize_types(df)
        
        # 2. Ordenar por índice si es fecha
        if isinstance(df.index, pd.DatetimeIndex):
            df = df.sort_index()

        # 3. Lógica de Relleno Financiero (Financial Imputation)
        # Identificamos columnas de Volumen
        vol_cols = [c for c in df.columns if 'volume' in c.lower() or 'vol' in c.lower()]
        price_cols = [c for c in df.columns if c not in vol_cols]
        
        # Precios: Forward Fill (El precio de ayer es la mejor predicción del precio de hoy si falta data)
        # Importante: NO usar interpolación lineal en precios, inventa movimientos que no existen.
        if price_cols:
            df[price_cols] = df[price_cols].ffill()
            
        # Volumen: Si falta, asumimos 0 actividad
        if vol_cols:
            df[vol_cols] = df[vol_cols].fillna(0)
            
        # Limpieza final de NaNs al inicio de la serie (donde no hay ffill posible)
        return df.dropna()

    def _optimize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Sub-rutina interna para inferencia de tipos."""
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Intenta limpiar comas de miles "1,000.50" y pasar a numérico
                    df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='raise')
                except:
                    try:
                        # Intenta pasar a fecha
                        df[col] = pd.to_datetime(df[col])
                    except:
                        pass # Se mantiene como objeto (ej. Tickers, Categorías)
        return df


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
        
        Args:
            series (pd.Series): Serie de tiempo (ej. Precios o Retornos).
            threshold (float): Valor p de corte (usualmente 0.05).
            
        Returns:
            Dict: Reporte con p-value, estadístico y recomendación.
        """
        clean_series = series.dropna()
        if len(clean_series) < 20:
            return {'error': 'Insuficientes datos para test ADF'}
            
        # maxlag=None deja que AIC elija el lag óptimo
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
    
    def get_returns(self, df: pd.DataFrame, columns: list = None, method: str = 'log') -> pd.DataFrame:
        
        df_ret = df.copy()
        cols = columns if columns else df.columns
        
        for col in cols:
            if method == 'log':
                df_ret[f"{col}_ret"] = np.log(df_ret[col] / df_ret[col].shift(1))
            else:
                df_ret[f"{col}_ret"] = df_ret[col].pct_change()
                
        return df_ret.dropna()

    def fractional_diff(self, series: pd.Series, d: float = 0.4, thres: float = 1e-5) -> pd.Series:
        
        # 1. Configuración de pesos y ventanas
        w = self.get_weights_ffd(d, thres, len(series))
        width = len(w) - 1
        
        # 2. Aplicación vectorial (Convolution)
        # Rellenamos con NaNs al principio porque perdemos 'width' datos
        df_temp = series.to_frame()
        output = {}
        
        column_name = series.name if series.name else 'close'
        
        # Loop optimizado (pandas rolling apply es lento para esto, usamos loops controlados o convolución)
        # Para series financieras largas, iterar sobre la serie con pesos fijos es eficiente
        series_values = series.dropna().values
        if len(series_values) < width:
            return pd.Series(index=series.index, data=np.nan)

        transformed = []
        # Aplicamos producto punto de los pesos sobre la ventana deslizante
        for i in range(width, len(series_values)):
            window0 = series_values[i-width : i+1]
            # Nota: w son los pesos, window0 es el precio. Producto punto.
            dot_prod = np.dot(w.T, window0)[0]
            transformed.append(dot_prod)
            
        # Reconstruir índice (alineando a la derecha, perdemos los primeros 'width' datos)
        new_index = series.dropna().index[width:]
        return pd.Series(data=transformed, index=new_index, name=f"{column_name}_frac_d{d}")
    
    @staticmethod
    def merge_datasets(dfs: list, ffill: bool = True) -> pd.DataFrame:
        """
        Une múltiples DataFrames por fecha (Index) y aplica relleno total.
        Ideal para armar carteras de múltiples activos.
        """
        # Unimos usando un 'outer join' para no perder días de ningún activo
        combined_df = pd.concat(dfs, axis=1, join='outer')
        
        if ffill:
            # Relleno hacia adelante para que todos los activos tengan precio el mismo día
            combined_df = combined_df.ffill()
            
        return combined_df.dropna() 
    
    def handle_outliers(self, df: pd.DataFrame, limits: list = [0.01, 0.01]) -> pd.DataFrame:
        """
        Aplica Winsorization para limitar valores extremos sin eliminarlos.
        Ideal para evitar que el ruido desvíe el entrenamiento de modelos de ML.
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

    def get_returns(self, df: pd.DataFrame, columns: list = None, method: str = 'log') -> pd.DataFrame:
        """
        Calcula retornos. Se agregó manejo de errores para evitar divisiones por cero.
        """
        df_ret = pd.DataFrame(index=df.index)
        cols = columns if columns else df.columns
        
        for col in cols:
            if method == 'log':
                # Log-returns: ln(Pt / Pt-1)
                df_ret[f"{col}_ret"] = np.log(df[col] / df[col].shift(1))
            else:
                df_ret[f"{col}_ret"] = df[col].pct_change()
        
        return df_ret.dropna()