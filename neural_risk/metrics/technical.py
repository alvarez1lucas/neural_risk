# neural_risk/metrics/technical.py
import pandas as pd
import numpy as np

class MarketIndicators:
    """Indicadores clásicos para inyectar al pipeline de features."""
    
    @staticmethod
    def add_all_indicators(df, close_col='Close'):
        df = MarketIndicators.momentum_indicators(df, close_col)
        df = MarketIndicators.trend_indicators(df, close_col)
        df = MarketIndicators.volatility_indicators(df, close_col)
        return df

    @staticmethod
    def momentum_indicators(df, close_col):
        # RSI: Fuerza relativa
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # MACD: Diferencia de medias rápidas/lentas
        ema12 = df[close_col].ewm(span=12, adjust=False).mean()
        ema26 = df[close_col].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema12 - ema26
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        return df

    @staticmethod
    def trend_indicators(df, close_col):
        # Distancia a la Media (Media como soporte/resistencia)
        df['SMA_20_Dist'] = (df[close_col] / df[close_col].rolling(window=20).mean()) - 1
        df['SMA_200_Dist'] = (df[close_col] / df[close_col].rolling(window=200).mean()) - 1
        return df

    @staticmethod
    def volatility_indicators(df, close_col):
        # Bollinger Bands %B (Mide qué tan 'apretado' está el precio)
        sma = df[close_col].rolling(window=20).mean()
        std = df[close_col].rolling(window=20).std()
        df['BB_Percent'] = (df[close_col] - (sma - 2*std)) / (4*std + 1e-9)
        return df
