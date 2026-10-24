# neural_risk/models/garch_volatility.py
"""
GARCH/EGARCH para Volatilidad Condicional
Perfecto para cripto: vol clustering, crashes, asymmetry
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from arch import arch_model
except ImportError:
    arch_model = None


class GARCHVolatilityExpert:
    """
    Expert en Volatilidad Condicional.
    
    Pros:
    - Vol clustering en crashes (2022, 2020)
    - Forecast de vol para hedging
    - Complementa XGB con persistencia
    - Bajo costo computacional
    - Robusto en beta alto
    
    Cons:
    - Asume heteroskedasticidad (ok para cripto)
    - Jumps news-driven → EGARCH maneja asymmetry
    
    Para Crypto:
    - Fit en ventanas [10, 30, 100, 1000]
    - EGARCH captura: negative returns ↑ volatility más
    - Output: vol_forecast para hedging signals
    """
    
    def __init__(self, model_type='egarch', p=1, q=1, o=1):
        """
        model_type: 'garch' o 'egarch'
        p, q, o: GARCH orders
        """
        if arch_model is None:
            raise ImportError("Install arch: pip install arch")
        
        self.model_type = model_type
        self.p = p
        self.q = q
        self.o = o if model_type == 'egarch' else None
        self.fitted_model = None
        self.last_return = None
        self.vol_history = []
        # Los retornos diarios (~0.01-0.05) están mal escalados para el
        # optimizador de arch -- se fitea sobre retornos*100 y se des-escala
        # al reportar volatilidades.
        self.scale = 100.0
        
    def fit(self, returns: np.ndarray) -> None:
        """
        Entrena GARCH/EGARCH en retornos.
        
        Args:
            returns: array de retornos (puede tener NaN)
        """
        # Limpiar
        returns_clean = np.array(returns).flatten()
        returns_clean = returns_clean[~np.isnan(returns_clean)]
        
        if len(returns_clean) < 50:
            print("⚠️  GARCH: Datos insuficientes (<50)")
            return
        
        # Fitear modelo
        try:
            self.fitted_model = self._build_model(returns_clean).fit(disp='off')
            self.last_return = returns_clean[-1]
        except Exception as e:
            print(f"⚠️  GARCH fit failed: {e}")
            self.fitted_model = None

    def _build_model(self, returns_clean: np.ndarray):
        if self.model_type == 'egarch':
            return arch_model(returns_clean * self.scale, vol='EGARCH',
                              p=self.p, q=self.q, o=self.o, rescale=False)
        return arch_model(returns_clean * self.scale, vol='Garch',
                          p=self.p, q=self.q, rescale=False)

    def forecast_volatility(self, horizon: int = 1,
                            recent_returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Forecast de volatilidad para hedging.

        Args:
            horizon: períodos adelante
            recent_returns: NUEVO -- si se pasa, se re-filtra la varianza
                condicional sobre estos retornos con los parámetros YA
                estimados (sin refitear). Sin esto, el forecast quedaba
                congelado en el último dato del entrenamiento entre
                reentrenamientos.

        Returns:
            {
                'vol_forecast': volatilidad predicha (escala de retornos),
                'vol_signal': HIGH/MID/LOW,
                'vol_zscore': z-score vs histórico
            }
        """
        if self.fitted_model is None:
            return {
                'vol_forecast': np.nan,
                'vol_signal': 'MID',
                'vol_zscore': 0.0,
                'hedging_intensity': 0.0
            }

        try:
            result = self.fitted_model
            if recent_returns is not None:
                rr = np.asarray(recent_returns, dtype=float).flatten()
                rr = rr[~np.isnan(rr)]
                if len(rr) >= 20:
                    result = self._build_model(rr).fix(self.fitted_model.params)

            # Forecast condicional
            forecast = result.forecast(horizon=horizon)
            variance_forecast = forecast.variance.iloc[-1, -1]
            vol_forecast = np.sqrt(variance_forecast) / self.scale

            # FIX: conditional_volatility ya es un desvío estándar (no una
            # varianza) y viene como ndarray -- antes se hacía
            # np.sqrt(...values), que tiraba AttributeError en TODOS los
            # forecasts (y aunque no tirara, comparaba std contra sqrt(std)).
            hist_vols = np.asarray(result.conditional_volatility, dtype=float) / self.scale
            hist_vols = hist_vols[~np.isnan(hist_vols)]
            vol_mean = np.mean(hist_vols[-100:]) if len(hist_vols) > 100 else np.mean(hist_vols)
            vol_std = np.std(hist_vols[-100:]) if len(hist_vols) > 100 else np.std(hist_vols)
            
            # Z-score
            vol_zscore = (vol_forecast - vol_mean) / (vol_std + 1e-6)
            
            # Signal
            if vol_zscore > 1.5:
                vol_signal = 'HIGH'
                hedging_intensity = 0.8
            elif vol_zscore < -1.5:
                vol_signal = 'LOW'
                hedging_intensity = 0.2
            else:
                vol_signal = 'MID'
                hedging_intensity = 0.5
            
            self.vol_history.append(vol_forecast)
            
            return {
                'vol_forecast': float(vol_forecast),
                'vol_signal': vol_signal,
                'vol_zscore': float(vol_zscore),
                'hedging_intensity': hedging_intensity,
                'vol_mean': float(vol_mean)
            }
        
        except Exception as e:
            print(f"⚠️  GARCH forecast failed: {e}")
            return {
                'vol_forecast': np.nan,
                'vol_signal': 'MID',
                'vol_zscore': 0.0,
                'hedging_intensity': 0.0
            }
    
    def get_volatility_regime(self) -> str:
        """Detecta régimen de volatilidad"""
        if not self.vol_history or len(self.vol_history) < 10:
            return 'STABLE'
        
        recent_vol = np.mean(self.vol_history[-10:])
        historical_vol = np.mean(self.vol_history[-100:]) if len(self.vol_history) >= 100 else recent_vol
        
        ratio = recent_vol / (historical_vol + 1e-6)
        
        if ratio > 1.5:
            return 'CRISIS'  # Vol spike (2022 style)
        elif ratio > 1.2:
            return 'HIGH'
        elif ratio < 0.8:
            return 'LOW'
        else:
            return 'STABLE'
    
    def get_parameters(self) -> Dict:
        """Retorna parámetros del modelo ajustado"""
        if self.fitted_model is None:
            return {}
        
        try:
            params = self.fitted_model.params.to_dict()
            return {
                'model_type': self.model_type,
                'parameters': params,
                'aic': float(self.fitted_model.aic),
                'bic': float(self.fitted_model.bic),
                'loglikelihood': float(self.fitted_model.loglikelihood)
            }
        except:
            return {}


class MultiWindowGARCH:
    """
    GARCH en múltiples ventanas temporales simultáneamente.
    Perfecto para crypto: análisis corto/medio/largo plazo.
    """
    
    def __init__(self, window_sizes: list = None):
        """
        window_sizes: [10, 30, 100, 1000] para cripto
        """
        self.window_sizes = window_sizes or [10, 30, 100, 1000]
        self.models = {ws: GARCHVolatilityExpert('egarch') for ws in self.window_sizes}
        self.forecasts = {}
    
    def fit_all(self, returns: np.ndarray) -> None:
        """Entrena GARCH en todas las ventanas"""
        self.fitted_windows = []
        for ws in self.window_sizes:
            if len(returns) >= ws + 20:
                window_returns = returns[-ws:] if len(returns) > ws else returns
                self.models[ws].fit(window_returns)
                if self.models[ws].fitted_model is not None:
                    self.fitted_windows.append(ws)
    
    def forecast_all(self, recent_returns: Optional[np.ndarray] = None) -> Dict:
        """
        Forecast de volatilidad en todas las ventanas. Solo se reportan las
        ventanas que efectivamente fitearon -- antes las no fiteadas (ej.
        10/30, por debajo del mínimo de 50 datos) entraban al promedio con
        valores por defecto y diluían la señal.
        """
        results = {}
        for ws in getattr(self, 'fitted_windows', self.window_sizes):
            rr = recent_returns[-ws:] if recent_returns is not None else None
            results[ws] = self.models[ws].forecast_volatility(horizon=1, recent_returns=rr)
        return results
    
    def get_hedging_signal(self, recent_returns: Optional[np.ndarray] = None) -> Dict:
        """
        Señal de hedging agregada.
        Si vol está alta en muchas ventanas → reduce exposición
        """
        all_forecasts = self.forecast_all(recent_returns)
        if not all_forecasts:
            return {'crisis_detected': False, 'high_signal_count': 0, 'avg_vol_zscore': 0.0,
                    'recommended_hedge_intensity': 0.5, 'hedging_intensity': 0.5,
                    'vol_forecast': np.nan, 'window_signals': {}}
        
        # Contar HIGH signals
        high_count = sum(1 for f in all_forecasts.values() if f['vol_signal'] == 'HIGH')
        crisis_detected = high_count >= max(1, (len(all_forecasts) + 1) // 2)  # mayoría de ventanas en HIGH
        
        avg_zscore = np.mean([f['vol_zscore'] for f in all_forecasts.values()])
        avg_intensity = np.mean([f['hedging_intensity'] for f in all_forecasts.values()])
        vols = [f['vol_forecast'] for f in all_forecasts.values() if np.isfinite(f['vol_forecast'])]
        
        # 'hedging_intensity' y 'vol_forecast' son las claves que lee
        # PortfolioAgent -- antes no existían en este dict, así que el
        # agente siempre caía a sus defaults.
        return {
            'crisis_detected': crisis_detected,
            'high_signal_count': high_count,
            'avg_vol_zscore': float(avg_zscore),
            'recommended_hedge_intensity': float(avg_intensity),
            'hedging_intensity': float(avg_intensity),
            'vol_forecast': float(np.mean(vols)) if vols else np.nan,
            'window_signals': {ws: forecasts['vol_signal'] 
                             for ws, forecasts in all_forecasts.items()}
        }
    


