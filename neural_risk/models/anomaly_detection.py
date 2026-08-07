# neural_risk/models/anomaly_detection.py
"""
Isolation Forest & Autoencoders para Anomaly Detection
Detecta outliers en cripto (rug pulls, flash crashes)

ESTADO: Parcheado (ver comentario FIX en predict_anomalies).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')


class AnomalyDetectionAutoencoder(nn.Module):
    """
    Autoencoder para detección de anomalías.
    Aprende distribución normal, desviaciones = anomalías.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class AnomalyDetector:
    """
    Expert en Anomaly Detection.
    Usa Isolation Forest (sobre retornos, univariado) + Autoencoder
    (sobre features completas, multivariado). Son dos detectores
    COMPLEMENTARIOS que miran cosas distintas -- por eso fit/predict
    de cada uno deben recibir consistentemente el mismo tipo de input
    que se usó para entrenarlos (ver FIX en predict_anomalies).
    """
    
    def __init__(self, contamination: float = 0.05, device=None):
        self.contamination = contamination
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self.autoencoder = None
        self.scaler = StandardScaler()
        self.ae_optimizer = None
        self.reconstruction_threshold = None
        
        self.anomaly_history = []
        self.scores_history = []
        
    def fit_isolation_forest(self, X: np.ndarray) -> None:
        """
        Entrena Isolation Forest.
        
        Se fitea SIEMPRE sobre datos univariados (retornos): X entra
        como array 1D y se reshapea a [-1, 1]. predict_anomalies() debe
        respetar esa misma forma al predecir (ver FIX ahí).
        """
        X_clean = np.array(X).reshape(-1, 1) if X.ndim == 1 else np.array(X)
        X_clean = np.nan_to_num(X_clean, nan=0)
        
        self.isolation_forest.fit(X_clean)
    
    def fit_autoencoder(self, X: pd.DataFrame, epochs: int = 20, 
                        batch_size: int = 32, lr: float = 1e-3) -> None:
        """Entrena Autoencoder sobre el set completo de features (multivariado)."""
        X_clean = X.fillna(0).values
        X_scaled = self.scaler.fit_transform(X_clean)
        
        self.autoencoder = AnomalyDetectionAutoencoder(
            input_dim=X_scaled.shape[1],
            encoding_dim=max(8, X_scaled.shape[1] // 2)
        ).to(self.device)
        
        self.ae_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                reconstructed = self.autoencoder(batch)
                loss = criterion(reconstructed, batch)
                
                self.ae_optimizer.zero_grad()
                loss.backward()
                self.ae_optimizer.step()
        
        with torch.no_grad():
            X_reconstructed = self.autoencoder(X_tensor)
            errors = torch.mean((X_tensor - X_reconstructed) ** 2, dim=1)
        
        self.reconstruction_threshold = np.percentile(errors.cpu().numpy(), 95)
    
    def predict_anomalies(self, X: pd.DataFrame, 
                         recent_returns: Optional[np.ndarray] = None) -> Dict:
        """
        Detecta anomalías usando ambos métodos.
        
        FIX: antes este método recibía UN solo argumento (X, el
        DataFrame multi-feature con todas las columnas que aprobó el
        jurado) y se lo pasaba directo a isolation_forest.predict(X).
        Pero fit_isolation_forest() se entrena con 'returns' -- un array
        1D reshapeado a 1 SOLA columna. Mismatch de dimensiones
        (ej. modelo entrenado con 1 feature, se le pide predecir con 15)
        -> sklearn tira ValueError EN TODOS LOS CICLOS, atrapado
        silenciosamente por el try/except de engine.py, dejando
        anomaly_detected=False siempre -- el "safety layer" nunca
        disparaba, sin que nada lo reportara como roto.
        
        Ahora el método recibe explícitamente los dos inputs que cada
        submodelo necesita, en el MISMO espacio en que cada uno se
        entrenó:
        - recent_returns: univariado, para Isolation Forest.
        - X: multivariado (features completas), para el Autoencoder.
        
        Args:
            X: DataFrame de features (mismo que se usó en fit_autoencoder)
            recent_returns: array de retornos recientes (mismo que se
                usó en fit_isolation_forest). Si no se pasa, cae a un
                fallback defensivo (ver abajo) que NO es tan confiable
                como pasarlo explícitamente -- se recomienda siempre
                pasar recent_returns.
        """
        if recent_returns is None:
            # Fallback defensivo: si no se pasa el array de retornos,
            # se aproxima con la primera columna numérica de X. Esto NO
            # es lo ideal (puede no ser semánticamente "retornos"), pero
            # evita un crash duro si algún caller viejo todavía no pasa
            # el segundo argumento.
            recent_returns = X.select_dtypes(include=[np.number]).iloc[:, 0].values
        
        returns_clean = np.array(recent_returns).flatten().reshape(-1, 1)
        returns_clean = np.nan_to_num(returns_clean, nan=0.0)
        
        # 1. Isolation Forest (univariado, sobre retornos)
        if_pred = self.isolation_forest.predict(returns_clean)
        if_scores = self.isolation_forest.score_samples(returns_clean)
        
        if_anomaly = bool(if_pred[-1] == -1)
        if_score = float(if_scores[-1]) if len(if_scores) > 0 else 0.0
        
        # 2. Autoencoder (multivariado, sobre features completas)
        X_clean = X.fillna(0).values
        ae_anomaly = False
        ae_score = 0.0
        
        if self.autoencoder is not None and self.reconstruction_threshold is not None:
            try:
                X_scaled = self.scaler.transform(X_clean)
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                
                with torch.no_grad():
                    X_reconstructed = self.autoencoder(X_tensor)
                    reconstruction_error = torch.mean(
                        (X_tensor - X_reconstructed) ** 2, dim=1
                    )
                
                ae_score = float(reconstruction_error[-1].cpu().numpy())
                ae_anomaly = ae_score > self.reconstruction_threshold
            except Exception as e:
                print(f"⚠️  Autoencoder prediction failed: {e}")
        
        # 3. Combinado
        combined_anomaly = if_anomaly or ae_anomaly
        
        if if_anomaly and ae_anomaly:
            confidence = 0.95
        elif if_anomaly or ae_anomaly:
            confidence = 0.70
        else:
            confidence = 0.05
        
        # FIX: antes se clasificaba con X_clean[-1] (el último valor de
        # una feature arbitraria del DataFrame, no necesariamente un
        # retorno). Los thresholds de _classify_anomaly (3-sigma, -0.2,
        # 0.2) están pensados para retornos, así que ahora se clasifica
        # con el propio array de retornos.
        anomaly_type = self._classify_anomaly(returns_clean.flatten())
        
        if combined_anomaly and confidence > 0.7:
            recommendation = 'IGNORE_SIGNALS_IMMEDIATELY'
        elif combined_anomaly:
            recommendation = 'REDUCE_POSITION_SIZE'
        else:
            recommendation = 'NORMAL_OPERATIONS'
        
        self.anomaly_history.append({
            'timestamp': pd.Timestamp.now(),
            'detected': combined_anomaly,
            'confidence': confidence,
            'type': anomaly_type
        })
        self.scores_history.append({'if_score': if_score, 'ae_score': ae_score})
        
        return {
            'anomaly_detected': combined_anomaly,
            'isolation_forest_anomaly': if_anomaly,
            'autoencoder_anomaly': ae_anomaly,
            'isolation_forest_score': if_score,
            'autoencoder_score': ae_score,
            'combined_score': float((if_score + ae_score) / 2),
            'anomaly_type': anomaly_type,
            'confidence': confidence,
            'recommendation': recommendation
        }
    
    def _classify_anomaly(self, data_point: np.ndarray) -> str:
        """Clasifica tipo de anomalía detectada (espera valores tipo retorno)."""
        if len(data_point) == 0:
            return 'UNKNOWN'
        
        last_val = data_point[-1]
        
        if np.isnan(last_val) or np.isinf(last_val):
            return 'DATA_ERROR'
        elif abs(last_val) > 3.0:
            return 'EXTREME_MOVEMENT'
        elif last_val < -0.2:
            return 'FLASH_CRASH'
        elif last_val > 0.2:
            return 'FLASH_SURGE'
        else:
            return 'MICRO_ANOMALY'
    
    def get_anomaly_summary(self) -> Dict:
        """Resumen de anomalías detectadas"""
        if not self.anomaly_history:
            return {
                'total_anomalies': 0,
                'anomaly_rate': 0.0,
                'anomaly_types': {},
                'recent_trend': 'STABLE'
            }
        
        total = len(self.anomaly_history)
        detected = sum(1 for a in self.anomaly_history if a['detected'])
        
        anomaly_types = {}
        for a in self.anomaly_history:
            atype = a['type']
            anomaly_types[atype] = anomaly_types.get(atype, 0) + 1
        
        recent = self.anomaly_history[-20:] if len(self.anomaly_history) > 20 else self.anomaly_history
        recent_rate = sum(1 for a in recent if a['detected']) / len(recent)
        
        trend = 'INCREASING' if recent_rate > 0.3 else 'DECREASING' if recent_rate < 0.05 else 'STABLE'
        
        return {
            'total_anomalies': detected,
            'anomaly_rate': float(detected / total),
            'anomaly_types': anomaly_types,
            'recent_trend': trend,
            'recommendation': 'INCREASE_MONITORING' if trend == 'INCREASING' else 'NORMAL'
        }


class DynamicAnomalyThreshold:
    """Umbral de anomalía dinámico que se adapta a cambios de régimen."""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        
    def update(self, X: np.ndarray) -> Tuple[float, float, float]:
        X_clean = np.array(X).flatten()
        X_clean = X_clean[~np.isnan(X_clean)]
        
        window = X_clean if len(X_clean) < self.window_size else X_clean[-self.window_size:]
        
        self.baseline_mean = np.mean(window)
        self.baseline_std = np.std(window)
        
        threshold = self.baseline_mean + self.sensitivity * self.baseline_std
        
        return threshold, self.baseline_mean, self.baseline_std
    
    def is_anomaly(self, value: float, threshold: Optional[float] = None) -> bool:
        if threshold is None:
            threshold = self.baseline_mean + self.sensitivity * self.baseline_std
        return abs(value - self.baseline_mean) > threshold