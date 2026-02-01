# neural_risk/models/anomaly_detection.py
"""
Isolation Forest & Autoencoders para Anomaly Detection
Detecta outliers en cripto (rug pulls, flash crashes)
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
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
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
    Usa Isolation Forest + Autoencoder.
    
    Pros:
    - Detecta rug pulls, flash crashes
    - Safety layer antes de signals
    - Robusto a outliers extremos
    
    Cons:
    - Falsos positivos en vol normal
    - No genera estrategias puras
    
    Para Crypto:
    - Post-jury: si anomaly detectada → signal=0
    - Dinámico: agente ignora signals temporalmente
    - Ventanas de 10/30/100 períodos
    """
    
    def __init__(self, contamination: float = 0.05, device=None):
        """
        contamination: % de anomalías esperadas
        """
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
        """Entrena Isolation Forest"""
        X_clean = np.array(X).reshape(-1, 1) if X.ndim == 1 else np.array(X)
        X_clean = np.nan_to_num(X_clean, nan=0)
        
        self.isolation_forest.fit(X_clean)
    
    def fit_autoencoder(self, X: pd.DataFrame, epochs: int = 20, 
                        batch_size: int = 32, lr: float = 1e-3) -> None:
        """Entrena Autoencoder"""
        X_clean = X.fillna(0).values
        X_scaled = self.scaler.fit_transform(X_clean)
        
        self.autoencoder = AnomalyDetectionAutoencoder(
            input_dim=X_scaled.shape[1],
            encoding_dim=max(8, X_scaled.shape[1] // 2)
        ).to(self.device)
        
        self.ae_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        # Training loop
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        for epoch in range(epochs):
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                # Forward
                reconstructed = self.autoencoder(batch)
                loss = criterion(reconstructed, batch)
                
                # Backward
                self.ae_optimizer.zero_grad()
                loss.backward()
                self.ae_optimizer.step()
        
        # Calcula reconstruction error threshold (95 percentil)
        with torch.no_grad():
            X_reconstructed = self.autoencoder(X_tensor)
            errors = torch.mean((X_tensor - X_reconstructed) ** 2, dim=1)
        
        self.reconstruction_threshold = np.percentile(
            errors.cpu().numpy(),
            95
        )
    
    def predict_anomalies(self, X: pd.DataFrame) -> Dict:
        """
        Detecta anomalías usando ambos métodos.
        
        Returns:
            {
                'anomaly_detected': bool,
                'isolation_forest_score': float,
                'autoencoder_score': float,
                'combined_score': float,
                'anomaly_type': str,
                'confidence': float,
                'recommendation': str
            }
        """
        X_clean = X.fillna(0).values
        
        # 1. Isolation Forest
        if len(X_clean.shape) == 1:
            X_clean = X_clean.reshape(-1, 1)
        
        if_pred = self.isolation_forest.predict(X_clean)
        if_scores = self.isolation_forest.score_samples(X_clean)
        
        # -1 = anomaly, 1 = normal
        if_anomaly = if_pred[-1] == -1
        if_score = float(if_scores[-1]) if len(if_scores) > 0 else 0
        
        # 2. Autoencoder
        ae_anomaly = False
        ae_score = 0.0
        
        if self.autoencoder is not None and self.reconstruction_threshold is not None:
            try:
                X_scaled = self.scaler.transform(X_clean)
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                
                with torch.no_grad():
                    X_reconstructed = self.autoencoder(X_tensor)
                    reconstruction_error = torch.mean(
                        (X_tensor - X_reconstructed) ** 2,
                        dim=1
                    )
                
                ae_score = float(reconstruction_error[-1].cpu().numpy())
                ae_anomaly = ae_score > self.reconstruction_threshold
            except Exception as e:
                print(f"⚠️  Autoencoder prediction failed: {e}")
        
        # 3. Combinado
        combined_anomaly = if_anomaly or ae_anomaly
        
        # Confidence: si ambos métodos acuerdan
        if if_anomaly and ae_anomaly:
            confidence = 0.95
        elif if_anomaly or ae_anomaly:
            confidence = 0.70
        else:
            confidence = 0.05
        
        # Detectar tipo de anomalía
        anomaly_type = self._classify_anomaly(X_clean[-1])
        
        # Recomendación
        if combined_anomaly and confidence > 0.7:
            recommendation = 'IGNORE_SIGNALS_IMMEDIATELY'
        elif combined_anomaly:
            recommendation = 'REDUCE_POSITION_SIZE'
        else:
            recommendation = 'NORMAL_OPERATIONS'
        
        # Log
        self.anomaly_history.append({
            'timestamp': pd.Timestamp.now(),
            'detected': combined_anomaly,
            'confidence': confidence,
            'type': anomaly_type
        })
        self.scores_history.append({
            'if_score': if_score,
            'ae_score': ae_score
        })
        
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
        """Clasifica tipo de anomalía detectada"""
        if len(data_point) == 0:
            return 'UNKNOWN'
        
        # Heurística simple
        last_val = data_point[-1] if isinstance(data_point, np.ndarray) else data_point
        
        if np.isnan(last_val) or np.isinf(last_val):
            return 'DATA_ERROR'
        elif abs(last_val) > 3.0:  # 3-sigma
            return 'EXTREME_MOVEMENT'
        elif last_val < -0.2:  # Caída fuerte
            return 'FLASH_CRASH'
        elif last_val > 0.2:  # Spike fuerte
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
        
        # Trend: ¿más anomalías últimamente?
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
    """
    Umbral de anomalía dinámico que se adapta a cambios de régimen.
    """
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.0):
        """
        window_size: datos para calcular baseline
        sensitivity: cuántos sigma para considerar anomalía
        """
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        
    def update(self, X: np.ndarray) -> Tuple[float, float, float]:
        """
        Actualiza baseline dinámicamente.
        
        Returns:
            threshold, baseline_mean, baseline_std
        """
        X_clean = np.array(X).flatten()
        X_clean = X_clean[~np.isnan(X_clean)]
        
        if len(X_clean) < self.window_size:
            window = X_clean
        else:
            window = X_clean[-self.window_size:]
        
        self.baseline_mean = np.mean(window)
        self.baseline_std = np.std(window)
        
        threshold = self.baseline_mean + self.sensitivity * self.baseline_std
        
        return threshold, self.baseline_mean, self.baseline_std
    
    def is_anomaly(self, value: float, threshold: Optional[float] = None) -> bool:
        """Verifica si valor es anomalía"""
        if threshold is None:
            threshold = self.baseline_mean + self.sensitivity * self.baseline_std
        
        return abs(value - self.baseline_mean) > threshold
