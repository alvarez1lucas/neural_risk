# neural_risk/models/lstm_transformer.py
"""
LSTM/Transformer para Forecasting Secuencial
Captura patterns largos en cripto (2021 bull runs, 2022 bear)

ESTADO: Parcheado (ver comentario FIX en train_step).
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class LSTMForecastingExpert(nn.Module):
    """
    LSTM para forecasting secuencial de retornos/precios.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, 
                 num_layers: int = 2, dropout: float = 0.2,
                 output_size: int = 1, forecast_horizon: int = 1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.forecast_horizon = forecast_horizon
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            mu, sigma: [batch, output_size]
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # [batch, hidden_size]
        
        x = self.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        
        mu = self.fc2(x)
        sigma = torch.nn.functional.softplus(self.fc2(x) * 0.1 + 1.0)
        
        return mu, sigma


class TransformerForecastingExpert(nn.Module):
    """
    Transformer para forecasting secuencial.
    Mejor que LSTM para capturar dependencias lejanas.
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.2, output_size: int = 1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Linear(input_size, hidden_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, seq_len, input_size]
        Returns:
            mu, sigma: [batch, output_size]
        """
        x = self.embedding(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # [batch, hidden_size]
        
        x = self.relu(self.fc1(x))
        x = self.dropout_layer(x)
        
        mu = self.fc2(x)
        sigma = torch.nn.functional.softplus(self.fc2(x) * 0.1 + 1.0)
        
        return mu, sigma


class SequentialForecastingEnsemble:
    """
    Ensemble LSTM + Transformer con stacking.
    Combina fortalezas de ambos.
    """
    
    def __init__(self, input_size: int, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = input_size
        
        self.lstm = LSTMForecastingExpert(
            input_size=input_size, hidden_size=64, num_layers=2, dropout=0.2
        ).to(self.device)
        
        self.transformer = TransformerForecastingExpert(
            input_size=input_size, hidden_size=64, num_heads=4,
            num_layers=2, dropout=0.2
        ).to(self.device)
        
        self.lstm_optimizer = torch.optim.Adam(self.lstm.parameters(), lr=1e-3)
        self.tf_optimizer = torch.optim.Adam(self.transformer.parameters(), lr=1e-3)
        
        self.criterion = nn.GaussianNLLLoss()
        
        self.lstm_weight = 0.5
        self.tf_weight = 0.5
        
    def train_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor):
        """
        Entrena ambos modelos un paso.
        
        FIX: nn.GaussianNLLLoss espera argumentos posicionales
        (input=media, target=valor real, var=varianza). La versión
        anterior llamaba self.criterion(mu, sigma, y_batch) -- es decir,
        usaba la incertidumbre PREDICHA por el propio modelo (sigma) como
        si fuera el target real, y el retorno real (y_batch) como si
        fuera la varianza. Como los retornos pueden ser negativos, y
        GaussianNLLLoss calcula log(var), esto producía log de un número
        negativo -> NaN en el loss, silenciosamente (PyTorch no crashea
        con NaN, el modelo simplemente nunca aprendía nada útil).
        
        Comparar con ensemble_trainer.py, que en la misma base de código
        hace la llamada análoga en el orden correcto:
        self.criterion(mu.squeeze(), y_batch, sigma.squeeze())
        """
        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)
        
        # LSTM
        self.lstm_optimizer.zero_grad()
        lstm_mu, lstm_sigma = self.lstm(X_batch)
        lstm_loss = self.criterion(lstm_mu.squeeze(), y_batch, lstm_sigma.squeeze())
        lstm_loss.backward()
        self.lstm_optimizer.step()
        
        # Transformer
        self.tf_optimizer.zero_grad()
        tf_mu, tf_sigma = self.transformer(X_batch)
        tf_loss = self.criterion(tf_mu.squeeze(), y_batch, tf_sigma.squeeze())
        tf_loss.backward()
        self.tf_optimizer.step()
        
        return float(lstm_loss), float(tf_loss)
    
    def predict_ensemble(self, X: torch.Tensor) -> Dict:
        """
        Predice con ambos modelos y blendea.
        """
        X = X.to(self.device)
        
        with torch.no_grad():
            lstm_mu, lstm_sigma = self.lstm(X)
            tf_mu, tf_sigma = self.transformer(X)
        
        lstm_conf = 1.0 / (lstm_sigma.mean() + 1e-6)
        tf_conf = 1.0 / (tf_sigma.mean() + 1e-6)
        
        total_conf = lstm_conf + tf_conf
        lstm_w = lstm_conf / total_conf
        tf_w = tf_conf / total_conf
        
        ensemble_mu = lstm_w * lstm_mu + tf_w * tf_mu
        ensemble_sigma = np.sqrt(
            (lstm_w ** 2) * (lstm_sigma ** 2) + 
            (tf_w ** 2) * (tf_sigma ** 2)
        )
        
        return {
            'ensemble_forecast': float(ensemble_mu.squeeze().cpu().numpy()),
            'lstm_forecast': float(lstm_mu.squeeze().cpu().numpy()),
            'transformer_forecast': float(tf_mu.squeeze().cpu().numpy()),
            'lstm_sigma': float(lstm_sigma.squeeze().cpu().numpy()),
            'tf_sigma': float(tf_sigma.squeeze().cpu().numpy()),
            'uncertainty': float(ensemble_sigma.squeeze().cpu().numpy()),
            'lstm_confidence': float(lstm_w.cpu().numpy()),
            'transformer_confidence': float(tf_w.cpu().numpy()),
            'model_agreement': float(1.0 - abs((lstm_mu - tf_mu) / (abs(lstm_mu) + 1e-6)).mean().cpu().numpy())
        }
    
    def update_weights_from_backtest(self, lstm_mae: float, tf_mae: float):
        """
        Actualiza pesos dinámicamente basado en backtest.
        Si LSTM tiene MAE mejor → sube peso.
        """
        total_error = lstm_mae + tf_mae
        self.lstm_weight = (1 - lstm_mae / total_error) if total_error > 0 else 0.5
        self.tf_weight = (1 - tf_mae / total_error) if total_error > 0 else 0.5
        
        print(f"   📊 LSTM weight: {self.lstm_weight:.3f}, TF weight: {self.tf_weight:.3f}")