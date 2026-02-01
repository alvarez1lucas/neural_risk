# neural_risk/models/ensemble_trainer.py
"""
Ensemble Training Module: Combina Neural + XGBoost + Kalman Filter
Para Paso 4 mejorado del engine.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class KalmanFilterRegime:
    """
    Kalman Filter para detección automática de regime shifts.
    Mantiene estado oculto de volatilidad y drift.
    """
    def __init__(self, q=1e-4, r=0.1):
        # q: proceso de ruido (pequeño = suavizado)
        # r: ruido de medición (pequeño = confiable)
        self.q = q
        self.r = r
        self.x = 0  # Estado: volatilidad implícita
        self.p = 1  # Varianza del estado
        self.history = []
    
    def update(self, z):
        """
        z: observación (volatilidad realizada)
        Retorna: estado filtrado y ganancia de Kalman
        """
        # Predicción
        x_pred = self.x
        p_pred = self.p + self.q
        
        # Actualización
        k = p_pred / (p_pred + self.r)  # Ganancia de Kalman
        self.x = x_pred + k * (z - x_pred)
        self.p = (1 - k) * p_pred
        
        self.history.append(self.x)
        return self.x, k
    
    def get_regime(self):
        """Detecta regime: LOW/MID/HIGH vol"""
        if len(self.history) < 2:
            return 'MID'
        recent_vol = np.mean(self.history[-10:])
        if recent_vol < 0.015:
            return 'LOW'
        elif recent_vol > 0.030:
            return 'HIGH'
        else:
            return 'MID'


class EnsembleTrainer:
    """
    Entrena 3 modelos en paralelo con pesos dinámicos:
    - Neural Risk Model (60%)
    - XGBoost (25%)
    - Kalman-adjusted baseline (15%)
    """
    
    def __init__(self, neural_model, lr=1e-3, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.neural_model = neural_model.to(self.device)
        
        # XGBoost model
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.05,
            random_state=42
        )
        
        # Kalman filter para regime detection
        self.kalman = KalmanFilterRegime(q=1e-4, r=0.1)
        
        # Optimizer y loss
        self.optimizer = torch.optim.Adam(
            self.neural_model.parameters(), 
            lr=lr, 
            weight_decay=1e-5
        )
        self.criterion = nn.GaussianNLLLoss()
        self.scaler = RobustScaler()
        
        # Historial de pesos
        self.weight_history = []
    
    def create_sequences(self, df, target, window_size=10):
        """Convierte DF plano en ventanas temporales [Batch, Time, Features]"""
        X, y = [], []
        for i in range(len(df) - window_size):
            X.append(df.iloc[i : i + window_size].values)
            y.append(target.iloc[i + window_size])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))
    
    def prepare_data(self, df_features, target, train_split=0.8, window_size=10):
        """
        Preparación MEJORADA con temporal CV y regime detection
        """
        df_numeric = df_features.select_dtypes(include=[np.number]).fillna(0)
        
        # Split temporal (crucial)
        split_idx = int(len(df_numeric) * train_split)
        train_df = df_numeric.iloc[:split_idx]
        test_df = df_numeric.iloc[split_idx:]
        
        # Escalamiento
        scaled_train = self.scaler.fit_transform(train_df)
        scaled_test = self.scaler.transform(test_df)
        
        # Crear secuencias
        X_train, y_train = self.create_sequences(
            pd.DataFrame(scaled_train), 
            target.iloc[:split_idx],
            window_size
        )
        X_test, y_test = self.create_sequences(
            pd.DataFrame(scaled_test), 
            target.iloc[split_idx:],
            window_size
        )
        
        # DataLoaders
        train_loader = DataLoader(
            TensorDataset(X_train, y_train), 
            batch_size=32, 
            shuffle=False  # CRÍTICO: no shuffle en finanzas
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test), 
            batch_size=32, 
            shuffle=False
        )
        
        # Datos planos para XGBoost
        return {
            'neural': (train_loader, test_loader),
            'xgb_X_train': scaled_train,
            'xgb_y_train': target.iloc[:split_idx].values,
            'xgb_X_test': scaled_test,
            'xgb_y_test': target.iloc[split_idx:].values,
            'X_test_np': scaled_test,
            'y_test': target.iloc[split_idx:].values
        }
    
    def train_ensemble(self, train_loader, test_loader, xgb_data, epochs=50, verbose=True):
        """
        Entrena los 3 modelos con arquitectura de ensemble adaptivo.
        """
        best_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        regime_weights = {'LOW': [0.5, 0.3, 0.2], 'MID': [0.6, 0.25, 0.15], 'HIGH': [0.7, 0.2, 0.1]}
        
        for epoch in range(epochs):
            # ===== FASE 1: Entrenar Neural =====
            train_loss_neural = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                self.optimizer.zero_grad()
                mu, sigma, _ = self.neural_model(X_batch)
                loss = self.criterion(mu.squeeze(), y_batch, sigma.squeeze())
                loss.backward()
                self.optimizer.step()
                train_loss_neural += loss.item()
            
            # ===== FASE 2: Entrenar XGBoost (cada N epochs) =====
            if epoch == 0 or epoch % 10 == 0:
                self.xgb_model.fit(
                    xgb_data['xgb_X_train'], 
                    xgb_data['xgb_y_train'],
                    verbose=False
                )
            
            # ===== FASE 3: Validación Ensemble =====
            val_loss_ensemble = 0
            neural_preds = []
            xgb_preds = []
            
            with torch.no_grad():
                for X_batch, y_batch in test_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    
                    # Neural prediction
                    mu, sigma, _ = self.neural_model(X_batch)
                    neural_loss = self.criterion(mu.squeeze(), y_batch, sigma.squeeze())
                    val_loss_ensemble += neural_loss.item()
                    neural_preds.append(mu.cpu().numpy())
            
            # XGBoost predictions en test
            xgb_pred = self.xgb_model.predict(xgb_data['xgb_X_test'])
            
            # Detectar regime
            returns_realized = np.std(xgb_data['y_test'])
            regime = self.kalman.get_regime()
            weights = regime_weights[regime]
            
            # Guardar pesos
            self.weight_history.append({'epoch': epoch, 'regime': regime, 'weights': weights})
            
            avg_loss = (train_loss_neural / len(train_loader)) if len(train_loader) > 0 else 0
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Regime: {regime} | Weights: {weights}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping en epoch {epoch+1}")
                break
        
        return self._get_ensemble_predictions(
            neural_preds, xgb_pred, weights
        )
    
    def _get_ensemble_predictions(self, neural_preds, xgb_pred, weights):
        """
        Combina predicciones de los 3 modelos con pesos dinámicos.
        """
        neural_mean = np.concatenate(neural_preds).mean()
        xgb_mean = xgb_pred.mean()
        
        # Baseline simple (media histórica)
        baseline = (neural_mean + xgb_mean) / 2
        
        # Ensemble ponderado
        ensemble_pred = (
            weights[0] * neural_mean +
            weights[1] * xgb_mean +
            weights[2] * baseline
        )
        
        # Incertidumbre estimada
        neural_std = np.concatenate(neural_preds).std() if neural_preds else 0.1
        xgb_std = np.std(xgb_pred) if len(xgb_pred) > 1 else 0.1
        ensemble_std = np.sqrt(
            weights[0]**2 * neural_std**2 +
            weights[1]**2 * xgb_std**2 +
            weights[2]**2 * ((neural_std + xgb_std) / 2)**2
        )
        
        return ensemble_pred, ensemble_std
    
    def predict_ensemble(self, X_test, y_test=None):
        """
        Predicción en datos nuevos usando ensemble.
        """
        # Neural
        self.neural_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_test).to(self.device)
            mu, sigma, _ = self.neural_model(X_tensor.unsqueeze(0))
            neural_pred = mu.squeeze().cpu().item()
            neural_std = sigma.squeeze().cpu().item()
        
        # XGBoost
        xgb_pred = self.xgb_model.predict(X_test.reshape(1, -1))[0]
        
        # Kalman update (si tenemos observación real)
        regime = self.kalman.get_regime()
        weights = {'LOW': [0.5, 0.3, 0.2], 'MID': [0.6, 0.25, 0.15], 'HIGH': [0.7, 0.2, 0.1]}[regime]
        
        # Ensemble final
        baseline = (neural_pred + xgb_pred) / 2
        ensemble_pred = (
            weights[0] * neural_pred +
            weights[1] * xgb_pred +
            weights[2] * baseline
        )
        ensemble_std = np.sqrt(
            weights[0]**2 * neural_std**2 +
            weights[1]**2 * (neural_std * 0.8)**2 +
            weights[2]**2 * ((neural_std + neural_std * 0.8) / 2)**2
        )
        
        return ensemble_pred, ensemble_std
