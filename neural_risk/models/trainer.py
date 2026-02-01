import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from torch.utils.data import TensorDataset
#trainer.py
class RiskTrainer:
    def __init__(self, model, lr=1e-3, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        self.criterion = nn.GaussianNLLLoss()
        self.scaler = RobustScaler() # Cambio a Robust para manejar e+09

    def create_sequences(self, df, target, window_size=10):
        """Convierte DF plano en ventanas temporales [Batch, Time, Features]"""
        X, y = [], []
        for i in range(len(df) - window_size):
            X.append(df.iloc[i : i + window_size].values)
            y.append(target.iloc[i + window_size])
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y))

    def prepare_data(self, df_features, target, train_split=0.8):
        """Prepara el pipeline de datos de punta a punta."""
        # 1. Limpieza de columnas no numéricas (como la 'date' con NaNs que mencionaste)
        df_numeric = df_features.select_dtypes(include=[np.number])
        
        # 2. Split cronológico (No aleatorio, para trading es vital)
        split_idx = int(len(df_numeric) * train_split)
        train_df = df_numeric.iloc[:split_idx]
        test_df = df_numeric.iloc[split_idx:]
        
        # 3. Escalamiento robusto (Ajustamos solo con train)
        scaled_train = self.scaler.fit_transform(train_df)
        scaled_test = self.scaler.transform(test_df) # Usamos los parámetros de train
        
        # 4. Crear secuencias para la LSTM/Attention
        X_train, y_train = self.create_sequences(pd.DataFrame(scaled_train), target.iloc[:split_idx])
        X_test, y_test = self.create_sequences(pd.DataFrame(scaled_test), target.iloc[split_idx:])
        
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32, shuffle=False)
        
        return train_loader, test_loader