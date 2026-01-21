import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class RiskTrainer:
    def __init__(self, model, lr=1e-3, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        
        # Usamos Gaussian NLL Loss: penaliza mala predicción y mala estimación de incertidumbre
        self.criterion = nn.GaussianNLLLoss()
        self.scaler = StandardScaler()

    def prepare_data(self, df):
        """Normaliza las 60+ features para que el Deep Learning converja."""
        # Fit del scaler solo con datos de entrenamiento (evitar data leakage)
        scaled_data = self.scaler.fit_transform(df)
        return pd.DataFrame(scaled_data, columns=df.columns, index=df.index)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # El modelo escupe: Media, Varianza, y los Pesos del VSN
            mu, sigma, feature_weights = self.model(batch_x)
            
            # batch_y debe tener forma [batch, 1]
            loss = self.criterion(mu, batch_y.unsqueeze(1), sigma)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def monitor_vsn_importance(self, dataloader, feature_names):
        """
        Extrae qué está 'pensando' el Feature Selector dinámico.
        Esencial para la '3era derivada' de decisiones.
        """
        self.model.eval()
        all_weights = []
        
        with torch.no_grad():
            for batch_x, _ in dataloader:
                _, _, weights = self.model(batch_x.to(self.device))
                all_weights.append(weights.mean(dim=(0, 1)).cpu().numpy())
        
        # Promediamos la importancia de cada feature en el dataset
        avg_weights = np.mean(all_weights, axis=0)
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': avg_weights.flatten()
        }).sort_values(by='importance', ascending=False)
        
        return importance_df