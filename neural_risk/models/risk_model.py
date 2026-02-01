import torch
import torch.nn as nn
from .layers import VariableSelectionNetwork, GatedResidualNetwork
#risk_model.py
class NeuralRiskModel(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_heads=4, dropout=0.1):
        super().__init__()
        
        # 1. Capa de Selección de Variables (Tu "Feature Selector" dinámico)
        # Esto procesa las 60+ features y decide cuáles importan en cada momento.
        self.vsn = VariableSelectionNetwork(
            num_inputs=num_features, 
            input_dim=1, # Cada feature entra como un escalar
            hidden_size=hidden_size, 
            dropout=dropout
        )
        
        # 2. Encoder Temporal (LSTM o Transformer Layer)
        # Captura la memoria de corto y mediano plazo.
        self.lstm = nn.LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            batch_first=True, 
            num_layers=2
        )
        
        # 3. Capa de Atención (Multi-Head)
        # Permite que el modelo "mire hacia atrás" a eventos de riesgo pasados.
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
        # 4. Output Heads (Probabilísticos)
        # Predice la Media (mu) y la Volatilidad de la predicción (sigma)
        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softplus() # Asegura que la volatilidad sea siempre positiva
        )

    def forward(self, x):
        """
        x: [Batch, Time_Window, Features]
        """
        # --- Fase 1: Selección Dinámica ---
        # embeddings shape: [Batch, Time, Hidden_Size]
        # weights: Importancia de cada feature (lo que querías monitorear)
        embeddings, feature_weights = self.vsn(x)
        
        # --- Fase 2: Memoria Temporal ---
        lstm_out, _ = self.lstm(embeddings)
        
        # --- Fase 3: Contexto Histórico (Attention) ---
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Solo tomamos el último timestamp para la predicción
        last_step = attn_out[:, -1, :]
        
        # --- Fase 4: Predicción de Riesgo ---
        mu = self.mu_head(last_step)
        sigma = self.sigma_head(last_step)
        
        return mu, sigma, feature_weights