import torch
import torch.nn as nn
from neural_risk.models.layers import VariableSelectionNetwork, GatedResidualNetwork

class NeuralRiskModel(nn.Module):
    def __init__(self, num_features, hidden_size=64, num_heads=4, dropout=0.1):
        super().__init__()
        self.vsn = VariableSelectionNetwork(
            num_inputs=num_features, input_dim=1, hidden_size=hidden_size, dropout=dropout
        )
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, num_layers=2)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.mu_head = nn.Linear(hidden_size, 1)
        self.sigma_head = nn.Sequential(nn.Linear(hidden_size, 1), nn.Softplus())

    def forward(self, x):
        embeddings, feature_weights = self.vsn(x)
        lstm_out, _ = self.lstm(embeddings)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        last_step = attn_out[:, -1, :]
        mu = self.mu_head(last_step)
        sigma = self.sigma_head(last_step)
        return mu, sigma, feature_weights 
    