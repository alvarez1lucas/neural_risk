# neural_risk/models/bayesian_model.py
from .risk_model import NeuralRiskModel
import torch

class BayesianNeuralRisk(NeuralRiskModel):
    """
    Variante Bayesiana que usa Dropout en tiempo de inferencia 
    para medir la incertidumbre epistémica (qué tanto 'no sabe' el modelo).
    """
    def predict_with_uncertainty(self, x, n_iter=50):
        self.train() # Activamos Dropout para inferencia (Técnica MC Dropout)
        mu_list = []
        
        with torch.no_grad():
            for _ in range(n_iter):
                mu, _, _ = self.forward(x) #
                mu_list.append(mu)
        
        mu_stack = torch.stack(mu_list)
        return mu_stack.mean(dim=0), mu_stack.std(dim=0) # Media y 'Duda' del modelo