# neural_risk/models/bayesian_model.py
# NOTA: predict_with_uncertainty (MC-Dropout) esta definido pero NUNCA
# se llama desde engine.py -- el entrenamiento usa forward() estandar
# via EnsembleTrainer. Capacidad "bayesiana" orfana, no rota, sin uso.
from neural_risk.models.risk_model import NeuralRiskModel
import torch

class BayesianNeuralRisk(NeuralRiskModel):
    def predict_with_uncertainty(self, x, n_iter=50):
        self.train()
        mu_list = []
        with torch.no_grad():
            for _ in range(n_iter):
                mu, _, _ = self.forward(x)
                mu_list.append(mu)
        mu_stack = torch.stack(mu_list)
        return mu_stack.mean(dim=0), mu_stack.std(dim=0)