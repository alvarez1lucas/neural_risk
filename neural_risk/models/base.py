from abc import ABC, abstractmethod
import pandas as pd
import numpy as np

class RiskModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
        self.metadata = {}

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pass

    def get_model_status(self):
        return {"name": self.model_name, "trained": self.is_trained, "metadata": self.metadata}