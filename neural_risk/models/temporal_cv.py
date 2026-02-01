# neural_risk/models/temporal_cv.py
"""
Temporal Cross-Validation: Walk-forward validation para datos financieros.
CRÍTICO: Evita data leakage en backtesting.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')


class TemporalCrossValidator:
    """
    Implementa walk-forward validation.
    Cada fold: entrena en pasado, valida en futuro inmediato.
    """
    
    def __init__(self, n_splits=5, initial_window=100, test_size=20):
        """
        n_splits: número de folds
        initial_window: mínimo de datos para primer entrenamiento
        test_size: tamaño de cada ventana de validación
        """
        self.n_splits = n_splits
        self.initial_window = initial_window
        self.test_size = test_size
        self.splits = []
    
    def split(self, X: pd.DataFrame, y: pd.Series = None) -> List[Tuple]:
        """
        Genera índices de train/test para walk-forward validation.
        
        Returns: Lista de (train_idx, test_idx) tuples
        """
        n_samples = len(X)
        step = (n_samples - self.initial_window - self.test_size) // (self.n_splits - 1) \
               if self.n_splits > 1 else n_samples - self.initial_window - self.test_size
        
        splits = []
        
        for i in range(self.n_splits):
            # Ventana de entrenamiento: crece gradualmente
            train_start = 0
            train_end = self.initial_window + i * step
            
            # Ventana de test: se desliza hacia adelante
            test_start = train_end
            test_end = min(train_end + self.test_size, n_samples)
            
            if test_end - test_start < self.test_size // 2:
                # Skip si el test es muy pequeño
                continue
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            splits.append((train_idx, test_idx))
        
        self.splits = splits
        return splits
    
    def get_splits_info(self) -> pd.DataFrame:
        """Información visual de los splits"""
        info = []
        for i, (train_idx, test_idx) in enumerate(self.splits):
            info.append({
                'fold': i + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'train_end': train_idx[-1] if len(train_idx) > 0 else 0,
                'test_start': test_idx[0] if len(test_idx) > 0 else 0,
                'test_end': test_idx[-1] if len(test_idx) > 0 else 0
            })
        
        return pd.DataFrame(info)
    
    def validate(self, train_fn, score_fn, X, y):
        """
        Ejecuta validación completa con walk-forward.
        
        train_fn: función que entrena (X_train, y_train) -> model
        score_fn: función que puntúa (model, X_test, y_test) -> score
        
        Returns: {
            'fold_scores': [scores],
            'mean_score': float,
            'std_score': float,
            'predictions': array,
            'actuals': array
        }
        """
        fold_scores = []
        predictions = np.array([])
        actuals = np.array([])
        
        for fold_idx, (train_idx, test_idx) in enumerate(self.splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            
            # Entrenar
            model = train_fn(X_train, y_train)
            
            # Puntuar
            score, preds = score_fn(model, X_test, y_test)
            fold_scores.append(score)
            predictions = np.concatenate([predictions, preds])
            actuals = np.concatenate([actuals, y_test.values])
        
        return {
            'fold_scores': fold_scores,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'predictions': predictions,
            'actuals': actuals,
            'n_folds': len(fold_scores)
        }


class WalkForwardValidator:
    """
    Versión simplificada de walk-forward para producción.
    Reentrenamiento periódico sin necesidad de múltiples folds.
    """
    
    def __init__(self, retrain_frequency=20):
        """
        retrain_frequency: reentrenar cada N observaciones nuevas
        """
        self.retrain_frequency = retrain_frequency
        self.retrains = 0
        self.model = None
        self.last_retrain_idx = 0
    
    def should_retrain(self, current_idx: int) -> bool:
        """¿Es hora de reentrenar?"""
        return (current_idx - self.last_retrain_idx) >= self.retrain_frequency
    
    def retrain(self, X_train, y_train, train_fn):
        """Reentrenar modelo con datos hasta ahora"""
        self.model = train_fn(X_train, y_train)
        self.retrains += 1
        self.last_retrain_idx = len(X_train)
    
    def predict(self, X_test):
        """Predicción con modelo actual"""
        if self.model is None:
            raise ValueError("Modelo no entrenado. Llama a retrain() primero.")
        return self.model.predict(X_test)


# ============================================================
# UTILIDADES PARA BACKTESTING REALISTA
# ============================================================

def calculate_realistic_metrics(y_actual, y_pred):
    """
    Calcula métricas realistas en walk-forward.
    """
    errors = y_actual - y_pred
    
    return {
        'mae': np.mean(np.abs(errors)),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mape': np.mean(np.abs(errors / (np.abs(y_actual) + 1e-6))),
        'direction_accuracy': np.mean((np.sign(y_pred) == np.sign(y_actual)) | (y_actual == 0)),
        'max_error': np.max(np.abs(errors)),
        'std_error': np.std(errors)
    }


def compare_cv_methods(X, y, train_fn, score_fn):
    """
    Compara K-fold (INCORRECTO) vs Walk-forward (CORRECTO).
    Demuestra cuánto error tiene K-fold en finanzas.
    """
    from sklearn.model_selection import KFold
    
    results = {}
    
    # K-fold (INCORRECTO para series temporales)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    kf_scores = []
    for train_idx, test_idx in kf.split(X):
        model = train_fn(X.iloc[train_idx], y.iloc[train_idx])
        score, _ = score_fn(model, X.iloc[test_idx], y.iloc[test_idx])
        kf_scores.append(score)
    results['kfold_mean'] = np.mean(kf_scores)
    results['kfold_std'] = np.std(kf_scores)
    
    # Walk-forward (CORRECTO para series temporales)
    wf = TemporalCrossValidator(n_splits=5)
    wf.split(X, y)
    wf_result = wf.validate(train_fn, score_fn, X, y)
    results['walkforward_mean'] = wf_result['mean_score']
    results['walkforward_std'] = wf_result['std_score']
    
    # Diferencia (típicamente -30%)
    results['performance_gap'] = (results['walkforward_mean'] - results['kfold_mean']) / results['kfold_mean']
    
    return results
