# neural_risk/research/walk_forward.py
"""
Walk-forward genérico: reentrena cada `retrain_every` barras usando SOLO
filas cuyo target ya se conocía (se cortan las últimas `h` barras antes del
inicio de cada bloque de test) y predice el bloque siguiente.

Cualquier modelo con la interfaz de scikit-learn (fit/predict) compite en
las mismas condiciones: mismas features, mismo target, mismos cortes.
"""

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def forward_return(f: pd.DataFrame, h: int) -> pd.Series:
    """Retorno desde el OPEN de la barra siguiente hasta el CLOSE de t+h."""
    close, open_ = f.attrs['close'], f.attrs['open']
    return close.shift(-h) / open_.shift(-1) - 1


def walk_forward(make_model: Callable[[], object], X: pd.DataFrame, y: pd.Series, h: int,
                 retrain_every: int, min_train: int, scale_y: bool = False,
                 clip: float = 5.0, min_obs: int = 500) -> pd.Series:
    """
    make_model: fábrica sin argumentos que devuelve un modelo NUEVO (fit/predict).
    scale_y:    divide el target por su desvío en el train (útil para redes y
                árboles con hiperparámetros sensibles a la escala).
    clip:       las features estandarizadas se recortan a [-clip, clip].
    """
    pred = pd.Series(np.nan, index=X.index)
    Xv, yv = X.values, y.values
    for start in range(min_train, len(X), retrain_every):
        train_end = start - h  # el target de las últimas h filas todavía no se conoce
        Xt, yt = Xv[:train_end], yv[:train_end]
        ok = ~np.isnan(Xt).any(axis=1) & ~np.isnan(yt)
        if ok.sum() < min_obs:
            continue
        sc = StandardScaler().fit(Xt[ok])
        target = yt[ok]
        if scale_y:
            target = target / target.std()
        model = make_model()
        model.fit(np.clip(sc.transform(Xt[ok]), -clip, clip), target)
        te = slice(start, min(start + retrain_every, len(X)))
        Xe = Xv[te]
        okp = ~np.isnan(Xe).any(axis=1)
        out = np.full(len(Xe), np.nan)
        if okp.any():
            out[okp] = model.predict(np.clip(sc.transform(Xe[okp]), -clip, clip))
        pred.iloc[te] = out
    return pred


def ridge_walk_forward(X: pd.DataFrame, y: pd.Series, h: int, retrain_every: int,
                       min_train: int, alpha: float = 10.0) -> pd.Series:
    """El modelo de referencia del pipeline de derivados."""
    from sklearn.linear_model import Ridge
    return walk_forward(lambda: Ridge(alpha=alpha), X, y, h, retrain_every, min_train)
