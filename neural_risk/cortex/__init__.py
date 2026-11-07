# neural_risk/cortex/__init__.py
# Imports perezosos: ambos módulos necesitan statsmodels (extra [ml]).
import importlib

_LAZY = {
    'FeatureJury': '.feature_jury',
    'CausalSelector': '.causal_selector',
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
