# neural_risk/data/__init__.py
# Imports perezosos: feature_engineering necesita torch (extra [ml]) y no
# debe cargarse al importar módulos livianos como binance_vision o
# derivatives_features.
import importlib

_LAZY = {
    'DataProcessor': '.data_processor',
    'RiskFeaturePipeline': '.feature_engineering',
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
