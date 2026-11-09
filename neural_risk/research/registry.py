# neural_risk/research/registry.py
"""
Registro de experimentos (append-only, JSONL).

Cada backtest que se corre se anota, INCLUIDOS los que se descartan. Sin
eso no hay forma de saber cuántas pruebas se hicieron, y el Deflated
Sharpe necesita ese número (ver validation.py).
"""

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd


def _git_commit() -> Optional[str]:
    try:
        return subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def _jsonable(x):
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (pd.Timestamp,)):
        return str(x)
    return x


class ExperimentRegistry:
    def __init__(self, path: str = "./backtest_results/experiments.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    @staticmethod
    def config_id(config: dict) -> str:
        blob = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def log(self, study: str, config: dict, metrics: dict) -> str:
        """study agrupa las variantes que compiten entre sí (ej. 'models_BTCUSDT_30m_h24')."""
        cid = self.config_id(config)
        row = {'time': datetime.now(timezone.utc).isoformat(timespec='seconds'),
               'study': study, 'config_id': cid, 'commit': _git_commit(),
               'config': config, 'metrics': {k: _jsonable(v) for k, v in metrics.items()}}
        with open(self.path, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + '\n')
        return cid

    def load(self, study: Optional[str] = None) -> pd.DataFrame:
        if not os.path.exists(self.path):
            return pd.DataFrame()
        with open(self.path, encoding='utf-8') as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        df = pd.json_normalize(rows)
        if study is not None and len(df):
            df = df[df['study'] == study]
        return df

    def n_trials(self, study: str) -> int:
        """Variantes DISTINTAS probadas en el estudio (repetir una corrida no suma)."""
        df = self.load(study)
        return int(df['config_id'].nunique()) if len(df) else 0
