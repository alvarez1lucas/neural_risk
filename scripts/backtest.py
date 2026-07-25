# scripts/backtest.py
"""
BACKTEST: valida el motor completo (Paso 1-5) sobre datos históricos,
sin tocar ninguna cuenta real ni el executor en vivo.

Simula walk-forward real (nunca mira el futuro):
- Reentrena el tier rápido cada N días de datos históricos (igual que
  train_models.py haría en producción)
- Reentrena el tier lento cada M días
- En los días intermedios, predice con los últimos modelos entrenados
- Simula fills y stop-loss, y alimenta el circuito de feedback
  (Thompson Sampling / hedging ponderado) con resultados reales -- así
  el backtest también valida que el aprendizaje funcione, no solo las
  señales puntuales.

LIMITACIONES CONOCIDAS (no resueltas en este script, marcadas a propósito):
- No aplica slippage_pct ni commission_pct de config.yaml -- los
  resultados están sobreestimados en esa medida. Fácil de agregar
  después si hace falta precisión.
- Un solo activo por corrida (no hay asignación de capital entre
  varios activos vía StrategyRouter -- eso sigue sin consumidor, como
  quedó anotado en engine.py).
- Entrada y salida simuladas al precio de Close del mismo día que
  generó la señal -- aproximación razonable para barras diarias, pero
  no modela el desfasaje real entre "se genera la señal" y "se ejecuta
  la orden" que sí existe en producción (executor corre 60-300s
  después del engine).

Uso:
    python scripts/backtest.py --asset BTC --data data/BTC_USD_data.csv
"""

import argparse
import logging
import numpy as np
import pandas as pd
import yaml

from neural_risk.engine import AutomatedRiskEngine
from neural_risk.data.data_processor import DataProcessor
from neural_risk.data.feature_engineering import RiskFeaturePipeline
from neural_risk.data.labeling import RiskLabeler
from neural_risk.cortex.feature_jury import FeatureJury
from neural_risk.agents.strategy_router import StrategyRouter
from neural_risk.agents.portfolio_agent import PortfolioAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BACKTEST] - %(message)s')
logger = logging.getLogger(__name__)


def load_ohlcv_csv(path: str) -> pd.DataFrame:
    """
    Carga un CSV con columnas Date,Open,High,Low,Close,Volume
    (case-insensitive) y lo deja en el formato que espera
    AutomatedRiskEngine.prepare_asset_features.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    col_map = {c.lower(): c for c in df.columns}
    rename = {}
    for wanted in ['date', 'open', 'high', 'low', 'close', 'volume']:
        if wanted in col_map:
            rename[col_map[wanted]] = wanted.capitalize()
    df = df.rename(columns=rename)

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').set_index('Date')

    return df[['Open', 'High', 'Low', 'Close', 'Volume']]


class BacktestRunner:
    def __init__(self, asset: str, config_path: str = "config/config.yaml",
                 fast_retrain_days: int = 1, slow_retrain_days: int = 5,
                 min_history: int = 250):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.asset = asset
        self.fast_retrain_days = fast_retrain_days
        self.slow_retrain_days = slow_retrain_days
        self.min_history = min_history  # velas mínimas antes del primer trade

        self.engine = AutomatedRiskEngine(
            processor=DataProcessor(),
            pipeline=RiskFeaturePipeline(),
            labeler=RiskLabeler(),
            jury=FeatureJury(),
            trainer_class=None,
            router=StrategyRouter(risk_appetite=self.config.get('risk_appetite', 0.7))
        )

        self.agent = PortfolioAgent(
            initial_capital=self.config['backtest']['initial_capital'],
            long_threshold=self.config['signals']['long_threshold'],
            short_threshold=self.config['signals']['short_threshold'],
            adaptive_sl_threshold=self.config['signals']['sl_adaptive_threshold'],
            base_staking_apy=self.config['signals']['base_staking_apy'],
            fractional_kelly=self.config['signals']['fractional_kelly'],
            max_position_size=self.config['signals']['max_position_size']
        )

        self.equity_curve = []
        self.fast_models = None
        self.slow_models = None
        self.anomaly_detector = None
        self.best_feats = None
        self.days_since_fast = 0
        self.days_since_slow = 0

    def _maybe_retrain(self, history_df: pd.DataFrame):
        """Reentrena tier rápido/lento si corresponde, SOLO con datos
        hasta el día actual (walk-forward, sin mirar el futuro)."""
        need_fast = self.fast_models is None or self.days_since_fast >= self.fast_retrain_days
        need_slow = self.slow_models is None or self.days_since_slow >= self.slow_retrain_days

        if not need_fast and not need_slow:
            return

        X_filtered, y, best_feats, returns, _ = self.engine.prepare_asset_features(
            self.asset, history_df
        )
        if X_filtered is None:
            return

        self.best_feats = best_feats

        if need_fast:
            self.fast_models, self.anomaly_detector = self.engine.fit_fast_tier_experts(
                X_filtered, y, returns, existing_anomaly=self.anomaly_detector
            )
            self.days_since_fast = 0

        if need_slow:
            self.slow_models, self.anomaly_detector = self.engine.fit_slow_tier_experts(
                X_filtered, y, len(best_feats), self.anomaly_detector
            )
            self.days_since_slow = 0

    def _check_open_position(self, current_price: float):
        """Cierra la posición abierta si el precio actual rompió el SL."""
        if self.asset not in self.agent.positions:
            return

        trade = self.agent.positions[self.asset]
        breached = (
            (trade.signal_type == 'LONG' and current_price <= trade.stop_loss) or
            (trade.signal_type == 'SHORT' and current_price >= trade.stop_loss)
        )
        if breached:
            self.agent.close_position(self.asset, current_price)

    def run(self, history_df: pd.DataFrame) -> pd.DataFrame:
        dates = history_df.index[self.min_history:]
        current_price = None

        for i, current_date in enumerate(dates):
            window = history_df.loc[:current_date]
            current_price = float(window['Close'].iloc[-1])

            self._check_open_position(current_price)

            self._maybe_retrain(window)
            self.days_since_fast += 1
            self.days_since_slow += 1

            if self.fast_models is None or self.best_feats is None:
                self.equity_curve.append({'date': current_date, 'equity': self.agent.portfolio_value})
                continue

            X_filtered, _, _, returns, df_features = self.engine.prepare_asset_features(
                self.asset, window, cached_best_feats=self.best_feats
            )
            if X_filtered is None:
                self.equity_curve.append({'date': current_date, 'equity': self.agent.portfolio_value})
                continue

            intelligence = self.engine.predict_with_cached_experts(
                self.asset, X_filtered, returns, df_features,
                self.fast_models, self.slow_models or {}, self.anomaly_detector
            )

            portfolio_decision = self.agent.execute_portfolio_decision(
                {self.asset: intelligence}, {self.asset: current_price}
            )
            decision = portfolio_decision['decisions'].get(self.asset)

            if decision and self.asset not in self.agent.positions and decision['signal'] in ('LONG', 'SHORT'):
                capital_to_risk = decision['position_size_pct'] * self.agent.portfolio_value
                quantity = capital_to_risk / current_price
                self.agent.update_positions(
                    ticker=self.asset,
                    signal=decision['signal'],
                    entry_price=current_price,
                    stop_loss=decision['stop_loss'],
                    position_size=quantity,
                    confidence=decision['confidence'],
                    expert_votes=decision['expert_signals']
                )

            self.equity_curve.append({
                'date': current_date,
                'equity': self.agent.portfolio_value + self.agent.cash
            })

            if (i + 1) % 30 == 0:
                logger.info(
                    f"{current_date.date()} | equity={self.agent.portfolio_value:.2f} | "
                    f"trades={len(self.agent.closed_trades)}"
                )

        if self.asset in self.agent.positions and current_price is not None:
            self.agent.close_position(self.asset, current_price)

        return pd.DataFrame(self.equity_curve).set_index('date')


def main():
    parser = argparse.ArgumentParser(description="Backtest del motor Neural Risk")
    parser.add_argument("--asset", required=True, help="Ticker, ej. BTC")
    parser.add_argument("--data", required=True, help="Ruta al CSV OHLCV")
    parser.add_argument("--fast-retrain-days", type=int, default=1)
    parser.add_argument("--slow-retrain-days", type=int, default=5)
    args = parser.parse_args()

    history_df = load_ohlcv_csv(args.data)
    logger.info(
        f"Datos cargados: {len(history_df)} filas, "
        f"{history_df.index[0].date()} a {history_df.index[-1].date()}"
    )

    runner = BacktestRunner(
        asset=args.asset,
        fast_retrain_days=args.fast_retrain_days,
        slow_retrain_days=args.slow_retrain_days
    )

    equity_df = runner.run(history_df)

    metrics = runner.agent.get_portfolio_metrics()
    logger.info("=" * 60)
    logger.info("RESULTADOS DEL BACKTEST")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    out_path = f"backtest_equity_{args.asset}.csv"
    equity_df.to_csv(out_path)
    logger.info(f"Curva de equity guardada en {out_path}")


if __name__ == "__main__":
    main()