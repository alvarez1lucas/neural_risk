# scripts/backtest.py
"""
BACKTEST walk-forward del motor completo (Paso 1-5) sobre datos históricos.

Diseño (reescrito):
- RiskFeaturePipeline es causal (cada fila usa solo el pasado), así que las
  features se calculan UNA vez sobre todo el histórico y el walk-forward
  solo las recorta. Antes se recalculaban 2 veces por barra (~27s c/u):
  inviable para más de unos cientos de barras.
- Reentrenamiento cada `retrain_every_bars`, tier rápido y lento JUNTOS
  (mismo set de features del jurado; antes el tier rápido pisaba
  best_feats y los modelos lentos quedaban entrenados con otras columnas).
- Decisión al CIERRE de la barra t, entrada al OPEN de t+1.
- Salidas: stop-loss / take-profit intrabarra (High/Low; si una barra toca
  ambos se asume el stop, conservador) o por tiempo (max_holding_bars),
  alineado con el triple-barrier labeling.
- Comisión + slippage por lado. Equity mark-to-market por barra.
- Benchmark Buy & Hold y diagnóstico de acierto direccional por experto.

Uso:
    python scripts/backtest.py --asset BTC --data "data/raw/BTC_UST Binance Historical Daily Data.csv"
    python scripts/backtest.py --asset ETH --data data/crypto/ETH_USDT_Binance_Historical_1H_Data.csv --resample 1D
"""

import argparse
import json
import logging
import os
import sys
import time

from _pathutils import resolve_path, project_root

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
from neural_risk.data.macro_context import MacroContextBuilder

# Los print() del paquete usan emojis: en consolas Windows cp1252 tiraban
# UnicodeEncodeError (incluso dentro de los except).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [BACKTEST] - %(message)s')
logging.getLogger('hmmlearn').setLevel(logging.ERROR)  # "Model is not converging" por cada fit
logger = logging.getLogger(__name__)

MACRO_CSVS = {
    'VIX': 'data/raw/VIX_Yahoo_Historical_Daily_Data.csv',
    'DXY': 'data/raw/DXY_Yahoo_Historical_Daily_Data.csv',
    'SPX': 'data/raw/SP500_Yahoo_Historical_Daily_Data.csv',
    'GOLD': 'data/raw/GOLD_Yahoo_Historical_Daily_Data.csv',
}
DIRECTIONAL_EXPERTS = ['XGB', 'CAUSAL', 'LSTM', 'ENSEMBLE']


def load_ohlcv_csv(path: str, resample: str = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    date_col = col_map.get('date') or col_map.get('timestamp')
    rename = {date_col: 'Date'}
    for wanted in ['open', 'high', 'low', 'close', 'volume']:
        if wanted in col_map:
            rename[col_map[wanted]] = wanted.capitalize()
    df = df.rename(columns=rename)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').drop_duplicates('Date').set_index('Date')
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].astype(float)
    if resample:
        df = df.resample(resample).agg({'Open': 'first', 'High': 'max', 'Low': 'min',
                                        'Close': 'last', 'Volume': 'sum'}).dropna()
    return df


def bars_per_year(index: pd.DatetimeIndex) -> float:
    step = pd.Series(index).diff().median()
    return pd.Timedelta(days=365) / step


def performance_stats(equity: pd.Series, periods_per_year: float) -> dict:
    rets = equity.pct_change().dropna()
    years = len(rets) / periods_per_year
    total = equity.iloc[-1] / equity.iloc[0] - 1
    dd = equity / equity.cummax() - 1
    downside = rets[rets < 0].std()
    return {
        'total_return': float(total),
        'cagr': float((1 + total) ** (1 / years) - 1) if years > 0 else 0.0,
        'ann_volatility': float(rets.std() * np.sqrt(periods_per_year)),
        'sharpe': float(rets.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else 0.0,
        'sortino': float(rets.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else 0.0,
        'max_drawdown': float(dd.min()),
        'calmar': float(((1 + total) ** (1 / years) - 1) / abs(dd.min())) if years > 0 and dd.min() < 0 else 0.0,
    }


class BacktestRunner:
    def __init__(self, asset: str, config_path: str = "config/config.yaml",
                 retrain_every: int = None, min_train: int = None, use_macro: bool = True):
        with open(resolve_path(config_path), encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        bt = self.config['backtest']
        sig = self.config['signals']

        self.asset = asset
        self.retrain_every = retrain_every or bt.get('retrain_every_bars', 30)
        self.min_train = min_train or bt.get('min_train_bars', 500)
        self.cost_rate = bt.get('fee_rate', 0.001) + bt.get('slippage_rate', 0.0005)
        self.max_holding = sig.get('max_holding_bars', self.config.get('labeling', {}).get('t_events', 5))
        self.use_macro = use_macro

        self.engine = AutomatedRiskEngine(
            processor=DataProcessor(), pipeline=RiskFeaturePipeline(), labeler=RiskLabeler(),
            jury=FeatureJury(), trainer_class=None,
            router=StrategyRouter(risk_appetite=self.config.get('risk_appetite', 0.7)),
            label_params=self.config.get('labeling')
        )
        self.agent = PortfolioAgent(
            initial_capital=bt['initial_capital'],
            long_threshold=sig['long_threshold'],
            short_threshold=sig['short_threshold'],
            min_confidence=sig['min_confidence'],
            sl_vol_mult=sig['sl_vol_mult'],
            tp_vol_mult=sig['tp_vol_mult'],
            base_staking_apy=sig['base_staking_apy'],
            fractional_kelly=sig['fractional_kelly'],
            max_position_size=sig['max_position_size'],
        )

        self.models = None
        self.best_feats = None
        self.pending = None          # decisión del cierre anterior, se ejecuta al open
        self.bars_held = 0
        self.records = []            # una fila por barra (equity, señal, votos)
        self.failures = {}
        self.retrain_log = []

    # ------------------------------------------------------------------
    def _load_macro(self) -> pd.DataFrame:
        if not self.use_macro:
            return pd.DataFrame()
        paths = {k: os.path.join(project_root(), v) for k, v in MACRO_CSVS.items()}
        paths = {k: v for k, v in paths.items() if os.path.exists(v)}
        if not paths:
            logger.info("Sin CSVs macro en data/raw -- se corre sin contexto macro")
            return pd.DataFrame()
        builder = MacroContextBuilder()
        return builder.compute_features(builder.load_raw_from_csv(paths))

    def _retrain(self, df_slice: pd.DataFrame):
        t0 = time.time()
        X, y, best_feats, returns, _ = self.engine.select_features(self.asset, df_slice)
        if X is None:
            return
        fast, anomaly = self.engine.fit_fast_tier_experts(X, y, returns)
        slow, anomaly = self.engine.fit_slow_tier_experts(X, y, len(best_feats), anomaly)
        self.models = {'fast': fast, 'slow': slow, 'anomaly': anomaly}
        self.best_feats = best_feats
        self.retrain_log.append({'date': str(df_slice.index[-1]), 'n_train': len(X),
                                 'n_feats': len(best_feats), 'feats': best_feats,
                                 'label_dist': y.value_counts().to_dict(),
                                 'seconds': round(time.time() - t0, 1)})
        logger.info(f"Reentreno @ {df_slice.index[-1]} | {len(X)} filas | {len(best_feats)} features "
                    f"| {time.time() - t0:.0f}s")

    def _manage_open_position(self, bar: pd.Series, ts: pd.Timestamp):
        """SL/TP intrabarra o salida por tiempo, sobre la barra que acaba de cerrar."""
        trade = self.agent.positions.get(self.asset)
        if trade is None:
            return
        self.bars_held += 1
        long = trade.signal_type == 'LONG'
        exit_price = None
        if long:
            if bar['Low'] <= trade.stop_loss:
                exit_price = min(bar['Open'], trade.stop_loss)
            elif trade.take_profit is not None and bar['High'] >= trade.take_profit:
                exit_price = max(bar['Open'], trade.take_profit)
        else:
            if bar['High'] >= trade.stop_loss:
                exit_price = max(bar['Open'], trade.stop_loss)
            elif trade.take_profit is not None and bar['Low'] <= trade.take_profit:
                exit_price = min(bar['Open'], trade.take_profit)
        if exit_price is None and self.bars_held >= self.max_holding:
            exit_price = bar['Close']
        if exit_price is not None:
            self.agent.close_position(self.asset, float(exit_price), timestamp=ts, fee_rate=self.cost_rate)

    def _open_pending(self, bar: pd.Series, ts: pd.Timestamp):
        d = self.pending
        self.pending = None
        if d is None or self.asset in self.agent.positions:
            return
        entry = float(bar['Open'])
        # SL/TP se re-anclan al precio real de entrada (misma distancia).
        shift = entry - d['entry_price']
        qty = d['position_size_pct'] * self.agent.portfolio_value / entry
        if qty <= 0:
            return
        self.agent.update_positions(
            ticker=self.asset, signal=d['signal'], entry_price=entry,
            stop_loss=d['stop_loss'] + shift, position_size=qty,
            confidence=d['confidence'], expert_votes=d['expert_signals'],
            regime_key=d.get('regime_key'),
            take_profit=(d['take_profit'] + shift) if d.get('take_profit') is not None else None,
            timestamp=ts,
        )
        self.bars_held = 0

    def _mark_to_market(self, price: float) -> float:
        equity = self.agent.portfolio_value
        trade = self.agent.positions.get(self.asset)
        if trade is not None:
            sign = 1 if trade.signal_type == 'LONG' else -1
            equity += sign * (price - trade.entry_price) * trade.position_size
        return equity

    # ------------------------------------------------------------------
    def run(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        t0 = time.time()
        logger.info(f"Calculando features una sola vez sobre {len(raw_df)} barras...")
        df_features = self.engine.build_features(self.asset, raw_df, self._load_macro())
        logger.info(f"Features listas en {time.time() - t0:.0f}s: {df_features.shape}")

        ohlc = raw_df.reindex(df_features.index)
        start = self.min_train
        if start >= len(df_features) - 10:
            raise ValueError(f"Historia insuficiente: {len(df_features)} barras de features, min_train={start}")

        for i in range(start, len(df_features)):
            ts = df_features.index[i]
            bar = ohlc.iloc[i]

            # 1) Ejecuciones dentro de la barra i (decisiones del cierre i-1)
            self._open_pending(bar, ts)
            self._manage_open_position(bar, ts)

            # 2) Reentreno con datos hasta el cierre de i
            df_slice = df_features.iloc[:i + 1]
            if self.models is None or (i - start) % self.retrain_every == 0:
                self._retrain(df_slice)

            record = {'date': ts, 'close': float(bar['Close']), 'signal': 'NONE'}
            if self.models is not None:
                X, _, _, returns, _ = self.engine.select_features(self.asset, df_slice, self.best_feats)
                if X is not None:
                    report = self.engine.predict_with_cached_experts(
                        self.asset, X, returns, df_slice,
                        self.models['fast'], self.models['slow'], self.models['anomaly'])
                    for name in report.get('failures', []):
                        self.failures[name] = self.failures.get(name, 0) + 1
                    decision = self.agent.execute_portfolio_decision(
                        {self.asset: report}, {self.asset: float(bar['Close'])}
                    )['decisions'].get(self.asset)
                    if decision:
                        record.update({'signal': decision['signal'], 'agreement': decision['agreement'],
                                       'confidence': decision['confidence'],
                                       'anomaly_override': decision['anomaly_override'],
                                       'regime': decision['regime_key']})
                        record.update({f'vote_{k}': v for k, v in decision['expert_signals'].items()})
                        if decision['signal'] in ('LONG', 'SHORT') and self.asset not in self.agent.positions:
                            self.pending = decision

            # 3) Equity al cierre de i
            record['equity'] = self._mark_to_market(float(bar['Close']))
            record['in_position'] = self.asset in self.agent.positions
            self.records.append(record)

            if (i - start + 1) % 100 == 0:
                logger.info(f"{ts} | equity={record['equity']:.0f} | trades={len(self.agent.closed_trades)} "
                            f"| {time.time() - t0:.0f}s")

        last = ohlc.iloc[-1]
        if self.asset in self.agent.positions:
            self.agent.close_position(self.asset, float(last['Close']), timestamp=df_features.index[-1],
                                      fee_rate=self.cost_rate)
            self.records[-1]['equity'] = self.agent.portfolio_value
        logger.info(f"Backtest completo en {time.time() - t0:.0f}s")
        return pd.DataFrame(self.records).set_index('date')

    # ------------------------------------------------------------------
    def summarize(self, results: pd.DataFrame) -> dict:
        ppy = bars_per_year(results.index)
        strat = performance_stats(results['equity'], ppy)
        bh_equity = results['close'] / results['close'].iloc[0] * results['equity'].iloc[0]
        bh = performance_stats(bh_equity, ppy)

        trades = self.agent.closed_trades
        pnl = np.array([t.pnl for t in trades]) if trades else np.array([])
        trade_stats = {
            'n_trades': len(trades),
            'n_long': sum(t.signal_type == 'LONG' for t in trades),
            'n_short': sum(t.signal_type == 'SHORT' for t in trades),
            'win_rate': float((pnl > 0).mean()) if len(pnl) else 0.0,
            'profit_factor': float(pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else None,
            'avg_trade_pct': float(np.mean([t.pnl_pct for t in trades])) if trades else 0.0,
            'exposure': float(results['in_position'].mean()),
        }

        # Acierto direccional out-of-sample de cada experto: signo del voto
        # vs signo del retorno de las próximas max_holding barras.
        fwd = results['close'].shift(-self.max_holding) / results['close'] - 1
        hit_rates = {}
        for name in DIRECTIONAL_EXPERTS + ['agreement']:
            col = f'vote_{name}' if name != 'agreement' else 'agreement'
            if col not in results:
                continue
            v = results[col]
            mask = v.notna() & fwd.notna() & (v != 0)
            if mask.sum() > 0:
                hit_rates[name] = {'hit_rate': float((np.sign(v[mask]) == np.sign(fwd[mask])).mean()),
                                   'corr_with_fwd_return': float(v[mask].corr(fwd[mask])),
                                   'n': int(mask.sum())}

        return {
            'asset': self.asset,
            'period': [str(results.index[0]), str(results.index[-1])],
            'bars': len(results),
            'strategy': strat,
            'buy_and_hold': bh,
            'trades': trade_stats,
            'signal_counts': results['signal'].value_counts().to_dict(),
            'expert_hit_rates': hit_rates,
            'expert_failures': self.failures,
            'final_expert_weights': {k: float(v) for k, v in self.agent.weighting.get_weights().items()},
            'n_retrains': len(self.retrain_log),
            'cost_per_side': self.cost_rate,
        }


def save_chart(results: pd.DataFrame, trades, path: str, asset: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    eq = results['equity']
    bh = results['close'] / results['close'].iloc[0] * eq.iloc[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(eq.index, eq, label='Neural Risk Engine', color='#1f77b4', lw=1.6)
    ax1.plot(bh.index, bh, label=f'Buy & Hold {asset}', color='#7f7f7f', ls='--', lw=1.2)
    for t in trades:
        if t.exit_time in eq.index:
            ax1.scatter(t.exit_time, eq.loc[t.exit_time], s=12,
                        color='#2ca02c' if (t.pnl or 0) > 0 else '#d62728', zorder=3)
    ax1.set_ylabel('Equity')
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.3)
    ax1.set_title(f'Backtest walk-forward {asset}')
    for series, color, label in [(eq, '#1f77b4', 'Estrategia'), (bh, '#7f7f7f', 'Buy & Hold')]:
        dd = (series / series.cummax() - 1) * 100
        ax2.plot(dd.index, dd, color=color, lw=1, label=label)
    ax2.set_ylabel('Drawdown %')
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Backtest walk-forward del motor Neural Risk")
    parser.add_argument("--asset", required=True, help="Ticker, ej. BTC")
    parser.add_argument("--data", required=True, help="Ruta al CSV OHLCV")
    parser.add_argument("--resample", default=None, help="Re-muestrear velas, ej. 1D o 4h")
    parser.add_argument("--start", default=None, help="Fecha inicial de los datos (opcional)")
    parser.add_argument("--retrain-every", type=int, default=None, help="Barras entre reentrenos")
    parser.add_argument("--min-train", type=int, default=None, help="Barras de historia antes del 1er reentreno")
    parser.add_argument("--no-macro", action="store_true", help="Correr sin contexto macro")
    parser.add_argument("--out", default="backtest_results", help="Carpeta de salida")
    args = parser.parse_args()

    raw_df = load_ohlcv_csv(resolve_path(args.data), args.resample)
    if args.start:
        raw_df = raw_df.loc[args.start:]
    logger.info(f"Datos: {len(raw_df)} barras, {raw_df.index[0]} a {raw_df.index[-1]}")

    runner = BacktestRunner(asset=args.asset, retrain_every=args.retrain_every,
                            min_train=args.min_train, use_macro=not args.no_macro)
    results = runner.run(raw_df)
    summary = runner.summarize(results)

    tf = args.resample or pd.Series(raw_df.index).diff().median().__str__().replace(' ', '').replace(':', '')
    out_dir = os.path.join(resolve_path('.') if os.path.isabs(args.out) else project_root(), args.out,
                           f"{args.asset}_{tf}")
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(os.path.join(out_dir, 'bars.csv'))
    pd.DataFrame([{
        'signal_type': t.signal_type, 'entry_time': t.entry_time, 'exit_time': t.exit_time,
        'entry_price': t.entry_price, 'exit_price': t.exit_price, 'stop_loss': t.stop_loss,
        'take_profit': t.take_profit, 'position_size': t.position_size, 'pnl': t.pnl,
        'pnl_pct': t.pnl_pct, 'confidence': t.confidence, 'regime': t.regime_key,
    } for t in runner.agent.closed_trades]).to_csv(os.path.join(out_dir, 'trades.csv'), index=False)
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str)
    with open(os.path.join(out_dir, 'retrains.json'), 'w', encoding='utf-8') as f:
        json.dump(runner.retrain_log, f, indent=2, default=str)
    save_chart(results, runner.agent.closed_trades, os.path.join(out_dir, 'equity.png'), args.asset)

    logger.info("=" * 60)
    logger.info(json.dumps(summary, indent=2, default=str))
    logger.info(f"Resultados en {out_dir}")


if __name__ == "__main__":
    main()
