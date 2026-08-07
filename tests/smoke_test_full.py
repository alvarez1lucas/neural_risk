"""
SMOKE TEST COMPLETO -- correr en TU entorno (donde podés instalar
torch/xgboost/hmmlearn/arch/statsmodels): pip install -r requirements.txt

Genera OHLCV sintético, corre el pipeline REAL de punta a punta:
  prepare_asset_features -> fit_fast_tier_experts -> fit_slow_tier_experts
  -> predict_with_cached_experts -> PortfolioAgent.execute_portfolio_decision

No necesita SQLite ni ningún script en background -- llama directo a los
métodos de AutomatedRiskEngine, igual que hacen train_models.py / 
run_engine.py / backtest.py, pero todo en un solo proceso para que
cualquier error de wiring aparezca inmediatamente con traceback completo.

Uso:
    pip install -r requirements.txt
    python tests/smoke_test_full.py
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

np.random.seed(42)


def generate_synthetic_ohlcv(n_days=400, start_price=45000, vol=0.02):
    """Genera un random walk geométrico con OHLC coherente (High>=Open,Close; Low<=Open,Close)."""
    dates = pd.date_range('2024-01-01', periods=n_days, freq='D')
    returns = np.random.randn(n_days) * vol
    close = start_price * np.cumprod(1 + returns)

    open_ = np.roll(close, 1)
    open_[0] = start_price

    daily_range = np.abs(np.random.randn(n_days)) * vol * close
    high = np.maximum(open_, close) + daily_range * 0.5
    low = np.minimum(open_, close) - daily_range * 0.5
    volume = np.random.uniform(1e6, 1e7, n_days)

    df = pd.DataFrame({
        'Open': open_, 'High': high, 'Low': low, 'Close': close, 'Volume': volume
    }, index=dates)
    return df


def main():
    print("=" * 70)
    print("SMOKE TEST COMPLETO -- pipeline real con los 9 expertos")
    print("=" * 70)

    print("\n[1/6] Generando OHLCV sintético (400 días, ~13 meses)...")
    raw_df = generate_synthetic_ohlcv(n_days=400)
    print(f"      {len(raw_df)} filas, {raw_df.index[0].date()} a {raw_df.index[-1].date()}")

    print("\n[2/6] Instanciando AutomatedRiskEngine con dependencias reales...")
    from neural_risk.engine import AutomatedRiskEngine
    from neural_risk.data.data_processor import DataProcessor
    from neural_risk.data.feature_engineering import RiskFeaturePipeline
    from neural_risk.data.labeling import RiskLabeler
    from neural_risk.cortex.feature_jury import FeatureJury
    from neural_risk.agents.strategy_router import StrategyRouter
    from neural_risk.agents.portfolio_agent import PortfolioAgent

    engine = AutomatedRiskEngine(
        processor=DataProcessor(), pipeline=RiskFeaturePipeline(), labeler=RiskLabeler(),
        jury=FeatureJury(), trainer_class=None, router=StrategyRouter(risk_appetite=0.7)
    )
    print("      OK -- import y construcción sin errores")

    print("\n[3/6] prepare_asset_features (feature engineering completo, modo ENTRENAMIENTO)...")
    print("      (esto puede tardar -- corre HMM en loop, IsolationForest, KMeans, etc.)")
    X_filtered, y, best_feats, returns, df_features = engine.prepare_asset_features('BTC', raw_df)

    assert X_filtered is not None, "FALLO: prepare_asset_features devolvió None -- revisar datos sintéticos o el pipeline"
    print(f"      OK -- {len(best_feats)} features aprobadas por el jurado: {best_feats[:8]}{'...' if len(best_feats) > 8 else ''}")
    print(f"      X_filtered shape: {X_filtered.shape}, y shape: {y.shape}, returns: {len(returns)} valores")

    print("\n[4/6] fit_fast_tier_experts (HMM, XGB, CAUSAL, GARCH, Isolation Forest)...")
    fast_models, anomaly = engine.fit_fast_tier_experts(X_filtered, y, returns)
    for name, model in fast_models.items():
        status = "OK" if model is not None else "FALLÓ (revisar log de arriba)"
        print(f"      {name}: {status}")

    print("\n[5/6] fit_slow_tier_experts (Ensemble neuronal, LSTM/Transformer, Autoencoder)...")
    print("      (esto es lo más lento -- entrena redes con backprop)")
    slow_models, anomaly = engine.fit_slow_tier_experts(X_filtered, y, len(best_feats), anomaly)
    for name, model in slow_models.items():
        status = "OK" if model is not None else "FALLÓ (revisar log de arriba)"
        print(f"      {name}: {status}")

    print("\n[6/6] predict_with_cached_experts + PortfolioAgent.execute_portfolio_decision...")
    asset_report = engine.predict_with_cached_experts(
        'BTC', X_filtered, returns, df_features, fast_models, slow_models, anomaly
    )
    print(f"      asset_report generado con {len(asset_report)} señales:")
    for k in ['hmm_regime', 'xgb_signal', 'causal_effect', 'garch_vol', 'lstm_forecast', 'anomaly', 'ensemble', 'hurst']:
        print(f"        {k}: {asset_report.get(k)}")

    current_price = float(df_features['BTC_Close'].iloc[-1])
    agent = PortfolioAgent(initial_capital=100000)
    portfolio_decision = agent.execute_portfolio_decision(
        {'BTC': asset_report}, {'BTC': current_price}
    )
    decision = portfolio_decision['decisions']['BTC']
    print(f"\n      DECISIÓN FINAL: signal={decision['signal']}, confidence={decision['confidence']:.3f}, "
          f"entry={decision['entry_price']:.2f}, SL={decision['stop_loss']:.2f}, "
          f"size={decision['position_size_pct']:.4f}")

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETO CORRIÓ SIN ERRORES DE PUNTA A PUNTA")
    print("=" * 70)
    print("\nSi llegaste hasta acá sin traceback, el engine real (con los 9")
    print("expertos, no simulados) está funcionalmente conectado de punta a punta.")


if __name__ == "__main__":
    main()