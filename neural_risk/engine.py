# neural_risk/engine.py
"""
AUTOMATED RISK ENGINE: Sistema multi-experto institucional para cripto
Paso 1-5: Data → Features → Jury → Multi-Experts → Router

ESTADO: Reestructurado para separar FIT (entrenamiento, caro, corre en
train_models.py con scheduling por tier) de PREDICT (barato, corre en
cada ciclo live de run_engine.py, nunca reentrena).

Métodos públicos, pensados para reutilizarse como librería:
- prepare_asset_features(ticker, raw_df, cached_best_feats=None)
- fit_fast_tier_experts(X_filtered, y, returns, existing_anomaly=None)
- fit_slow_tier_experts(X_filtered, y, num_features, anomaly_detector)
- predict_with_cached_experts(ticker, X_filtered, returns, df_features,
                               fast_models, slow_models, anomaly_detector)
- run_portfolio_automation(assets_data, train=True): wrapper de
  conveniencia fit+predict de punta a punta, para backtest/uso puntual.

PENDIENTE (marcado, no resuelto):
- El costo de prepare_asset_features (RiskFeaturePipeline.transform) NO
  está cacheado -- corre en cada ciclo live, entrenamiento o predicción.
- TemporalCrossValidator se calcula en fit_slow_tier_experts pero sus
  folds no se usan para entrenar todavía.
- StrategyRouter.allocate_capital solo se invoca desde
  run_portfolio_automation, que run_engine.py y backtest.py NO llaman
  (usan prepare_asset_features + predict_with_cached_experts
  directamente) -- el Router hoy no tiene consumidor en el camino real.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List, Optional
import warnings

from neural_risk.models.classic import XGBoostVolModel
from neural_risk.models.hmm_model import RegimeHMMModel
from neural_risk.models.bayesian_model import BayesianNeuralRisk
from neural_risk.models.causal_strategy import CausalInferenceModel
from neural_risk.models.temporal_cv import TemporalCrossValidator
from neural_risk.models.ensemble_trainer import EnsembleTrainer
from neural_risk.models.garch_volatility import GARCHVolatilityExpert, MultiWindowGARCH
from neural_risk.models.lstm_transformer import SequentialForecastingEnsemble
from neural_risk.models.reinforcement_learning import RLAllocationExpert, MultiArmedBanditExpert
from neural_risk.models.copula_expert import CopulaExpert, MultiAssetCopulaExpert
from neural_risk.models.anomaly_detection import AnomalyDetector

warnings.filterwarnings('ignore')


class AutomatedRiskEngine:
    """
    ORQUESTADOR INSTITUCIONAL:
    Coordina Data -> Features -> Jury -> MULTI-MODELOS -> Router.
    """

    def __init__(self, processor, pipeline, labeler, jury, trainer_class, router):
        self.processor = processor
        self.pipeline = pipeline
        self.labeler = labeler
        self.jury = jury
        self.TrainerClass = trainer_class
        self.router = router
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.window_sizes = {
            'short': 10,
            'medium': 30,
            'long': 100,
            'strategic': 1000
        }

    # ========================================================================
    # PREPARACIÓN DE FEATURES -- camino COMPARTIDO entre entrenamiento y
    # predicción, para que ambos siempre usen exactamente el mismo feature
    # engineering.
    # ========================================================================
    def prepare_asset_features(self, ticker: str, raw_df: pd.DataFrame,
                              cached_best_feats: Optional[List[str]] = None,
                              macro_context: Optional[pd.DataFrame] = None
                              ) -> Tuple:
        """
        Si cached_best_feats es None -> modo ENTRENAMIENTO: calcula el
        target (triple barrier) y corre el jurado desde cero. Retorna y real.

        Si cached_best_feats viene seteado -> modo PREDICCIÓN: salta el
        jurado por completo. Retorna y=None.

        NUEVO (mejora #3 de 3, contexto macro): macro_context es un
        DataFrame opcional con features derivadas de índices FUERA del
        universo cripto (VIX, DXY, SPX, Oro -- ver
        neural_risk/data/macro_context.py), indexado por fecha. Si se
        provee, se alinea (ffill, sin look-ahead) y se agrega como
        columnas adicionales ANTES de la selección de features -- el
        jurado decide por sí mismo si aportan señal real, igual que
        cualquier otra columna. Si es None (default), el comportamiento
        es idéntico al de antes de esta mejora.

        Returns: (X_filtered, y_or_None, best_feats, returns, df_features)
                 o (None, None, None, None, None) si no hay datos suficientes.
        """
        df_clean = self.processor.auto_clean(raw_df)
        df_prefixed = self.processor.rename_columns(df_clean, ticker)
        df_features = self.pipeline.transform(df_prefixed, asset_names=[ticker])

        if macro_context is not None and not macro_context.empty:
            from neural_risk.data.macro_context import merge_macro_context
            df_features = merge_macro_context(df_features, macro_context)

        close_col = f'{ticker}_Close'
        returns = np.log(df_features[close_col] / df_features[close_col].shift(1)).dropna().values

        if cached_best_feats is None:
            target = self.labeler.triple_barrier_label(df_features[close_col])
            common_idx = df_features.index.intersection(target.index)

            if len(common_idx) < 100:
                return None, None, None, None, None

            X = df_features.loc[common_idx].select_dtypes(include=[np.number])
            y = target.loc[common_idx]

            best_feats = self.jury.evaluate(X, y)
            if not best_feats:
                best_feats = X.columns[:10].tolist()

            X_filtered = X[best_feats]
            return X_filtered, y, best_feats, returns, df_features

        else:
            if len(df_features) < 100:
                return None, None, None, None, None

            X = df_features.select_dtypes(include=[np.number])
            available_feats = [f for f in cached_best_feats if f in X.columns]
            if not available_feats:
                return None, None, None, None, None

            X_filtered = X[available_feats]
            return X_filtered, None, available_feats, returns, df_features

    # ========================================================================
    # FIT -- tier rápido (barato, diario): HMM, XGB, CAUSAL, GARCH,
    # Isolation Forest del Anomaly Detector.
    # ========================================================================
    def fit_fast_tier_experts(self, X_filtered: pd.DataFrame, y: pd.Series,
                             returns: np.ndarray,
                             existing_anomaly: Optional[AnomalyDetector] = None
                             ) -> Tuple[Dict, AnomalyDetector]:
        models = {}

        try:
            hmm = RegimeHMMModel(n_components=3)
            hmm.fit(X_filtered)
            models['HMM'] = hmm
        except Exception as e:
            print(f"⚠️  HMM fit failed: {e}")
            models['HMM'] = None

        try:
            xgb_model = XGBoostVolModel()
            xgb_model.fit(X_filtered, y)
            models['XGB'] = xgb_model
        except Exception as e:
            print(f"⚠️  XGB fit failed: {e}")
            models['XGB'] = None

        try:
            causal_model = CausalInferenceModel()
            causal_model.fit(X_filtered, y)
            models['CAUSAL'] = causal_model
        except Exception as e:
            print(f"⚠️  CAUSAL fit failed: {e}")
            models['CAUSAL'] = None

        try:
            garch = MultiWindowGARCH(window_sizes=[
                self.window_sizes['short'], self.window_sizes['medium'],
                self.window_sizes['long'], self.window_sizes['strategic']
            ])
            garch.fit_all(returns)
            models['GARCH'] = garch
        except Exception as e:
            print(f"⚠️  GARCH fit failed: {e}")
            models['GARCH'] = None

        anomaly = existing_anomaly or AnomalyDetector(contamination=0.05, device=self.device)
        try:
            anomaly.fit_isolation_forest(returns)
        except Exception as e:
            print(f"⚠️  ANOMALY (isolation forest) fit failed: {e}")

        return models, anomaly

    # ========================================================================
    # FIT -- tier lento (caro, cada N días): Ensemble neuronal, LSTM/Transformer,
    # Autoencoder del Anomaly.
    # ========================================================================
    def fit_slow_tier_experts(self, X_filtered: pd.DataFrame, y: pd.Series,
                             num_features: int,
                             anomaly_detector: AnomalyDetector
                             ) -> Tuple[Dict, AnomalyDetector]:
        models = {}

        try:
            deep_tft = BayesianNeuralRisk(num_features=num_features).to(self.device)

            # TODO: los folds de TemporalCrossValidator se calculan pero
            # todavía no se usan para entrenar.
            cv = TemporalCrossValidator(n_splits=5, initial_window=100, test_size=20)
            cv.split(X_filtered, y)

            ensemble = EnsembleTrainer(deep_tft, device=self.device)
            data_bundle = ensemble.prepare_data(X_filtered, y)
            ensemble.train_ensemble(
                train_loader=data_bundle['neural'][0],
                test_loader=data_bundle['neural'][1],
                xgb_data=data_bundle,
                epochs=20
            )
            models['ENSEMBLE_TRAINER'] = ensemble
        except Exception as e:
            print(f"⚠️  ENSEMBLE fit failed: {e}")
            models['ENSEMBLE_TRAINER'] = None

        try:
            lstm_tf = SequentialForecastingEnsemble(input_size=num_features, device=self.device)

            if len(X_filtered) > self.window_sizes['strategic']:
                window_data = X_filtered.iloc[-self.window_sizes['strategic']:].values
            else:
                window_data = X_filtered.values

            y_window = y.iloc[-len(window_data):].values

            for _ in range(3):
                X_tensor = torch.FloatTensor(window_data).unsqueeze(0).to(self.device)
                y_tensor = torch.FloatTensor([y_window[-1]]).to(self.device)
                lstm_tf.train_step(X_tensor, y_tensor)

            models['LSTM_TF'] = lstm_tf
        except Exception as e:
            print(f"⚠️  LSTM/TF fit failed: {e}")
            models['LSTM_TF'] = None

        try:
            anomaly_detector.fit_autoencoder(X_filtered, epochs=10)
        except Exception as e:
            print(f"⚠️  ANOMALY (autoencoder) fit failed: {e}")

        return models, anomaly_detector

    # ========================================================================
    # PREDICT -- SIN fit. Usa modelos ya entrenados.
    # ========================================================================
    def predict_with_cached_experts(self, ticker: str, X_filtered: pd.DataFrame,
                                   returns: np.ndarray, df_features: pd.DataFrame,
                                   fast_models: Dict, slow_models: Dict,
                                   anomaly_detector: Optional[AnomalyDetector]
                                   ) -> Dict:
        asset_report = {}
        xgb_signal = 0.0
        causal_impact = 0.0

        hmm = fast_models.get('HMM')
        if hmm is not None:
            try:
                asset_report['hmm_regime'] = int(hmm.predict(X_filtered.iloc[-1:])[-1])
            except Exception as e:
                print(f"⚠️  HMM predict failed: {e}")
                asset_report['hmm_regime'] = 1
        else:
            asset_report['hmm_regime'] = 1

        causal_model = fast_models.get('CAUSAL')
        if causal_model is not None:
            try:
                causal_impact = float(causal_model.estimate_effect(X_filtered.iloc[-1:])[-1])
                asset_report['causal_effect'] = causal_impact
            except Exception as e:
                print(f"⚠️  CAUSAL predict failed: {e}")
                asset_report['causal_effect'] = 0.0
        else:
            asset_report['causal_effect'] = 0.0

        xgb_model = fast_models.get('XGB')
        if xgb_model is not None:
            try:
                xgb_signal = float(xgb_model.predict(X_filtered.iloc[-1:])[-1])
                asset_report['xgb_signal'] = xgb_signal
            except Exception as e:
                print(f"⚠️  XGB predict failed: {e}")
                asset_report['xgb_signal'] = 0.0
        else:
            asset_report['xgb_signal'] = 0.0

        garch = fast_models.get('GARCH')
        if garch is not None:
            try:
                asset_report['garch_vol'] = garch.get_hedging_signal()
            except Exception as e:
                print(f"⚠️  GARCH predict failed: {e}")
                asset_report['garch_vol'] = {'crisis_detected': False}
        else:
            asset_report['garch_vol'] = {'crisis_detected': False}

        lstm_tf = slow_models.get('LSTM_TF')
        if lstm_tf is not None:
            try:
                X_test = torch.FloatTensor(X_filtered.values[-20:]).unsqueeze(0).to(self.device)
                asset_report['lstm_forecast'] = lstm_tf.predict_ensemble(X_test)
            except Exception as e:
                print(f"⚠️  LSTM/TF predict failed: {e}")
                asset_report['lstm_forecast'] = {'ensemble_forecast': 0.0}
        else:
            asset_report['lstm_forecast'] = {'ensemble_forecast': 0.0}

        if anomaly_detector is not None:
            try:
                asset_report['anomaly'] = anomaly_detector.predict_anomalies(
                    X_filtered.iloc[-10:], recent_returns=returns[-10:]
                )
            except Exception as e:
                print(f"⚠️  ANOMALY predict failed: {e}")
                asset_report['anomaly'] = {'anomaly_detected': False}
        else:
            asset_report['anomaly'] = {'anomaly_detected': False}

        ensemble = slow_models.get('ENSEMBLE_TRAINER')
        if ensemble is not None:
            try:
                if len(returns) >= 20:
                    ensemble.kalman.update(float(np.std(returns[-20:])))

                mu_ens, sigma_ens = ensemble.predict_ensemble(X_filtered.values[-10:])
                asset_report['ensemble'] = {
                    'mu': float(mu_ens),
                    'sigma': float(sigma_ens),
                    'regime': ensemble.kalman.get_regime()
                }
            except Exception as e:
                print(f"⚠️  ENSEMBLE predict failed: {e}")
                asset_report['ensemble'] = {'mu': 0.0, 'sigma': 1.0, 'regime': 'MID'}
        else:
            asset_report['ensemble'] = {'mu': 0.0, 'sigma': 1.0, 'regime': 'MID'}

        asset_report['copula'] = {'tail_dependence': 'N/A_por_activo'}
        asset_report['rl_allocation'] = {'action': 'HOLD', 'confidence': 0.5}

        hurst_col = f'{ticker}_hurst'
        asset_report['hurst'] = float(df_features[hurst_col].iloc[-1]) if hurst_col in df_features.columns else 0.5

        asset_report['multi_window_signals'] = {}
        windows_list = [
            ('short', self.window_sizes['short']), ('medium', self.window_sizes['medium']),
            ('long', self.window_sizes['long']), ('strategic', self.window_sizes['strategic'])
        ]
        for window_name, ws in windows_list:
            if len(X_filtered) < ws:
                continue
            signals = {
                'xgb': xgb_signal, 'causal': causal_impact,
                'ensemble_mu': asset_report['ensemble']['mu']
            }
            asset_report['multi_window_signals'][window_name] = {
                'window_size': ws, 'data_points': ws,
                'signals': signals, 'agreement': np.mean(list(signals.values()))
            }

        return asset_report

    def _build_signals(self, portfolio_intelligence: Dict) -> Dict[str, Tuple[float, float]]:
        signals = {}
        for ticker, report in portfolio_intelligence.items():
            ensemble = report.get('ensemble', {'mu': 0.0, 'sigma': 1.0})
            signals[ticker] = (ensemble.get('mu', 0.0), ensemble.get('sigma', 1.0))
        return signals

    def _build_market_state(self, portfolio_intelligence: Dict) -> Dict:
        if not portfolio_intelligence:
            return {'hurst': 0.5, 'volatility': 0.02, 'crash_prob': 0.1}

        hursts = [r.get('hurst', 0.5) for r in portfolio_intelligence.values()]
        n = len(portfolio_intelligence)
        risk_flags = sum(
            1 for r in portfolio_intelligence.values()
            if r.get('garch_vol', {}).get('crisis_detected', False)
            or r.get('anomaly', {}).get('anomaly_detected', False)
        )

        return {
            'hurst': float(np.mean(hursts)),
            'volatility': 0.02,  # TODO: reemplazar por vol real del GARCH
            'crash_prob': risk_flags / n if n > 0 else 0.1
        }

    def run_portfolio_automation(self, assets_data: Dict[str, pd.DataFrame],
                               train: bool = True) -> Dict:
        """
        Wrapper de conveniencia: fit + predict de punta a punta para uso
        en BACKTEST o corridas puntuales/manuales. NO es el camino que
        usa el loop live (run_engine.py) ni backtest.py -- esos llaman
        directamente a prepare_asset_features + predict_with_cached_experts.
        """
        print("\n" + "="*60)
        print("🚀 INICIANDO ENGINE MULTI-MODELO (INSTITUTIONAL GRADE)")
        print("="*60 + "\n")

        portfolio_intelligence = {}
        portfolio_models = {}

        for ticker, raw_df in assets_data.items():
            print(f"\n⚙️  Procesando Activo: {ticker}")

            if not train:
                print(f"⚠️  train=False requiere model_cache -- usar prepare_asset_features + predict_with_cached_experts directamente. Saltando {ticker}.")
                continue

            X_filtered, y, best_feats, returns, df_features = self.prepare_asset_features(ticker, raw_df)
            if X_filtered is None:
                print(f"⚠️  Datos insuficientes para {ticker}. Saltando...")
                continue

            print(f"✅  Features aprobadas: {len(best_feats)}")
            print(f"🧠  Entrenando tier rápido...")
            fast_models, anomaly = self.fit_fast_tier_experts(X_filtered, y, returns)

            print(f"🧠  Entrenando tier lento...")
            slow_models, anomaly = self.fit_slow_tier_experts(
                X_filtered, y, len(best_feats), anomaly
            )

            asset_report = self.predict_with_cached_experts(
                ticker, X_filtered, returns, df_features, fast_models, slow_models, anomaly
            )

            print(f"✅  Reporte Inteligente {ticker}: {len(asset_report)} señales")
            portfolio_intelligence[ticker] = asset_report
            portfolio_models[ticker] = {
                'fast': fast_models, 'slow': slow_models,
                'anomaly_detector': anomaly, 'best_feats': best_feats
            }

        print("\n💰 FASE 5: ASIGNACIÓN DE CAPITAL")
        print("-" * 60)

        signals = self._build_signals(portfolio_intelligence)
        market_state = self._build_market_state(portfolio_intelligence)
        df_master = pd.DataFrame({
            ticker: {'mu': s[0], 'sigma': s[1]} for ticker, s in signals.items()
        }).T

        final_allocations = self.router.allocate_capital(signals, df_master, market_state)
        print(f"✅  Pesos Finales del Portfolio:\n{final_allocations}")

        return {
            'allocations': final_allocations,
            'intelligence': portfolio_intelligence,
            'models': portfolio_models
        }