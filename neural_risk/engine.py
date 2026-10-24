# neural_risk/engine.py
"""
AUTOMATED RISK ENGINE: Sistema multi-experto institucional para cripto
Paso 1-5: Data → Features → Jury → Multi-Experts → Router

Métodos públicos, pensados para reutilizarse como librería:
- build_features(ticker, raw_df, macro_context=None) -- Paso 1-2. Como
  RiskFeaturePipeline es causal (cada fila usa solo el pasado), se puede
  correr UNA vez sobre todo el histórico y recortar por fecha después.
- select_features(ticker, df_features, cached_best_feats=None) -- Paso 3
  (target + jurado en modo entrenamiento; solo recorte en modo predicción).
- prepare_asset_features(ticker, raw_df, cached_best_feats=None, macro_context=None)
  = build_features + select_features.
- prepare_asset_features_cached(ticker, raw_df, cache, ...) -- usa
  FeatureCache para no recalcular el feature engineering pesado.
- fit_fast_tier_experts(X_filtered, y, returns, existing_anomaly=None)
- fit_slow_tier_experts(X_filtered, y, num_features, anomaly_detector)
- predict_with_cached_experts(ticker, X_filtered, returns, df_features,
                               fast_models, slow_models, anomaly_detector)
- run_portfolio_automation(assets_data, train=True): wrapper de
  conveniencia fit+predict, para backtest/uso puntual.

PENDIENTE:
- TemporalCrossValidator todavía no se usa para entrenar/evaluar.
- StrategyRouter.allocate_capital solo se invoca desde
  run_portfolio_automation (run_engine.py y backtest.py no lo llaman).
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Optional
import warnings

from neural_risk.models.classic import XGBoostVolModel
from neural_risk.models.hmm_model import RegimeHMMModel
from neural_risk.models.bayesian_model import BayesianNeuralRisk
from neural_risk.models.causal_strategy import CausalInferenceModel
from neural_risk.models.ensemble_trainer import EnsembleTrainer
from neural_risk.models.garch_volatility import MultiWindowGARCH
from neural_risk.models.lstm_transformer import SequentialForecastingEnsemble
from neural_risk.models.anomaly_detection import AnomalyDetector

warnings.filterwarnings('ignore')

# Columnas crudas de precio/volumen: no estacionarias, no entran como features.
RAW_COLUMNS = ('Open', 'High', 'Low', 'Close', 'Volume')


class AutomatedRiskEngine:
    def __init__(self, processor, pipeline, labeler, jury, trainer_class, router,
                 seed: int = 42, label_params: Optional[Dict] = None):
        self.processor = processor
        self.pipeline = pipeline
        self.labeler = labeler
        self.jury = jury
        self.TrainerClass = trainer_class
        self.router = router
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.seed = seed
        # Barreras SIMÉTRICAS por defecto: con [2, 1] la barrera de -1σ se
        # toca mucho más seguido que la de +2σ, el target queda sesgado a
        # -1 y un "-1" solo significa "un LONG habría tocado el stop", no
        # "un SHORT gana". Simétrico = target direccional.
        self.label_params = label_params or {'t_events': 5, 'pt_sl': [2, 2]}

        self.window_sizes = {
            'short': 10, 'medium': 30, 'long': 100, 'strategic': 1000
        }
        # GARCH necesita >=50 datos por ventana: las ventanas 10/30 nunca
        # llegaban a fitear y entraban al promedio con valores por defecto.
        self.garch_windows = [100, 250, 1000]
        self.lstm_seq_len = 20
        self.ensemble_window = 10

    # ------------------------------------------------------------------
    # Paso 1-3
    # ------------------------------------------------------------------
    def build_features(self, ticker: str, raw_df: pd.DataFrame,
                       macro_context: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        df_clean = self.processor.auto_clean(raw_df)
        df_prefixed = self.processor.rename_columns(df_clean, ticker)
        df_features = self.pipeline.transform(df_prefixed, asset_names=[ticker])

        if macro_context is not None and not macro_context.empty:
            from neural_risk.data.macro_context import merge_macro_context
            df_features = merge_macro_context(df_features, macro_context)
        return df_features

    def _feature_matrix(self, ticker: str, df_features: pd.DataFrame) -> pd.DataFrame:
        raw_cols = {f'{ticker}_{c}' for c in RAW_COLUMNS}
        X = df_features.select_dtypes(include=[np.number])
        return X[[c for c in X.columns if c not in raw_cols]]

    def select_features(self, ticker: str, df_features: pd.DataFrame,
                        cached_best_feats: Optional[List[str]] = None,
                        fixed_feats: Optional[List[str]] = None) -> Tuple:
        """
        Si cached_best_feats es None -> modo ENTRENAMIENTO: calcula el
        target y corre el jurado desde cero (o usa fixed_feats, si se pasa,
        para reentrenar con el MISMO set de columnas). Retorna y real.
        Si cached_best_feats viene seteado -> modo PREDICCIÓN: salta el
        jurado. Retorna y=None.

        Returns: (X_filtered, y_or_None, best_feats, returns, df_features)
                 o (None, None, None, None, None) si no hay datos suficientes.
        """
        close_col = f'{ticker}_Close'
        returns = np.log(df_features[close_col] / df_features[close_col].shift(1)).dropna().values
        X = self._feature_matrix(ticker, df_features)

        if cached_best_feats is None:
            target = self.labeler.triple_barrier_label(df_features[close_col], **self.label_params)
            common_idx = X.index.intersection(target.index)
            if len(common_idx) < 100:
                return None, None, None, None, None

            X = X.loc[common_idx]
            y = target.loc[common_idx]

            if fixed_feats is not None:
                if any(f not in X.columns for f in fixed_feats):
                    return None, None, None, None, None
                best_feats = list(fixed_feats)
            else:
                best_feats = self.jury.evaluate(X, y)
                if not best_feats:
                    best_feats = X.columns[:10].tolist()

            return X[best_feats], y, best_feats, returns, df_features

        if len(df_features) < 100:
            return None, None, None, None, None
        available_feats = [f for f in cached_best_feats if f in X.columns]
        if len(available_feats) != len(cached_best_feats):
            # Los modelos se entrenaron con exactamente estas columnas: si
            # falta alguna, predecir con un subset daría columnas corridas.
            return None, None, None, None, None
        return X[available_feats], None, available_feats, returns, df_features

    def prepare_asset_features(self, ticker: str, raw_df: pd.DataFrame,
                              cached_best_feats: Optional[List[str]] = None,
                              macro_context: Optional[pd.DataFrame] = None
                              ) -> Tuple:
        df_features = self.build_features(ticker, raw_df, macro_context)
        return self.select_features(ticker, df_features, cached_best_feats)

    def prepare_asset_features_cached(self, ticker: str, raw_df: pd.DataFrame,
                                     cache, cached_best_feats: Optional[List[str]] = None,
                                     macro_context: Optional[pd.DataFrame] = None,
                                     force_refresh: bool = False) -> Tuple:
        """
        Envoltorio de prepare_asset_features() que usa un FeatureCache
        (neural_risk/data/feature_cache.py) para NO recalcular el feature
        engineering pesado si ya se corrió antes con los mismos datos.
        Cachea el df_features (Paso 1-2) y vuelve a correr la selección
        (Paso 3), así el mismo cache sirve para modo entrenamiento y
        predicción.
        """
        has_macro = macro_context is not None and not macro_context.empty
        df_features = None if force_refresh else cache.load(ticker, raw_df, variant=f'macro={has_macro}')
        if df_features is not None:
            print(f"✅  FeatureCache HIT para {ticker} -- se salta el cálculo pesado")
        else:
            print(f"⏳  FeatureCache MISS para {ticker} -- calculando features (puede tardar)...")
            df_features = self.build_features(ticker, raw_df, macro_context)
            cache.save(ticker, raw_df, df_features, variant=f'macro={has_macro}')
        return self.select_features(ticker, df_features, cached_best_feats)

    # ------------------------------------------------------------------
    # Paso 4: FIT
    # ------------------------------------------------------------------
    def fit_fast_tier_experts(self, X_filtered: pd.DataFrame, y: pd.Series,
                             returns: np.ndarray,
                             existing_anomaly: Optional[AnomalyDetector] = None
                             ) -> Tuple[Dict, AnomalyDetector]:
        models = {}

        try:
            hmm = RegimeHMMModel(n_components=3)
            hmm.model.random_state = self.seed
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
            garch = MultiWindowGARCH(window_sizes=self.garch_windows)
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

    def _fit_lstm(self, X_filtered: pd.DataFrame, y: pd.Series, num_features: int,
                  epochs: int = 5, batch_size: int = 64) -> SequentialForecastingEnsemble:
        """
        FIX: antes el LSTM/Transformer "entrenaba" con UNA sola muestra (la
        secuencia completa + solo la última etiqueta, 3 pasos) y predecía
        con ventanas de otro largo. Ahora entrena con todas las ventanas
        de largo seq_len (las mismas que ve en predicción), escaladas con
        un scaler que viaja con el modelo.
        """
        seq_len = self.lstm_seq_len
        lstm_tf = SequentialForecastingEnsemble(input_size=num_features, device=self.device)
        scaler = StandardScaler().fit(np.nan_to_num(X_filtered.values))
        Xs = scaler.transform(np.nan_to_num(X_filtered.values))

        windows = np.lib.stride_tricks.sliding_window_view(Xs, seq_len, axis=0).transpose(0, 2, 1)
        targets = y.values[seq_len - 1:].astype(np.float32)
        X_all = torch.FloatTensor(np.ascontiguousarray(windows))
        y_all = torch.FloatTensor(targets)

        gen = torch.Generator().manual_seed(self.seed)
        for _ in range(epochs):
            perm = torch.randperm(len(X_all), generator=gen)
            for i in range(0, len(X_all), batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) < 2:
                    continue
                lstm_tf.train_step(X_all[idx], y_all[idx])

        lstm_tf.scaler = scaler
        lstm_tf.seq_len = seq_len
        return lstm_tf

    def fit_slow_tier_experts(self, X_filtered: pd.DataFrame, y: pd.Series,
                             num_features: int,
                             anomaly_detector: AnomalyDetector
                             ) -> Tuple[Dict, AnomalyDetector]:
        models = {}
        torch.manual_seed(self.seed)

        try:
            deep_tft = BayesianNeuralRisk(num_features=num_features).to(self.device)
            ensemble = EnsembleTrainer(deep_tft, device=self.device)
            data_bundle = ensemble.prepare_data(X_filtered, y, window_size=self.ensemble_window)
            ensemble.train_ensemble(
                train_loader=data_bundle['neural'][0],
                test_loader=data_bundle['neural'][1],
                xgb_data=data_bundle,
                epochs=20,
                verbose=False
            )
            models['ENSEMBLE_TRAINER'] = ensemble
        except Exception as e:
            print(f"⚠️  ENSEMBLE fit failed: {e}")
            models['ENSEMBLE_TRAINER'] = None

        try:
            models['LSTM_TF'] = self._fit_lstm(X_filtered, y, num_features)
        except Exception as e:
            print(f"⚠️  LSTM/TF fit failed: {e}")
            models['LSTM_TF'] = None

        try:
            anomaly_detector.fit_autoencoder(X_filtered, epochs=10)
        except Exception as e:
            print(f"⚠️  ANOMALY (autoencoder) fit failed: {e}")

        return models, anomaly_detector

    # ------------------------------------------------------------------
    # Paso 4: PREDICT
    # ------------------------------------------------------------------
    def predict_with_cached_experts(self, ticker: str, X_filtered: pd.DataFrame,
                                   returns: np.ndarray, df_features: pd.DataFrame,
                                   fast_models: Dict, slow_models: Dict,
                                   anomaly_detector: Optional[AnomalyDetector]
                                   ) -> Dict:
        asset_report = {'failures': []}
        xgb_signal = 0.0
        causal_impact = 0.0

        # Volatilidad realizada (escala de retornos) -- base para stops.
        asset_report['realized_vol'] = float(np.std(returns[-20:])) if len(returns) >= 20 else float('nan')

        hmm = fast_models.get('HMM')
        asset_report['hmm_regime'] = 1
        if hmm is not None:
            try:
                # FIX: Viterbi sobre UNA sola observación no tiene contexto
                # temporal -- se decodifica una ventana y se toma el último estado.
                asset_report['hmm_regime'] = int(hmm.predict(X_filtered.iloc[-50:])[-1])
            except Exception as e:
                print(f"⚠️  HMM predict failed: {e}")
                asset_report['failures'].append('HMM')

        causal_model = fast_models.get('CAUSAL')
        asset_report['causal_effect'] = 0.0
        if causal_model is not None:
            try:
                causal_impact = float(causal_model.estimate_effect(X_filtered.iloc[-1:])[-1])
                asset_report['causal_effect'] = causal_impact
            except Exception as e:
                print(f"⚠️  CAUSAL predict failed: {e}")
                asset_report['failures'].append('CAUSAL')

        xgb_model = fast_models.get('XGB')
        asset_report['xgb_signal'] = 0.0
        if xgb_model is not None:
            try:
                xgb_signal = float(xgb_model.predict(X_filtered.iloc[-1:])[-1])
                asset_report['xgb_signal'] = xgb_signal
            except Exception as e:
                print(f"⚠️  XGB predict failed: {e}")
                asset_report['failures'].append('XGB')

        garch = fast_models.get('GARCH')
        asset_report['garch_vol'] = {'crisis_detected': False}
        if garch is not None:
            try:
                asset_report['garch_vol'] = garch.get_hedging_signal(recent_returns=returns)
            except Exception as e:
                print(f"⚠️  GARCH predict failed: {e}")
                asset_report['failures'].append('GARCH')

        lstm_tf = slow_models.get('LSTM_TF')
        asset_report['lstm_forecast'] = {'ensemble_forecast': 0.0}
        if lstm_tf is not None:
            try:
                seq_len = getattr(lstm_tf, 'seq_len', self.lstm_seq_len)
                X_seq = np.nan_to_num(X_filtered.values[-seq_len:])
                if hasattr(lstm_tf, 'scaler'):
                    X_seq = lstm_tf.scaler.transform(X_seq)
                X_test = torch.FloatTensor(X_seq).unsqueeze(0).to(self.device)
                asset_report['lstm_forecast'] = lstm_tf.predict_ensemble(X_test)
            except Exception as e:
                print(f"⚠️  LSTM/TF predict failed: {e}")
                asset_report['failures'].append('LSTM')

        asset_report['anomaly'] = {'anomaly_detected': False}
        if anomaly_detector is not None:
            try:
                asset_report['anomaly'] = anomaly_detector.predict_anomalies(
                    X_filtered.iloc[-10:], recent_returns=returns[-10:]
                )
            except Exception as e:
                print(f"⚠️  ANOMALY predict failed: {e}")
                asset_report['failures'].append('ANOMALY')

        ensemble = slow_models.get('ENSEMBLE_TRAINER')
        asset_report['ensemble'] = {'mu': 0.0, 'sigma': 1.0, 'regime': 'MID'}
        if ensemble is not None:
            try:
                if len(returns) >= 20:
                    ensemble.kalman.update(float(np.std(returns[-20:])))
                mu_ens, sigma_ens = ensemble.predict_ensemble(X_filtered.values[-self.ensemble_window:])
                asset_report['ensemble'] = {
                    'mu': float(mu_ens), 'sigma': float(sigma_ens),
                    'regime': ensemble.kalman.get_regime()
                }
            except Exception as e:
                print(f"⚠️  ENSEMBLE predict failed: {e}")
                asset_report['failures'].append('ENSEMBLE')

        hurst_col = f'{ticker}_hurst'
        asset_report['hurst'] = float(df_features[hurst_col].iloc[-1]) if hurst_col in df_features.columns else 0.5

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
        vols = [r.get('realized_vol') for r in portfolio_intelligence.values()]
        vols = [v for v in vols if v is not None and np.isfinite(v)]
        n = len(portfolio_intelligence)
        risk_flags = sum(
            1 for r in portfolio_intelligence.values()
            if r.get('garch_vol', {}).get('crisis_detected', False)
            or r.get('anomaly', {}).get('anomaly_detected', False)
        )

        return {
            'hurst': float(np.mean(hursts)),
            'volatility': float(np.mean(vols)) if vols else 0.02,
            'crash_prob': risk_flags / n if n > 0 else 0.1
        }

    def run_portfolio_automation(self, assets_data: Dict[str, pd.DataFrame],
                               train: bool = True) -> Dict:
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
            fast_models, anomaly = self.fit_fast_tier_experts(X_filtered, y, returns)
            slow_models, anomaly = self.fit_slow_tier_experts(X_filtered, y, len(best_feats), anomaly)

            asset_report = self.predict_with_cached_experts(
                ticker, X_filtered, returns, df_features, fast_models, slow_models, anomaly
            )

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
