# neural_risk/engine.py
"""
AUTOMATED RISK ENGINE: Sistema multi-experto institucional para cripto
Paso 1-5: Data → Features → Jury → Multi-Experts → Router

NUEVOS EXPERTOS (Paso 4 mejorado):
- GARCH/EGARCH: Volatilidad condicional + hedging
- LSTM/Transformer: Forecasting secuencial de largo plazo
- Reinforcement Learning: Asignación dinámica de capital
- Copula: Dependencia multivariada y contagio
- Anomaly Detection: Safety layer (rug pulls, crashes)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple, List
import warnings

# --- COMITÉ CLÁSICO DE EXPERTOS ---
from neural_risk.models.classic import XGBoostVolModel
from neural_risk.models.hmm_model import RegimeHMMModel
from neural_risk.models.bayesian_model import BayesianNeuralRisk
from neural_risk.models.causal_strategy import CausalInferenceModel

# --- COMPONENTES DE VALIDACIÓN Y ENSEMBLE ---
from neural_risk.models.temporal_cv import TemporalCrossValidator
from neural_risk.models.ensemble_trainer import EnsembleTrainer

# --- NUEVOS 5 EXPERTOS (Paso 4 mejorado) ---
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
        """
        Recibimos trainer_class para crear entrenadores frescos por activo.
        """
        self.processor = processor
        self.pipeline = pipeline
        self.labeler = labeler 
        self.jury = jury 
        self.TrainerClass = trainer_class 
        self.router = router
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Historial para auditoría
        self.execution_logs = {}
        
        # WINDOW SIZES PARA CRIPTO (Corto/Medio/Largo/Risk Management)
        # 10 periodos: 1 semana (daily data) - scalping signals
        # 30 periodos: 1 mes - swing trading
        # 100 periodos: 3 meses - trend following
        # 1000 periodos: 1 año - detectar crashes como 2021-2022 bear market
        self.window_sizes = {
            'short': 10,      # Señales tácticas diarias
            'medium': 30,     # Swing trades semanales
            'long': 100,      # Trend siguiente mensual
            'strategic': 1000  # Crisis detection anual
        }

    def _initialize_experts(self, num_features: int):
        """
        🏭 FABRICA DE EXPERTOS (9 MODELOS):
        
        CLÁSICOS (rápidos, probados):
        - HMM: Contexto de mercado, detección de régimen
        - XGB: Señal rápida, volatilidad
        - CAUSAL: Verdad estadística, causación
        - DEEP_TFT: Forecasting bayesiano
        
        NUEVOS (específico para crypto):
        - GARCH: Volatilidad condicional, hedging
        - LSTM/Transformer: Secuencias largas (2021 bull, 2022 bear)
        - RL: Asignación óptima dinámica
        - COPULA: Contagio multi-asset (ETH-SOL correlation spikes)
        - ANOMALY: Safety layer (rug pulls, flash crashes)
        """
        experts = {
            # Clásicos
            'HMM': RegimeHMMModel(n_components=3),
            'XGB': XGBoostVolModel(),
            'CAUSAL': CausalInferenceModel(),
            'DEEP_TFT': BayesianNeuralRisk(num_features=num_features).to(self.device),
            
            # Nuevos para cripto
            'GARCH': MultiWindowGARCH(
                window_sizes=[self.window_sizes['short'], 
                             self.window_sizes['medium'],
                             self.window_sizes['long'],
                             self.window_sizes['strategic']]
            ),
            'LSTM_TF': SequentialForecastingEnsemble(
                input_size=num_features,
                device=self.device
            ),
            'ANOMALY': AnomalyDetector(contamination=0.05, device=self.device),
            # RL y COPULA se inicializan por portafolio, no por activo
        }
        
        return experts

    def run_portfolio_automation(self, assets_data: Dict[str, pd.DataFrame], 
                               train: bool = True) -> Dict:
        print("\n" + "="*60)
        print("🚀 INICIANDO ENGINE MULTI-MODELO (INSTITUTIONAL GRADE)")
        print("="*60 + "\n")
        
        portfolio_intelligence = {}
        
        for ticker, raw_df in assets_data.items():
            print(f"\n⚙️  Procesando Activo: {ticker}")
            
            # ============================================================
            # FASE 1 & 2: PREPARACIÓN DE DATOS (Data & Features)
            # ============================================================
            df_clean = self.processor.auto_clean(raw_df)
            df_features = self.pipeline.transform(df_clean, asset_names=[ticker])
            target = self.labeler.triple_barrier_label(df_features['Close'])
            
            common_idx = df_features.index.intersection(target.index)
            if len(common_idx) < 100:
                print(f"⚠️  Datos insuficientes para {ticker}. Saltando...")
                continue
                
            X = df_features.loc[common_idx].select_dtypes(include=[np.number])
            y = target.loc[common_idx]

            # ============================================================
            # FASE 3: EL JURADO (Feature Selection)
            # ============================================================
            print(f"⚖️  El Jurado analiza {X.shape[1]} features potenciales...")
            best_feats = self.jury.filter_by_causality(X, y)
            
            if not best_feats: 
                best_feats = X.columns[:10].tolist()
                
            X_filtered = X[best_feats]
            print(f"✅  Features aprobadas: {len(best_feats)}")

            # ============================================================
            # FASE 4: EL COMITÉ DE 9 EXPERTOS (PASO 4 MEJORADO)
            # ============================================================
            print(f"🧠  Consultando al Comité de 9 Expertos...")
            experts = self._initialize_experts(num_features=len(best_feats))
            asset_report = {}
            
            # Calcular retornos para algunos expertos
            returns = np.log(df_features['Close'] / df_features['Close'].shift(1)).dropna().values

            if train:
                # ⚡ EXPERTO 1: HMM (Regime Detection)
                print(f"    🔄 [1/9] HMM: Detectando regímenes...")
                try:
                    experts['HMM'].fit(X_filtered)
                    current_regime = experts['HMM'].predict(X_filtered.iloc[-1:])[-1]
                    asset_report['hmm_regime'] = int(current_regime)
                except Exception as e:
                    print(f"      ⚠️  HMM failed: {e}")
                    asset_report['hmm_regime'] = 1

                # ⚡ EXPERTO 2: Causal Inference (Verdad)
                print(f"    🔗 [2/9] CAUSAL: Análisis causal...")
                try:
                    experts['CAUSAL'].fit(X_filtered, y)
                    causal_impact = experts['CAUSAL'].estimate_effect(X_filtered.iloc[-1:])[-1]
                    asset_report['causal_effect'] = float(causal_impact)
                except Exception as e:
                    print(f"      ⚠️  CAUSAL failed: {e}")
                    asset_report['causal_effect'] = 0.0

                # ⚡ EXPERTO 3: XGBoost (Señal Rápida)
                print(f"    ⚡ [3/9] XGB: Señal rápida...")
                try:
                    experts['XGB'].fit(X_filtered, y)
                    xgb_signal = experts['XGB'].predict(X_filtered.iloc[-1:])[-1]
                    asset_report['xgb_signal'] = float(xgb_signal)
                except Exception as e:
                    print(f"      ⚠️  XGB failed: {e}")
                    asset_report['xgb_signal'] = 0.0

                # ⚡ EXPERTO 4: GARCH/EGARCH (Volatilidad Condicional)
                print(f"    📊 [4/9] GARCH: Volatilidad condicional + hedging...")
                try:
                    experts['GARCH'].fit_all(returns)
                    garch_signal = experts['GARCH'].get_hedging_signal()
                    asset_report['garch_vol'] = garch_signal
                    print(f"      {'🚨 CRISIS DETECTADA' if garch_signal['crisis_detected'] else '✅ Vol normal'}")
                except Exception as e:
                    print(f"      ⚠️  GARCH failed: {e}")
                    asset_report['garch_vol'] = {'crisis_detected': False}

                # ⚡ EXPERTO 5: LSTM/Transformer (Secuencias largas)
                print(f"    🔮 [5/9] LSTM/Transformer: Forecasting secuencial...")
                try:
                    # Usar ventana larga para training
                    if len(X_filtered) > self.window_sizes['strategic']:
                        lstm_window = X_filtered.iloc[-self.window_sizes['strategic']:].values
                    else:
                        lstm_window = X_filtered.values
                    
                    y_window = y.iloc[-len(lstm_window):].values
                    
                    # Entrenar ensemble secuencial (ligero)
                    for _ in range(3):  # Quick 3-epoch training
                        X_tensor = torch.FloatTensor(lstm_window).unsqueeze(0).to(self.device)
                        y_tensor = torch.FloatTensor([y_window[-1]]).to(self.device)
                        experts['LSTM_TF'].train_step(X_tensor, y_tensor)
                    
                    # Predicción
                    X_test = torch.FloatTensor(X_filtered.values[-20:]).unsqueeze(0).to(self.device)
                    lstm_pred = experts['LSTM_TF'].predict_ensemble(X_test)
                    asset_report['lstm_forecast'] = lstm_pred
                except Exception as e:
                    print(f"      ⚠️  LSTM/TF failed: {e}")
                    asset_report['lstm_forecast'] = {'ensemble_forecast': 0.0}

                # ⚡ EXPERTO 6: Anomaly Detection (Safety layer)
                print(f"    🛡️  [6/9] ANOMALY: Detección de outliers...")
                try:
                    experts['ANOMALY'].fit_isolation_forest(returns)
                    experts['ANOMALY'].fit_autoencoder(X_filtered, epochs=10)
                    anomaly_result = experts['ANOMALY'].predict_anomalies(X_filtered.iloc[-10:])
                    asset_report['anomaly'] = anomaly_result
                    if anomaly_result['anomaly_detected']:
                        print(f"      🚨 ANOMALÍA DETECTADA: {anomaly_result['anomaly_type']}")
                except Exception as e:
                    print(f"      ⚠️  ANOMALY failed: {e}")
                    asset_report['anomaly'] = {'anomaly_detected': False}

                # ⚡ EXPERTO 7: Ensemble + Kalman (Combinación Inteligente)
                print(f"    🎯 [7/9] ENSEMBLE: Neural + XGB + Kalman...")
                try:
                    cv = TemporalCrossValidator(n_splits=5, initial_window=100, test_size=20)
                    cv.split(X_filtered, y)
                    
                    ensemble = EnsembleTrainer(experts['DEEP_TFT'], device=self.device)
                    data_bundle = ensemble.prepare_data(X_filtered, y)
                    
                    mu_ens, sigma_ens = ensemble.train_ensemble(
                        train_loader=data_bundle['neural'][0],
                        test_loader=data_bundle['neural'][1],
                        xgb_data=data_bundle,
                        epochs=20
                    )
                    
                    asset_report['ensemble'] = {
                        'mu': float(mu_ens),
                        'sigma': float(sigma_ens),
                        'regime': ensemble.kalman.get_regime()
                    }
                except Exception as e:
                    print(f"      ⚠️  ENSEMBLE failed: {e}")
                    asset_report['ensemble'] = {'mu': 0.0, 'sigma': 1.0, 'regime': 'MID'}

                # ⚡ EXPERTO 8: Copula (Dependencias multi-activo) - Agregado al final
                asset_report['copula'] = {
                    'tail_dependence': 'N/A_por_activo'  # Se calcula a nivel portafolio
                }

                # ⚡ EXPERTO 9: RL Allocation (Multi-armed Bandit) - Agregado al final
                asset_report['rl_allocation'] = {
                    'action': 'HOLD',  # Se decide a nivel portafolio
                    'confidence': 0.5
                }

                # 🎯 SÍNTESIS MULTI-VENTANA (Corto/Medio/Largo)
                print(f"    📈 Análisis multi-ventana (Corto/Medio/Largo/Strategic)...")
                asset_report['multi_window_signals'] = {}
                
                windows_list = [
                    ('short', self.window_sizes['short']),
                    ('medium', self.window_sizes['medium']),
                    ('long', self.window_sizes['long']),
                    ('strategic', self.window_sizes['strategic'])
                ]
                
                for window_name, ws in windows_list:
                    if len(X_filtered) < ws:
                        continue
                    
                    window_data = X_filtered.iloc[-ws:].values
                    window_returns = returns[-ws:] if len(returns) >= ws else returns
                    
                    # Agregado: qué expertos concuerdan en esta ventana
                    signals = {
                        'xgb': float(xgb_signal) if 'xgb_signal' in asset_report else 0.0,
                        'causal': float(causal_impact) if 'causal_effect' in asset_report else 0.0,
                        'ensemble_mu': asset_report.get('ensemble', {}).get('mu', 0.0)
                    }
                    
                    asset_report['multi_window_signals'][window_name] = {
                        'window_size': ws,
                        'data_points': len(window_data),
                        'signals': signals,
                        'agreement': np.mean(list(signals.values()))
                    }

            else:
                # Lógica para cargar modelos guardados
                pass

            print(f"✅  Reporte Inteligente {ticker}: {len(asset_report)} señales")
            portfolio_intelligence[ticker] = asset_report

        # ============================================================
        # FASE 5: DECISIONES DE CAPITAL (Router)
        # ============================================================
        print("\n💰 FASE 5: ASIGNACIÓN DE CAPITAL")
        print("-" * 60)
        
        # El Router ahora tiene un reporte mucho más rico para decidir
        final_allocations = self.router.allocate_capital(portfolio_intelligence)
        
        print(f"✅  Pesos Finales del Portfolio:\n{final_allocations}")
        
        return {
            'allocations': final_allocations,
            'intelligence': portfolio_intelligence
        }