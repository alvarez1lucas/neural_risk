# STATUS DEL PROYECTO — fuente de verdad

Este archivo (y el resto del árbol en este sandbox) es el estado REAL
y VERIFICADO de cada archivo, persistido en disco. Antes de mostrar
cualquier archivo modificado, se lee/edita ACÁ (no de memoria de la
conversación) y se re-verifica sintaxis antes de presentarlo.

## ✅ Archivos con bugs reales arreglados

| Archivo | Bug arreglado |
|---|---|
| `neural_risk/engine.py` | Import roto, método jury incorrecto, mismatch de columnas, NameError latente, reentrena-todo-cada-ciclo → reestructurado en prepare/fit_fast/fit_slow/predict |
| `neural_risk/agents/portfolio_agent.py` | Anomaly no-op, Kelly hardcodeado, feedback desconectado, hedging no ponderado por trust, fractional_kelly/max_position_size ignorados |
| `neural_risk/agents/strategy_router.py` | Kelly colapsaba a piso en señales negativas |
| `neural_risk/models/anomaly_detection.py` | Mismatch de dimensiones isolation forest vs autoencoder |
| `neural_risk/models/lstm_transformer.py` | Orden de argumentos de GaussianNLLLoss invertido |
| `neural_risk/models/ensemble_trainer.py` | `kalman.update()` nunca se llamaba |
| `neural_risk/models/hmm_model.py` | Sin fallback si jurado no aprueba columnas hurst/garch |
| `neural_risk/data/data_processor.py` | Métodos duplicados (`_optimize_types`, `get_returns`) |
| `scripts/run_executor.py` | Circuit breaker de daily loss muerto, sizing sobre capital inicial, stop-loss no ejecutado, intervalo hardcodeado, dedup por fecha en vez de status. AHORA: ejecuta rebalanceo real (parcial) leyendo `portfolio_hedge_recommendations` -- aumenta exposición con orden nueva, reduce exposición cerrando parcialmente órdenes FIFO existentes |
| `scripts/run_data_fetcher.py` | Faltaba columna `open`, intervalo hardcodeado. AHORA: conecta `BinanceLoader` real (ccxt) detrás de `config['exchanges']['use_mock']` (default true, no rompe nada existente) |
| `scripts/run_engine.py` | Reescrito: solo predice, nunca fitea, carga cache de `train_models.py` |
| `scripts/train_models.py` | Reescrito: pipeline real (antes era incompatible con engine.py), scheduling por tier |
| `scripts/backtest.py` | Nuevo, walk-forward real |
| `config/model_schedule.yaml` | Nuevo |

## 🟢 Revisados, sin bugs encontrados (sin cambios)

- `neural_risk/models/classic.py` (XGBoostVolModel)
- `neural_risk/models/garch_volatility.py` (GARCH/MultiWindowGARCH)
- `neural_risk/models/base.py`, `layers.py`, `risk_model.py`
- `neural_risk/cortex/feature_jury.py` (solo el CALL SITE en engine.py estaba mal)
- `neural_risk/data/labeling.py`

## 🟡 Menores, identificados pero NO arreglados a propósito

- `neural_risk/models/causal_strategy.py`: shockea todas las columnas por igual al 1% (metodológicamente flojo, no rompe)
- `neural_risk/models/temporal_cv.py`: folds calculados en `fit_slow_tier_experts` pero nunca usados (TODO)
- `neural_risk/models/bayesian_model.py`: `predict_with_uncertainty` (MC-Dropout) nunca se llama, capacidad orfana

## ⚪ Diferido a propósito (decisión explícita del usuario, no bugs)

- `neural_risk/models/reinforcement_learning.py`: alcance futuro de librería. `MarketEnvironment` no es un `gym.Env` válido — roto si se intentara usar hoy.
- `neural_risk/models/copula_expert.py`: alcance futuro de librería (risk management). Mismatch conceptual n_assets vs features, sin resolver.
- `neural_risk/cortex/causal_selector.py`: huérfano, revisar en ronda de Paso 3.
- `neural_risk/models/trainer.py` (RiskTrainer): huérfano, casi duplicado de EnsembleTrainer, candidato a borrar.

## ✅ Recibidos, implementados y CONECTADOS (ronda 3)

| Archivo | Estado real |
|---|---|
| `neural_risk/data/loaders.py` | Implementado: `BinanceLoader` (ccxt) + `YahooFinanceLoader` (yfinance). **Conectado** a `run_data_fetcher.py` (detrás de `exchanges.use_mock`) |
| `neural_risk/optimization/hedging.py` | Implementado: `PortfolioHedgeOptimizer` (Differential Evolution sobre Expected Shortfall vía `RiskAnalytics`). **Conectado** vía `scripts/run_portfolio_hedge.py` (nuevo, servicio asesor no auto-ejecutable) |
| `scripts/run_portfolio_hedge.py` | Servicio asesor: calcula pesos óptimos entre activos, compara contra posiciones reales del executor, guarda recomendación en `portfolio_hedge_recommendations`. Corre 1x/día. **Ejecución real conectada** en `run_executor.execute_rebalance()` (ver arriba) |
| `neural_risk/metrics/performance.py` | Implementado, sin bugs. **Conectado**: `PortfolioAgent.get_portfolio_metrics()` usa `PerformanceMetrics.max_drawdown` sobre una curva de equity real (nueva: `_build_equity_curve()`) |
| `neural_risk/metrics/risk_analytics.py` | Implementado, sin bugs. **Conectado**: `get_portfolio_metrics()` agrega `var_95`/`cvar_95` vía `RiskAnalytics`. También usado por `PortfolioHedgeOptimizer` |
| `neural_risk/metrics/technical.py` | Implementado, sin bugs. **Sigue huérfano** — nadie lo importa todavía (posible duplicación con `RiskFeaturePipeline`, ahora que la tenemos completa se podría auditar) |
| `neural_risk/data/feature_engineering.py` (RiskFeaturePipeline) | **Recibido completo** (ronda 3). 2 bugs reales arreglados: look-ahead bias en `synthetic_delta` (crítico), off-by-one en `dynamic_volume_profile_stats`/`rolling_coint_spread`. 2 features-ruido documentadas sin arreglar (`book_skew_asymmetry`, `autoencoder_bottleneck` — decisión de diseño pendiente) |

## 🐛 Bug nuevo encontrado en esta ronda

- `PortfolioAgent.portfolio_value` nunca se actualizaba tras cerrar posiciones (quedaba congelado en `initial_capital` para siempre) → el sizing de Kelly en `backtest.py` nunca reaccionaba a ganancias/pérdidas reales. **Arreglado**: `close_position()` ahora actualiza `portfolio_value = initial_capital + pnl realizado acumulado`.

## Archivo real pendiente: ninguno

Con `feature_engineering.py` recibido, **ya no queda ningún archivo del árbol original sin contenido real** — el placeholder de esa ruta fue reemplazado por el archivo verdadero.

## Problemas de arquitectura conocidos, sin resolver

1. **`StrategyRouter.allocate_capital` sin consumidor real**: solo se invoca desde `AutomatedRiskEngine.run_portfolio_automation()`, que ni `run_engine.py` ni `backtest.py` llaman (usan `prepare_asset_features` + `predict_with_cached_experts` directamente).
2. **Paso 2 (feature engineering) sin cachear**: `prepare_asset_features` corre `RiskFeaturePipeline.transform()` completo en CADA ciclo (entrenamiento o predicción) — potencialmente el cuello de botella de performance más grande hoy. Diferido a revisión de esa capa.
3. **Config muerto**: `execution.order_type`, `execution.limit_price_deviation`, `signals.staking_min_amount` están en `config.yaml` pero ningún código los lee.
4. **`PortfolioAgent.positions` (libro interno) solo se usa en backtest**: en producción (`run_engine.py`), la fuente de verdad de posiciones es la DB del executor — decisión de arquitectura ya tomada, documentada en el código.

## 🧪 Smoke test REAL ejecutado (no solo revisión estática)

**Limitación del sandbox**: sin acceso a internet, no se pudieron instalar `torch`/`xgboost`/`hmmlearn`/`arch`/`statsmodels`/`ccxt`/`yfinance` acá. Se corrió lo que sí es instalable (`tests/smoke_test_light.py`) y se dejó preparado `tests/smoke_test_full.py` para correr en el entorno del usuario con todas las dependencias (`requirements.txt`).

**`tests/smoke_test_light.py` — corrido de verdad, TODOS los asserts pasaron:**
- `PortfolioAgent.execute_portfolio_decision` con 3 activos sintéticos, sin errores.
- 30 trades simulados → `portfolio_value` se actualiza correctamente (fix verificado con assert, no a ojo).
- Thompson Sampling deja de ser uniforme tras feedback (fix verificado con assert).
- `StrategyRouter.allocate_capital` → pesos suman 1.0.
- Fix del bug de Kelly verificado: `Kelly(mu=-0.05) == Kelly(mu=+0.05)` (antes colapsaba a 0.05 en el caso negativo).
- `PortfolioHedgeOptimizer.optimize` con Differential Evolution real → converge, pesos suman 1.0, `suggest_rebalance` funciona.

**Bugs nuevos encontrados y arreglados durante esta verificación (no por revisión de código, sino por intentar ejecutar):**
1. `neural_risk/models/__init__.py` nunca se había creado (paquete inconsistente).
2. `neural_risk/__init__.py` importaba TODO el árbol de forma ansiosa → cualquier submódulo liviano (ej. `PortfolioAgent`, que solo necesita numpy/pandas/scipy) requería tener instaladas TODAS las dependencias pesadas de la librería para poder importarse. Reescrito con imports defensivos (`try/except` por símbolo, con `ImportWarning` claro).
3. `DataProcessor` importaba `statsmodels` a nivel de módulo aunque solo `check_stationarity()` lo usa -- import movido a local, ahora el resto de la clase funciona sin `statsmodels` instalado.

Estos 3 hallazgos son directamente relevantes al objetivo de escalabilidad: antes, agregar cualquier pieza nueva a la librería heredaba el problema de "necesitás instalar todo para usar una parte".

**Pendiente de correr por el usuario**: ~~`tests/smoke_test_full.py`~~ ✅ **CORRIDO CON ÉXITO en el entorno real del usuario** (Windows, Python 3.11.9, venv limpio con `torch==2.5.1` CPU-only). Pipeline completo con los 9 expertos REALES (no simulados) corrió de punta a punta sin errores: `prepare_asset_features` → `fit_fast_tier_experts` → `fit_slow_tier_experts` → `predict_with_cached_experts` → `PortfolioAgent.execute_portfolio_decision`. Esto es la primera validación end-to-end del sistema completo, no solo revisión estática de código.

**Esto confirma en la práctica:**
- El feature engineering completo de `RiskFeaturePipeline` (con los 2 fixes: look-ahead bias y off-by-one) corre sin crashear.
- Los 9 expertos (HMM, XGB, CAUSAL, GARCH, ENSEMBLE, LSTM_TF, ANOMALY -- COPULA/RL siguen fuera del engine activo por decisión de diseño) se instancian, entrenan y predicen correctamente encadenados.
- La reestructuración fit/predict de `engine.py` (separar entrenamiento caro de predicción barata) funciona de verdad, no solo en el papel.
- El fix de imports defensivos en `neural_risk/__init__.py` + `data_processor.py` no rompió nada al instalar las dependencias reales.


## Presentación / CV (ronda posterior a la validación end-to-end)

- `scripts/plot_backtest.py` -- **nuevo, corrido de verdad en el sandbox** con datos sintéticos (equity/trades/metrics/OHLCV), generó un PNG válido de 1934x1622px sin errores. Genera: equity vs Buy&Hold con marcadores de trades ganadores/perdedores, panel de drawdown, distribución de PnL por trade, caja de métricas (Sharpe/Sortino/MaxDD/VaR/CVaR/WinRate/ProfitFactor).
- `scripts/backtest.py` -- extendido para exportar `backtest_trades_<ASSET>.csv` y `backtest_metrics_<ASSET>.json` (antes solo exportaba la curva de equity), necesarios para que `plot_backtest.py` no tenga que recalcular nada.
- `README.md` -- **nuevo**. Diagrama de arquitectura en Mermaid (renderiza nativo en GitHub, sin necesidad de imagen aparte), tabla de scheduling por tiers, sección de "ingeniería y debugging destacado" (5 bugs representativos de toda la auditoría, pensada como señal de rigor para un reclutador), instrucciones de uso.
- `requirements.txt` -- agregado `matplotlib>=3.7.0`.
- **Recomendación de presentación**: README + diagrama Mermaid + chart embebido en GitHub, priorizado sobre un dashboard hosteado (Streamlit) por costo de mantenimiento cero y ser lo que un reclutador efectivamente abre primero. Streamlit queda como mejora opcional futura, no bloqueante.

## Mejoras del aprendizaje adaptativo (ronda 1 de 3: reward por experto)

**Bug nuevo encontrado y arreglado**: `TradeRecord.close_trade()` no distinguía LONG de SHORT -- para un SHORT, un precio a la baja (ganancia real) se calculaba como pérdida y viceversa. Solo afectaba a `backtest.py` (el executor real ya lo hacía bien por separado). **Verificado con test real**: SHORT entry=100/exit=90 ahora da `pnl=+10.00` (antes daba negativo).

**Fix pedido por el usuario -- precisión del reward de Thompson Sampling**: antes, TODOS los expertos que participaron en una señal recibían el MISMO reward (ganó/perdió el trade AGREGADO), sin importar si el voto INDIVIDUAL de cada uno acertó la dirección real del precio. Esto diluía la señal de aprendizaje. Ahora `record_expert_feedback()` acepta `price_return` (movimiento real de precio, signo puro) y cada experto recibe reward según si SU voto coincidió en signo con lo que realmente pasó. `is_winning` en `ExpertEvaluator` (Sharpe/Sortino/Win Rate por experto) sigue atado al resultado real del trade a propósito -- son preguntas distintas ("¿acertó la dirección?" vs "¿qué tan buenas son las métricas de riesgo cuando participó?"). **Verificado con test dirigido**: dos expertos artificiales (uno siempre acierta dirección, otro siempre se equivoca), participando en la MISMA cantidad de trades ganadores/perdedores agregados -> terminaron con pesos 0.95 vs 0.05 (con el bug viejo, estos dos pesos habrían quedado idénticos, ya que el reward viejo no podía distinguirlos).

Conectado en producción: `run_engine.py::sync_feedback_from_fills()` ahora calcula y pasa `price_return` también.

**Pendiente (rondas 2 y 3, a pedido del usuario)**:
2. Pesos condicionados a régimen (usar `hmm_regime`/`garch_vol.crisis_detected` para mantener sets de pesos DISTINTOS por contexto de mercado, en vez de un único Thompson Sampling global que deriva lento) -- reactivo hoy, esto lo acercaría a "anticipatorio".
3. Variables macro externas (VIX, DXY, SPX, oro vía `YahooFinanceLoader` ya implementado) como columnas de contexto en Paso 1/2, para dar señal fuera del universo cripto.

## Cómo seguir trabajando de forma confiable






1. Antes de mostrar un archivo modificado: `view` la versión actual en este sandbox.
2. Aplicar el cambio con `str_replace` o reescribir con `create_file`/heredoc.
3. Correr `python3 -m py_compile <archivo>` para confirmar sintaxis.
4. Si el cambio afecta una firma pública (nombre de método, argumentos), `grep -rn` el nombre viejo en todo el árbol para encontrar todos los call sites.
5. Recién ahí, pegar el contenido actualizado en el chat.
6. Actualizar este STATUS.md con el cambio.
