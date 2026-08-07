# Neural Risk Engine

Motor de trading algorítmico multi-experto para cripto, con arquitectura institucional de 5 capas: ingesta de datos → ingeniería de features → selección estadística de features → comité de 9 modelos → agente de decisión con aprendizaje adaptativo (Thompson Sampling) y hedging a nivel portafolio.

## Arquitectura

```mermaid
flowchart TB
    subgraph L1["Paso 1 · Datos"]
        A1[BinanceLoader / YahooFinanceLoader] --> A2[DataProcessor<br/>limpieza, tipos, timezone]
    end

    subgraph L2["Paso 2 · Feature Engineering"]
        B1[RiskFeaturePipeline<br/>~30 features: volatilidad, microestructura,<br/>order flow, cointegración, HMM, anomaly score]
    end

    subgraph L3["Paso 3 · Cortex"]
        C1[FeatureJury<br/>Granger + Mutual Info + Lasso<br/>consenso 2 de 3 jueces]
        C2[CausalSelector<br/>causalidad bidireccional]
    end

    subgraph L4["Paso 4 · Comité de 9 Expertos"]
        direction LR
        D1[HMM<br/>régimen]
        D2[XGBoost<br/>señal rápida]
        D3[Causal<br/>inference]
        D4[GARCH<br/>volatilidad]
        D5[Ensemble<br/>Neural+XGB+Kalman]
        D6[LSTM/Transformer<br/>forecasting]
        D7[Anomaly<br/>Detector]
        D8[Copula*]
        D9[RL*]
    end

    subgraph L5["Paso 5 · Agente + Ejecución"]
        E1[PortfolioAgent<br/>Thompson Sampling, Kelly, Stop-Loss dinámico]
        E2[StrategyRouter<br/>asignación de capital]
        E3[PortfolioHedgeOptimizer<br/>Differential Evolution sobre CVaR]
        E4[OrderExecutor<br/>circuit breakers, rebalanceo real]
    end

    A2 --> B1 --> C1 & C2 --> D1 & D2 & D3 & D4 & D5 & D6 & D7 & D8 & D9
    D1 & D2 & D3 & D4 & D5 & D6 & D7 --> E1
    E1 --> E4
    E1 -.-> E2
    E3 --> E4

    style D8 stroke-dasharray: 5 5
    style D9 stroke-dasharray: 5 5
```
<sub>*Copula y RL son piezas de la librería pensadas para escalabilidad futura — implementadas pero no conectadas al motor activo hoy.</sub>

### Scheduling de entrenamiento (dos velocidades)

| Tier | Expertos | Frecuencia | Por qué |
|---|---|---|---|
| **Rápido** | HMM, XGBoost, Causal, GARCH, Isolation Forest | 1x/día | Modelos deterministas/basados en árboles — baratos de reentrenar |
| **Lento** | Ensemble neuronal, LSTM/Transformer, Autoencoder | cada 5 días | Redes con backprop — costosas de reentrenar |

El ciclo **live** (60-300s) nunca reentrena — solo predice con los modelos cacheados por `train_models.py`. Esto separa por completo el costo de entrenamiento del costo de inferencia.

## Resultados de backtest

![Backtest chart](backtest_chart_BTC.png)

*(Chart generado con `scripts/plot_backtest.py` — equity de la estrategia vs. Buy & Hold, drawdown, y distribución de PnL por trade.)*

| Métrica | Valor |
|---|---|
| Sharpe Ratio | *(completar con tu corrida real)* |
| Sortino Ratio | |
| Max Drawdown | |
| Win Rate | |
| Profit Factor | |
| VaR / CVaR (95%) | |

## Ingeniería y debugging destacado

Este proyecto pasó por una auditoría exhaustiva de principio a fin. Algunos hallazgos representativos:

- **Look-ahead bias en una feature de "delta sintético"**: usaba `series.shift(-1)` (precio del período siguiente) para calcular el valor de "hoy" — invalidaba cualquier backtest que la usara. Corregido a una diferencia hacia atrás (solo pasado/presente).
- **Circuit breaker de riesgo diario roto**: dependía de una tabla que nunca se poblaba (`update_fills()` era un `pass`) — el límite de pérdida diaria nunca se activaba, pase lo que pase. Reconectado con cierre real de stop-loss.
- **Sesgo de Kelly Criterion**: un error de orden de operaciones colapsaba el sizing a 0.05 en toda señal de reversión a la media, sin importar su magnitud real.
- **Reentrenamiento constante**: el motor reentrenaba los 9 modelos *desde cero en cada ciclo de 60s* — rediseñado con scheduling por tiers (arriba) para separar entrenamiento de inferencia.
- **Escalabilidad de imports**: el `__init__.py` del paquete importaba todo el árbol de forma ansiosa — cualquier submódulo liviano exigía instalar todas las dependencias pesadas (torch, xgboost, hmmlearn, arch). Reescrito con imports defensivos por símbolo.

## Stack técnico

`Python` · `PyTorch` · `XGBoost` · `hmmlearn` · `arch (GARCH)` · `statsmodels` · `scikit-learn` · `SQLite` · `ccxt` (Binance) · `scipy.optimize` (Differential Evolution) · `matplotlib`

## Cómo correrlo

```bash
pip install -r requirements.txt

# Backtest sobre datos históricos
python scripts/backtest.py --asset BTC --data data/BTC_USD_data.csv
python scripts/plot_backtest.py --asset BTC --data data/BTC_USD_data.csv

# Sistema en vivo (4 procesos independientes, comunicados por SQLite)
python scripts/run_data_fetcher.py    # Layer 1: ingesta
python scripts/train_models.py        # Layer 2: entrenamiento por tiers
python scripts/run_engine.py          # Layer 3-5: predicción + decisión
python scripts/run_executor.py        # Layer 6: ejecución + circuit breakers
python scripts/run_portfolio_hedge.py # Hedging a nivel portafolio (asesor)
```

## Estado del proyecto

Ver [`STATUS.md`](STATUS.md) para el detalle completo de qué está validado, qué quedó documentado como limitación conocida, y qué es alcance futuro de la librería.
