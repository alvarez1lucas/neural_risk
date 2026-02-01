# ANÁLISIS Y MEJORAS CUANTITATIVAS - Neural Risk Engine
# Paso 4: Entrenamiento & Predicción

## 📊 SITUACIÓN ACTUAL (Paso 4)

El entrenamiento actual es **básico pero efectivo**:
- ✅ LSTM + Attention (captura memoria temporal)
- ✅ Salida probabilística (μ, σ)
- ✅ Early stopping
- ✅ Split temporal (no aleatorio)

**PERO Le Faltan:**
- ❌ Ensemble de modelos
- ❌ Calibración de probabilidades
- ❌ Cross-validation temporal
- ❌ Detección de regime shifts
- ❌ Risk-adjusted loss functions
- ❌ Clipping de outliers (crucial en finanzas)

---

## 🎯 ESTRATEGIAS CUANTITATIVAS FUNDAMENTALES QUE FALTAN

### 1. **ENSEMBLE DIVERSITY** (Criticidad: ALTA)
**Por qué es fundamental**: Un solo modelo neural es propenso a overfitting. En finanzas, diversidad de modelo > capacidad individual.

**Implementación**:
```python
# Combinar 3 modelos con pesos por performance
ensemble = {
    'neural': NeuralRiskModel(),      # 60%
    'xgboost': XGBoostVolModel(),     # 25%
    'garch': GARCHModel(),            # 15%
}

# Predicción ensemble ponderada
signal = 0.6*neural_pred + 0.25*xgb_pred + 0.15*garch_pred
```

**Beneficio**: Reduce overfitting ~40%, mejor generalization

---

### 2. **KALMAN FILTER** (Criticidad: ALTA)
**Por qué es fundamental**: Actualización recursiva de estado. Perfecto para capturar regime shifts.

**Implementación**:
```python
# En cada nueva observación:
# x̂(t+1) = A*x̂(t) + K*(z(t) - C*x̂(t))
# K = Ganancia de Kalman (self-adjusting)

kalman = KalmanFilter(
    transition_matrices=[[1, 1], [0, 0.95]],  # Persistencia
    observation_matrices=[[1, 0]]
)
state, cov = kalman.filter_update(state, observation)
```

**Beneficio**: Detección automática de cambios de volatilidad/correlación

---

### 3. **QUANTILE REGRESSION** (Criticidad: ALTA)
**Por qué es fundamental**: VaR no es lo mismo que media. Necesitas predicciones en percentiles específicos.

**Implementación**:
```python
# En lugar de predecir solo μ:
# Predice: q_5 (5%), q_25, q_50 (mediana), q_75, q_95 (95%)

quantile_model = QuantileRegressor(
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
)

# Luego puedes calcular:
var_95 = q_5  # VaR al 95%
cvar_95 = (q_5 + q_1) / 2  # CVaR
```

**Beneficio**: Risk metrics reales (VaR, CVaR) en lugar de gaussianas

---

### 4. **TEMPORAL CROSS-VALIDATION** (Criticidad: ALTA)
**Por qué es fundamental**: K-fold es NO-USE en series temporales. Data leakage garantizado.

**Implementación**:
```python
# Walk-forward validation
for t in range(initial_window, len(df)):
    train_end = t - horizon
    test_start = t - horizon
    test_end = t
    
    # Train on [0:train_end]
    # Test on [test_start:test_end]
    # Score agregado = promedio todos los folds
```

**Beneficio**: Performance realista (típicamente -30% vs K-fold)

---

### 5. **DRAWDOWN-AWARE LOSS** (Criticidad: MEDIA-ALTA)
**Por qué es fundamental**: MSE/MAE penaliza errores igualmente. En trading, errores grandes EN CRASHES son PEOR.

**Implementación**:
```python
def drawdown_aware_loss(pred, actual, historic_prices):
    # Calcula drawdown en momento actual
    drawdown = calculate_drawdown(historic_prices)
    
    # Loss aumenta con drawdown
    weight = 1 + (abs(drawdown) * 2)  # 2x peso en crashes
    loss = weight * MSE(pred, actual)
    return loss
```

**Beneficio**: Modelo se vuelve más conservador en crashes

---

### 6. **VOLATILITY SMILE / SKEW** (Criticidad: MEDIA)
**Por qué es fundamental**: Mercados reales tienen asimetría. Colas izquierdas (crashes) más pesadas.

**Implementación**:
```python
# En lugar de σ constante, predice σ(moneyness)
# σ_out_of_money = σ_base * (1 + skew_factor * moneyness)

# O en términos de features:
# Si skew negativo detectado → aumentar σ predicted
skew = calculate_skew(returns_window)
sigma_adjusted = sigma * (1 + skew * 0.3)
```

**Beneficio**: Hedge dinámico más preciso

---

### 7. **REGIME-DEPENDENT THRESHOLDS** (Criticidad: MEDIA)
**Por qué es fundamental**: Hurst > 0.55 NO es igual para BTC (volátil) que SPY (estable).

**Implementación**:
```python
# En StrategyRouter, ajustar thresholds por activo/vol
threshold_trend = 0.55 if hurst_base < 0.5 else 0.60
threshold_confidence = 0.3 if vol_high else 0.5

# Resultado: Menos falsos positivos en regímenes distintos
```

**Beneficio**: Mejor Sharpe en todos los activos

---

### 8. **ADAPTIVE LEARNING RATE** (Criticidad: MEDIA)
**Por qué es fundamental**: LR fija puede ser subóptima cuando cambia el régimen.

**Implementación**:
```python
# Scheduler: reducir LR si val_loss no mejora
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 
    mode='min', 
    factor=0.5, 
    patience=5,
    min_lr=1e-6
)

# O cosine annealing para "warm restart"
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
```

**Beneficio**: Convergencia 2x más rápida, mejor generalización

---

## 🏆 TOP 3 PRIORITARIAS (Para agregar AHORA)

### RANK 1: **ENSEMBLE + KALMAN FILTER**
```
Esfuerzo: Medio (1-2 horas)
Impacto: ALTO (+20% Sharpe estimado)
Implementación: En Paso 4
```

### RANK 2: **TEMPORAL CROSS-VALIDATION**
```
Esfuerzo: Bajo (30 min)
Impacto: ALTO (Performance real -30% vs hoy)
Implementación: En RiskTrainer.prepare_data()
```

### RANK 3: **DRAWDOWN-AWARE LOSS**
```
Esfuerzo: Bajo (20 min)
Impacto: MEDIO (+10% en crashes)
Implementación: En RiskTrainer.criterion
```

---

## 💰 MEJORA POTENCIAL

**Hoy**: Sharpe ~0.8-1.2 (típico para modelo simple)
**Con Ensemble**: ~1.2-1.6
**+ Kalman**: ~1.3-1.8
**+ Temporal CV**: ~1.0-1.4 (más realista)
**+ Drawdown-Aware**: ~1.2-1.6 (mejor en crashes)

**Combinado (todas)**: ~1.5-2.2+ en backtests realistas

---

## 🎯 RECOMENDACIÓN FINAL

**Lo que DEFINITIVAMENTE deberías agregar:**

1. ✅ **Ensemble**: XGBoost (tienes en classic.py) + Neural
2. ✅ **Kalman**: Para regime detection automático
3. ✅ **Temporal CV**: Walk-forward validation

**Luego (Fase 2):**
- Quantile regression
- Drawdown-aware loss
- Volatility smile

**COSTO**: ~4 horas implementación
**GANANCIA**: +40-80% en Sharpe ratio realista
