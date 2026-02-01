#  Neural Risk Engine v0.2.0

**Automated cryptocurrency risk management and portfolio optimization system**

A production-ready microservices architecture for real-time multi-asset trading with neural ensemble models and portfolio optimization.

---

##  What This System Does

**Input**: Real-time market data (Binance API)  
**Process**: 9 ensemble experts + portfolio optimization  
**Output**: Trading signals with entry/exit levels every 5 minutes  
**Deployment**: 4 independent services running 24/7

```
Market Data → Data Fetcher → Engine (9 Experts) → Executor → Live Trading
                                         ↓
                                   Portfolio Agent
                                   (Kelly Criterion)
```

---

##  Architecture Overview

### 5-Layer Microservices

**1. Data Fetcher** (independent process)
- Fetches market data every 5 minutes from Binance
- Caches and stores in SQLite database
- Handles API rate limits and errors gracefully

**2. Model Training** (daily cron job)
- Trains 9 expert models offline
- Runs daily at 00:00 UTC
- Saves trained models to pickle cache

**3. Engine** (independent process)
- Loads trained models every 5 minutes
- Runs 9 experts in parallel via ThreadPool:
  - 3 classical models (MA, Volatility, RSI)
  - 3 ML models (Random Forest, XGBoost, SVM)
  - 3 statistical models (ARIMA, GARCH, HMM)
- PASO 5: PortfolioAgent combines signals with Kelly Criterion
- Outputs trading decisions with confidence levels

**4. Executor** (independent process)
- Reads trading decisions from database
- Validates position sizes and daily loss limits
- Places orders on Binance API
- Logs all transactions for audit trail

**5. Dashboard** (optional monitoring)
- Real-time portfolio metrics
- PnL tracking and performance charts
- Alert system for risk thresholds

### Database Schema
```sql
market_data           -- Raw OHLCV from Binance
engine_decisions      -- Trading signals with confidence
orders                -- Order execution log
fills                 -- Filled orders with PnL
```

---

## ⚡ Quick Start

### 1. Prerequisites

```bash
# Python 3.9+
python --version

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/config.yaml`:
```yaml
mode: production              # or development/backtest
assets: [BTC, ETH]            # Start with 2
cycle_time: 300               # 5 minutes
position_size_pct: 10         # Max 10% per trade
daily_loss_limit: -5          # Pause if -5% loss
```

### 3. Validate System

```bash
python scripts/pre_deploy_check.py
# Expected output: [OK] PRE-DEPLOY CHECK PASSED - READY TO DEPLOY!
```

### 4. Deploy 4 Services

**Terminal 1** (Data Fetcher):
```bash
python scripts/run_data_fetcher.py
# Expected: "Fetching market data... [OK]" every 300 seconds
```

**Terminal 2** (wait 30 seconds then run Engine):
```bash
python scripts/run_engine.py
# Expected: "Processing signals... [OK]" every 300 seconds
```

**Terminal 3** (wait 30 seconds then run Executor):
```bash
python scripts/run_executor.py
# Expected: "Checking orders... [OK]" every 300 seconds
```

**Terminal 4** (optional Dashboard):
```bash
jupyter notebook notebooks/live_dashboard.ipynb
# Opens live monitoring dashboard
```

### 5. Monitor

```bash
# Watch real-time logs
tail -f logs/neural_risk.log

# Check database growth (should increase every cycle)
ls -lh data/neural_risk.db

# Query trading signals
sqlite3 data/neural_risk.db "SELECT * FROM engine_decisions LIMIT 5;"
```

---

##  What to Expect

### During First Hour
- Data Fetcher: Fetches 5min candlestick every 300s ✓
- Engine: Generates signals with confidence scores ✓
- Executor: Validates and places mock orders (until API keys configured) ✓
- Dashboard: Shows 12+ cycles completed ✓

### Performance Metrics
After 24 hours of live trading, you'll see:
- **Sortino Ratio** (risk-adjusted returns)
- **Win Rate** (% of profitable trades)
- **Profit Factor** (wins ÷ losses)
- **Daily PnL** (exact profit/loss tracking)

### Files to Monitor
```
logs/neural_risk.log          -- Real-time activity log
data/neural_risk.db           -- Trading history (grows each cycle)
data/trained_models.pkl       -- Model cache (updated daily)
config/config.yaml            -- Live configuration
```

---

##  Key Features

✅ **Multi-Asset**: Supports 2-10+ cryptocurrencies  
✅ **Ensemble**: 9 diverse expert models for robust signals  
✅ **Parallel**: ThreadPool-based concurrent processing  
✅ **Automated**: 100% automatic after deployment  
✅ **Scalable**: Easy to add new assets or experts  
✅ **Safe**: Daily loss limits + position size controls  
✅ **Auditable**: Complete transaction logging  
✅ **Real-time**: Decisions every 5 minutes  

---

##  Project Structure

```
neural-risk/
├── neural_risk/
│   ├── data/                 -- Data loading & preprocessing
│   ├── models/               -- ML model implementations
│   ├── metrics/              -- Backtest & risk metrics
│   ├── optimization/         -- Portfolio optimization (Kelly)
│   ├── agents/               -- Strategy routers
│   └── cortex/               -- Causal feature selection
├── scripts/
│   ├── run_data_fetcher.py   -- Service 1: Data collection
│   ├── run_engine.py         -- Service 2: Signal generation
│   ├── run_executor.py       -- Service 3: Order execution
│   ├── train_models.py       -- Daily model training
│   └── pre_deploy_check.py   -- Deployment validation
├── config/
│   └── config.yaml           -- Production configuration
├── data/
│   ├── neural_risk.db        -- SQLite database (auto-created)
│   ├── BTC_USD_data.csv      -- Historical prices
│   └── trained_models.pkl    -- Model cache (auto-created)
├── logs/
│   └── neural_risk.log       -- Activity logs
└── notebooks/
    ├── live_dashboard.ipynb  -- Real-time monitoring
    └── backtest_analysis.ipynb -- Historical analysis
```

---

##  Configuration Guide

### `config/config.yaml`

```yaml
# Mode: production, development, or backtest
mode: production

# Assets to trade (start with 2, scale to 5+)
assets:
  - BTC
  - ETH

# Cycle timing (300s = 5 min, SAFE for free APIs)
cycle_timing:
  data_fetcher: 300      # Fetch market data every 5 min
  engine: 300            # Generate signals every 5 min
  executor: 300          # Check and execute orders every 5 min

# Trading thresholds
signals:
  long_threshold: 0.60   # BUY if consensus > 60%
  short_threshold: -0.60 # SELL if consensus < -60%
  min_confidence: 0.50   # Minimum expert agreement

# Risk management
risk:
  max_position_size: 0.10       # Max 10% per trade
  daily_loss_limit: -0.05       # Pause if -5% daily loss
  use_kelly_criterion: true     # Fractional Kelly (safer)
  kelly_fraction: 0.25          # Use 25% of Kelly (conservative)

# Model training
training:
  schedule: "00:00"             # Run daily at 00:00 UTC
  lookback_days: 180            # Use 6 months of historical data
  validation_split: 0.20        # 80% train, 20% validation
  retraining_frequency: "daily"  # Can be: daily, weekly, monthly
```

---

##  Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'neural_risk'"

**Solution**: Install package in development mode
```bash
cd f:\quant_risk\neural-risk
pip install -e .
```

### Issue: "No module named 'ensemble_trainer'"

**Solution**: Verify installation
```bash
python -c "from neural_risk.models.ensemble_trainer import EnsembleTrainer; print('OK')"
```

### Issue: "API key not configured"

**Solution**: System runs in MOCK mode (simulated orders)
- To use real trading, set `BINANCE_API_KEY` and `BINANCE_API_SECRET` in environment
- For testing, MOCK mode is fine

### Issue: "Database is locked"

**Solution**: Only one engine can write to DB at a time
- Ensure only ONE `run_engine.py` instance is running
- Kill any zombie processes: `taskkill /F /IM python.exe` (Windows)

---

##  Performance Targets

After 30 days of live trading:
- Sortino Ratio: > 1.5
- Win Rate: > 50%
- Profit Factor: > 1.5
- Max Drawdown: < 15%
- Daily Sharpe: > 0.5

---

##  Updating Models

Models are trained daily at 00:00 UTC using the last 180 days of data. To manually retrain:

```bash
python scripts/train_models.py
# Output: [OK] Models saved to data/trained_models.pkl
```

---

##  Development

### Add a New Expert Model

1. Create model class in `neural_risk/models/`
2. Inherit from `BaseModel`
3. Implement `fit()` and `predict()` methods
4. Register in `EnsembleTrainer`

### Add a New Asset

1. Edit `config/config.yaml` assets list
2. Restart all 4 services
3. System automatically handles new asset

### Adjust Signal Thresholds

1. Edit `config/config.yaml` signals section
2. No restart needed (config reloaded each cycle)
3. Changes take effect next cycle

---

##  Support

**For issues**:
1. Check `logs/neural_risk.log` for error messages
2. Run `python scripts/pre_deploy_check.py` to validate system
3. Review configuration in `config/config.yaml`

---

## 📜 License

Proprietary - All rights reserved

---

##  Quick Validation

After deployment, run this to confirm everything works:

```bash
# Check all services are running
tasklist | findstr python

# Verify database exists and growing
dir /s data\neural_risk.db

# View latest trading signals
sqlite3 data/neural_risk.db "SELECT timestamp, asset, signal, confidence FROM engine_decisions ORDER BY timestamp DESC LIMIT 5;"

# Expected: 5+ rows with recent timestamps and signals between -1.0 and 1.0
```

---

**Status**: ✅ Production Ready | **Version**: 0.2.0 | **Last Updated**: Feb 1, 2026
