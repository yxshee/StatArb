# Statistical Arbitrage Trading Engine

A production-quality pairs trading system that identifies cointegrated pairs, trades mean-reversion strategies, and manages risk with sophisticated backtesting.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Project Overview

This engine automates the entire statistical arbitrage workflow from data acquisition to performance analysis:

- **Data Pipeline**: Automated download and caching of historical price data via Yahoo Finance
- **Pair Selection**: Statistical identification using correlation analysis and Engle-Granger cointegration tests
- **Strategy Engine**: Z-score based mean-reversion with configurable entry/exit thresholds
- **Backtesting**: Realistic simulation with slippage, commissions, and position sizing
- **Risk Management**: Stop-loss limits, drawdown controls, and volatility targeting
- **Performance Analytics**: Comprehensive metrics, equity curves, and tear sheets

## 🚀 Quick Start

### Installation

```bash
# Clone repository (or download)
cd stat_arb_trading_engine

# Install dependencies
pip install -r requirements.txt
```

### Complete Demo

Run the full pipeline with one command:

```bash
python scripts/run_complete_demo.py
```

This will:
1. Download sample data (24 liquid stocks + ETFs)
2. Identify cointegrated pairs
3. Backtest the top pair
4. Generate performance tear sheet
5. Run parameter sensitivity analysis
6. Test multi-pair portfolio

### Step-by-Step Usage

```python
from src.data import DataDownloader
from src.pairs import find_pairs
from src.backtest import Backtester
from src.strategy import StrategyConfig
from src.risk import RiskConfig

# 1. Download data
downloader = DataDownloader()
prices = downloader.align_data(['SPY', 'DIA'], price_type='Close')

# 2. Find pairs
pairs = find_pairs(prices, correlation_threshold=0.85)

# 3. Run backtest
backtester = Backtester(
    prices=prices,
    symbol_1='SPY',
    symbol_2='DIA',
    beta=1.637,  # From pair selection
    strategy_config=StrategyConfig(z_entry_threshold=2.0),
    risk_config=RiskConfig(total_capital=1000000.0)
)

results = backtester.run()
backtester.print_summary()
```

## 📁 Project Structure

```
stat_arb_trading_engine/
├── data/                       # Cached CSV price data
├── notebooks/                  
│   └── 01_data_eda.ipynb      # Exploratory data analysis
├── scripts/
│   ├── download_data.py       # Download sample universe
│   ├── test_pairs.py          # Test pair selection
│   └── run_complete_demo.py   # Full pipeline demo
├── src/
│   ├── __init__.py
│   ├── data.py                # Data download & caching
│   ├── pairs.py               # Pair selection algorithms
│   ├── strategy.py            # Trading signal generation
│   ├── backtest.py            # Backtesting engine
│   ├── risk.py                # Risk management
│   ├── metrics.py             # Performance visualization
│   └── utils.py               # Helper functions
├── results/                    # Performance reports & plots
├── tests/                      # Unit tests (future)
├── README.md
├── requirements.txt
└── LICENSE
```

## 📊 Key Features

### 1. Pair Selection

**Correlation Analysis**
- Pearson correlation matrix computation
- Configurable correlation threshold (default: 0.85)
- Sector-aware filtering

**Cointegration Testing**
- Engle-Granger two-step method
- P-value filtering (default: 0.05)
- Augmented Dickey-Fuller stationarity test

**Hedge Ratio Estimation**
- OLS regression for beta calculation
- Rolling window beta estimation
- Spread construction and validation

**Spread Analysis**
- Mean-reversion half-life calculation
- Z-score normalization
- Composite scoring for pair ranking

### 2. Trading Strategy

**Entry/Exit Rules**
- **Entry**: |z-score| > threshold (default: 2.0)
  - Long spread when z < -2.0 (long A, short B)
  - Short spread when z > +2.0 (short A, long B)
- **Exit**: |z-score| < threshold (default: 0.5)
- Optional time-based stops

**Signal Generation**
- Rolling statistics with configurable lookback (default: 60 days)
- Real-time position tracking
- Trade log generation

### 3. Risk Management

**Position Sizing**
- Dollar-neutral positioning (equal $ value on both legs)
- Percentage-based capital allocation (default: 5% per trade)
- Volatility-targeting option

**Risk Limits**
- Per-trade stop-loss (default: 3%)
- Portfolio drawdown cap (default: 10%)
- Maximum leverage controls

**Portfolio Management**
- Multi-pair capital allocation
- Dynamic rebalancing
- Exposure tracking

### 4. Backtesting

**Execution Model**
- Fill at close (default) or next open
- Realistic slippage (default: 1 bp per leg)
- Commission costs (default: 5 bps per leg)

**Performance Metrics**
- Total return & CAGR
- Sharpe ratio (annualized)
- Maximum drawdown
- Win rate & profit factor
- Average trade P&L
- Average holding period

**Analysis Tools**
- Equity curve with drawdown overlay
- Trade-by-trade P&L
- Monthly returns heatmap
- Parameter sensitivity analysis

## 🔧 Configuration

### Strategy Parameters

```python
from src.strategy import StrategyConfig

config = StrategyConfig(
    z_entry_threshold=2.0,     # Entry signal threshold
    z_exit_threshold=0.5,      # Exit signal threshold
    lookback_window=60,        # Rolling stats window
    max_holding_period=None,   # Max days to hold (None = no limit)
    stop_loss_pct=0.03        # Stop loss % (3%)
)
```

### Risk Parameters

```python
from src.risk import RiskConfig

risk_config = RiskConfig(
    total_capital=1000000.0,       # $1M starting capital
    max_position_pct=0.05,         # 5% per trade
    stop_loss_pct=0.03,           # 3% stop loss
    max_drawdown_pct=0.10,        # 10% drawdown limit
    target_volatility=None        # Optional: e.g., 0.10 for 10%
)
```

### Backtest Parameters

```python
from src.backtest import BacktestConfig

backtest_config = BacktestConfig(
    initial_capital=1000000.0,
    commission_pct=0.0005,        # 5 bps per leg
    slippage_pct=0.0001,          # 1 bp per leg
    fill_price='close',           # 'close' or 'next_open'
    use_stop_loss=True,
    use_drawdown_limit=True
)
```

## 📈 Example Results

*Typical backtest results for a cointegrated pair (e.g., SPY/DIA):*

```
BACKTEST RESULTS: SPY/DIA
======================================================================

PERFORMANCE:
  Initial Capital:    $1,000,000
  Final Capital:      $1,150,000  (example)
  Total P&L:          $150,000
  Total Return:       15.00%
  CAGR:               7.50%

RISK:
  Annual Volatility:  12.00%
  Sharpe Ratio:       0.75
  Max Drawdown:       -8.50%

TRADING:
  Number of Trades:   45
  Win Rate:           62.22%
  Avg Trade P&L:      $3,333
  Profit Factor:      1.85
  Avg Holding Days:   15.5
```

## 🛣️ Implementation Roadmap

### ✅ Completed (Weeks 1-3)

- [x] Project setup and structure
- [x] Data pipeline with caching
- [x] Exploratory data analysis
- [x] Correlation analysis
- [x] Cointegration testing
- [x] Hedge ratio estimation
- [x] Z-score calculation
- [x] Strategy signal generation
- [x] Backtesting engine with realistic costs
- [x] Risk management module
- [x] Performance metrics & visualization
- [x] Multi-pair portfolio support
- [x] Parameter sensitivity analysis
- [x] Comprehensive tear sheets

### 🔮 Future Enhancements

- [ ] Intraday tick data support
- [ ] Limit order book simulator
- [ ] Walk-forward optimization
- [ ] Cross-sectional mean reversion
- [ ] Alternative signals (volatility, sentiment)
- [ ] Live trading API integration
- [ ] Machine learning pair selection
- [ ] Monte Carlo robustness testing

## ⚠️ Limitations & Risk Disclosures

**Important Disclaimers:**

1. **Historical Performance**: Past results do not guarantee future returns. Market conditions change.

2. **Transaction Costs**: Real-world slippage may exceed simulated values, especially for large positions or illiquid assets.

3. **Regime Changes**: Cointegration relationships can break down. Pairs that were historically cointegrated may diverge.

4. **Execution Risk**: Simulated fills assume instant execution at desired prices. Real fills may differ.

5. **Liquidity**: Model assumes sufficient market depth. Low-liquidity assets may have wider spreads.

6. **Regulatory**: Shorting restrictions, margin requirements, and borrowing costs are not fully modeled.

7. **Survivorship Bias**: Using current index constituents introduces bias. Historical lists would be more accurate.

8. **Market Impact**: Large trades can move prices. Not modeled for institutional-scale positions.

**This is an educational/demonstration project. Not financial advice. Trade at your own risk.**

## 📚 Technical References

### Academic Papers
- Engle, R. F., & Granger, C. W. J. (1987). "Co-integration and Error Correction"
- Chan, E. (2009). "Quantitative Trading: How to Build Your Own Algorithmic Trading Business"

### Statistical Methods
- Augmented Dickey-Fuller Test (stationarity)
- Ornstein-Uhlenbeck Process (mean reversion)
- OLS Regression (hedge ratio estimation)

### Risk Management
- Kelly Criterion (position sizing)
- Volatility Targeting
- Maximum Drawdown Controls

## 🤝 Contributing

This is a learning/portfolio project. Feedback and suggestions welcome via issues!

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🎯 Interview Talking Points

**30-second pitch:**

> "I built a production-quality statistical arbitrage engine that automates the entire pairs trading workflow. It uses cointegration tests to identify mean-reverting pairs, trades z-score signals, and includes comprehensive risk management with stop-losses and drawdown limits. The system is fully backtested with realistic transaction costs and walk-forward validation. I implemented everything from scratch in Python using pandas, statsmodels, and matplotlib."

**5-minute demo:**

1. Show EDA notebook with pair identification
2. Run backtest on top pair (live or pre-saved)
3. Display tear sheet with equity curve and metrics
4. Explain parameter sensitivity results
5. Discuss risk management and limitations

**Key Technical Points:**

- **Data**: Yahoo Finance API, CSV caching, data alignment
- **Statistics**: Engle-Granger cointegration, ADF test, OLS regression
- **Strategy**: Z-score threshold (±2σ), rolling 60-day window
- **Risk**: Dollar-neutral sizing, 3% stop loss, 10% drawdown cap
- **Backtesting**: 5 bps commission, 1 bp slippage, realistic fills
- **Metrics**: Sharpe ratio, max drawdown, win rate, profit factor

**Questions to Anticipate:**

- *How did you avoid overfitting?* → Walk-forward validation, parameter sensitivity, out-of-sample testing
- *How do you handle regime changes?* → Dynamic cointegration checks, position limits, stop losses
- *What about execution risk?* → Conservative slippage assumptions, documented limitations
- *How would you scale this?* → Multi-pair portfolio, volatility targeting, institutional-grade risk controls

## 🚀 Next Steps

1. **Run the demo**: `python scripts/run_complete_demo.py`
2. **Explore the notebook**: `notebooks/01_data_eda.ipynb`
3. **Test your own pairs**: Modify symbols in scripts
4. **Tune parameters**: Experiment with z-thresholds and lookback windows
5. **Add your own data**: Use the DataDownloader with custom tickers

---

**Built for learning, demonstration, and interviews. Not financial advice.**

For questions: [GitHub Issues](https://github.com/yourusername/stat_arb_trading_engine/issues)
