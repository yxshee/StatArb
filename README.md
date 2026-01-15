# Statistical Arbitrage Trading Engine

A production-quality pairs trading system that identifies cointegrated pairs, trades mean-reversion strategies, and manages risk with sophisticated backtesting.

## 🎯 Project Overview

This engine automates the entire statistical arbitrage workflow:
- **Data Pipeline**: Downloads and caches historical price data
- **Pair Selection**: Identifies tradeable pairs using correlation and cointegration
- **Strategy**: Z-score based mean-reversion with configurable thresholds
- **Backtesting**: Simulates realistic trading with slippage and commissions
- **Risk Management**: Position sizing, stop losses, and drawdown controls
- **Analytics**: Comprehensive performance metrics and tear sheets

## 🚀 Quick Start

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage
```python
from src.data import download_symbol
from src.pairs import find_cointegrated_pairs
from src.strategy import generate_signals
from src.backtest import run_backtest

# Download data
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
for sym in symbols:
    download_symbol(sym, start='2015-01-01')

# Find pairs and backtest
# (More detailed examples coming soon)
```

## 📁 Project Structure

```
stat_arb_trading_engine/
├── data/               # Cached price data (CSVs)
├── notebooks/          # Analysis and demos
├── src/
│   ├── data.py        # Data download and caching
│   ├── pairs.py       # Pair selection algorithms
│   ├── strategy.py    # Trading signal generation
│   ├── backtest.py    # Backtesting engine
│   ├── risk.py        # Risk management
│   └── utils.py       # Helper functions
├── results/           # Performance reports and plots
├── tests/             # Unit tests
└── requirements.txt
```

## 📊 Key Features

### Pair Selection
- Correlation analysis (Pearson coefficient)
- Cointegration testing (Engle-Granger)
- Hedge ratio estimation (OLS regression)
- Spread stationarity validation

### Strategy
- Z-score based entry/exit signals
- Configurable thresholds and lookback windows
- Time-based exit stops
- Dollar-neutral positioning

### Risk Management
- Per-trade position sizing
- Stop-loss limits
- Portfolio drawdown caps
- Volatility targeting

### Performance Metrics
- Total return, CAGR, Sharpe ratio
- Maximum drawdown
- Win rate and profit factor
- Trade statistics

## 🔧 Configuration

Key parameters can be tuned in strategy files:
- `Z_ENTRY_THRESHOLD`: Entry signal (default: 2.0)
- `Z_EXIT_THRESHOLD`: Exit signal (default: 0.5)
- `LOOKBACK_WINDOW`: Rolling stats window (default: 60)
- `MAX_POSITION_SIZE`: Per-trade capital allocation
- `STOP_LOSS_PCT`: Stop loss threshold

## 📈 Performance

*(Performance metrics and backtesting results will be added as the project progresses)*

## 🛣️ Roadmap

- [x] Project setup and structure
- [ ] Data pipeline implementation
- [ ] Pair selection module
- [ ] Strategy logic
- [ ] Backtesting engine
- [ ] Risk management
- [ ] Performance analytics
- [ ] Multi-pair portfolio
- [ ] Walk-forward validation
- [ ] Demo notebook

## ⚠️ Limitations & Risks

- **Historical bias**: Past performance doesn't guarantee future results
- **Transaction costs**: Real-world slippage may vary
- **Regime changes**: Cointegration relationships can break down
- **Execution risk**: Simulated fills differ from live trading
- **Liquidity**: Assumes sufficient market depth

## 📚 Resources

- Cointegration: Engle-Granger (1987)
- Mean-reversion strategies: Chan (2009)
- Risk management: Kelly criterion, volatility targeting

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

This is a learning project. Feedback and suggestions welcome!

---

**Built for learning and demonstration purposes. Not financial advice.**
