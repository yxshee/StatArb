"""
Backtesting engine with realistic execution, transaction costs, and performance metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from .strategy import PairsStrategy, StrategyConfig, SignalType
    from .risk import RiskManager, RiskConfig
except ImportError:  # pragma: no cover
    from strategy import PairsStrategy, StrategyConfig, SignalType  # noqa: F401
    from risk import RiskManager, RiskConfig  # noqa: F401


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    
    # Initial capital
    initial_capital: float = 1000000.0
    
    # Transaction costs
    commission_pct: float = 0.0005  # 0.05% per leg
    slippage_pct: float = 0.0001    # 0.01% per leg
    
    # Execution
    fill_price: str = 'close'  # 'open', 'close', or 'next_open'
    
    # Risk management
    use_stop_loss: bool = True
    use_drawdown_limit: bool = True
    
    def __str__(self):
        return (f"BacktestConfig(capital=${self.initial_capital:,.0f}, "
                f"commission={self.commission_pct*100}%, "
                f"slippage={self.slippage_pct*100}%)")


@dataclass
class Trade:
    """Represents a completed trade."""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    symbol_1: str
    symbol_2: str
    direction: int  # 1 = long spread, -1 = short spread
    
    # Entry
    entry_price_1: float
    entry_price_2: float
    shares_1: int
    shares_2: int
    entry_value: float
    
    # Exit
    exit_price_1: float
    exit_price_2: float
    exit_value: float
    
    # P&L
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    
    # Metadata
    entry_zscore: float
    exit_zscore: float
    holding_days: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'entry_date': self.entry_date,
            'exit_date': self.exit_date,
            'symbol_1': self.symbol_1,
            'symbol_2': self.symbol_2,
            'direction': 'LONG' if self.direction == 1 else 'SHORT',
            'shares_1': self.shares_1,
            'shares_2': self.shares_2,
            'entry_value': self.entry_value,
            'exit_value': self.exit_value,
            'gross_pnl': self.gross_pnl,
            'commission': self.commission,
            'slippage': self.slippage,
            'net_pnl': self.net_pnl,
            'entry_zscore': self.entry_zscore,
            'exit_zscore': self.exit_zscore,
            'holding_days': self.holding_days
        }


class Backtester:
    """
    Comprehensive backtesting engine for pairs trading.
    """
    
    def __init__(
        self,
        prices: pd.DataFrame,
        symbol_1: str,
        symbol_2: str,
        beta: float,
        strategy_config: Optional[StrategyConfig] = None,
        risk_config: Optional[RiskConfig] = None,
        backtest_config: Optional[BacktestConfig] = None
    ):
        """
        Initialize backtester.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data
        symbol_1, symbol_2 : str
            Pair symbols
        beta : float
            Hedge ratio
        strategy_config : StrategyConfig, optional
        risk_config : RiskConfig, optional
        backtest_config : BacktestConfig, optional
        """
        self.prices = prices
        self.symbol_1 = symbol_1
        self.symbol_2 = symbol_2
        self.beta = beta
        
        self.strategy_config = strategy_config if strategy_config else StrategyConfig()
        self.risk_config = risk_config if risk_config else RiskConfig()
        self.backtest_config = backtest_config if backtest_config else BacktestConfig()
        
        # Initialize components
        self.strategy = PairsStrategy(prices, symbol_1, symbol_2, beta, self.strategy_config)
        self.risk_manager = RiskManager(self.risk_config)
        
        # Results
        self.trades: List[Trade] = []
        self.equity_curve: Optional[pd.Series] = None
        
    def _get_fill_price(self, date: pd.Timestamp, symbol: str, is_entry: bool) -> float:
        """Get fill price based on execution model."""
        try:
            if self.backtest_config.fill_price == 'close':
                price = self.prices.loc[date, symbol]
            elif self.backtest_config.fill_price == 'open':
                price = self.prices.loc[date, symbol]  # Simplified: using close
            else:  # next_open
                # Get next day's price
                next_idx = self.prices.index.get_loc(date) + 1
                if next_idx < len(self.prices):
                    price = self.prices.iloc[next_idx][symbol]
                else:
                    price = self.prices.loc[date, symbol]
            
            # Apply slippage
            if is_entry:
                price *= (1 + self.backtest_config.slippage_pct)
            else:
                price *= (1 - self.backtest_config.slippage_pct)
            
            return price
        except:
            return np.nan
    
    def run(self) -> Dict:
        """
        Run backtest.
        
        Returns:
        --------
        Dict : Backtest results with metrics
        """
        print(f"\nRunning backtest for {self.symbol_1}/{self.symbol_2}...")
        print(f"Strategy: {self.strategy_config}")
        print(f"Capital: ${self.risk_config.total_capital:,.0f}")
        
        # Generate signals
        signals = self.strategy.generate_signals()
        
        if len(signals) == 0:
            print("⚠ No signals generated")
            return self._empty_results()
        
        # Track state
        position = None  # Current open position
        equity = pd.Series(self.risk_config.total_capital, index=self.prices.index)
        
        # Process signals
        for signal in signals:
            # Check if trading is allowed
            if not self.risk_manager.is_trading_allowed():
                print(f"⚠ Trading halted at {signal.date.date()} due to drawdown limit")
                break
            
            if signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
                # Entry signal
                if position is not None:
                    continue  # Already in position
                
                # Get prices
                entry_price_1 = self._get_fill_price(signal.date, self.symbol_1, is_entry=True)
                entry_price_2 = self._get_fill_price(signal.date, self.symbol_2, is_entry=True)
                
                if pd.isna(entry_price_1) or pd.isna(entry_price_2):
                    continue
                
                # Calculate position size
                shares_1, shares_2 = self.risk_manager.calculate_position_size(
                    self.symbol_1, self.symbol_2,
                    entry_price_1, entry_price_2,
                    self.beta
                )
                
                if shares_1 == 0 or shares_2 == 0:
                    continue
                
                # Record position
                direction = 1 if signal.signal_type == SignalType.ENTRY_LONG else -1
                entry_value = entry_price_1 * shares_1 + entry_price_2 * shares_2
                
                position = {
                    'entry_date': signal.date,
                    'entry_price_1': entry_price_1,
                    'entry_price_2': entry_price_2,
                    'shares_1': shares_1,
                    'shares_2': shares_2,
                    'entry_value': entry_value,
                    'entry_zscore': signal.zscore,
                    'direction': direction
                }
                
            elif signal.signal_type == SignalType.EXIT and position is not None:
                # Exit signal
                exit_price_1 = self._get_fill_price(signal.date, self.symbol_1, is_entry=False)
                exit_price_2 = self._get_fill_price(signal.date, self.symbol_2, is_entry=False)
                
                if pd.isna(exit_price_1) or pd.isna(exit_price_2):
                    continue
                
                # Calculate P&L
                exit_value = exit_price_1 * position['shares_1'] + exit_price_2 * position['shares_2']
                
                # Gross P&L based on spread movement
                spread_entry = position['entry_price_1'] - self.beta * position['entry_price_2']
                spread_exit = exit_price_1 - self.beta * exit_price_2
                gross_pnl = (spread_exit - spread_entry) * position['shares_1'] * position['direction']
                
                # Transaction costs
                total_value = position['entry_value'] + exit_value
                commission = total_value * self.backtest_config.commission_pct
                slippage_cost = total_value * self.backtest_config.slippage_pct
                
                net_pnl = gross_pnl - commission - slippage_cost
                
                # Create trade record
                trade = Trade(
                    entry_date=position['entry_date'],
                    exit_date=signal.date,
                    symbol_1=self.symbol_1,
                    symbol_2=self.symbol_2,
                    direction=position['direction'],
                    entry_price_1=position['entry_price_1'],
                    entry_price_2=position['entry_price_2'],
                    shares_1=position['shares_1'],
                    shares_2=position['shares_2'],
                    entry_value=position['entry_value'],
                    exit_price_1=exit_price_1,
                    exit_price_2=exit_price_2,
                    exit_value=exit_value,
                    gross_pnl=gross_pnl,
                    commission=commission,
                    slippage=slippage_cost,
                    net_pnl=net_pnl,
                    entry_zscore=position['entry_zscore'],
                    exit_zscore=signal.zscore,
                    holding_days=(signal.date - position['entry_date']).days
                )
                
                self.trades.append(trade)
                
                # Update risk manager
                self.risk_manager.update_capital(net_pnl)
                
                # Update equity curve
                equity[signal.date:] = self.risk_manager.current_capital
                
                # Clear position
                position = None
        
        self.equity_curve = equity
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        print(f"\n✓ Backtest complete: {len(self.trades)} trades")
        return metrics
    
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        if len(self.trades) == 0:
            return self._empty_results()
        
        # Trade statistics
        trade_df = pd.DataFrame([t.to_dict() for t in self.trades])
        
        winning_trades = trade_df[trade_df['net_pnl'] > 0]
        losing_trades = trade_df[trade_df['net_pnl'] < 0]
        
        # Returns
        total_return = (self.risk_manager.current_capital / self.risk_config.total_capital) - 1
        
        # Time period
        start_date = self.equity_curve.index[0]
        end_date = self.equity_curve.index[-1]
        years = (end_date - start_date).days / 365.25
        
        # CAGR
        cagr = (1 + total_return) ** (1 / max(years, 0.1)) - 1 if years > 0 else 0
        
        # Volatility (annualized)
        returns = self.equity_curve.pct_change().dropna()
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe = (cagr / annual_vol) if annual_vol > 0 else 0
        
        # Maximum drawdown
        rolling_max = self.equity_curve.expanding().max()
        drawdown = (self.equity_curve - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            # Overall
            'initial_capital': self.risk_config.total_capital,
            'final_capital': self.risk_manager.current_capital,
            'total_pnl': self.risk_manager.current_capital - self.risk_config.total_capital,
            'total_return_pct': total_return * 100,
            'cagr_pct': cagr * 100,
            
            # Risk metrics
            'annual_volatility_pct': annual_vol * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown * 100,
            
            # Trade statistics
            'n_trades': len(trade_df),
            'win_rate_pct': (len(winning_trades) / len(trade_df)) * 100,
            'avg_trade_pnl': trade_df['net_pnl'].mean(),
            'avg_win': winning_trades['net_pnl'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['net_pnl'].mean() if len(losing_trades) > 0 else 0,
            'profit_factor': (winning_trades['net_pnl'].sum() / abs(losing_trades['net_pnl'].sum())) 
                             if len(losing_trades) > 0 and losing_trades['net_pnl'].sum() < 0 else np.inf,
            'avg_holding_days': trade_df['holding_days'].mean(),
            
            # Transaction costs
            'total_commission': trade_df['commission'].sum(),
            'total_slippage': trade_df['slippage'].sum(),
            
            # Period
            'start_date': start_date,
            'end_date': end_date,
            'years': years,
            
            # Data
            'equity_curve': self.equity_curve,
            'trade_log': trade_df
        }
    
    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'initial_capital': self.risk_config.total_capital,
            'final_capital': self.risk_config.total_capital,
            'total_pnl': 0,
            'total_return_pct': 0,
            'cagr_pct': 0,
            'annual_volatility_pct': 0,
            'sharpe_ratio': 0,
            'max_drawdown_pct': 0,
            'n_trades': 0,
            'win_rate_pct': 0,
            'avg_trade_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'avg_holding_days': 0,
            'total_commission': 0,
            'total_slippage': 0,
            'equity_curve': pd.Series(self.risk_config.total_capital),
            'trade_log': pd.DataFrame()
        }
    
    def print_summary(self, results: Optional[Dict] = None):
        """Print backtest summary."""
        if results is None:
            results = self.calculate_metrics()
        
        print("\n" + "="*70)
        print(f"BACKTEST RESULTS: {self.symbol_1}/{self.symbol_2}")
        print("="*70)
        
        print(f"\nPERFORMANCE:")
        print(f"  Initial Capital:    ${results['initial_capital']:>12,.0f}")
        print(f"  Final Capital:      ${results['final_capital']:>12,.0f}")
        print(f"  Total P&L:          ${results['total_pnl']:>12,.0f}")
        print(f"  Total Return:       {results['total_return_pct']:>12.2f}%")
        print(f"  CAGR:               {results['cagr_pct']:>12.2f}%")
        
        print(f"\nRISK:")
        print(f"  Annual Volatility:  {results['annual_volatility_pct']:>12.2f}%")
        print(f"  Sharpe Ratio:       {results['sharpe_ratio']:>12.2f}")
        print(f"  Max Drawdown:       {results['max_drawdown_pct']:>12.2f}%")
        
        print(f"\nTRADING:")
        print(f"  Number of Trades:   {results['n_trades']:>12d}")
        print(f"  Win Rate:           {results['win_rate_pct']:>12.2f}%")
        print(f"  Avg Trade P&L:      ${results['avg_trade_pnl']:>12,.0f}")
        print(f"  Avg Win:            ${results['avg_win']:>12,.0f}")
        print(f"  Avg Loss:           ${results['avg_loss']:>12,.0f}")
        print(f"  Profit Factor:      {results['profit_factor']:>12.2f}")
        print(f"  Avg Holding Days:   {results['avg_holding_days']:>12.1f}")
        
        print(f"\nCOSTS:")
        print(f"  Total Commission:   ${results['total_commission']:>12,.0f}")
        print(f"  Total Slippage:     ${results['total_slippage']:>12,.0f}")


if __name__ == "__main__":
    print("Backtesting module loaded successfully!")
    print("Use Backtester class for comprehensive backtesting")
