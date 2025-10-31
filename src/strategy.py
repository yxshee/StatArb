"""
Trading strategy module - implements z-score based mean-reversion signals.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    ENTRY_LONG = "entry_long"      # Long spread (long A, short B)
    ENTRY_SHORT = "entry_short"    # Short spread (short A, long B)
    EXIT = "exit"                   # Close position
    HOLD = "hold"                   # No action


@dataclass
class StrategyConfig:
    """Configuration for the mean-reversion strategy."""
    
    # Z-score thresholds
    z_entry_threshold: float = 2.0    # Enter when |z| > this
    z_exit_threshold: float = 0.5     # Exit when |z| < this
    
    # Lookback window for rolling stats
    lookback_window: int = 60
    
    # Position limits
    max_position_size: float = 1.0    # Max position size (1.0 = full notional)
    
    # Time-based exits
    max_holding_period: Optional[int] = None  # Max days to hold, None = no limit
    
    # Stop loss
    stop_loss_pct: Optional[float] = None  # Stop loss as % of entry value
    
    def __str__(self):
        return (f"StrategyConfig(z_entry={self.z_entry_threshold}, "
                f"z_exit={self.z_exit_threshold}, "
                f"lookback={self.lookback_window})")


@dataclass
class Signal:
    """Trading signal with metadata."""
    date: pd.Timestamp
    signal_type: SignalType
    zscore: float
    spread_value: float
    
    def __str__(self):
        return f"{self.date.date()}: {self.signal_type.value} (z={self.zscore:.2f})"


class PairsStrategy:
    """
    Z-score based mean-reversion strategy for pairs trading.
    """
    
    def __init__(
        self,
        prices: pd.DataFrame,
        symbol_1: str,
        symbol_2: str,
        beta: float,
        config: Optional[StrategyConfig] = None
    ):
        """
        Initialize pairs trading strategy.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data with symbols as columns
        symbol_1 : str
            First symbol in the pair
        symbol_2 : str
            Second symbol in the pair
        beta : float
            Hedge ratio (from OLS regression)
        config : StrategyConfig, optional
            Strategy configuration
        """
        self.prices = prices
        self.symbol_1 = symbol_1
        self.symbol_2 = symbol_2
        self.beta = beta
        self.config = config if config else StrategyConfig()
        
        # Calculate spread and z-score
        self.spread = None
        self.zscore = None
        self._calculate_spread_and_zscore()
        
    def _calculate_spread_and_zscore(self):
        """Calculate spread and z-score."""
        # Get price series
        price_1 = self.prices[self.symbol_1]
        price_2 = self.prices[self.symbol_2]
        
        # Calculate spread
        self.spread = price_1 - self.beta * price_2
        
        # Calculate rolling statistics
        rolling_mean = self.spread.rolling(
            window=self.config.lookback_window,
            min_periods=20
        ).mean()
        
        rolling_std = self.spread.rolling(
            window=self.config.lookback_window,
            min_periods=20
        ).std()
        
        # Calculate z-score
        self.zscore = (self.spread - rolling_mean) / rolling_std
        
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on z-score.
        
        Returns:
        --------
        List[Signal] : List of trading signals
        """
        signals = []
        position = 0  # 0 = no position, 1 = long spread, -1 = short spread
        entry_date = None
        
        # Start after lookback period to have valid z-scores
        start_idx = self.config.lookback_window
        
        for i in range(start_idx, len(self.zscore)):
            date = self.zscore.index[i]
            z = self.zscore.iloc[i]
            spread_val = self.spread.iloc[i]
            
            # Skip if z-score is NaN
            if pd.isna(z):
                continue
            
            # Check time-based exit
            if position != 0 and self.config.max_holding_period is not None:
                if entry_date is not None:
                    days_held = (date - entry_date).days
                    if days_held >= self.config.max_holding_period:
                        signals.append(Signal(date, SignalType.EXIT, z, spread_val))
                        position = 0
                        entry_date = None
                        continue
            
            # Generate signals based on position state
            if position == 0:
                # No position - look for entry
                if z < -self.config.z_entry_threshold:
                    # Spread is too low -> go long spread
                    signals.append(Signal(date, SignalType.ENTRY_LONG, z, spread_val))
                    position = 1
                    entry_date = date
                    
                elif z > self.config.z_entry_threshold:
                    # Spread is too high -> go short spread
                    signals.append(Signal(date, SignalType.ENTRY_SHORT, z, spread_val))
                    position = -1
                    entry_date = date
                    
            else:
                # Have position - check for exit
                if position == 1:
                    # Long spread position
                    if z > -self.config.z_exit_threshold:
                        # Spread has mean-reverted
                        signals.append(Signal(date, SignalType.EXIT, z, spread_val))
                        position = 0
                        entry_date = None
                        
                elif position == -1:
                    # Short spread position
                    if z < self.config.z_exit_threshold:
                        # Spread has mean-reverted
                        signals.append(Signal(date, SignalType.EXIT, z, spread_val))
                        position = 0
                        entry_date = None
        
        return signals
    
    def generate_signal_series(self) -> pd.Series:
        """
        Generate a time series of signals.
        
        Returns:
        --------
        pd.Series : Signal values (1 = long, -1 = short, 0 = no position)
        """
        signals = self.generate_signals()
        
        # Create signal series
        signal_series = pd.Series(0, index=self.zscore.index)
        
        position = 0
        for signal in signals:
            if signal.signal_type == SignalType.ENTRY_LONG:
                position = 1
            elif signal.signal_type == SignalType.ENTRY_SHORT:
                position = -1
            elif signal.signal_type == SignalType.EXIT:
                position = 0
            
            # Fill forward from signal date
            signal_series[signal.date:] = position
        
        return signal_series
    
    def get_trade_log(self) -> pd.DataFrame:
        """
        Get a structured trade log from signals.
        
        Returns:
        --------
        pd.DataFrame : Trade log with entry/exit info
        """
        signals = self.generate_signals()
        
        trades = []
        current_trade = None
        
        for signal in signals:
            if signal.signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
                # Start new trade
                current_trade = {
                    'entry_date': signal.date,
                    'entry_type': signal.signal_type.value,
                    'entry_zscore': signal.zscore,
                    'entry_spread': signal.spread_value,
                    'direction': 1 if signal.signal_type == SignalType.ENTRY_LONG else -1
                }
                
            elif signal.signal_type == SignalType.EXIT and current_trade is not None:
                # Close trade
                current_trade.update({
                    'exit_date': signal.date,
                    'exit_zscore': signal.zscore,
                    'exit_spread': signal.spread_value,
                    'spread_pnl': (signal.spread_value - current_trade['entry_spread']) * current_trade['direction'],
                    'holding_days': (signal.date - current_trade['entry_date']).days
                })
                trades.append(current_trade)
                current_trade = None
        
        return pd.DataFrame(trades)
    
    def backtest_simple(self) -> Dict:
        """
        Simple backtest returning basic statistics.
        
        Returns:
        --------
        Dict : Backtest results
        """
        trade_log = self.get_trade_log()
        
        if len(trade_log) == 0:
            return {
                'n_trades': 0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'total_pnl': 0.0,
                'avg_holding_days': 0.0
            }
        
        # Calculate metrics
        winning_trades = trade_log[trade_log['spread_pnl'] > 0]
        
        return {
            'n_trades': len(trade_log),
            'win_rate': len(winning_trades) / len(trade_log) if len(trade_log) > 0 else 0,
            'avg_pnl': trade_log['spread_pnl'].mean(),
            'total_pnl': trade_log['spread_pnl'].sum(),
            'avg_holding_days': trade_log['holding_days'].mean(),
            'trade_log': trade_log
        }


def optimize_parameters(
    prices: pd.DataFrame,
    symbol_1: str,
    symbol_2: str,
    beta: float,
    z_entry_values: List[float] = [1.5, 2.0, 2.5],
    lookback_values: List[int] = [30, 60, 120]
) -> pd.DataFrame:
    """
    Grid search over strategy parameters.
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    symbol_1, symbol_2 : str
        Pair symbols
    beta : float
        Hedge ratio
    z_entry_values : List[float]
        Z-entry thresholds to test
    lookback_values : List[int]
        Lookback windows to test
        
    Returns:
    --------
    pd.DataFrame : Results for each parameter combination
    """
    results = []
    
    for z_entry in z_entry_values:
        for lookback in lookback_values:
            config = StrategyConfig(
                z_entry_threshold=z_entry,
                z_exit_threshold=0.5,  # Fixed
                lookback_window=lookback
            )
            
            strategy = PairsStrategy(prices, symbol_1, symbol_2, beta, config)
            metrics = strategy.backtest_simple()
            
            results.append({
                'z_entry': z_entry,
                'lookback': lookback,
                **metrics
            })
    
    return pd.DataFrame(results).sort_values('total_pnl', ascending=False)


# Convenience function
def calculate_spread_zscore(
    prices: pd.DataFrame,
    symbol_1: str,
    symbol_2: str,
    beta: float,
    lookback_window: int = 60
) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate spread and z-score for a pair (convenience function).
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    symbol_1, symbol_2 : str
        Pair symbols
    beta : float
        Hedge ratio
    lookback_window : int
        Window for rolling stats
        
    Returns:
    --------
    Tuple[pd.Series, pd.Series] : (spread, zscore)
    """
    config = StrategyConfig(lookback_window=lookback_window)
    strategy = PairsStrategy(prices, symbol_1, symbol_2, beta, config)
    return strategy.spread, strategy.zscore


if __name__ == "__main__":
    print("Strategy module loaded successfully!")
    print("Use PairsStrategy class or convenience functions")
