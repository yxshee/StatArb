"""
Risk management module - position sizing, stop loss, drawdown controls.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class RiskConfig:
    """Risk management configuration."""
    
    # Capital allocation
    total_capital: float = 1000000.0  # Total portfolio capital
    max_position_pct: float = 0.05    # Max % of capital per position (5%)
    
    # Stop loss
    stop_loss_pct: float = 0.03       # Stop loss threshold (3%)
    
    # Portfolio limits
    max_drawdown_pct: float = 0.10    # Max drawdown before halting (10%)
    max_leverage: float = 1.0         # Max leverage ratio
    
    # Volatility targeting
    target_volatility: Optional[float] = None  # Target annual volatility (e.g., 0.10 for 10%)
    
    def __str__(self):
        return (f"RiskConfig(capital=${self.total_capital:,.0f}, "
                f"max_pos={self.max_position_pct*100}%, "
                f"stop_loss={self.stop_loss_pct*100}%)")


class RiskManager:
    """
    Manages position sizing and risk limits.
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize risk manager.
        
        Parameters:
        -----------
        config : RiskConfig, optional
            Risk configuration
        """
        self.config = config if config else RiskConfig()
        
        # Track portfolio state
        self.current_capital = self.config.total_capital
        self.peak_capital = self.config.total_capital
        self.current_drawdown = 0.0
        self.positions = {}  # symbol -> position size
        
    def calculate_position_size(
        self,
        symbol_1: str,
        symbol_2: str,
        price_1: float,
        price_2: float,
        beta: float,
        volatility: Optional[float] = None
    ) -> Tuple[int, int]:
        """
        Calculate dollar-neutral position sizes for a pair.
        
        Parameters:
        -----------
        symbol_1, symbol_2 : str
            Pair symbols
        price_1, price_2 : float
            Current prices
        beta : float
            Hedge ratio
        volatility : float, optional
            Pair volatility for vol targeting
            
        Returns:
        --------
        Tuple[int, int] : (shares_1, shares_2) - positive values
        """
        # Base capital allocation per position
        base_capital = self.current_capital * self.config.max_position_pct
        
        # Adjust for volatility targeting if enabled
        if self.config.target_volatility is not None and volatility is not None:
            # Scale position size inversely with volatility
            vol_scalar = self.config.target_volatility / max(volatility, 0.01)
            capital_allocation = base_capital * min(vol_scalar, 2.0)  # Cap at 2x
        else:
            capital_allocation = base_capital
        
        # Calculate dollar-neutral shares
        # We want: price_1 * shares_1 = beta * price_2 * shares_2
        # And: price_1 * shares_1 = capital_allocation / 2
        
        dollar_value_per_leg = capital_allocation / 2
        
        shares_1 = int(dollar_value_per_leg / price_1)
        shares_2 = int((price_1 * shares_1) / (beta * price_2))
        
        return shares_1, shares_2
    
    def check_stop_loss(
        self,
        entry_value: float,
        current_value: float,
        direction: int
    ) -> bool:
        """
        Check if stop loss is triggered.
        
        Parameters:
        -----------
        entry_value : float
            Position entry value
        current_value : float
            Current position value
        direction : int
            1 for long, -1 for short
            
        Returns:
        --------
        bool : True if stop loss triggered
        """
        # Calculate P&L as percentage
        pnl_pct = ((current_value - entry_value) / entry_value) * direction
        
        return pnl_pct <= -self.config.stop_loss_pct
    
    def update_capital(self, pnl: float) -> None:
        """
        Update capital after trade P&L.
        
        Parameters:
        -----------
        pnl : float
            Trade profit/loss
        """
        self.current_capital += pnl
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Update drawdown
        self.current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital
    
    def is_trading_allowed(self) -> bool:
        """
        Check if trading is allowed given current drawdown.
        
        Returns:
        --------
        bool : True if trading is allowed
        """
        return self.current_drawdown < self.config.max_drawdown_pct
    
    def get_portfolio_stats(self) -> Dict:
        """
        Get current portfolio statistics.
        
        Returns:
        --------
        Dict : Portfolio stats
        """
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'total_pnl': self.current_capital - self.config.total_capital,
            'total_return_pct': (self.current_capital / self.config.total_capital - 1) * 100,
            'current_drawdown_pct': self.current_drawdown * 100,
            'trading_allowed': self.is_trading_allowed()
        }
    
    def reset(self) -> None:
        """Reset risk manager to initial state."""
        self.current_capital = self.config.total_capital
        self.peak_capital = self.config.total_capital
        self.current_drawdown = 0.0
        self.positions = {}


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float,
    max_fraction: float = 0.25
) -> float:
    """
    Calculate Kelly criterion position size.
    
    Parameters:
    -----------
    win_rate : float
        Probability of winning (0-1)
    avg_win : float
        Average win amount
    avg_loss : float
        Average loss amount (positive value)
    max_fraction : float
        Maximum fraction to use (for safety)
        
    Returns:
    --------
    float : Position fraction (0-1)
    """
    if avg_loss <= 0:
        return 0.0
    
    # Kelly formula: f = (p * b - q) / b
    # where p = win_rate, q = 1-p, b = avg_win / avg_loss
    
    b = avg_win / avg_loss
    kelly = (win_rate * b - (1 - win_rate)) / b
    
    # Conservative Kelly (use fraction of kelly)
    kelly = max(0, kelly) * 0.5  # Use half-Kelly for safety
    
    # Cap at max_fraction
    return min(kelly, max_fraction)


def calculate_volatility_target_weights(
    returns: pd.DataFrame,
    target_vol: float = 0.10,
    lookback: int = 60
) -> pd.Series:
    """
    Calculate position weights for volatility targeting.
    
    Parameters:
    -----------
    returns : pd.DataFrame
        Returns for each pair/strategy
    target_vol : float
        Target annual volatility
    lookback : int
        Lookback period for vol estimation
        
    Returns:
    --------
    pd.Series : Volatility-scaled weights
    """
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=lookback).std() * np.sqrt(252)
    
    # Calculate inverse volatility weights
    inv_vol = 1 / rolling_vol.replace(0, np.nan)
    
    # Scale to target volatility
    weights = target_vol * inv_vol
    
    # Normalize
    weights = weights.div(weights.sum(axis=1), axis=0)
    
    return weights


if __name__ == "__main__":
    print("Risk management module loaded successfully!")
    print("Use RiskManager class for position sizing and risk controls")
