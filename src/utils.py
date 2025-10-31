"""
Utility functions for the trading engine.
"""
import os
from datetime import datetime
from typing import Optional
import pandas as pd


def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(directory, exist_ok=True)


def get_project_root() -> str:
    """Get the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_dir() -> str:
    """Get the data directory path."""
    root = get_project_root()
    data_dir = os.path.join(root, 'data')
    ensure_dir(data_dir)
    return data_dir


def get_results_dir() -> str:
    """Get the results directory path."""
    root = get_project_root()
    results_dir = os.path.join(root, 'results')
    ensure_dir(results_dir)
    return results_dir


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Validate that a DataFrame contains required columns.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to validate
    required_columns : list
        List of required column names
        
    Returns:
    --------
    bool : True if valid, False otherwise
    """
    if df is None or df.empty:
        return False
    
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return False
    
    return True


def format_currency(value: float, currency: str = '$') -> str:
    """Format a number as currency."""
    return f"{currency}{value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a number as percentage."""
    return f"{value:.{decimals}f}%"


def calculate_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate simple returns from price series.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
        
    Returns:
    --------
    pd.Series : Returns series
    """
    return prices.pct_change()


def calculate_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calculate log returns from price series.
    
    Parameters:
    -----------
    prices : pd.Series
        Price series
        
    Returns:
    --------
    pd.Series : Log returns series
    """
    import numpy as np
    return np.log(prices / prices.shift(1))
