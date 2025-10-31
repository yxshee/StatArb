"""
Statistical Arbitrage Trading Engine

A production-quality pairs trading system for identifying cointegrated pairs
and trading mean-reversion strategies.
"""

__version__ = '0.1.0'
__author__ = 'Quant Trading Team'

from .data import DataDownloader, download_symbol, download_multiple, load_symbol
from .utils import (
    get_data_dir,
    get_results_dir,
    calculate_returns,
    calculate_log_returns
)

__all__ = [
    'DataDownloader',
    'download_symbol',
    'download_multiple',
    'load_symbol',
    'get_data_dir',
    'get_results_dir',
    'calculate_returns',
    'calculate_log_returns',
]
