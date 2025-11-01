"""
Data download and caching module.
Handles fetching historical price data from Yahoo Finance and caching to CSV.
"""
import os
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from .utils import get_data_dir, validate_dataframe
except ImportError:  # pragma: no cover
    from utils import get_data_dir, validate_dataframe  # noqa: F401


class DataDownloader:
    """
    Handles downloading and caching of historical price data.
    """
    
    def __init__(self, data_dir: Optional[str] = None):
        """
        Initialize the data downloader.
        
        Parameters:
        -----------
        data_dir : str, optional
            Directory to store cached data. If None, uses default data/ folder.
        """
        self.data_dir = data_dir if data_dir else get_data_dir()
        
    def get_cache_path(self, symbol: str) -> str:
        """Get the cache file path for a symbol."""
        return os.path.join(self.data_dir, f"{symbol}.csv")
    
    def is_cached(self, symbol: str, max_age_days: int = 1) -> bool:
        """
        Check if symbol data is cached and fresh.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        max_age_days : int
            Maximum age of cached data in days
            
        Returns:
        --------
        bool : True if cached and fresh
        """
        cache_path = self.get_cache_path(symbol)
        
        if not os.path.exists(cache_path):
            return False
        
        # Check file age
        file_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
        age = datetime.now() - file_time
        
        return age.days < max_age_days
    
    def download_symbol(
        self, 
        symbol: str, 
        start: str = '2015-01-01', 
        end: Optional[str] = None,
        use_cache: bool = True,
        force_download: bool = False
    ) -> pd.DataFrame:
        """
        Download historical data for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol (e.g., 'AAPL', 'MSFT')
        start : str
            Start date in 'YYYY-MM-DD' format
        end : str, optional
            End date in 'YYYY-MM-DD' format. If None, uses current date.
        use_cache : bool
            Whether to use cached data if available
        force_download : bool
            Force re-download even if cache exists
            
        Returns:
        --------
        pd.DataFrame : Historical OHLCV data with DatetimeIndex
        """
        cache_path = self.get_cache_path(symbol)
        
        # Check cache
        if use_cache and not force_download and self.is_cached(symbol):
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                print(f"✓ Loaded {symbol} from cache ({len(df)} rows)")
                return df
            except Exception as e:
                print(f"⚠ Cache read failed for {symbol}: {e}")
        
        # Download from Yahoo Finance
        try:
            print(f"↓ Downloading {symbol}...")
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start, end=end, auto_adjust=True)
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Clean up columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            df.index.name = 'Date'
            
            # Save to cache
            df.to_csv(cache_path)
            print(f"✓ Downloaded {symbol} ({len(df)} rows, {df.index[0].date()} to {df.index[-1].date()})")
            
            return df
            
        except Exception as e:
            print(f"✗ Failed to download {symbol}: {e}")
            return pd.DataFrame()
    
    def download_multiple(
        self,
        symbols: List[str],
        start: str = '2015-01-01',
        end: Optional[str] = None,
        use_cache: bool = True,
        force_download: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of ticker symbols
        start : str
            Start date
        end : str, optional
            End date
        use_cache : bool
            Whether to use cached data
        force_download : bool
            Force re-download
            
        Returns:
        --------
        Dict[str, pd.DataFrame] : Dictionary mapping symbols to DataFrames
        """
        data = {}
        
        print(f"\nDownloading {len(symbols)} symbols...")
        for symbol in tqdm(symbols):
            df = self.download_symbol(
                symbol, 
                start=start, 
                end=end, 
                use_cache=use_cache,
                force_download=force_download
            )
            if not df.empty:
                data[symbol] = df
        
        print(f"\n✓ Successfully downloaded {len(data)}/{len(symbols)} symbols")
        return data
    
    def load_symbol(self, symbol: str) -> pd.DataFrame:
        """
        Load a symbol from cache.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
            
        Returns:
        --------
        pd.DataFrame : Cached data, or empty DataFrame if not found
        """
        cache_path = self.get_cache_path(symbol)
        
        if not os.path.exists(cache_path):
            print(f"⚠ No cached data for {symbol}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df
        except Exception as e:
            print(f"✗ Failed to load {symbol}: {e}")
            return pd.DataFrame()
    
    def get_price_series(
        self, 
        symbol: str, 
        price_type: str = 'Close'
    ) -> pd.Series:
        """
        Get a price series for a symbol.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
        price_type : str
            Price type: 'Open', 'High', 'Low', 'Close'
            
        Returns:
        --------
        pd.Series : Price series
        """
        df = self.load_symbol(symbol)
        
        if df.empty:
            return pd.Series()
        
        if price_type not in df.columns:
            raise ValueError(f"Price type '{price_type}' not found. Available: {df.columns.tolist()}")
        
        return df[price_type]
    
    def align_data(
        self, 
        symbols: List[str], 
        price_type: str = 'Close'
    ) -> pd.DataFrame:
        """
        Load and align price data for multiple symbols.
        
        Parameters:
        -----------
        symbols : List[str]
            List of symbols
        price_type : str
            Price type to extract
            
        Returns:
        --------
        pd.DataFrame : Aligned price data with symbols as columns
        """
        prices = {}
        
        for symbol in symbols:
            series = self.get_price_series(symbol, price_type)
            if not series.empty:
                prices[symbol] = series
        
        if not prices:
            return pd.DataFrame()
        
        # Combine and forward-fill missing values
        df = pd.DataFrame(prices)
        df = df.ffill().dropna()
        
        return df
    
    def get_data_summary(self, symbol: str) -> Dict:
        """
        Get summary statistics for a symbol's data.
        
        Parameters:
        -----------
        symbol : str
            Ticker symbol
            
        Returns:
        --------
        Dict : Summary statistics
        """
        df = self.load_symbol(symbol)
        
        if df.empty:
            return {}
        
        return {
            'symbol': symbol,
            'start_date': df.index[0].date(),
            'end_date': df.index[-1].date(),
            'num_rows': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'avg_volume': df['Volume'].mean(),
            'price_range': {
                'min': df['Close'].min(),
                'max': df['Close'].max(),
                'current': df['Close'].iloc[-1]
            }
        }


# Convenience functions
def download_symbol(symbol: str, start: str = '2015-01-01', 
                   end: Optional[str] = None, use_cache: bool = True) -> pd.DataFrame:
    """Download a single symbol (convenience function)."""
    downloader = DataDownloader()
    return downloader.download_symbol(symbol, start, end, use_cache)


def download_multiple(symbols: List[str], start: str = '2015-01-01',
                     end: Optional[str] = None, use_cache: bool = True) -> Dict[str, pd.DataFrame]:
    """Download multiple symbols (convenience function)."""
    downloader = DataDownloader()
    return downloader.download_multiple(symbols, start, end, use_cache)


def load_symbol(symbol: str) -> pd.DataFrame:
    """Load a symbol from cache (convenience function)."""
    downloader = DataDownloader()
    return downloader.load_symbol(symbol)


# Example usage and testing
if __name__ == "__main__":
    # Example: Download some tech stocks
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
    
    downloader = DataDownloader()
    data = downloader.download_multiple(symbols, start='2020-01-01')
    
    # Print summaries
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    for symbol in symbols:
        summary = downloader.get_data_summary(symbol)
        if summary:
            print(f"\n{symbol}:")
            print(f"  Date range: {summary['start_date']} to {summary['end_date']}")
            print(f"  Rows: {summary['num_rows']}")
            print(f"  Price: ${summary['price_range']['current']:.2f}")
