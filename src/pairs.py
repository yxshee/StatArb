"""
Pair selection module for statistical arbitrage.
Implements correlation analysis, cointegration testing, and pair ranking.
"""
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from statsmodels.tsa.stattools import coint
from statsmodels.api import OLS, add_constant
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from .utils import get_data_dir, get_results_dir
except ImportError:
    from utils import get_data_dir, get_results_dir


class PairSelector:
    """
    Identifies and ranks tradeable pairs using statistical methods.
    """
    
    def __init__(
        self,
        prices: pd.DataFrame,
        correlation_threshold: float = 0.85,
        coint_pvalue_threshold: float = 0.05,
        lookback_window: int = 252
    ):
        """
        Initialize pair selector.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Price data with symbols as columns, dates as index
        correlation_threshold : float
            Minimum correlation for pair candidates (default: 0.85)
        coint_pvalue_threshold : float
            Maximum p-value for cointegration test (default: 0.05)
        lookback_window : int
            Window for rolling statistics (default: 252 days)
        """
        self.prices = prices
        self.correlation_threshold = correlation_threshold
        self.coint_pvalue_threshold = coint_pvalue_threshold
        self.lookback_window = lookback_window
        
        self.correlation_matrix = None
        self.high_corr_pairs = None
        self.cointegrated_pairs = None
        
    def compute_correlation(self) -> pd.DataFrame:
        """
        Compute correlation matrix on price data.
        
        Returns:
        --------
        pd.DataFrame : Correlation matrix
        """
        print("Computing correlation matrix...")
        self.correlation_matrix = self.prices.corr()
        return self.correlation_matrix
    
    def find_high_correlation_pairs(self) -> pd.DataFrame:
        """
        Find pairs with correlation above threshold.
        
        Returns:
        --------
        pd.DataFrame : Pairs with high correlation
        """
        if self.correlation_matrix is None:
            self.compute_correlation()
        
        pairs = []
        n = len(self.correlation_matrix)
        
        # Get upper triangle indices (avoid duplicates)
        for i in range(n):
            for j in range(i + 1, n):
                corr_val = self.correlation_matrix.iloc[i, j]
                
                if corr_val >= self.correlation_threshold:
                    pairs.append({
                        'symbol_1': self.correlation_matrix.index[i],
                        'symbol_2': self.correlation_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        self.high_corr_pairs = pd.DataFrame(pairs).sort_values('correlation', ascending=False)
        
        print(f"✓ Found {len(self.high_corr_pairs)} pairs with correlation >= {self.correlation_threshold}")
        return self.high_corr_pairs
    
    def test_cointegration(
        self,
        series_a: pd.Series,
        series_b: pd.Series
    ) -> Tuple[float, float, Dict]:
        """
        Test cointegration between two price series using Engle-Granger method.
        
        Parameters:
        -----------
        series_a : pd.Series
            First price series
        series_b : pd.Series
            Second price series
            
        Returns:
        --------
        Tuple[float, float, Dict] : (test_statistic, p_value, details_dict)
        """
        # Align series
        aligned = pd.DataFrame({'A': series_a, 'B': series_b}).dropna()
        
        if len(aligned) < 30:  # Minimum data points
            return np.nan, 1.0, {}
        
        # Run cointegration test
        score, pvalue, _ = coint(aligned['A'], aligned['B'])
        
        # Additional details
        details = {
            'n_obs': len(aligned),
            'test_statistic': score
        }
        
        return score, pvalue, details
    
    def find_cointegrated_pairs(
        self,
        candidate_pairs: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Test cointegration for candidate pairs.
        
        Parameters:
        -----------
        candidate_pairs : pd.DataFrame, optional
            Pairs to test. If None, uses high_corr_pairs.
            
        Returns:
        --------
        pd.DataFrame : Cointegrated pairs with test results
        """
        if candidate_pairs is None:
            if self.high_corr_pairs is None:
                self.find_high_correlation_pairs()
            candidate_pairs = self.high_corr_pairs
        
        print(f"\nTesting cointegration for {len(candidate_pairs)} pairs...")
        
        results = []
        
        for idx, row in candidate_pairs.iterrows():
            sym1, sym2 = row['symbol_1'], row['symbol_2']
            
            # Get price series
            series_a = self.prices[sym1]
            series_b = self.prices[sym2]
            
            # Test cointegration
            score, pvalue, details = self.test_cointegration(series_a, series_b)
            
            if pvalue <= self.coint_pvalue_threshold:
                results.append({
                    'symbol_1': sym1,
                    'symbol_2': sym2,
                    'correlation': row['correlation'],
                    'coint_pvalue': pvalue,
                    'coint_stat': score,
                    'n_obs': details.get('n_obs', len(series_a))
                })
        
        self.cointegrated_pairs = pd.DataFrame(results).sort_values('coint_pvalue')
        
        print(f"✓ Found {len(self.cointegrated_pairs)} cointegrated pairs (p-value <= {self.coint_pvalue_threshold})")
        return self.cointegrated_pairs
    
    def estimate_hedge_ratio(
        self,
        symbol_1: str,
        symbol_2: str,
        method: str = 'ols'
    ) -> float:
        """
        Estimate hedge ratio (beta) between two assets.
        
        Parameters:
        -----------
        symbol_1 : str
            First symbol (dependent variable)
        symbol_2 : str
            Second symbol (independent variable)
        method : str
            Method: 'ols' or 'tls' (total least squares)
            
        Returns:
        --------
        float : Hedge ratio (beta)
        """
        series_a = self.prices[symbol_1].dropna()
        series_b = self.prices[symbol_2].dropna()
        
        # Align series
        aligned = pd.DataFrame({'A': series_a, 'B': series_b}).dropna()
        
        if method == 'ols':
            # OLS regression: A = alpha + beta * B + epsilon
            X = add_constant(aligned['B'])
            model = OLS(aligned['A'], X).fit()
            beta = model.params['B']
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return beta
    
    def calculate_spread(
        self,
        symbol_1: str,
        symbol_2: str,
        beta: Optional[float] = None
    ) -> pd.Series:
        """
        Calculate spread between two assets.
        
        Parameters:
        -----------
        symbol_1 : str
            First symbol
        symbol_2 : str
            Second symbol
        beta : float, optional
            Hedge ratio. If None, estimates using OLS.
            
        Returns:
        --------
        pd.Series : Spread time series
        """
        if beta is None:
            beta = self.estimate_hedge_ratio(symbol_1, symbol_2)
        
        series_a = self.prices[symbol_1]
        series_b = self.prices[symbol_2]
        
        spread = series_a - beta * series_b
        return spread
    
    def calculate_zscore(
        self,
        spread: pd.Series,
        window: Optional[int] = None
    ) -> pd.Series:
        """
        Calculate z-score of spread using rolling statistics.
        
        Parameters:
        -----------
        spread : pd.Series
            Spread time series
        window : int, optional
            Lookback window. If None, uses self.lookback_window.
            
        Returns:
        --------
        pd.Series : Z-score time series
        """
        if window is None:
            window = self.lookback_window
        
        rolling_mean = spread.rolling(window=window, min_periods=30).mean()
        rolling_std = spread.rolling(window=window, min_periods=30).std()
        
        zscore = (spread - rolling_mean) / rolling_std
        return zscore
    
    def analyze_spread_stability(
        self,
        spread: pd.Series
    ) -> Dict:
        """
        Analyze spread characteristics for trading suitability.
        
        Parameters:
        -----------
        spread : pd.Series
            Spread time series
            
        Returns:
        --------
        Dict : Statistics about spread stability
        """
        # Test for stationarity using Augmented Dickey-Fuller
        from statsmodels.tsa.stattools import adfuller
        
        clean_spread = spread.dropna()
        
        if len(clean_spread) < 30:
            return {}
        
        adf_result = adfuller(clean_spread, maxlag=1)
        
        return {
            'mean': clean_spread.mean(),
            'std': clean_spread.std(),
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'is_stationary': adf_result[1] < 0.05,
            'half_life': self._calculate_half_life(clean_spread)
        }
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate mean-reversion half-life using Ornstein-Uhlenbeck process.
        
        Parameters:
        -----------
        spread : pd.Series
            Spread time series
            
        Returns:
        --------
        float : Half-life in days
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Align
        aligned = pd.DataFrame({
            'lag': spread_lag,
            'diff': spread_diff
        }).dropna()
        
        if len(aligned) < 10:
            return np.nan
        
        # Regression: diff = lambda * lag + epsilon
        X = aligned['lag']
        y = aligned['diff']
        
        model = OLS(y, X).fit()
        lambda_param = model.params[0]
        
        if lambda_param >= 0:
            return np.nan  # Not mean-reverting
        
        half_life = -np.log(2) / lambda_param
        return half_life
    
    def rank_pairs(self) -> pd.DataFrame:
        """
        Rank cointegrated pairs by trading suitability.
        
        Returns:
        --------
        pd.DataFrame : Ranked pairs with scores
        """
        if self.cointegrated_pairs is None or len(self.cointegrated_pairs) == 0:
            print("No cointegrated pairs to rank")
            return pd.DataFrame()
        
        print("\nRanking pairs by trading suitability...")
        
        ranked = []
        
        for idx, row in self.cointegrated_pairs.iterrows():
            sym1, sym2 = row['symbol_1'], row['symbol_2']
            
            # Estimate beta
            beta = self.estimate_hedge_ratio(sym1, sym2)
            
            # Calculate spread
            spread = self.calculate_spread(sym1, sym2, beta)
            
            # Analyze spread
            spread_stats = self.analyze_spread_stability(spread)
            
            # Calculate composite score
            # Lower p-value = better
            # Higher correlation = better
            # Lower half-life = better (faster mean reversion)
            # Lower ADF p-value = better (more stationary)
            
            score = (
                (1 - row['coint_pvalue']) * 0.4 +  # 40% weight
                row['correlation'] * 0.3 +           # 30% weight
                (1 - spread_stats.get('adf_pvalue', 1)) * 0.3  # 30% weight
            )
            
            ranked.append({
                'symbol_1': sym1,
                'symbol_2': sym2,
                'correlation': row['correlation'],
                'coint_pvalue': row['coint_pvalue'],
                'beta': beta,
                'spread_mean': spread_stats.get('mean', np.nan),
                'spread_std': spread_stats.get('std', np.nan),
                'half_life': spread_stats.get('half_life', np.nan),
                'is_stationary': spread_stats.get('is_stationary', False),
                'adf_pvalue': spread_stats.get('adf_pvalue', np.nan),
                'composite_score': score
            })
        
        ranked_df = pd.DataFrame(ranked).sort_values('composite_score', ascending=False)
        
        print(f"✓ Ranked {len(ranked_df)} pairs")
        return ranked_df
    
    def run_full_pipeline(self) -> pd.DataFrame:
        """
        Run the complete pair selection pipeline.
        
        Returns:
        --------
        pd.DataFrame : Final ranked pairs
        """
        print("="*70)
        print("PAIR SELECTION PIPELINE")
        print("="*70)
        
        # Step 1: Correlation
        self.find_high_correlation_pairs()
        
        # Step 2: Cointegration
        self.find_cointegrated_pairs()
        
        # Step 3: Ranking
        ranked_pairs = self.rank_pairs()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70)
        print(f"✓ High correlation pairs: {len(self.high_corr_pairs) if self.high_corr_pairs is not None else 0}")
        print(f"✓ Cointegrated pairs: {len(self.cointegrated_pairs) if self.cointegrated_pairs is not None else 0}")
        print(f"✓ Final ranked pairs: {len(ranked_pairs)}")
        
        return ranked_pairs


# Convenience functions
def find_pairs(
    prices: pd.DataFrame,
    correlation_threshold: float = 0.85,
    coint_pvalue_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Find and rank tradeable pairs (convenience function).
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    correlation_threshold : float
        Minimum correlation
    coint_pvalue_threshold : float
        Maximum cointegration p-value
        
    Returns:
    --------
    pd.DataFrame : Ranked pairs
    """
    selector = PairSelector(
        prices=prices,
        correlation_threshold=correlation_threshold,
        coint_pvalue_threshold=coint_pvalue_threshold
    )
    
    return selector.run_full_pipeline()


def test_pair(
    prices: pd.DataFrame,
    symbol_1: str,
    symbol_2: str
) -> Dict:
    """
    Test a specific pair (convenience function).
    
    Parameters:
    -----------
    prices : pd.DataFrame
        Price data
    symbol_1 : str
        First symbol
    symbol_2 : str
        Second symbol
        
    Returns:
    --------
    Dict : Test results
    """
    selector = PairSelector(prices=prices)
    
    # Correlation
    corr = prices[[symbol_1, symbol_2]].corr().iloc[0, 1]
    
    # Cointegration
    score, pvalue, details = selector.test_cointegration(
        prices[symbol_1],
        prices[symbol_2]
    )
    
    # Beta
    beta = selector.estimate_hedge_ratio(symbol_1, symbol_2)
    
    # Spread
    spread = selector.calculate_spread(symbol_1, symbol_2, beta)
    spread_stats = selector.analyze_spread_stability(spread)
    
    return {
        'symbol_1': symbol_1,
        'symbol_2': symbol_2,
        'correlation': corr,
        'coint_pvalue': pvalue,
        'coint_stat': score,
        'beta': beta,
        **spread_stats
    }


if __name__ == "__main__":
    # Example usage
    print("Pairs module loaded successfully!")
    print("Use PairSelector class or convenience functions find_pairs() and test_pair()")
