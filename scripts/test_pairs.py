"""
Test script for pairs module - demonstrates pair selection pipeline
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataDownloader
from src.pairs import PairSelector, find_pairs, test_pair
import pandas as pd

def main():
    """Test the pair selection pipeline."""
    
    print("="*70)
    print("TESTING PAIR SELECTION MODULE")
    print("="*70)
    
    # Load data
    downloader = DataDownloader()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
               'JPM', 'BAC', 'GS', 'WFC', 'MS',
               'WMT', 'HD', 'NKE', 'MCD', 'SBUX',
               'XOM', 'CVX', 'COP',
               'SPY', 'QQQ', 'IWM', 'DIA']
    
    prices = downloader.align_data(symbols, price_type='Close')
    print(f"\n✓ Loaded price data: {prices.shape}")
    
    # Run full pipeline
    ranked_pairs = find_pairs(
        prices=prices,
        correlation_threshold=0.85,
        coint_pvalue_threshold=0.05
    )
    
    # Display top pairs
    print("\n" + "="*70)
    print("TOP 10 TRADEABLE PAIRS")
    print("="*70)
    print(ranked_pairs.head(10).to_string(index=False))
    
    # Save results
    import os
    from src.utils import get_results_dir
    output_path = os.path.join(get_results_dir(), 'ranked_pairs.csv')
    ranked_pairs.to_csv(output_path, index=False)
    print(f"\n✓ Saved ranked pairs to: {output_path}")
    
    # Test a specific pair
    if len(ranked_pairs) > 0:
        print("\n" + "="*70)
        print("DETAILED ANALYSIS OF TOP PAIR")
        print("="*70)
        
        top_pair = ranked_pairs.iloc[0]
        sym1, sym2 = top_pair['symbol_1'], top_pair['symbol_2']
        
        print(f"\nPair: {sym1} / {sym2}")
        print(f"Correlation: {top_pair['correlation']:.4f}")
        print(f"Cointegration p-value: {top_pair['coint_pvalue']:.6f}")
        print(f"Hedge ratio (beta): {top_pair['beta']:.4f}")
        print(f"Spread mean: {top_pair['spread_mean']:.4f}")
        print(f"Spread std: {top_pair['spread_std']:.4f}")
        print(f"Half-life: {top_pair['half_life']:.2f} days")
        print(f"Stationary: {top_pair['is_stationary']}")
        print(f"Composite score: {top_pair['composite_score']:.4f}")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
