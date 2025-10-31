"""
Test script for data module - downloads sample data for testing
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataDownloader

def main():
    """Download sample data for the trading engine."""
    
    # Define a diverse universe of liquid stocks
    # Tech stocks
    tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
    
    # Financial stocks
    financials = ['JPM', 'BAC', 'GS', 'WFC', 'MS']
    
    # Consumer stocks
    consumer = ['WMT', 'HD', 'NKE', 'MCD', 'SBUX']
    
    # Energy stocks
    energy = ['XOM', 'CVX', 'COP']
    
    # ETFs (good for pairs)
    etfs = ['SPY', 'QQQ', 'IWM', 'DIA']
    
    # Combine all symbols
    all_symbols = tech_stocks + financials + consumer + energy + etfs
    
    print("="*70)
    print("DOWNLOADING SAMPLE DATA FOR STATISTICAL ARBITRAGE ENGINE")
    print("="*70)
    print(f"\nTotal symbols: {len(all_symbols)}")
    print(f"Date range: 2018-01-01 to present")
    print(f"Categories: Tech ({len(tech_stocks)}), Finance ({len(financials)}), "
          f"Consumer ({len(consumer)}), Energy ({len(energy)}), ETFs ({len(etfs)})")
    
    # Download data
    downloader = DataDownloader()
    data = downloader.download_multiple(
        symbols=all_symbols,
        start='2018-01-01',
        use_cache=True
    )
    
    # Print detailed summaries
    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    
    for symbol in all_symbols:
        if symbol in data:
            summary = downloader.get_data_summary(symbol)
            print(f"\n{symbol:6s} | {summary['start_date']} to {summary['end_date']} "
                  f"| {summary['num_rows']:4d} days | Last: ${summary['price_range']['current']:8.2f}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("DOWNLOAD COMPLETE")
    print("="*70)
    print(f"✓ Successfully downloaded: {len(data)}/{len(all_symbols)} symbols")
    print(f"✓ Data stored in: {downloader.data_dir}")
    print("\nReady for pair selection and analysis!")

if __name__ == "__main__":
    main()
