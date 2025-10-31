"""
Complete end-to-end demo of the statistical arbitrage trading engine.
Demonstrates: data loading, pair selection, strategy backtesting, and performance analysis.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import DataDownloader
from src.pairs import PairSelector
from src.strategy import StrategyConfig, PairsStrategy
from src.backtest import Backtester, BacktestConfig
from src.risk import RiskConfig
from src.metrics import create_tear_sheet, plot_equity_curve, plot_trade_analysis
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def main():
    """Run complete pipeline demonstration."""
    
    print("="*80)
    print(" STATISTICAL ARBITRAGE TRADING ENGINE - COMPLETE DEMO")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA LOADING")
    print("="*80)
    
    downloader = DataDownloader()
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',
               'JPM', 'BAC', 'GS', 'WFC', 'MS',
               'WMT', 'HD', 'MCD', 'SBUX',
               'XOM', 'CVX', 'COP',
               'SPY', 'QQQ', 'IWM', 'DIA']
    
    prices = downloader.align_data(symbols, price_type='Close')
    print(f"✓ Loaded {len(symbols)} symbols")
    print(f"✓ Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"✓ Total trading days: {len(prices)}")
    
    # ========================================================================
    # STEP 2: Pair Selection
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 2: PAIR SELECTION")
    print("="*80)
    
    selector = PairSelector(
        prices=prices,
        correlation_threshold=0.85,
        coint_pvalue_threshold=0.05,
        lookback_window=60
    )
    
    # Run full pipeline
    ranked_pairs = selector.run_full_pipeline()
    
    if len(ranked_pairs) == 0:
        print("⚠ No pairs found. Exiting.")
        return
    
    # Display top 5 pairs
    print("\n📊 TOP 5 PAIRS:")
    print(ranked_pairs[['symbol_1', 'symbol_2', 'correlation', 'coint_pvalue', 
                        'beta', 'half_life', 'composite_score']].head(5).to_string(index=False))
    
    # ========================================================================
    # STEP 3: Single-Pair Backtest
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 3: SINGLE-PAIR BACKTESTING")
    print("="*80)
    
    # Select top pair
    top_pair = ranked_pairs.iloc[0]
    symbol_1 = top_pair['symbol_1']
    symbol_2 = top_pair['symbol_2']
    beta = top_pair['beta']
    
    print(f"\nTesting pair: {symbol_1} / {symbol_2}")
    print(f"  Correlation: {top_pair['correlation']:.4f}")
    print(f"  Cointegration p-value: {top_pair['coint_pvalue']:.6f}")
    print(f"  Hedge ratio (beta): {beta:.4f}")
    print(f"  Half-life: {top_pair['half_life']:.1f} days")
    
    # Configure strategy
    strategy_config = StrategyConfig(
        z_entry_threshold=2.0,
        z_exit_threshold=0.5,
        lookback_window=60
    )
    
    # Configure risk
    risk_config = RiskConfig(
        total_capital=1000000.0,
        max_position_pct=0.10,  # Use 10% of capital per trade
        stop_loss_pct=0.03,
        max_drawdown_pct=0.15
    )
    
    # Configure backtest
    backtest_config = BacktestConfig(
        initial_capital=1000000.0,
        commission_pct=0.0005,  # 5 bps per leg
        slippage_pct=0.0001,    # 1 bp per leg
        use_stop_loss=True
    )
    
    # Run backtest
    backtester = Backtester(
        prices=prices,
        symbol_1=symbol_1,
        symbol_2=symbol_2,
        beta=beta,
        strategy_config=strategy_config,
        risk_config=risk_config,
        backtest_config=backtest_config
    )
    
    results = backtester.run()
    backtester.print_summary(results)
    
    # ========================================================================
    # STEP 4: Performance Analysis
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: PERFORMANCE VISUALIZATION")
    print("="*80)
    
    # Create tear sheet
    create_tear_sheet(results, symbol_1, symbol_2)
    
    # ========================================================================
    # STEP 5: Parameter Optimization (Optional)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    print("\nTesting different z-score thresholds...")
    
    sensitivity_results = []
    
    for z_entry in [1.5, 2.0, 2.5]:
        for lookback in [30, 60, 90]:
            config = StrategyConfig(
                z_entry_threshold=z_entry,
                z_exit_threshold=0.5,
                lookback_window=lookback
            )
            
            bt = Backtester(
                prices=prices,
                symbol_1=symbol_1,
                symbol_2=symbol_2,
                beta=beta,
                strategy_config=config,
                risk_config=risk_config,
                backtest_config=backtest_config
            )
            
            res = bt.run()
            
            sensitivity_results.append({
                'z_entry': z_entry,
                'lookback': lookback,
                'total_return_%': res['total_return_pct'],
                'sharpe_ratio': res['sharpe_ratio'],
                'max_drawdown_%': res['max_drawdown_pct'],
                'n_trades': res['n_trades'],
                'win_rate_%': res['win_rate_pct']
            })
    
    # Display results
    sensitivity_df = pd.DataFrame(sensitivity_results)
    sensitivity_df = sensitivity_df.sort_values('sharpe_ratio', ascending=False)
    
    print("\n📊 PARAMETER SENSITIVITY RESULTS (sorted by Sharpe):")
    print(sensitivity_df.to_string(index=False))
    
    # Save results
    from src.utils import get_results_dir
    output_path = os.path.join(get_results_dir(), f'sensitivity_{symbol_1}_{symbol_2}.csv')
    sensitivity_df.to_csv(output_path, index=False)
    print(f"\n✓ Saved sensitivity analysis to: {output_path}")
    
    # ========================================================================
    # STEP 6: Multi-Pair Portfolio (Top 3 Pairs)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: MULTI-PAIR PORTFOLIO ANALYSIS")
    print("="*80)
    
    print("\nBacktesting top 3 pairs...")
    
    portfolio_results = []
    
    for i in range(min(3, len(ranked_pairs))):
        pair = ranked_pairs.iloc[i]
        sym1, sym2 = pair['symbol_1'], pair['symbol_2']
        beta_val = pair['beta']
        
        print(f"\n  Pair {i+1}: {sym1}/{sym2}")
        
        bt = Backtester(
            prices=prices,
            symbol_1=sym1,
            symbol_2=sym2,
            beta=beta_val,
            strategy_config=strategy_config,
            risk_config=RiskConfig(
                total_capital=1000000.0 / 3,  # Split capital equally
                max_position_pct=0.10
            ),
            backtest_config=backtest_config
        )
        
        res = bt.run()
        
        portfolio_results.append({
            'pair': f"{sym1}/{sym2}",
            'total_return_%': res['total_return_pct'],
            'sharpe': res['sharpe_ratio'],
            'max_dd_%': res['max_drawdown_pct'],
            'n_trades': res['n_trades']
        })
    
    portfolio_df = pd.DataFrame(portfolio_results)
    print("\n📊 PORTFOLIO RESULTS:")
    print(portfolio_df.to_string(index=False))
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("DEMO COMPLETE!")
    print("="*80)
    
    print(f"\n✓ Generated comprehensive analysis for {symbol_1}/{symbol_2}")
    print(f"✓ Performance tear sheet saved to results/")
    print(f"✓ Parameter sensitivity analysis completed")
    print(f"✓ Multi-pair portfolio analyzed")
    
    print("\n🎯 Key Findings:")
    print(f"  • Best pair: {symbol_1}/{symbol_2}")
    print(f"  • Total return: {results['total_return_pct']:.2f}%")
    print(f"  • Sharpe ratio: {results['sharpe_ratio']:.2f}")
    print(f"  • Win rate: {results['win_rate_pct']:.1f}%")
    print(f"  • Number of trades: {results['n_trades']}")
    
    print("\n" + "="*80)
    print("Ready for interviews! 🚀")
    print("="*80)


if __name__ == "__main__":
    main()
