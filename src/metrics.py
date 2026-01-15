"""
Performance metrics and visualization module.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os

try:
    from .utils import get_results_dir
except ImportError:
    from utils import get_results_dir


def plot_equity_curve(
    equity_curve: pd.Series,
    title: str = "Equity Curve",
    save_path: Optional[str] = None
):
    """Plot equity curve with drawdown."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1])
    
    # Equity curve
    ax1.plot(equity_curve.index, equity_curve.values, linewidth=2, label='Equity')
    ax1.fill_between(equity_curve.index, equity_curve.values, alpha=0.3)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=11)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # Drawdown
    rolling_max = equity_curve.expanding().max()
    drawdown = (equity_curve - rolling_max) / rolling_max * 100
    
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
    ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=11)
    ax2.set_ylabel('Drawdown (%)', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()


def plot_trade_analysis(
    trade_log: pd.DataFrame,
    title: str = "Trade Analysis",
    save_path: Optional[str] = None
):
    """Plot trade analysis charts."""
    if len(trade_log) == 0:
        print("No trades to plot")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. P&L distribution
    axes[0, 0].hist(trade_log['net_pnl'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].axvline(trade_log['net_pnl'].mean(), color='green', linestyle='--', 
                       linewidth=2, label=f"Mean: ${trade_log['net_pnl'].mean():,.0f}")
    axes[0, 0].set_title('P&L Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Net P&L ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cumulative P&L
    cumulative_pnl = trade_log['net_pnl'].cumsum()
    axes[0, 1].plot(range(len(cumulative_pnl)), cumulative_pnl.values, linewidth=2, color='green')
    axes[0, 1].fill_between(range(len(cumulative_pnl)), cumulative_pnl.values, alpha=0.3, color='green')
    axes[0, 1].set_title('Cumulative P&L', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Trade Number')
    axes[0, 1].set_ylabel('Cumulative P&L ($)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].ticklabel_format(style='plain', axis='y')
    
    # 3. Holding period vs P&L
    colors = ['green' if x > 0 else 'red' for x in trade_log['net_pnl']]
    axes[1, 0].scatter(trade_log['holding_days'], trade_log['net_pnl'], 
                      c=colors, alpha=0.6, s=50)
    axes[1, 0].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_title('Holding Period vs P&L', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Holding Days')
    axes[1, 0].set_ylabel('Net P&L ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Monthly returns heatmap
    if 'exit_date' in trade_log.columns:
        trade_log['exit_date'] = pd.to_datetime(trade_log['exit_date'])
        monthly_pnl = trade_log.set_index('exit_date')['net_pnl'].resample('M').sum()
        
        if len(monthly_pnl) > 0:
            monthly_returns = pd.DataFrame({
                'year': monthly_pnl.index.year,
                'month': monthly_pnl.index.month,
                'returns': monthly_pnl.values
            })
            
            pivot = monthly_returns.pivot(index='month', columns='year', values='returns')
            
            sns.heatmap(pivot, annot=True, fmt='.0f', cmap='RdYlGn', center=0, 
                       ax=axes[1, 1], cbar_kws={'label': 'P&L ($)'})
            axes[1, 1].set_title('Monthly P&L Heatmap', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Month')
            axes[1, 1].set_xlabel('Year')
        else:
            axes[1, 1].text(0.5, 0.5, 'Insufficient data for heatmap', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved plot to {save_path}")
    
    plt.show()


def create_tear_sheet(
    results: Dict,
    symbol_1: str,
    symbol_2: str,
    save_dir: Optional[str] = None
):
    """Create comprehensive performance tear sheet."""
    if save_dir is None:
        save_dir = get_results_dir()
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 20))
    gs = fig.add_gridspec(5, 2, hspace=0.3, wspace=0.3)
    
    pair_name = f"{symbol_1}/{symbol_2}"
    fig.suptitle(f"Performance Tear Sheet: {pair_name}", fontsize=16, fontweight='bold', y=0.995)
    
    # 1. Equity curve
    ax1 = fig.add_subplot(gs[0, :])
    equity = results['equity_curve']
    ax1.plot(equity.index, equity.values, linewidth=2, color='steelblue')
    ax1.fill_between(equity.index, equity.values, alpha=0.3, color='steelblue')
    ax1.set_title('Equity Curve', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    rolling_max = equity.expanding().max()
    drawdown = (equity - rolling_max) / rolling_max * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
    ax2.plot(drawdown.index, drawdown.values, color='darkred', linewidth=1)
    ax2.set_title('Drawdown', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance metrics table
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Return', f"{results['total_return_pct']:.2f}%"],
        ['CAGR', f"{results['cagr_pct']:.2f}%"],
        ['Sharpe Ratio', f"{results['sharpe_ratio']:.2f}"],
        ['Max Drawdown', f"{results['max_drawdown_pct']:.2f}%"],
        ['Annual Volatility', f"{results['annual_volatility_pct']:.2f}%"],
    ]
    
    table1 = ax3.table(cellText=metrics_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table1.auto_set_font_size(False)
    table1.set_fontsize(10)
    table1.scale(1, 2)
    
    # Style header
    for i in range(2):
        table1[(0, i)].set_facecolor('#4472C4')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Performance Metrics', fontsize=12, fontweight='bold', pad=20)
    
    # 4. Trade statistics table
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    trade_data = [
        ['Metric', 'Value'],
        ['# Trades', f"{results['n_trades']}"],
        ['Win Rate', f"{results['win_rate_pct']:.1f}%"],
        ['Avg Trade P&L', f"${results['avg_trade_pnl']:,.0f}"],
        ['Profit Factor', f"{results['profit_factor']:.2f}"],
        ['Avg Holding Days', f"{results['avg_holding_days']:.1f}"],
    ]
    
    table2 = ax4.table(cellText=trade_data, cellLoc='left', loc='center',
                      colWidths=[0.6, 0.4])
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)
    
    for i in range(2):
        table2[(0, i)].set_facecolor('#4472C4')
        table2[(0, i)].set_text_props(weight='bold', color='white')
    
    ax4.set_title('Trade Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # 5. P&L distribution
    if len(results['trade_log']) > 0:
        ax5 = fig.add_subplot(gs[3, 0])
        trade_log = results['trade_log']
        ax5.hist(trade_log['net_pnl'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2)
        ax5.set_title('P&L Distribution', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Net P&L ($)')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
        
        # 6. Cumulative P&L
        ax6 = fig.add_subplot(gs[3, 1])
        cumulative_pnl = trade_log['net_pnl'].cumsum()
        ax6.plot(range(len(cumulative_pnl)), cumulative_pnl.values, linewidth=2, color='green')
        ax6.fill_between(range(len(cumulative_pnl)), cumulative_pnl.values, alpha=0.3, color='green')
        ax6.set_title('Cumulative P&L', fontsize=12, fontweight='bold')
        ax6.set_xlabel('Trade Number')
        ax6.set_ylabel('Cumulative P&L ($)')
        ax6.grid(True, alpha=0.3)
        ax6.ticklabel_format(style='plain', axis='y')
        
        # 7. Returns over time
        ax7 = fig.add_subplot(gs[4, :])
        returns = equity.pct_change().dropna()
        ax7.bar(returns.index, returns.values * 100, alpha=0.6, color=['g' if x > 0 else 'r' for x in returns.values])
        ax7.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax7.set_title('Daily Returns', fontsize=12, fontweight='bold')
        ax7.set_xlabel('Date')
        ax7.set_ylabel('Daily Return (%)')
        ax7.grid(True, alpha=0.3, axis='y')
    
    # Save
    output_path = os.path.join(save_dir, f'tear_sheet_{symbol_1}_{symbol_2}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved tear sheet to {output_path}")
    
    plt.show()


if __name__ == "__main__":
    print("Performance metrics module loaded successfully!")
