"""
Portfolio Plotting Module

This module provides visualization functions for portfolio optimization results,
including efficient frontier plots.
"""

from typing import Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .optimizer import OptimizationResult


def plot_efficient_frontier(
    returns: np.ndarray,
    volatilities: np.ndarray,
    sharpe_ratios: Optional[np.ndarray] = None,
    min_var_result: Optional[OptimizationResult] = None,
    max_sharpe_result: Optional[OptimizationResult] = None,
    individual_assets: Optional[pd.DataFrame] = None,
    user_portfolio: Optional[dict] = None,
    risk_free_rate: float = 0.0,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot the efficient frontier with optional optimal portfolios.
    
    Parameters
    ----------
    returns : np.ndarray
        Array of portfolio returns for efficient frontier points
    volatilities : np.ndarray
        Array of portfolio volatilities for efficient frontier points
    sharpe_ratios : Optional[np.ndarray], default=None
        Array of Sharpe ratios for efficient frontier points
    min_var_result : Optional[OptimizationResult], default=None
        Minimum variance portfolio result to plot
    max_sharpe_result : Optional[OptimizationResult], default=None
        Maximum Sharpe ratio portfolio result to plot
    individual_assets : Optional[pd.DataFrame], default=None
        DataFrame with individual asset returns and volatilities (columns: 'return', 'volatility')
    risk_free_rate : float, default=0.0
        Risk-free rate for reference line
    save_path : Optional[str], default=None
        Path to save the figure. If None, figure is not saved.
    figsize : Tuple[int, int], default=(12, 8)
        Figure size (width, height)
    
    Returns
    -------
    plt.Figure
        Matplotlib figure object
    
    Examples
    --------
    >>> from portfolio.plotting import plot_efficient_frontier
    >>> 
    >>> fig = plot_efficient_frontier(
    ...     returns=ef_returns,
    ...     volatilities=ef_volatilities,
    ...     min_var_result=min_var_portfolio,
    ...     max_sharpe_result=max_sharpe_portfolio
    ... )
    >>> plt.show()
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot efficient frontier
    ax.plot(volatilities, returns, 'b-', linewidth=2, label='Efficient Frontier', alpha=0.7)
    
    # Plot individual assets if provided
    if individual_assets is not None:
        ax.scatter(
            individual_assets['volatility'],
            individual_assets['return'],
            s=100,
            alpha=0.6,
            c='gray',
            marker='o',
            label='Individual Assets',
            edgecolors='black',
            linewidths=1
        )
        
        # Annotate asset names
        for idx, row in individual_assets.iterrows():
            ax.annotate(
                idx,
                (row['volatility'], row['return']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9,
                alpha=0.7
            )
    
    # Plot user portfolio point if provided
    if user_portfolio is not None:
        ax.scatter(
            user_portfolio.get('volatility'),
            user_portfolio.get('expected_return'),
            s=180,
            c='blue',
            marker='D',
            label=user_portfolio.get('label', 'User Portfolio'),
            edgecolors='black',
            linewidths=2,
            zorder=6
        )

    # Plot minimum variance portfolio
    if min_var_result is not None:
        ax.scatter(
            min_var_result.volatility,
            min_var_result.expected_return,
            s=200,
            c='green',
            marker='*',
            label=f'Min Variance Portfolio\n(Sharpe: {min_var_result.sharpe_ratio:.3f})',
            edgecolors='black',
            linewidths=2,
            zorder=5
        )
    
    # Plot maximum Sharpe ratio portfolio
    if max_sharpe_result is not None:
        ax.scatter(
            max_sharpe_result.volatility,
            max_sharpe_result.expected_return,
            s=200,
            c='red',
            marker='*',
            label=f'Max Sharpe Portfolio\n(Sharpe: {max_sharpe_result.sharpe_ratio:.3f})',
            edgecolors='black',
            linewidths=2,
            zorder=5
        )
    
    # Plot risk-free rate line (Capital Market Line)
    if risk_free_rate > 0 and max_sharpe_result is not None:
        # Draw line from risk-free rate through max Sharpe portfolio
        x_line = np.array([0, max_sharpe_result.volatility * 1.2])
        y_line = risk_free_rate + (max_sharpe_result.sharpe_ratio * x_line)
        ax.plot(x_line, y_line, 'r--', linewidth=1.5, alpha=0.5, label='Capital Market Line')
    
    # Formatting
    ax.set_xlabel('Volatility (Annualized)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Expected Return (Annualized)', fontsize=12, fontweight='bold')
    ax.set_title('Efficient Frontier', fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=10)
    
    # Format axes as percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    return fig

def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    title: str = "Correlation Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot correlation matrix as a heatmap.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Use seaborn if available, else matplotlib
    try:
        import seaborn as sns
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            square=True,
            ax=ax,
            cbar_kws={"shrink": .8}
        )
    except ImportError:
        # Fallback to matplotlib
        cax = ax.imshow(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
        fig.colorbar(cax)
        
        # Annotations
        for i in range(len(correlation_matrix)):
            for j in range(len(correlation_matrix)):
                text = ax.text(j, i, f"{correlation_matrix.iloc[i, j]:.2f}",
                               ha="center", va="center", color="black", fontsize=9)
                               
        # Ticks
        ax.set_xticks(np.arange(len(correlation_matrix)))
        ax.set_yticks(np.arange(len(correlation_matrix)))
        ax.set_xticklabels(correlation_matrix.columns, rotation=45, ha="right")
        ax.set_yticklabels(correlation_matrix.index)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
        
    return fig

def plot_drawdown_chart(
    cumulative_returns: pd.Series,
    drawdowns: pd.Series,
    title: str = "Portfolio Drawdown Analysis",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot Cumulative Returns and Drawdowns.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot Cumulative Returns
    ax1.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Return', color='blue', linewidth=1.5)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.set_ylabel('Growth of $1')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot Drawdowns (Area Chart)
    ax2.fill_between(drawdowns.index, drawdowns, 0, color='red', alpha=0.3, label='Drawdown')
    ax2.plot(drawdowns.index, drawdowns, color='red', linewidth=0.5, alpha=0.8)
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_xlabel('Date')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower left')
    
    # Format Y-axis as percent for drawdown
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Drawdown chart saved to: {save_path}")
        
    return fig
