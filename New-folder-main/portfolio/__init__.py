"""
Portfolio Optimization Module

This module provides portfolio optimization capabilities including:
- Mean-variance optimization (Markowitz)
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Efficient frontier visualization
"""

__version__ = "0.1.0"

from .optimizer import (
    PortfolioOptimizer,
    compute_daily_returns,
    compute_expected_returns,
    compute_covariance_matrix
)
from .plotting import plot_efficient_frontier

__all__ = [
    'PortfolioOptimizer',
    'compute_daily_returns',
    'compute_expected_returns',
    'compute_covariance_matrix',
    'plot_efficient_frontier'
]
