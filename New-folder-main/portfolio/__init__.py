"""
Portfolio Optimization Module

This module provides portfolio optimization capabilities including:
- Mean-variance optimization (Markowitz)
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Efficient frontier visualization
- Investor-friendly output and explanations
"""

__version__ = "0.1.0"

from .optimizer import (
    PortfolioOptimizer,
    compute_daily_returns,
    compute_expected_returns,
    compute_covariance_matrix,
    compare_covariance_methods,
    compute_weight_dispersion
)
from .plotting import plot_efficient_frontier
from .investor_guide import (
    format_portfolio_summary,
    format_comparison_summary,
    create_investor_friendly_csv,
    create_simple_weights_csv,
    format_risk_summary,
    format_scenario_result,
    format_stress_test_result,
    get_scenario_menu,
    interpret_var,
    interpret_volatility,
    interpret_sharpe_ratio,
    interpret_diversification
)

__all__ = [
    'PortfolioOptimizer',
    'compute_daily_returns',
    'compute_expected_returns',
    'compute_covariance_matrix',
    'plot_efficient_frontier',
    'compare_covariance_methods',
    'compute_weight_dispersion',
    'format_portfolio_summary',
    'format_comparison_summary',
    'create_investor_friendly_csv',
    'create_simple_weights_csv',
    'format_risk_summary',
    'format_scenario_result',
    'format_stress_test_result',
    'get_scenario_menu',
    'interpret_var',
    'interpret_volatility',
    'interpret_sharpe_ratio',
    'interpret_diversification'
]
