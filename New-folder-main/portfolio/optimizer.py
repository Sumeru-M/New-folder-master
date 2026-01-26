"""
Portfolio Optimizer Module

This module implements Markowitz mean-variance portfolio optimization
using cvxpy for convex optimization.

Features:
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio
- Efficient frontier computation
- Ledoit-Wolf covariance shrinkage for robust estimation
- Weight dispersion metrics for stability analysis
"""

from typing import Dict, Optional, Tuple
import pandas as pd
import numpy as np
import cvxpy as cp
from dataclasses import dataclass


@dataclass
class OptimizationResult:
    """Container for portfolio optimization results."""
    weights: pd.Series
    expected_return: float
    volatility: float
    sharpe_ratio: float
    optimization_type: str


def compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price data.
    
    Uses log returns: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with close prices. Columns are tickers, index is datetime.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with daily log returns. First row will be NaN.
    
    Examples
    --------
    >>> returns = compute_daily_returns(prices)
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_expected_returns(returns: pd.DataFrame, annualized: bool = True) -> pd.Series:
    """
    Compute expected annual returns from daily log returns using geometric mean (CAGR).
    
    For log returns, the cumulative return is:
    exp(sum of log returns) - 1
    
    Then annualized return is:
    (1 + cumulative_return)^(252/n) - 1
    
    This represents the actual compound annual growth rate achieved
    over the period, accounting for volatility drag.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with daily log returns
    annualized : bool, default=True
        If True, annualize returns using geometric mean
    
    Returns
    -------
    pd.Series
        Expected annual returns for each asset
    
    Examples
    --------
    >>> expected_returns = compute_expected_returns(daily_returns)
    """
    if annualized:
        # For log returns: cumulative return = exp(sum of log returns) - 1
        n_periods = len(returns)
        cumulative_return = np.exp(returns.sum()) - 1
        
        # Annualize using CAGR formula
        mean_returns = (1 + cumulative_return) ** (252 / n_periods) - 1
    else:
        # Simple mean of daily returns
        mean_returns = returns.mean()
    
    return mean_returns


def compute_covariance_matrix(returns: pd.DataFrame, annualized: bool = True) -> pd.DataFrame:
    """
    Compute sample covariance matrix of returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with daily log returns
    annualized : bool, default=True
        If True, annualize covariance by multiplying by 252 (trading days per year)
    
    Returns
    -------
    pd.DataFrame
        Sample covariance matrix (annualized if annualized=True)
    
    Examples
    --------
    >>> cov_matrix = compute_covariance_matrix(daily_returns)
    """
    cov_matrix = returns.cov()
    
    if annualized:
        # Annualize: multiply by number of trading days per year
        cov_matrix = cov_matrix * 252
    
    return cov_matrix


def compute_ledoit_wolf_shrinkage(returns: pd.DataFrame, annualized: bool = True) -> Tuple[pd.DataFrame, float]:
    """
    Compute Ledoit-Wolf shrinkage covariance matrix estimator.
    
    The Ledoit-Wolf estimator shrinks the sample covariance matrix towards
    a structured estimator (identity matrix scaled by average variance) to reduce
    estimation error, especially when the number of observations is small relative
    to the number of assets.
    
    Formula: Sigma_shrink = (1 - delta) * S + delta * F
    where:
    - S is the sample covariance matrix
    - F is the shrinkage target (scaled identity matrix)
    - delta is the optimal shrinkage intensity
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with daily log returns. Rows are time periods, columns are assets.
    annualized : bool, default=True
        If True, annualize covariance by multiplying by 252 (trading days per year)
    
    Returns
    -------
    Tuple[pd.DataFrame, float]
        Shrinkage covariance matrix and shrinkage intensity (delta)
    
    References
    ---------
    Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional
    covariance matrices. Journal of multivariate analysis, 88(2), 365-411.
    
    Examples
    --------
    >>> cov_shrink, delta = compute_ledoit_wolf_shrinkage(daily_returns)
    >>> print(f"Shrinkage intensity: {delta:.4f}")
    """
    # Convert to numpy array
    X = returns.values  # Shape: (T, N) where T = time periods, N = assets
    T, N = X.shape
    
    if T < 2:
        raise ValueError("Need at least 2 observations for covariance estimation")
    
    # Center the data (subtract mean)
    X_centered = X - X.mean(axis=0, keepdims=True)
    
    # Sample covariance matrix (unannualized)
    S = (X_centered.T @ X_centered) / (T - 1)
    
    # Shrinkage target: F = mu * I, where mu is average of diagonal elements
    mu = np.trace(S) / N
    F = mu * np.eye(N)
    
    # Compute optimal shrinkage intensity (delta)
    # This is the Ledoit-Wolf formula for optimal shrinkage
    # We compute the bias-variance tradeoff
    
    # Compute pi: sum of squared off-diagonal elements of S
    pi_hat = 0.0
    for i in range(N):
        for j in range(N):
            if i != j:
                # Estimate E[(S_ij - F_ij)^2]
                # Using sample estimator
                pi_hat += S[i, j] ** 2
    
    # Compute rho: sum of diagonal elements of S^2
    rho_hat = np.trace(S @ S)
    
    # Compute gamma: sum of squared differences between S and F
    gamma_hat = np.sum((S - F) ** 2)
    
    # Optimal shrinkage intensity
    # delta = max(0, min(1, kappa / gamma))
    # where kappa = pi - rho + gamma
    kappa_hat = pi_hat - rho_hat + gamma_hat
    
    if gamma_hat > 0:
        delta = max(0, min(1, kappa_hat / (T * gamma_hat)))
    else:
        delta = 1.0  # If gamma is zero, use full shrinkage
    
    # Shrinkage covariance matrix
    Sigma_shrink = (1 - delta) * S + delta * F
    
    # Ensure positive definiteness (add small value to diagonal if needed)
    min_eigenval = np.linalg.eigvalsh(Sigma_shrink).min()
    if min_eigenval <= 0:
        Sigma_shrink += (abs(min_eigenval) + 1e-8) * np.eye(N)
    
    # Annualize if requested
    if annualized:
        Sigma_shrink = Sigma_shrink * 252
    
    # Convert back to DataFrame with original column/index names
    cov_shrink = pd.DataFrame(
        Sigma_shrink,
        index=returns.columns,
        columns=returns.columns
    )
    
    return cov_shrink, delta


def compute_weight_dispersion(weights: pd.Series) -> Dict[str, float]:
    """
    Compute weight dispersion metrics to quantify portfolio stability.
    
    These metrics help assess how concentrated or diversified a portfolio is,
    and how stable the weights are across different estimation methods.
    
    Metrics computed:
    1. Effective Number of Assets (ENA): 1 / sum(w_i^2)
       - Measures concentration (lower = more concentrated)
       - Maximum = N (equal weights), Minimum = 1 (single asset)
    
    2. Gini Coefficient: Measures inequality in weight distribution
       - Range: 0 (equal weights) to 1 (single asset)
    
    3. Weight Entropy: -sum(w_i * log(w_i))
       - Higher = more diversified, Lower = more concentrated
    
    4. Max Weight: Largest single weight in portfolio
    
    5. Weight Standard Deviation: Standard deviation of weights
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (should sum to 1)
    
    Returns
    -------
    Dict[str, float]
        Dictionary containing all dispersion metrics
    
    Examples
    --------
    >>> weights = pd.Series([0.3, 0.4, 0.3], index=['A', 'B', 'C'])
    >>> metrics = compute_weight_dispersion(weights)
    >>> print(f"Effective N: {metrics['effective_n']:.2f}")
    """
    w = weights.values
    w = w[w > 0]  # Only consider positive weights
    
    if len(w) == 0:
        return {
            'effective_n': 0.0,
            'gini_coefficient': 1.0,
            'weight_entropy': 0.0,
            'max_weight': 0.0,
            'weight_std': 0.0
        }
    
    # Normalize to ensure sum = 1
    w = w / w.sum()
    
    # 1. Effective Number of Assets (ENA)
    # ENA = 1 / sum(w_i^2)
    # Also known as Herfindahl-Hirschman Index inverse
    effective_n = 1.0 / np.sum(w ** 2)
    
    # 2. Gini Coefficient
    # Measures inequality in weight distribution
    w_sorted = np.sort(w)
    n = len(w_sorted)
    cumsum = np.cumsum(w_sorted)
    # Gini = (2 * sum(i * w_i)) / (n * sum(w_i)) - (n + 1) / n
    gini = (2 * np.sum(np.arange(1, n + 1) * w_sorted)) / (n * np.sum(w_sorted)) - (n + 1) / n
    
    # 3. Weight Entropy
    # Entropy = -sum(w_i * log(w_i))
    # Higher entropy = more diversified
    weight_entropy = -np.sum(w * np.log(w + 1e-10))
    
    # 4. Max Weight
    max_weight = np.max(w)
    
    # 5. Weight Standard Deviation
    weight_std = np.std(w)
    
    return {
        'effective_n': float(effective_n),
        'gini_coefficient': float(gini),
        'weight_entropy': float(weight_entropy),
        'max_weight': float(max_weight),
        'weight_std': float(weight_std)
    }


class PortfolioOptimizer:
    """
    Portfolio optimizer implementing Markowitz mean-variance optimization.
    
    Supports two optimization modes:
    1. Minimum variance portfolio
    2. Maximum Sharpe ratio portfolio
    """
    
    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize portfolio optimizer.
        
        Parameters
        ----------
        expected_returns : pd.Series
            Expected annual returns for each asset
        covariance_matrix : pd.DataFrame
            Annualized covariance matrix of returns
        risk_free_rate : float, default=0.0
            Annual risk-free rate (e.g., 0.05 for 5%)
        """
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        
        # Validate inputs
        if len(expected_returns) != covariance_matrix.shape[0]:
            raise ValueError("Expected returns and covariance matrix must have same dimensions")
        
        if not np.allclose(covariance_matrix, covariance_matrix.T):
            raise ValueError("Covariance matrix must be symmetric")
    
    def optimize_min_variance(self) -> OptimizationResult:
        """
        Optimize for minimum variance portfolio.
        
        Minimizes portfolio variance subject to:
        - Sum of weights = 1
        - All weights >= 0 (long-only constraint)
        
        Returns
        -------
        OptimizationResult
            Optimal weights, expected return, volatility, and Sharpe ratio
        """
        # Convert to numpy arrays
        mu = self.expected_returns.values
        Sigma = self.covariance_matrix.values
        
        # Optimization variables
        w = cp.Variable(self.n_assets)
        
        # Portfolio variance
        portfolio_variance = cp.quad_form(w, Sigma)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0  # Long-only (no short selling)
        ]
        
        # Objective: minimize variance
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
        problem.solve()
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(f"Optimization failed with status: {problem.status}")
        
        # Extract optimal weights
        optimal_weights = pd.Series(w.value, index=self.expected_returns.index)
        
        # Compute portfolio metrics
        portfolio_return = np.dot(optimal_weights.values, mu)
        portfolio_volatility = np.sqrt(optimal_weights.values @ Sigma @ optimal_weights.values)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        return OptimizationResult(
            weights=optimal_weights,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            optimization_type="Minimum Variance"
        )
    
    def optimize_max_sharpe(self) -> OptimizationResult:
        """
        Optimize for maximum Sharpe ratio portfolio.
        
        Maximizes Sharpe ratio = (Return - RiskFreeRate) / Volatility
        subject to:
        - Sum of weights = 1
        - All weights >= 0 (long-only constraint)
        
        Uses a quadratic programming reformulation to solve this as a convex problem.
        The method minimizes portfolio variance for a given level of excess return,
        then scales to find the maximum Sharpe ratio.
        
        Returns
        -------
        OptimizationResult
            Optimal weights, expected return, volatility, and Sharpe ratio
        """
        # Convert to numpy arrays
        mu = self.expected_returns.values
        Sigma = self.covariance_matrix.values
        excess_returns = mu - self.risk_free_rate
        
        # Optimization variables
        w = cp.Variable(self.n_assets)
        kappa = cp.Variable()  # auxiliary variable for reformulation
        
        # Reformulation: minimize w^T Sigma w / (excess_returns^T w)^2
        # Equivalent to: minimize w^T Sigma w subject to excess_returns^T w = kappa
        # Then maximize kappa^2 / (w^T Sigma w)
        
        # We use the substitution y = w / (excess_returns^T w) to make it convex
        # Then solve: minimize y^T Sigma y subject to excess_returns^T y = 1, sum(y) = kappa
        
        # Direct approach: Use multiple target returns and find max Sharpe
        best_sharpe = -np.inf
        best_weights = None
        
        # Try a range of target returns to find maximum Sharpe
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        
        # Search over possible target returns
        for target_return in np.linspace(min_return, max_return, 100):
            try:
                # Constraints
                constraints = [
                    cp.sum(w) == 1,
                    w >= 0,
                    mu @ w == target_return  # Target return constraint
                ]
                
                # Minimize variance for this target return
                problem = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
                problem.solve()
                
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    weights = w.value
                    portfolio_return = np.dot(weights, mu)
                    portfolio_variance = weights @ Sigma @ weights
                    
                    if portfolio_variance > 0:
                        portfolio_volatility = np.sqrt(portfolio_variance)
                        sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                        
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_weights = weights.copy()
            except:
                continue
        
        if best_weights is None:
            raise RuntimeError("Maximum Sharpe optimization failed - no feasible solution found")
        
        # Extract optimal weights
        optimal_weights = pd.Series(best_weights, index=self.expected_returns.index)
        
        # Normalize weights to ensure they sum to exactly 1
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        # Compute portfolio metrics
        portfolio_return = np.dot(optimal_weights.values, mu)
        portfolio_volatility = np.sqrt(optimal_weights.values @ Sigma @ optimal_weights.values)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        return OptimizationResult(
            weights=optimal_weights,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            optimization_type="Maximum Sharpe Ratio"
        )
    
    def compute_efficient_frontier(
        self,
        n_points: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute efficient frontier.
        
        Parameters
        ----------
        n_points : int, default=50
            Number of points on the efficient frontier
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Returns (returns_array, volatilities_array, sharpe_ratios_array)
            Each array has shape (n_points,)
        """
        mu = self.expected_returns.values
        Sigma = self.covariance_matrix.values
        
        # Get minimum and maximum expected returns
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        
        # Create target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        returns_array = []
        volatilities_array = []
        sharpe_ratios_array = []
        
        w = cp.Variable(self.n_assets)
        
        for target_return in target_returns:
            # Constraints
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                mu @ w >= target_return  # Target return constraint
            ]
            
            # Objective: minimize variance
            problem = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
            problem.solve()
            
            if problem.status in ["optimal", "optimal_inaccurate"]:
                weights = w.value
                weights = weights / weights.sum()  # Normalize
                
                portfolio_return = np.dot(weights, mu)
                portfolio_volatility = np.sqrt(weights @ Sigma @ weights)
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
                
                returns_array.append(portfolio_return)
                volatilities_array.append(portfolio_volatility)
                sharpe_ratios_array.append(sharpe_ratio)
        
        return (
            np.array(returns_array),
            np.array(volatilities_array),
            np.array(sharpe_ratios_array)
        )


@dataclass
class ComparisonResult:
    """Container for comparing sample vs shrinkage covariance optimization results."""
    sample_result: OptimizationResult
    shrinkage_result: OptimizationResult
    sample_dispersion: Dict[str, float]
    shrinkage_dispersion: Dict[str, float]
    shrinkage_intensity: float
    weight_difference: pd.Series
    volatility_difference: float
    sharpe_difference: float


def compare_covariance_methods(
    returns: pd.DataFrame,
    expected_returns: pd.Series,
    risk_free_rate: float = 0.05,
    optimization_type: str = "max_sharpe"
) -> ComparisonResult:
    """
    Compare portfolio optimization using sample vs Ledoit-Wolf shrinkage covariance.
    
    This function demonstrates the benefits of shrinkage estimation:
    - Reduced estimation error
    - Improved stability (less extreme weights)
    - Better out-of-sample performance
    
    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns
    expected_returns : pd.Series
        Expected annual returns for each asset
    risk_free_rate : float, default=0.05
        Annual risk-free rate
    optimization_type : str, default="max_sharpe"
        Optimization type: "min_variance" or "max_sharpe"
    
    Returns
    -------
    ComparisonResult
        Comparison results including weights, metrics, and dispersion measures
    
    Examples
    --------
    >>> comparison = compare_covariance_methods(
    ...     daily_returns,
    ...     expected_returns,
    ...     risk_free_rate=0.05,
    ...     optimization_type="max_sharpe"
    ... )
    >>> print(f"Shrinkage intensity: {comparison.shrinkage_intensity:.4f}")
    >>> print(f"Effective N (sample): {comparison.sample_dispersion['effective_n']:.2f}")
    >>> print(f"Effective N (shrinkage): {comparison.shrinkage_dispersion['effective_n']:.2f}")
    """
    # Compute sample covariance
    cov_sample = compute_covariance_matrix(returns, annualized=True)
    
    # Compute shrinkage covariance
    cov_shrink, shrinkage_intensity = compute_ledoit_wolf_shrinkage(returns, annualized=True)
    
    # Optimize with sample covariance
    optimizer_sample = PortfolioOptimizer(
        expected_returns=expected_returns,
        covariance_matrix=cov_sample,
        risk_free_rate=risk_free_rate
    )
    
    if optimization_type == "min_variance":
        result_sample = optimizer_sample.optimize_min_variance()
    else:
        result_sample = optimizer_sample.optimize_max_sharpe()
    
    # Optimize with shrinkage covariance
    optimizer_shrink = PortfolioOptimizer(
        expected_returns=expected_returns,
        covariance_matrix=cov_shrink,
        risk_free_rate=risk_free_rate
    )
    
    if optimization_type == "min_variance":
        result_shrink = optimizer_shrink.optimize_min_variance()
    else:
        result_shrink = optimizer_shrink.optimize_max_sharpe()
    
    # Compute weight dispersion metrics
    dispersion_sample = compute_weight_dispersion(result_sample.weights)
    dispersion_shrink = compute_weight_dispersion(result_shrink.weights)
    
    # Compute differences
    weight_diff = result_shrink.weights - result_sample.weights
    volatility_diff = result_shrink.volatility - result_sample.volatility
    sharpe_diff = result_shrink.sharpe_ratio - result_sample.sharpe_ratio
    
    return ComparisonResult(
        sample_result=result_sample,
        shrinkage_result=result_shrink,
        sample_dispersion=dispersion_sample,
        shrinkage_dispersion=dispersion_shrink,
        shrinkage_intensity=shrinkage_intensity,
        weight_difference=weight_diff,
        volatility_difference=volatility_diff,
        sharpe_difference=sharpe_diff
    )