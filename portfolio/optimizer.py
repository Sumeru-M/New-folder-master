"""
Portfolio Optimizer Module - FIXED VERSION

This module implements Markowitz mean-variance portfolio optimization
using cvxpy for convex optimization.

Features:
- Minimum variance portfolio
- Maximum Sharpe ratio portfolio with CONCENTRATION LIMITS
- Efficient frontier computation
- Ledoit-Wolf covariance shrinkage for robust estimation
- Weight dispersion metrics for stability analysis

FIXES:
1. Added max_weight parameter to prevent over-concentration
2. Fixed shrinkage covariance calculation
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
    
    This function also validates and removes columns with excessive missing data.
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with close prices. Columns are tickers, index is datetime.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with daily log returns. First row will be NaN.
    
    Raises
    ------
    ValueError
        If no valid data remains after removing missing values
    
    Examples
    --------
    >>> returns = compute_daily_returns(prices)
    """
    # Check for columns with excessive NaN values (more than 50% missing)
    if prices.isna().any().any():
        nan_threshold = 0.5
        nan_pct = prices.isna().sum() / len(prices)
        invalid_columns = nan_pct[nan_pct >= nan_threshold].index.tolist()
        
        if invalid_columns:
            print(f"⚠️  Warning: Removing ticker(s) with >{nan_threshold*100:.0f}% missing data: {', '.join(invalid_columns)}")
            prices = prices.drop(columns=invalid_columns)
    
    if prices.empty:
        raise ValueError("No valid price data available after removing columns with missing data")
    
    # Compute log returns
    returns = np.log(prices / prices.shift(1))
    
    # Drop first row (will be NaN due to shift)
    returns = returns.iloc[1:]
    
    # Drop any remaining rows with NaN values
    initial_rows = len(returns)
    returns = returns.dropna()
    dropped_rows = initial_rows - len(returns)
    
    if dropped_rows > 0:
        print(f"   Removed {dropped_rows} day(s) with missing data")
    
    if returns.empty:
        raise ValueError("No valid returns data after removing missing values. Check if tickers have overlapping trading days.")
    
    return returns


def compute_expected_returns(daily_returns, annualized=True):
    """
    Compute expected returns from LOG returns.
    
    IMPORTANT: This function expects LOG returns (from np.log(prices/prices.shift(1)))
    not simple returns. The annualization formula is different for log returns.
    
    Parameters
    ----------
    daily_returns : pd.DataFrame or pd.Series
        Daily LOG returns (not simple returns)
    annualized : bool, default=True
        If True, annualize the returns using exponential growth
    
    Returns
    -------
    pd.Series
        Expected returns (annualized if annualized=True)
    """
    if annualized:
    # Arithmetic annualized return (lognormal adjustment: geometric + sigma^2/2)
    # Required for mean-variance optimization (Markowitz uses arithmetic returns)
        mean_daily_log_return = daily_returns.mean()
        var_daily_log_return  = daily_returns.var()
        annualized_return = np.exp(mean_daily_log_return * 252 + 0.5 * var_daily_log_return * 252) - 1
        return annualized_return
    else:
        return daily_returns.mean()


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
    
    # Ensure perfect symmetry (fix numerical precision issues)
    cov_matrix = (cov_matrix + cov_matrix.T) / 2
    
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
    
    # Compute optimal shrinkage intensity using Ledoit-Wolf formula
    # This is a simplified version - for production, use sklearn.covariance.LedoitWolf
    
    # Estimate the expected squared Frobenius norm of (S - Sigma_true)
    # This is approximated by the sum of squared sample covariances
    # Analytical LW formula for scaled-identity target (Ledoit-Wolf 2004, Appendix B)
# pi_hat: sum of asymptotic variances of sample covariance entries
    pi_hat = 0.0
    for i in range(N):
        for j in range(N):
            cross_prod = X_centered[:, i] * X_centered[:, j]
            pi_hat += np.var(cross_prod, ddof=1)
            pi_hat = pi_hat / T

# rho_hat: correction term for scaled-identity target (missing in original)
# rho_hat = (1/T) * sum_i [ AsymVar(S_ii) * (S_ii - mu) / mu ]
    rho_hat = 0.0
    for i in range(N):
        cross_prod_ii = X_centered[:, i] ** 2
        asym_var_ii = np.var(cross_prod_ii, ddof=1) / T
        rho_hat += asym_var_ii * (S[i, i] - mu) / mu if mu > 0 else 0.0

# gamma_hat: squared Frobenius norm of (S - F)
        gamma_hat = np.sum((S - F) ** 2)

# Optimal shrinkage intensity (full formula: delta* = (pi - rho) / gamma)
    kappa_hat = (pi_hat - rho_hat) / gamma_hat if gamma_hat > 0 else 1.0

# Shrinkage intensity delta
    delta = max(0.0, min(1.0, kappa_hat))  # Clamp between 0 and 1
    
    # Apply shrinkage
    Sigma_shrink = (1 - delta) * S + delta * F
    
    # Ensure symmetry
    Sigma_shrink = (Sigma_shrink + Sigma_shrink.T) / 2
    
    # Convert back to DataFrame with proper index/columns
    cov_shrink = pd.DataFrame(
        Sigma_shrink,
        index=returns.columns,
        columns=returns.columns
    )
    
    # Annualize if requested
    if annualized:
        cov_shrink = cov_shrink * 252
    
    return cov_shrink, delta


def compute_weight_dispersion(weights: pd.Series) -> Dict[str, float]:
    """
    Compute weight dispersion metrics for a portfolio.
    
    These metrics help assess portfolio concentration and stability:
    - Herfindahl Index: Sum of squared weights (concentration measure)
    - Effective N: 1/HHI, interpretable as "number of effective assets"
    - Max Weight: Largest single position
    - Entropy: Information-theoretic dispersion measure
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (should sum to 1)
    
    Returns
    -------
    Dict[str, float]
        Dictionary with dispersion metrics
    
    Examples
    --------
    >>> dispersion = compute_weight_dispersion(optimal_weights)
    >>> print(f"Effective N: {dispersion['effective_n']:.2f}")
    """
    w = weights.values
    
    # Herfindahl-Hirschman Index (HHI)
    hhi = np.sum(w ** 2)
    
    # Effective number of assets
    effective_n = 1 / hhi if hhi > 0 else 0
    
    # Max weight
    max_weight = np.max(w)
    
    # Shannon entropy (with small epsilon to avoid log(0))
    epsilon = 1e-10
    w_pos = w[w > epsilon]
    entropy = -np.sum(w_pos * np.log(w_pos + epsilon)) if len(w_pos) > 0 else 0.0
    
    return {
        "herfindahl_index": hhi,
        "effective_n": effective_n,
        "max_weight": max_weight,
        "entropy": entropy,
        "n_assets": len(weights),
        "n_nonzero_weights": np.sum(w > 0.001)  # Count weights > 0.1%
    }


class PortfolioOptimizer:
    """
    Portfolio optimizer using Markowitz mean-variance framework.
    
    This class provides methods to compute optimal portfolios:
    - Minimum variance portfolio
    - Maximum Sharpe ratio portfolio
    - Efficient frontier
    
    Parameters
    ----------
    expected_returns : pd.Series
        Expected annual returns for each asset
    covariance_matrix : pd.DataFrame
        Annualized covariance matrix of returns
    risk_free_rate : float, default=0.05
        Annual risk-free rate (for Sharpe ratio calculation)
    
    Examples
    --------
    >>> optimizer = PortfolioOptimizer(
    ...     expected_returns=annual_returns,
    ...     covariance_matrix=annual_cov_matrix,
    ...     risk_free_rate=0.05
    ... )
    >>> min_var_result = optimizer.optimize_min_variance()
    >>> max_sharpe_result = optimizer.optimize_max_sharpe()
    """
    
    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        risk_free_rate: float = 0.05
    ):
        """Initialize portfolio optimizer."""
        self.expected_returns = expected_returns
        self.covariance_matrix = covariance_matrix
        self.risk_free_rate = risk_free_rate
        self.n_assets = len(expected_returns)
        
        # Validate inputs
        if not self.expected_returns.index.equals(self.covariance_matrix.index):
            raise ValueError("Expected returns and covariance matrix must have matching indices")
        
        if not self.covariance_matrix.index.equals(self.covariance_matrix.columns):
            raise ValueError("Covariance matrix must be square with matching row/column indices")
    
    def optimize_min_variance(self, max_weight: Optional[float] = None) -> OptimizationResult:
        """
        Compute minimum variance portfolio.
        
        This portfolio minimizes total risk (volatility) without regard to returns.
        Useful for highly risk-averse investors.
        
        Parameters
        ----------
        max_weight : Optional[float], default=None
            Maximum weight allowed per asset (e.g., 0.3 for 30% max).
            If None, no maximum weight constraint is applied.
        
        Returns
        -------
        OptimizationResult
            Optimization results including weights, return, volatility, and Sharpe ratio
        
        Examples
        --------
        >>> result = optimizer.optimize_min_variance(max_weight=0.3)
        >>> print(result.weights)
        """
        # Define optimization variables
        w = cp.Variable(self.n_assets)
        
        # Convert to numpy arrays for cvxpy
        Sigma = self.covariance_matrix.values
        
        # ============ AUTO-ADJUSTMENT FOR SMALL PORTFOLIOS ============
        # Ensure max_weight constraint is mathematically feasible
        n_assets = len(self.expected_returns)
        
        if max_weight is not None:
            # Calculate minimum max_weight needed for feasibility
            min_feasible_weight = 1.0 / n_assets
            
            # If max_weight is too restrictive, adjust it with a warning
            if max_weight < min_feasible_weight:
                original_max_weight = max_weight
                max_weight = min(0.60, min_feasible_weight + 0.05)  # Cap at 60%, add 5% buffer
                print(f"\n⚠️  Constraint Auto-Adjustment (Min Variance):")
                print(f"   Portfolio has only {n_assets} assets.")
                print(f"   Max {original_max_weight*100:.0f}% per asset constraint adjusted to {max_weight*100:.0f}%.")
        # ============================================================
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,  # Weights sum to 1
            w >= 0           # No short-selling
        ]
        
        # Add maximum weight constraint if specified
        if max_weight is not None:
            constraints.append(w <= max_weight)
        
        # Objective: minimize portfolio variance
        objective = cp.Minimize(cp.quad_form(w, Sigma))
        
        # Solve optimization problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP, verbose=False)
        
        if problem.status not in ["optimal", "optimal_inaccurate"]:
            raise RuntimeError(
                f"Optimization failed with status: {problem.status}. "
                "Check if covariance matrix is positive semi-definite."
            )
        
        # Extract solution
        optimal_weights = pd.Series(w.value, index=self.expected_returns.index)
        
        # Ensure weights are non-negative and sum to 1
        optimal_weights = optimal_weights.clip(lower=0)
        optimal_weights = optimal_weights / optimal_weights.sum()
        
        # Compute portfolio metrics
        portfolio_return = np.dot(optimal_weights.values, self.expected_returns.values)
        portfolio_variance = optimal_weights.values @ Sigma @ optimal_weights.values
        portfolio_volatility = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        return OptimizationResult(
            weights=optimal_weights,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            optimization_type="Minimum Variance"
        )
    
    def optimize_max_sharpe(self, max_weight: Optional[float] = 0.40) -> OptimizationResult:
        """
        Compute maximum Sharpe ratio portfolio with CONCENTRATION LIMITS.
        
        This portfolio maximizes risk-adjusted returns (Sharpe ratio).
        Best for investors seeking optimal risk-return tradeoff.
        
        FIXED: Now includes default max_weight=0.40 (40% per asset) to prevent
        over-concentration in a single stock.
        
        Parameters
        ----------
        max_weight : Optional[float], default=0.40
            Maximum weight allowed per asset (e.g., 0.4 for 40% max).
            DEFAULT IS NOW 40% to prevent concentration risk.
            Set to None to remove this constraint (not recommended).
        
        Returns
        -------
        OptimizationResult
            Optimization results including weights, return, volatility, and Sharpe ratio
        
        Examples
        --------
        >>> # Default: max 40% per asset
        >>> result = optimizer.optimize_max_sharpe()
        >>> 
        >>> # Custom: max 30% per asset
        >>> result = optimizer.optimize_max_sharpe(max_weight=0.30)
        >>> 
        >>> # No limit (not recommended - can lead to concentration)
        >>> result = optimizer.optimize_max_sharpe(max_weight=None)
        """
        mu = self.expected_returns.values
        Sigma = self.covariance_matrix.values
        
        # ============ AUTO-ADJUSTMENT FOR SMALL PORTFOLIOS ============
        # Ensure max_weight constraint is mathematically feasible
        n_assets = len(self.expected_returns)
        
        if max_weight is not None:
            # Calculate minimum max_weight needed for feasibility
            # For n assets, minimum is 1/n (equal weight)
            min_feasible_weight = 1.0 / n_assets
            
            # If max_weight is too restrictive, adjust it with a warning
            if max_weight < min_feasible_weight:
                original_max_weight = max_weight
                max_weight = min(0.60, min_feasible_weight + 0.05)  # Cap at 60%, add 5% buffer
                print(f"\n⚠️  Constraint Auto-Adjustment:")
                print(f"   Portfolio has only {n_assets} assets.")
                print(f"   Original max {original_max_weight*100:.0f}% per asset is mathematically infeasible.")
                print(f"   Auto-adjusted to {max_weight*100:.0f}% per asset.")
                print(f"   💡 Recommendation: Use 4-5 stocks for better diversification and")
                print(f"      more flexible constraint enforcement.\n")
        # ============================================================
        
        # We use the approach of maximizing return subject to varying target volatility levels
        # Then select the portfolio with the highest Sharpe ratio
        
        # Get range of possible returns
        min_return = self.expected_returns.min()
        max_return = self.expected_returns.max()
        
        # Create grid of target returns to search
        n_points = 100
        target_returns = np.linspace(min_return, max_return, n_points)
        
        best_sharpe = -np.inf
        best_weights = None
        
        w = cp.Variable(self.n_assets)
        
        for target_return in target_returns:
            try:
                # Constraints
                constraints = [
                    cp.sum(w) == 1,               # Weights sum to 1
                    w >= 0,                        # No short-selling
                    mu @ w == target_return        # Target return constraint
                ]
                
                # Add maximum weight constraint if specified
                if max_weight is not None:
                    constraints.append(w <= max_weight)
                
                # Minimize variance for this target return
                problem = cp.Problem(cp.Minimize(cp.quad_form(w, Sigma)), constraints)
                problem.solve(solver=cp.OSQP, verbose=False)
                
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    weights = w.value
                    
                    if weights is not None:
                        # Clean up small negative values from numerical precision
                        weights = np.maximum(weights, 0)
                        weights = weights / weights.sum()  # Renormalize
                        
                        # Calculate portfolio metrics
                        portfolio_return = np.dot(weights, mu)
                        portfolio_variance = weights @ Sigma @ weights
                        
                        if portfolio_variance > 1e-10:  # Avoid division by zero
                            portfolio_volatility = np.sqrt(portfolio_variance)
                            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_volatility
                            
                            # Track best Sharpe (can be negative!)
                            if sharpe > best_sharpe:
                                best_sharpe = sharpe
                                best_weights = weights.copy()
                                
            except Exception:
                # Skip this target return if optimization fails
                continue
        
        if best_weights is None:
            raise RuntimeError(
                "Maximum Sharpe optimization failed - no feasible solution found. "
                "This can happen if the constraints are too restrictive or data is invalid."
            )
        
        # Final cleanup and results
        optimal_weights = pd.Series(best_weights, index=self.expected_returns.index)
        optimal_weights = optimal_weights / optimal_weights.sum()  # Ensure exact sum to 1
        
        # Compute final metrics
        portfolio_return = np.dot(optimal_weights.values, mu)
        portfolio_volatility = np.sqrt(optimal_weights.values @ Sigma @ optimal_weights.values)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        constraint_msg = f" (Max {max_weight*100:.0f}% per asset)" if max_weight else ""
        
        return OptimizationResult(
            weights=optimal_weights,
            expected_return=portfolio_return,
            volatility=portfolio_volatility,
            sharpe_ratio=sharpe_ratio,
            optimization_type=f"Maximum Sharpe Ratio{constraint_msg}"
        )
    
    def compute_efficient_frontier(
        self,
        n_points: int = 50,
        max_weight: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute efficient frontier.
        
        Parameters
        ----------
        n_points : int, default=50
            Number of points on the efficient frontier
        max_weight : Optional[float], default=None
            Maximum weight allowed per asset
        
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
            
            # Add maximum weight constraint if specified
            if max_weight is not None:
                constraints.append(w <= max_weight)
            
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
    optimization_type: str = "max_sharpe",
    max_weight: Optional[float] = 0.40
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
    max_weight : Optional[float], default=0.40
        Maximum weight allowed per asset (40% default to prevent concentration)
    
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
    ...     optimization_type="max_sharpe",
    ...     max_weight=0.40
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
        result_sample = optimizer_sample.optimize_min_variance(max_weight=max_weight)
    else:
        result_sample = optimizer_sample.optimize_max_sharpe(max_weight=max_weight)
    
    # Optimize with shrinkage covariance
    optimizer_shrink = PortfolioOptimizer(
        expected_returns=expected_returns,
        covariance_matrix=cov_shrink,
        risk_free_rate=risk_free_rate
    )
    
    if optimization_type == "min_variance":
        result_shrink = optimizer_shrink.optimize_min_variance(max_weight=max_weight)
    else:
        result_shrink = optimizer_shrink.optimize_max_sharpe(max_weight=max_weight)
    
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