"""
Risk Metrics Module

This module computes advanced risk metrics such as Value at Risk (VaR),
Conditional Value at Risk (CVaR/Expected Shortfall), and Component VaR.

All calculations are consistent with log returns methodology used in other modules.
"""

from typing import Dict, Union
import pandas as pd
import numpy as np
from scipy.stats import norm

def compute_parametric_var(
    weights: Union[pd.Series, np.ndarray],
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    confidence_level: float = 0.95,
    time_horizon_days: int = 1,
    portfolio_value: float = 1.0
) -> Dict[str, float]:
    """
    Compute Parametric (Analytical) Value at Risk.
    
    Assumes normal distribution of returns. Uses daily parameters derived
    from annualized inputs.
    
    VaR = -Z_alpha * sigma * sqrt(T)
    
    For short horizons (1-10 days), mean return is typically ignored as it's
    small relative to volatility. This is standard practice in risk management.
    
    Parameters
    ----------
    weights : pd.Series or np.ndarray
        Portfolio weights
    expected_returns : pd.Series
        Annualized expected returns
    covariance_matrix : pd.DataFrame
        Annualized covariance matrix
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95%)
    time_horizon_days : int, default=1
        Time horizon in days
    portfolio_value : float, default=1.0
        Total value of portfolio (to get VaR in currency units)
        
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - var_percent: VaR as a percentage (positive value = loss)
        - var_amount: VaR in currency units
        - portfolio_return: Expected return over horizon
        - portfolio_volatility: Volatility over horizon
    """
    if isinstance(weights, pd.Series):
        w = weights.values
    else:
        w = weights
        
    # Convert annualized parameters to daily
    mu_daily = expected_returns.values / 252
    sigma_daily = covariance_matrix.values / 252
    
    # Portfolio Mean and Variance (Daily)
    port_mean_daily = np.dot(w, mu_daily)
    port_var_daily = np.dot(w.T, np.dot(sigma_daily, w))
    port_std_daily = np.sqrt(port_var_daily)
    
    # Scale to time horizon
    horizon_mean = port_mean_daily * time_horizon_days
    horizon_std = port_std_daily * np.sqrt(time_horizon_days)
    
    # Z-score for confidence level
    # For 95% confidence, we look at the 5% left tail
    alpha = 1 - confidence_level
    z_score = norm.ppf(alpha)  # This will be negative (e.g., -1.645 for 95%)
    
    # VaR calculation (without mean for short horizons - standard practice)
    # VaR = -Z_alpha * sigma * sqrt(T)
    # Since z_score is negative, we negate it to get positive VaR
    var_percent = -z_score * horizon_std
    
    # Alternative: Include mean (more conservative for long horizons)
    # var_with_mean = -(horizon_mean + z_score * horizon_std)
    
    return {
        "var_percent": var_percent,
        "var_amount": var_percent * portfolio_value,
        "portfolio_return": horizon_mean,
        "portfolio_volatility": horizon_std
    }


def compute_historical_var(
    returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 1.0
) -> Dict[str, float]:
    """
    Compute Historical Value at Risk from a return series.
    
    Uses empirical distribution of returns (no parametric assumptions).
    
    Parameters
    ----------
    returns : pd.Series
        Historical daily returns of the portfolio (should be log returns
        for consistency with other modules)
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95%)
    portfolio_value : float, default=1.0
        Portfolio value for scaling
    
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - var_percent: VaR as percentage (positive = loss)
        - var_amount: VaR in currency units
    """
    if returns.empty:
        return {"var_percent": 0.0, "var_amount": 0.0}
        
    # Calculate the percentile
    # For 95% confidence, we want the 5th percentile of returns (left tail)
    alpha = 1 - confidence_level
    alpha_percentile = alpha * 100
    
    # Get the percentile value (this will be negative for losses)
    percentile_return = np.percentile(returns, alpha_percentile)
    
    # VaR is reported as positive value representing potential loss
    var_percent = -percentile_return
    
    return {
        "var_percent": var_percent,
        "var_amount": var_percent * portfolio_value
    }


def compute_cvar(
    returns: pd.Series,
    confidence_level: float = 0.95,
    portfolio_value: float = 1.0
) -> Dict[str, float]:
    """
    Compute Conditional Value at Risk (CVaR) / Expected Shortfall.
    
    CVaR is the expected loss given that the loss exceeds VaR.
    It represents the average of the worst (1-confidence_level)% of outcomes.
    
    Parameters
    ----------
    returns : pd.Series
        Historical daily returns of the portfolio (log returns)
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95%)
    portfolio_value : float, default=1.0
        Portfolio value for scaling
    
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - cvar_percent: CVaR as percentage (positive = loss)
        - cvar_amount: CVaR in currency units
        - var_percent: VaR for reference
    """
    if returns.empty:
        return {"cvar_percent": 0.0, "cvar_amount": 0.0, "var_percent": 0.0}
        
    alpha = 1 - confidence_level
    
    # Get VaR threshold
    var_threshold = np.percentile(returns, alpha * 100)
    
    # Filter returns worse than (or equal to) VaR
    tail_losses = returns[returns <= var_threshold]
    
    if tail_losses.empty:
        # Edge case: use VaR as fallback
        cvar_percent = -var_threshold
    else:
        # CVaR is the average of the tail losses
        cvar_percent = -tail_losses.mean()
        
    return {
        "cvar_percent": cvar_percent,
        "cvar_amount": cvar_percent * portfolio_value,
        "var_percent": -var_threshold
    }


def compute_component_var(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Compute Component VaR (contribution of each asset to portfolio VaR).
    
    Component VaR decomposes total portfolio VaR into contributions from each asset.
    This uses the Euler decomposition principle.
    
    Mathematical Formula:
    - Marginal VaR_i = Z_alpha * (Sigma * w)_i / sigma_portfolio
    - Component VaR_i = w_i * Marginal VaR_i
    - Sum of Component VaRs = Portfolio VaR
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (sum to 1)
    covariance_matrix : pd.DataFrame
        Annualized covariance matrix
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95%)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - Weight: Asset weight in portfolio
        - Marginal VaR: Marginal contribution to VaR
        - Component VaR: Total contribution (Weight × Marginal VaR)
        - % Contribution: Percentage of total portfolio VaR
    """
    w = weights.values
    sigma = covariance_matrix.values / 252  # Convert to daily
    
    # Portfolio variance and volatility
    port_var = np.dot(w.T, np.dot(sigma, w))
    port_std = np.sqrt(port_var)
    
    if port_std == 0:
        # Edge case: zero volatility portfolio
        return pd.DataFrame({
            "Weight": w,
            "Marginal VaR": np.zeros(len(w)),
            "Component VaR": np.zeros(len(w)),
            "% Contribution": np.zeros(len(w))
        }, index=weights.index)
        
    # Z-score for confidence level
    alpha = 1 - confidence_level
    z_score = abs(norm.ppf(alpha))  # Use absolute value
    
    # Marginal VaR: derivative of portfolio VaR with respect to each weight
    # dVaR/dw_i = Z * (Sigma * w)_i / sigma_portfolio
    marginal_contribution = np.dot(sigma, w) / port_std
    marginal_var = z_score * marginal_contribution
    
    # Component VaR: weight times marginal contribution
    component_var = w * marginal_var
    
    # Percentage contribution to total VaR
    total_var = np.sum(component_var)
    
    if total_var > 0:
        percent_contribution = (component_var / total_var) * 100
    else:
        percent_contribution = np.zeros(len(w))
    
    return pd.DataFrame({
        "Weight": w,
        "Marginal VaR": marginal_var,
        "Component VaR": component_var,
        "% Contribution": percent_contribution
    }, index=weights.index)


def compute_portfolio_risk_metrics(
    weights: Union[pd.Series, np.ndarray],
    returns: pd.Series,
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    confidence_level: float = 0.95,
    portfolio_value: float = 1000000.0
) -> Dict:
    """
    Compute comprehensive risk metrics for a portfolio.
    
    Combines parametric, historical, and component risk measures.
    
    Parameters
    ----------
    weights : pd.Series or np.ndarray
        Portfolio weights
    returns : pd.Series
        Historical portfolio returns (daily log returns)
    expected_returns : pd.Series
        Annualized expected returns per asset
    covariance_matrix : pd.DataFrame
        Annualized covariance matrix
    confidence_level : float, default=0.95
        Confidence level for VaR/CVaR
    portfolio_value : float, default=1000000
        Total portfolio value
        
    Returns
    -------
    Dict
        Comprehensive risk metrics including VaR, CVaR, Component VaR
    """
    # Parametric VaR
    parametric_var = compute_parametric_var(
        weights, expected_returns, covariance_matrix,
        confidence_level, time_horizon_days=1, portfolio_value=portfolio_value
    )
    
    # Historical VaR
    historical_var = compute_historical_var(
        returns, confidence_level, portfolio_value
    )
    
    # CVaR
    cvar = compute_cvar(
        returns, confidence_level, portfolio_value
    )
    
    # Component VaR
    if isinstance(weights, np.ndarray):
        weights_series = pd.Series(weights, index=expected_returns.index)
    else:
        weights_series = weights
        
    component_var = compute_component_var(
        weights_series, covariance_matrix, confidence_level
    )
    
    return {
        "parametric_var": parametric_var,
        "historical_var": historical_var,
        "cvar": cvar,
        "component_var": component_var,
        "confidence_level": confidence_level
    }