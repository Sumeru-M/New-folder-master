"""
Risk Metrics Module (CORRECTED VERSION)

This module computes advanced risk metrics such as Value at Risk (VaR),
Conditional Value at Risk (CVaR/Expected Shortfall), and Component VaR.

All calculations are consistent with log returns methodology used in other modules.

FIXES:
- Corrected drawdown calculations to use log returns properly
- Added input validation
- Improved error handling
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
    portfolio_value: float = 1.0,
    include_mean: bool = False  # NEW: Option to include mean for long horizons
    ) -> Dict[str, float]:
    """
    Compute Parametric (Analytical) Value at Risk.
    
    Assumes normal distribution of returns. Uses daily parameters derived
    from annualized inputs.
    
    VaR = -Z_alpha * sigma * sqrt(T)  (short horizon, mean ignored)
    VaR = -(mu*T + Z_alpha * sigma * sqrt(T))  (long horizon, mean included)
    
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
        Total value of portfolio in INR (to get VaR in INR)
    include_mean : bool, default=False
        Whether to include expected return in VaR calculation.
        Standard practice: False for short horizons (<30 days), True for longer.
        
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - var_percent: VaR as a percentage (positive value = loss)
        - var_amount: VaR in INR
        - portfolio_return: Expected return over horizon
        - portfolio_volatility: Volatility over horizon
    """
    # Input validation
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
    
    if portfolio_value <= 0:
        raise ValueError(f"Portfolio value must be positive, got {portfolio_value}")
    
    if time_horizon_days <= 0:
        raise ValueError(f"Time horizon must be positive, got {time_horizon_days}")
    
    if isinstance(weights, pd.Series):
        w = weights.values
    else:
        w = weights
    
    # Validate weights sum to approximately 1
    if not np.isclose(w.sum(), 1.0, atol=0.01):
        raise ValueError(f"Weights must sum to 1.0, got {w.sum():.4f}")
    
    # Validate weights are non-negative
    if (w < -1e-6).any():  # Allow small numerical errors
        raise ValueError("Weights cannot be negative (no short selling)")
        
    # Convert annualized parameters to daily
    mu_daily = expected_returns.values / 252
    sigma_daily = covariance_matrix.values / 252
    
    # Portfolio Mean and Variance (Daily)
    port_mean_daily = np.dot(w, mu_daily)
    port_var_daily = np.dot(w.T, np.dot(sigma_daily, w))
    
    if port_var_daily < 0:
        raise ValueError("Negative portfolio variance - check covariance matrix")
    
    port_std_daily = np.sqrt(port_var_daily)
    
    # Scale to time horizon
    horizon_mean = port_mean_daily * time_horizon_days
    horizon_std = port_std_daily * np.sqrt(time_horizon_days)
    
    # Z-score for confidence level
    # For 95% confidence, we look at the 5% left tail
    alpha = 1 - confidence_level
    z_score = norm.ppf(alpha)  # This will be negative (e.g., -1.645 for 95%)
    
    # VaR calculation
    # Auto-include mean for longer horizons
    if include_mean or time_horizon_days > 30:
        # Include expected return (more conservative)
        var_percent = -(horizon_mean + z_score * horizon_std)
    else:
        # Standard practice for short horizons: ignore mean
        var_percent = -z_score * horizon_std
    
    return {
        "var_percent": var_percent,
        "var_amount": var_percent * portfolio_value,
        "portfolio_return": horizon_mean,
        "portfolio_volatility": horizon_std,
        "included_mean": include_mean or time_horizon_days > 30
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
        Portfolio value in INR for scaling
    
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - var_percent: VaR as percentage (positive = loss)
        - var_amount: VaR in INR
    """
    if returns.empty:
        return {"var_percent": 0.0, "var_amount": 0.0}
    
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
        
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
        Portfolio value in INR for scaling
    
    Returns
    -------
    Dict[str, float]
        Dictionary with:
        - cvar_percent: CVaR as percentage (positive = loss)
        - cvar_amount: CVaR in INR
        - var_percent: VaR for reference
    """
    if returns.empty:
        return {"cvar_percent": 0.0, "cvar_amount": 0.0, "var_percent": 0.0}
    
    if confidence_level <= 0 or confidence_level >= 1:
        raise ValueError(f"Confidence level must be between 0 and 1, got {confidence_level}")
        
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
    if not np.isclose(weights.sum(), 1.0, atol=0.01):
        raise ValueError(f"Weights must sum to 1.0, got {weights.sum():.4f}")
    
    w = weights.values
    sigma = covariance_matrix.values / 252  # Convert to daily
    
    # Portfolio variance and volatility
    port_var = np.dot(w.T, np.dot(sigma, w))
    
    if port_var < 0:
        raise ValueError("Negative portfolio variance - check covariance matrix")
    
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
    z_score = -norm.ppf(alpha)  # Positive value (e.g., 1.645 for 95%)
    
    # Marginal VaR for each asset
    # Marginal VaR_i = Z_alpha * (Sigma * w)_i / sigma_portfolio
    sigma_times_w = np.dot(sigma, w)
    marginal_var = z_score * sigma_times_w / port_std
    
    # Component VaR = weight × marginal VaR
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
    portfolio_value: float = 10_00_000.0
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
    portfolio_value : float, default=10_00_000
        Total portfolio value in INR (₹10,00,000 = 10 lakhs)
        
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


def compute_max_drawdown(returns: pd.Series) -> Dict[str, float]:
    """
    Compute Maximum Drawdown (MDD) and related metrics.
    
    Maximum Drawdown is the maximum observed loss from a peak to a trough of a portfolio,
    before a new peak is attained.
    
    FIXED: Now properly handles log returns.
    
    Parameters
    ----------
    returns : pd.Series
        Historical daily returns (log returns)
        
    Returns
    -------
    Dict[str, float]
        - max_drawdown: Maximum drawdown (positive value representing loss)
        - max_drawdown_duration: Number of days of the longest drawdown
        - current_drawdown: Drawdown at the last observation
    """
    if returns.empty:
        return {"max_drawdown": 0.0, "duration": 0.0, "current": 0.0}

    # FIXED: Convert log returns to cumulative wealth index properly
    # For log returns: wealth = exp(cumsum(log_returns))
    cumulative_log_returns = returns.cumsum()
    wealth_index = np.exp(cumulative_log_returns)
    
    # Calculate running maximum
    previous_peaks = wealth_index.cummax()
    
    # Drawdown as percentage from peak
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    max_drawdown = abs(drawdowns.min())  # Convert to positive number
    
    # Calculate duration (max consecutive days below peak)
    is_underwater = drawdowns < 0
    # Group by consecutive True values
    drawdown_periods = is_underwater.ne(is_underwater.shift()).cumsum()
    # Filter only underwater periods
    underwater_periods = drawdown_periods[is_underwater]
    if underwater_periods.empty:
        max_duration = 0
    else:
        max_duration = underwater_periods.value_counts().max()

    return {
        "max_drawdown": max_drawdown,
        "max_drawdown_duration_days": float(max_duration),
        "current_drawdown": abs(drawdowns.iloc[-1])
    }


def compute_ulcer_index(returns: pd.Series, period: int = 14) -> float:
    """
    Compute Ulcer Index.
    
    The Ulcer Index is a measure of the depth and duration of drawdowns in prices.
    Unlike standard deviation, it only penalizes downside volatility.
    
    sqrt(mean(drawdowns^2))
    
    FIXED: Now properly handles log returns.
    
    Parameters
    ----------
    returns : pd.Series
        Daily log returns
    period : int, default=14
        Not used in this implementation (uses full history)
        
    Returns
    -------
    float
        Ulcer Index as decimal (e.g., 0.05 for 5%)
    """
    if returns.empty:
        return 0.0
    
    # FIXED: Convert log returns to wealth index properly
    cumulative_log_returns = returns.cumsum()
    wealth_index = np.exp(cumulative_log_returns)
    
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    drawdowns_pct = drawdowns * 100  # Convert to percent
    
    squared_drawdowns = drawdowns_pct ** 2
    ulcer_index = np.sqrt(squared_drawdowns.mean())
    
    return float(ulcer_index / 100)  # Return as decimal


def detect_market_regime(returns: pd.Series, window: int = 21) -> Dict[str, Union[str, float]]:
    """
    Detect Market Regime based on Volatility.
    
    Uses rolling volatility Z-score to classify market state.
    
    Regimes:
    - Low Volatility (Z < -1)
    - Normal (1 >= Z >= -1)
    - High Volatility (Z > 1)
    - Crisis (Z > 2)
    
    Parameters
    ----------
    returns : pd.Series
        Market returns (e.g. Nifty) or Portfolio returns
    window : int
        Rolling window for current volatility (default 21 days ~ 1 month)
        
    Returns
    -------
    Dict
        - regime: str (Low, Normal, High, Crisis)
        - z_score: float
        - current_vol: float (annualized)
        - long_term_vol: float (annualized)
    """
    if len(returns) < window + 2:
        return {"regime": "Unknown", "z_score": 0.0, "current_vol": 0.0}
        
    daily_vol = returns.rolling(window=window).std()
    current_vol = daily_vol.iloc[-1]
    
    # Long term stats (using full history available)
    long_term_mean_vol = daily_vol.mean()
    long_term_std_vol = daily_vol.std()
    
    if long_term_std_vol == 0:
        z_score = 0
    else:
        z_score = (current_vol - long_term_mean_vol) / long_term_std_vol
        
    if z_score > 2.0:
        regime = "Crisis"
    elif z_score > 1.0:
        regime = "High Volatility"
    elif z_score < -1.0:
        regime = "Low Volatility"
    else:
        regime = "Normal"
        
    return {
        "regime": regime,
        "z_score": float(z_score),
        "current_vol_annualized": float(current_vol * np.sqrt(252)),
        "long_term_vol_annualized": float(long_term_mean_vol * np.sqrt(252))
    }


def compute_rolling_correlations(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """
    Compute average pairwise rolling correlations.
    
    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns
    window : int
        Rolling window size
        
    Returns
    -------
    pd.DataFrame
        Average pairwise correlation over time.
    """
    if returns.empty or len(returns.columns) < 2:
         return pd.DataFrame()
         
    # pairwise correlation for each window
    rolling_corr = returns.rolling(window=window).corr()
    
    # We want to extract a single time series representing "Average Correlation"
    # This is useful to see if assets are moving together (Crisis) or apart (Diversification)
    
    avg_corrs = []
    dates = returns.index[window-1:]
    
    for dt in dates:
        # Get corr matrix for this date
        try:
            corr_mat = rolling_corr.loc[dt]
            # Average off-diagonal elements
            # Mask diagonal
            mask = np.ones_like(corr_mat, dtype=bool)
            np.fill_diagonal(mask, False)
            avg_corr = corr_mat.values[mask].mean()
            avg_corrs.append(avg_corr)
        except KeyError:
            avg_corrs.append(np.nan)
            
    return pd.DataFrame({"Average Correlation": avg_corrs}, index=dates)