"""
Portfolio Engine Module

This module computes portfolio metrics from price data including:
- Daily returns
- Annualized returns and volatility
- Covariance and correlation matrices
- Rolling volatility
- Maximum drawdown

All metrics are returned in a structured dictionary for easy access.
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime


def compute_portfolio_metrics(
    price_data: pd.DataFrame,
    risk_free_rate: float = 0.0
) -> Dict:
    """
    Compute comprehensive portfolio metrics from price data.
    
    This function calculates various risk and return metrics for a portfolio
    of assets. All metrics are annualized where appropriate (using 252 trading days).
    
    Parameters
    ----------
    price_data : pd.DataFrame
        DataFrame with Close prices. Columns are tickers, index is datetime.
        Can also accept OHLCV data with MultiIndex columns - will extract Close prices.
    risk_free_rate : float, default=0.0
        Annual risk-free rate (e.g., 0.05 for 5%). Used for Sharpe ratio calculation.
    
    Returns
    -------
    Dict
        Dictionary containing:
        - 'daily_returns': pd.DataFrame - Daily returns for each asset
        - 'annualized_returns': pd.Series - Annualized mean returns
        - 'annualized_volatility': pd.Series - Annualized volatility (std dev)
        - 'covariance_matrix': pd.DataFrame - Covariance matrix of returns
        - 'correlation_matrix': pd.DataFrame - Correlation matrix of returns
        - 'rolling_volatility_30d': pd.DataFrame - 30-day rolling volatility
        - 'max_drawdown': pd.Series - Maximum drawdown for each asset
        - 'sharpe_ratio': pd.Series - Sharpe ratio (annualized)
        - 'price_data': pd.DataFrame - Original price data (Close prices)
    
    Examples
    --------
    >>> from src.data_loader import fetch_market_data, get_close_prices
    >>> 
    >>> # Fetch data
    >>> ohlcv_data = fetch_market_data(["RELIANCE.NS", "TCS.NS"])
    >>> prices = get_close_prices(ohlcv_data)
    >>> 
    >>> # Compute metrics
    >>> metrics = compute_portfolio_metrics(prices)
    >>> print(metrics['annualized_returns'])
    """
    # Extract close prices if OHLCV data is provided
    if isinstance(price_data.columns, pd.MultiIndex):
        prices = _extract_close_from_ohlcv(price_data)
    else:
        prices = price_data.copy()
    
    # Ensure data is sorted by date
    prices = prices.sort_index()
    
    # Remove any rows with all NaN
    prices = prices.dropna(how='all')
    
    # Compute daily returns (log returns)
    daily_returns = _compute_daily_returns(prices)
    
    # Compute annualized metrics
    annualized_returns = _compute_annualized_returns(daily_returns)
    annualized_volatility = _compute_annualized_volatility(daily_returns)
    
    # Compute covariance and correlation matrices
    covariance_matrix = _compute_covariance_matrix(daily_returns)
    correlation_matrix = _compute_correlation_matrix(daily_returns)
    
    # Compute rolling 30-day volatility
    rolling_volatility_30d = _compute_rolling_volatility(daily_returns, window=30)
    
    # Compute maximum drawdown
    max_drawdown = _compute_max_drawdown(prices)
    
    # Compute Sharpe ratio
    sharpe_ratio = _compute_sharpe_ratio(annualized_returns, annualized_volatility, risk_free_rate)
    
    # Compile results
    results = {
        'daily_returns': daily_returns,
        'annualized_returns': annualized_returns,
        'annualized_volatility': annualized_volatility,
        'covariance_matrix': covariance_matrix,
        'correlation_matrix': correlation_matrix,
        'rolling_volatility_30d': rolling_volatility_30d,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'price_data': prices
    }
    
    return results


def _extract_close_from_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Extract Close prices from OHLCV data."""
    close_data = {}
    if isinstance(data.columns, pd.MultiIndex):
        for ticker in data.columns.get_level_values(0).unique():
            if (ticker, 'Close') in data.columns:
                close_data[ticker] = data[(ticker, 'Close')]
    return pd.DataFrame(close_data)


def _compute_daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from prices.
    
    Log returns: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    """
    return np.log(prices / prices.shift(1)).dropna()


def _compute_annualized_returns(daily_returns: pd.DataFrame) -> pd.Series:
    """
    Compute annualized returns using geometric mean (CAGR).
    
    Uses compound growth rate formula:
    Annualized return = [(1+r₁) × (1+r₂) × ... × (1+rₙ)]^(252/n) - 1
    
    This represents the actual compound annual growth rate achieved
    over the period, accounting for volatility drag.
    
    Parameters
    ----------
    daily_returns : pd.DataFrame
        DataFrame with daily percentage returns
    
    Returns
    -------
    pd.Series
        Annualized returns for each asset
    """
    n_periods = len(daily_returns)
    cumulative_return = (1 + daily_returns).prod()
    return cumulative_return ** (252 / n_periods) - 1


def _compute_annualized_volatility(daily_returns: pd.DataFrame) -> pd.Series:
    """
    Compute annualized volatility (standard deviation).
    
    Annualized volatility = std(daily_return) * sqrt(252)
    where 252 is the number of trading days per year.
    """
    return daily_returns.std() * np.sqrt(252)


def _compute_covariance_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute covariance matrix of daily returns.
    
    Returns annualized covariance matrix.
    """
    # Daily covariance
    daily_cov = daily_returns.cov()
    
    # Annualize: multiply by 252 (trading days per year)
    annualized_cov = daily_cov * 252
    
    return annualized_cov


def _compute_correlation_matrix(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix of daily returns.
    
    Correlation is scale-invariant, so no annualization needed.
    """
    return daily_returns.corr()


def _compute_rolling_volatility(
    daily_returns: pd.DataFrame,
    window: int = 30
) -> pd.DataFrame:
    """
    Compute rolling volatility over a specified window.
    
    Parameters
    ----------
    daily_returns : pd.DataFrame
        Daily returns
    window : int, default=30
        Rolling window size in days
    
    Returns
    -------
    pd.DataFrame
        Rolling volatility (annualized)
    """
    # Compute rolling standard deviation
    rolling_std = daily_returns.rolling(window=window).std()
    
    # Annualize
    rolling_vol = rolling_std * np.sqrt(252)
    
    return rolling_vol


def _compute_max_drawdown(prices: pd.DataFrame) -> pd.Series:
    """
    Compute maximum drawdown for each asset.
    
    Maximum drawdown is the largest peak-to-trough decline in price.
    Returns as a percentage (e.g., 0.25 for 25% drawdown).
    
    Formula: MDD = max((Peak - Trough) / Peak)
    """
    max_dd = {}
    
    for ticker in prices.columns:
        price_series = prices[ticker].dropna()
        
        if len(price_series) == 0:
            max_dd[ticker] = np.nan
            continue
        
        # Calculate running maximum (peak)
        running_max = price_series.expanding().max()
        
        # Calculate drawdown at each point
        drawdown = (running_max - price_series) / running_max
        
        # Maximum drawdown
        max_dd[ticker] = drawdown.max()
    
    return pd.Series(max_dd)


def _compute_sharpe_ratio(
    annualized_returns: pd.Series,
    annualized_volatility: pd.Series,
    risk_free_rate: float = 0.0
) -> pd.Series:
    """
    Compute Sharpe ratio.
    
    Sharpe Ratio = (Return - RiskFreeRate) / Volatility
    
    Parameters
    ----------
    annualized_returns : pd.Series
        Annualized returns
    annualized_volatility : pd.Series
        Annualized volatility
    risk_free_rate : float, default=0.0
        Annual risk-free rate
    
    Returns
    -------
    pd.Series
        Sharpe ratio for each asset
    """
    excess_returns = annualized_returns - risk_free_rate
    sharpe = excess_returns / annualized_volatility
    
    # Handle division by zero
    sharpe = sharpe.replace([np.inf, -np.inf], np.nan)
    
    return sharpe
