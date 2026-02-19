"""
Data Ingestion Module

This module handles downloading historical market data using yfinance
and computing various financial metrics including returns, volatility, and correlations.
"""

from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def download_etf_data(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "5y"
) -> pd.DataFrame:
    """
    Download historical price data for a list of tickers (ETFs or stocks).
    
    For Indian markets, use .NS suffix for NSE-listed instruments (e.g., 'NIFTYBEES.NS').
    For US markets, use standard tickers (e.g., 'SPY').
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols (e.g., ['NIFTYBEES.NS', 'BANKBEES.NS'] for Indian markets
        or ['SPY', 'QQQ'] for US markets)
    start_date : Optional[str], default=None
        Start date in 'YYYY-MM-DD' format. If None, uses period parameter.
    end_date : Optional[str], default=None
        End date in 'YYYY-MM-DD' format. If None, uses today's date.
    period : str, default='5y'
        Valid periods: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        Only used if start_date is None.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with columns as tickers and index as dates.
        Contains 'Close' prices for each ticker.
    
    Raises
    ------
    ValueError
        If tickers list is empty or if data download fails.
    
    Examples
    --------
    >>> # Indian markets
    >>> data = download_etf_data(['NIFTYBEES.NS', 'BANKBEES.NS'], period='2y')
    >>> # US markets
    >>> data = download_etf_data(['SPY', 'QQQ'], period='2y')
    >>> data.head()
    """
    if not tickers:
        raise ValueError("Tickers list cannot be empty")
    
    # Download data using yfinance
    if start_date and end_date:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    else:
        data = yf.download(tickers, period=period, progress=False)
    
    # Handle different yfinance return formats
    if len(tickers) == 1:
        # Single ticker: yfinance returns DataFrame with columns [Open, High, Low, Close, Volume, ...]
        if isinstance(data, pd.DataFrame):
            if 'Close' in data.columns:
                data = pd.DataFrame(data['Close'])
            else:
                # If structure is different, try to extract Close
                data = pd.DataFrame(data.iloc[:, 3])  # Close is typically 4th column (index 3)
            data.columns = tickers
        else:
            # If it's a Series, convert to DataFrame
            data = pd.DataFrame(data, columns=tickers)
    else:
        # Multiple tickers: yfinance returns DataFrame with MultiIndex columns
        # Structure: (Ticker, OHLCV) or (OHLCV, Ticker)
        if isinstance(data.columns, pd.MultiIndex):
            # Check if first level is OHLCV or Ticker
            first_level = data.columns.get_level_values(0)
            if 'Close' in first_level:
                # Format: (OHLCV, Ticker)
                data = data.xs('Close', level=0, axis=1)
            else:
                # Format: (Ticker, OHLCV) - extract Close from second level
                data = data.xs('Close', level=1, axis=1)
        else:
            # Fallback: assume Close is in column names somehow
            if 'Close' in data.columns:
                data = data['Close']
    
    # Ensure columns match ticker order and handle missing tickers
    available_tickers = [t for t in tickers if t in data.columns]
    if not available_tickers:
        raise ValueError(f"Failed to download data for any of the tickers: {tickers}")
    
    data = data[available_tickers]
    
    # Remove any rows with all NaN values
    data = data.dropna(how='all')
    
    if data.empty:
        raise ValueError(f"Failed to download data for tickers: {tickers}")
    
    return data


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily log returns from price data.
    
    Log returns are calculated as: ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1})
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with prices, where columns are assets and index is dates.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with log returns. First row will be NaN.
    
    Examples
    --------
    >>> prices = download_etf_data(['SPY', 'QQQ'])
    >>> log_returns = compute_log_returns(prices)
    """
    return np.log(prices / prices.shift(1))


def compute_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly returns from daily price data.
    
    Monthly returns are calculated as: (P_end_of_month / P_start_of_month) - 1
    
    Parameters
    ----------
    prices : pd.DataFrame
        DataFrame with daily prices, where columns are assets and index is dates.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with monthly returns, indexed by month-end dates.
    
    Examples
    --------
    >>> prices = download_etf_data(['SPY', 'QQQ'])
    >>> monthly_returns = compute_monthly_returns(prices)
    """
    # Resample to monthly and take last value of each month
    monthly_prices = prices.resample('M').last()
    
    # Compute monthly returns: (P_t / P_{t-1}) - 1
    monthly_returns = monthly_prices.pct_change()
    
    return monthly_returns


def compute_volatility(returns: pd.DataFrame, annualized: bool = True) -> pd.Series:
    """
    Compute volatility (standard deviation) of returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with returns (log returns or simple returns).
    annualized : bool, default=True
        If True, annualize volatility by multiplying by sqrt(252) for daily returns
        or sqrt(12) for monthly returns.
    
    Returns
    -------
    pd.Series
        Series with volatility for each asset.
    
    Examples
    --------
    >>> log_returns = compute_log_returns(prices)
    >>> volatility = compute_volatility(log_returns)
    """
    # Calculate standard deviation
    volatility = returns.std()
    
    if annualized:
        # Determine if returns are daily or monthly based on frequency
        # Approximate: if average time between returns < 5 days, assume daily
        if len(returns) > 1:
            avg_days_between = (returns.index[-1] - returns.index[0]).days / len(returns)
            if avg_days_between < 5:
                # Daily returns - annualize with sqrt(252)
                volatility = volatility * np.sqrt(252)
            else:
                # Monthly returns - annualize with sqrt(12)
                volatility = volatility * np.sqrt(12)
    
    return volatility


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute correlation matrix for asset returns.
    
    Parameters
    ----------
    returns : pd.DataFrame
        DataFrame with returns, where columns are assets and index is dates.
    
    Returns
    -------
    pd.DataFrame
        Correlation matrix with assets as both rows and columns.
        Values range from -1 to 1.
    
    Examples
    --------
    >>> log_returns = compute_log_returns(prices)
    >>> corr_matrix = compute_correlation_matrix(log_returns)
    """
    return returns.corr()


def get_etf_summary_stats(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "5y"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to download data and compute all summary statistics.
    
    Parameters
    ----------
    tickers : List[str]
        List of ETF ticker symbols.
    start_date : Optional[str], default=None
        Start date in 'YYYY-MM-DD' format.
    end_date : Optional[str], default=None
        End date in 'YYYY-MM-DD' format.
    period : str, default='5y'
        Period to download if start_date is None.
    
    Returns
    -------
    Tuple containing:
        - prices: pd.DataFrame with historical prices
        - daily_log_returns: pd.DataFrame with daily log returns
        - monthly_returns: pd.DataFrame with monthly returns
        - volatility: pd.Series with annualized volatility
        - correlation_matrix: pd.DataFrame with correlation matrix
    """
    # Download price data
    prices = download_etf_data(tickers, start_date=start_date, end_date=end_date, period=period)
    
    # Compute returns
    daily_log_returns = compute_log_returns(prices)
    monthly_returns = compute_monthly_returns(prices)
    
    # Compute volatility and correlation
    volatility = compute_volatility(daily_log_returns, annualized=True)
    correlation_matrix = compute_correlation_matrix(daily_log_returns)
    
    return prices, daily_log_returns, monthly_returns, volatility, correlation_matrix
