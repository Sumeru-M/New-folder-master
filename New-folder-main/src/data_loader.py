"""
Data Loader Module

This module handles fetching, cleaning, and caching historical market data
for Indian stocks (NSE) using yfinance.

Features:
- Fetches OHLCV price data via yfinance
- Supports lists of tickers with .NS suffix for NSE
- Supports date range parameters
- CSV-based caching for faster subsequent loads
- Data cleaning and normalization
"""

from typing import List, Optional
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import os
import hashlib


def _generate_cache_filename(tickers: List[str], start_date: Optional[str], end_date: Optional[str], period: str = "1y") -> str:
    """
    Generate a cache filename based on tickers and date range.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols
    start_date : Optional[str]
        Start date string
    end_date : Optional[str]
        End date string
    period : str
        Period string (e.g., '1y', 'max')
    
    Returns
    -------
    str
        Cache filename
    """
    # Create a unique identifier from tickers and dates
    ticker_str = "_".join(sorted(tickers))
    date_str = f"{start_date or 'None'}_{end_date or 'None'}_{period}"
    combined = f"{ticker_str}_{date_str}"
    
    # Create hash for filename (to handle long names)
    hash_obj = hashlib.md5(combined.encode())
    hash_str = hash_obj.hexdigest()[:8]
    
    # Create cache directory if it doesn't exist
    cache_dir = "data_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    return os.path.join(cache_dir, f"market_data_{hash_str}.csv")


def _load_from_cache(cache_file: str) -> Optional[pd.DataFrame]:
    """
    Load data from cache file if it exists.
    
    Parameters
    ----------
    cache_file : str
        Path to cache file
    
    Returns
    -------
    Optional[pd.DataFrame]
        Cached data if file exists, None otherwise
    """
    if os.path.exists(cache_file):
        try:
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True, date_format='ISO8601')
            print(f"Loaded data from cache: {cache_file}")
            return data
        except Exception as e:
            print(f"Error loading cache: {e}. Fetching fresh data...")
            return None
    return None


def _save_to_cache(data: pd.DataFrame, cache_file: str) -> None:
    """
    Save data to cache file.
    
    Parameters
    ----------
    data : pd.DataFrame
        Data to cache
    cache_file : str
        Path to cache file
    """
    try:
        data.to_csv(cache_file)
        print(f"Data cached to: {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save to cache: {e}")


def fetch_market_data(
    tickers: List[str],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True,
    period: str = "1y"
) -> pd.DataFrame:
    """
    Fetch historical OHLCV market data for Indian stocks (NSE).
    
    This function downloads data using yfinance and optionally caches it to CSV
    for faster subsequent loads. Data is cleaned and normalized before returning.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols with .NS suffix for NSE (e.g., ["RELIANCE.NS", "TCS.NS"])
    start_date : Optional[str], default=None
        Start date in 'YYYY-MM-DD' format. If None, fetches last 1 year.
    end_date : Optional[str], default=None
        End date in 'YYYY-MM-DD' format. If None, uses today's date.
    use_cache : bool, default=True
        If True, checks for cached data and saves new data to cache.
    period : str, default="1y"
        Period of data to fetch if dates are not provided.
        Valid values: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex columns: (Ticker, OHLCV)
        Index is datetime. Contains Open, High, Low, Close, Volume for each ticker.
        Example structure:
                    RELIANCE.NS              TCS.NS
                    Open  High  Low  Close  Open  High  Low  Close
        2020-01-01  100   105   99   103    200   205   198  202
        ...
    
    Raises
    ------
    ValueError
        If tickers list is empty, if period is invalid, or if data download fails.
    
    Examples
    --------
    >>> # Fetch data for last 1 year
    >>> data = fetch_market_data(["RELIANCE.NS", "TCS.NS"])
    
    >>> # Fetch data for specific date range
    >>> data = fetch_market_data(
    ...     ["RELIANCE.NS", "TCS.NS"],
    ...     start_date="2020-01-01",
    ...     end_date="2023-12-31"
    ... )
    """
    if not tickers:
        raise ValueError("Tickers list cannot be empty")
    
    # Validate period if using period (not date range)
    if not (start_date and end_date):
        valid_periods = ['1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max']
        if period not in valid_periods:
            raise ValueError(
                f"Invalid period '{period}'. Valid periods are: {', '.join(valid_periods)}. "
                f"Did you mean '1y' or '1d'?"
            )
    
    # Generate cache filename
    cache_file = _generate_cache_filename(tickers, start_date, end_date, period) if use_cache else None
    
    # Try to load from cache
    if use_cache and cache_file:
        cached_data = _load_from_cache(cache_file)
        if cached_data is not None:
            # Skip empty cache files
            if cached_data.empty or cached_data.shape[0] == 0:
                print(f"Warning: Cache file is empty, fetching fresh data...")
            else:   
                 # Some code with proper indentation
                    if cached_data is not None and hasattr(cached_data, 'columns'):
                        cached_tickers = [col[0] for col in cached_data.columns.levels[0]]
                    else:
                     cached_tickers = []
                   # ✅ Safely handles None case
                     if set(tickers).issubset(set(cached_tickers)):
                        return cached_data
    
    # Download data using yfinance
    print(f"Fetching market data for {len(tickers)} ticker(s)...")
    
    if start_date and end_date:
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
    else:
        # Use provided period
        data = yf.download(tickers, period=period, progress=False)
    
    # Handle different yfinance return formats
    # yfinance typically returns: MultiIndex columns (OHLCV, Ticker) for multiple tickers
    # or flat columns [Open, High, Low, Close, Volume] for single ticker
    
    if isinstance(data.columns, pd.MultiIndex):
        # Multiple tickers: yfinance format is (OHLCV, Ticker)
        # We want (Ticker, OHLCV) for consistency
        level_0 = data.columns.get_level_values(0)
        level_1 = data.columns.get_level_values(1)
        
        # Check if first level is OHLCV (standard yfinance format)
        if 'Open' in level_0 or 'Close' in level_0:
            # Format: (OHLCV, Ticker) - need to swap
            data_dict = {}
            for ticker in tickers:
                for metric in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    try:
                        data_dict[(ticker, metric)] = data[(metric, ticker)]
                    except KeyError:
                        # Ticker might not be in data, skip
                        pass
            if data_dict:
                data = pd.DataFrame(data_dict)
            else:
                raise ValueError(f"Could not extract data for tickers: {tickers}")
        else:
            # Format: (Ticker, OHLCV) - already correct
            pass
    else:
        # Single ticker: flat columns
        if len(tickers) == 1:
            data_dict = {}
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns:
                    data_dict[(tickers[0], col)] = data[col]
            if data_dict:
                data = pd.DataFrame(data_dict)
            else:
                raise ValueError(f"Could not extract OHLCV data for {tickers[0]}")
        else:
            raise ValueError("Unexpected data format: flat columns with multiple tickers")
    
    # Check if data is empty
    if data.empty or data.shape[0] == 0:
        error_msg = f"No data found for tickers: {tickers}"
        if not (start_date and end_date):
            error_msg += f" with period='{period}'"
        else:
            error_msg += f" with date range {start_date} to {end_date}"
        error_msg += "\nPossible reasons:"
        error_msg += "\n  - Tickers may be incorrect or delisted"
        error_msg += "\n  - Period may be too short (try '1y' or '2y')"
        error_msg += "\n  - Network connection issues"
        error_msg += "\n  - Market data unavailable for the specified period"
        raise ValueError(error_msg)
    
    # Clean and normalize data
    data = _clean_data(data, tickers)
    
    # Check which tickers actually have data after cleaning
    if isinstance(data.columns, pd.MultiIndex):
        available_tickers = list(data.columns.get_level_values(0).unique())
        missing_tickers = set(tickers) - set(available_tickers)
        if missing_tickers:
            print(f"Warning: No data found for ticker(s): {', '.join(missing_tickers)}")
            if not available_tickers:
                error_msg = f"All tickers failed: {tickers}"
                if not (start_date and end_date):
                    error_msg += f" with period='{period}'"
                else:
                    error_msg += f" with date range {start_date} to {end_date}"
                raise ValueError(error_msg)
    
    # Save to cache
    if use_cache and cache_file:
        _save_to_cache(data, cache_file)
    
    available_count = len(data.columns.get_level_values(0).unique()) if isinstance(data.columns, pd.MultiIndex) else len(tickers)
    print(f"Successfully fetched data: {data.shape[0]} rows, {available_count} ticker(s)")
    return data


def _clean_data(data: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Clean and normalize market data.
    
    Parameters
    ----------
    data : pd.DataFrame
        Raw market data
    tickers : List[str]
        List of ticker symbols
    
    Returns
    -------
    pd.DataFrame
        Cleaned data with proper structure
    """
    # Remove any rows with all NaN values
    data = data.dropna(how='all')
    
    # Ensure proper column structure
    if isinstance(data.columns, pd.MultiIndex):
        # Ensure we have the required columns for each ticker
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_tickers = []
        
        for ticker in tickers:
            ticker_cols = [col for col in data.columns if col[0] == ticker]
            if ticker_cols:
                available_tickers.append(ticker)
            else:
                print(f"Warning: No data found for {ticker}")
        
        # Reorder columns to match ticker order
        if available_tickers:
            ordered_cols = []
            for ticker in available_tickers:
                for col in required_cols:
                    col_tuple = (ticker, col)
                    if col_tuple in data.columns:
                        ordered_cols.append(col_tuple)
            if ordered_cols:
                data = data[ordered_cols]
    else:
        # Convert flat structure to MultiIndex if needed
        print("Warning: Data does not have MultiIndex columns. Attempting to restructure...")
    
    # Sort by date
    data = data.sort_index()
    
    # Remove duplicate dates (keep last)
    data = data[~data.index.duplicated(keep='last')]
    
    return data


def get_close_prices(data: pd.DataFrame) -> pd.DataFrame:
    """
    Extract close prices from OHLCV data.
    
    Parameters
    ----------
    data : pd.DataFrame
        OHLCV data with MultiIndex columns (Ticker, OHLCV)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with Close prices, columns are tickers, index is dates
    """
    if isinstance(data.columns, pd.MultiIndex):
        close_data = {}
        for ticker in data.columns.get_level_values(0).unique():
            if (ticker, 'Close') in data.columns:
                close_data[ticker] = data[(ticker, 'Close')]
        return pd.DataFrame(close_data)
    else:
        raise ValueError("Data must have MultiIndex columns (Ticker, OHLCV)")


def get_stock_data(
    tickers: List[str],
    period: str = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Simple wrapper to get stock data for multiple tickers.
    Works just like yfinance.download() but cleaner and returns close prices.
    
    This is a convenience function that fetches data and returns close prices
    in a simple format (tickers as columns, dates as index).
    
    Parameters
    ----------
    tickers : List[str] or str
        List of tickers (e.g., ["RELIANCE.NS", "TCS.NS"]) or single ticker string.
        For Indian markets, use .NS suffix for NSE-listed stocks.
    period : str, default="1y"
        Period of data to fetch. Valid values: 
        '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        Only used if start_date and end_date are None.
    start_date : Optional[str], default=None
        Start date in 'YYYY-MM-DD' format. If provided, overrides period.
    end_date : Optional[str], default=None
        End date in 'YYYY-MM-DD' format. If None, uses today's date.
    use_cache : bool, default=True
        If True, uses cached data if available and saves new data to cache.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with close prices. Columns are tickers, index is dates.
        Example:
                    RELIANCE.NS  TCS.NS  INFY.NS
        2023-01-01     2450.50  3456.75  1523.25
        2023-01-02     2455.20  3460.10  1525.50
        ...
    
    Examples
    --------
    >>> # Simple usage with period
    >>> prices = get_stock_data(["RELIANCE.NS", "TCS.NS"], period="2y")
    
    >>> # Single ticker (string)
    >>> prices = get_stock_data("RELIANCE.NS", period="1y")
    
    >>> # With date range
    >>> prices = get_stock_data(
    ...     ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    ...     start_date="2023-01-01",
    ...     end_date="2024-01-01"
    ... )
    
    >>> # Use with portfolio engine
    >>> from src.portfolio_engine import compute_portfolio_metrics
    >>> prices = get_stock_data(["RELIANCE.NS", "TCS.NS"], period="1y")
    >>> metrics = compute_portfolio_metrics(prices)
    """
    # Convert single ticker string to list
    if isinstance(tickers, str):
        tickers = [tickers]
    
    if not tickers:
        raise ValueError("Tickers list cannot be empty")
    
    # Fetch OHLCV data
    ohlcv_data = fetch_market_data(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
        period=period
    )
    
    # Return close prices (most common use case)
    return get_close_prices(ohlcv_data)


def fetch_live_quotes(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch live/current market quotes for Indian stocks (NSE).
    
    This function fetches real-time or near real-time price data using yfinance.
    Note: yfinance provides delayed data (typically 15-20 minutes delay for free tier).
    For true real-time data, you would need a paid API service.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols with .NS suffix for NSE (e.g., ["RELIANCE.NS", "TCS.NS"])
    
    Returns
    -------
    pd.DataFrame
        DataFrame with current market data. Columns include:
        - 'currentPrice' or 'regularMarketPrice': Current/last traded price
        - 'bid': Bid price
        - 'ask': Ask price
        - 'volume': Current day volume
        - 'marketCap': Market capitalization
        - 'previousClose': Previous day's close
        - 'dayHigh': Day's high
        - 'dayLow': Day's low
        - 'open': Day's open price
    
    Examples
    --------
    >>> quotes = fetch_live_quotes(["RELIANCE.NS", "TCS.NS"])
    >>> print(quotes)
    """
    if not tickers:
        raise ValueError("Tickers list cannot be empty")
    
    print(f"Fetching live quotes for {len(tickers)} ticker(s)...")
    
    quotes_data = {}
    
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.fast_info  # Fast info for quick access
            
            # Get current price and other live data
            quote = {
                'ticker': ticker,
                'currentPrice': getattr(info, 'lastPrice', None) or getattr(info, 'regularMarketPrice', None),
                'previousClose': getattr(info, 'previousClose', None),
                'open': getattr(info, 'open', None),
                'dayHigh': getattr(info, 'dayHigh', None),
                'dayLow': getattr(info, 'dayLow', None),
                'volume': getattr(info, 'volume', None),
                'marketCap': getattr(info, 'marketCap', None),
            }
            
            # Try to get more detailed info if fast_info doesn't have everything
            try:
                detailed_info = ticker_obj.info
                quote['bid'] = detailed_info.get('bid', None)
                quote['ask'] = detailed_info.get('ask', None)
                quote['bidSize'] = detailed_info.get('bidSize', None)
                quote['askSize'] = detailed_info.get('askSize', None)
            except:
                pass
            
            quotes_data[ticker] = quote
            
        except Exception as e:
            print(f"Warning: Could not fetch live data for {ticker}: {e}")
            quotes_data[ticker] = {'ticker': ticker, 'error': str(e)}
    
    # Convert to DataFrame
    quotes_df = pd.DataFrame(quotes_data).T
    quotes_df.index.name = 'ticker'
    
    return quotes_df


def fetch_intraday_data(
    tickers: List[str],
    interval: str = "5m",
    period: str = "1d"
) -> pd.DataFrame:
    """
    Fetch intraday (live) market data for Indian stocks (NSE).
    
    This function fetches intraday price data with minute-level granularity.
    Useful for real-time monitoring and intraday analysis.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols with .NS suffix for NSE (e.g., ["RELIANCE.NS", "TCS.NS"])
    interval : str, default="5m"
        Data interval. Valid values: '1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo'
        Note: '1m' data is only available for last 7 days
    period : str, default="1d"
        Period of data to fetch. Valid values: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
        For intraday, typically use '1d' or '5d'
    
    Returns
    -------
    pd.DataFrame
        DataFrame with MultiIndex columns: (Ticker, OHLCV)
        Index is datetime with minute-level precision.
        Contains Open, High, Low, Close, Volume for each ticker.
    
    Examples
    --------
    >>> # Fetch 5-minute intraday data for today
    >>> intraday = fetch_intraday_data(["RELIANCE.NS", "TCS.NS"], interval="5m", period="1d")
    
    >>> # Fetch 1-minute data (last 7 days max)
    >>> intraday = fetch_intraday_data(["RELIANCE.NS"], interval="1m", period="5d")
    """
    if not tickers:
        raise ValueError("Tickers list cannot be empty")
    
    print(f"Fetching intraday data for {len(tickers)} ticker(s) (interval: {interval}, period: {period})...")
    
    # Download intraday data
    data = yf.download(
        tickers,
        period=period,
        interval=interval,
        progress=False
    )
    
    # Handle different yfinance return formats (same as fetch_market_data)
    if isinstance(data.columns, pd.MultiIndex):
        level_0 = data.columns.get_level_values(0)
        level_1 = data.columns.get_level_values(1)
        
        if 'Open' in level_0 or 'Close' in level_0:
            # Format: (OHLCV, Ticker) - need to swap
            data_dict = {}
            for ticker in tickers:
                for metric in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    try:
                        data_dict[(ticker, metric)] = data[(metric, ticker)]
                    except KeyError:
                        pass
            if data_dict:
                data = pd.DataFrame(data_dict)
        # else: Format: (Ticker, OHLCV) - already correct
    else:
        # Single ticker: flat columns
        if len(tickers) == 1:
            data_dict = {}
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in data.columns:
                    data_dict[(tickers[0], col)] = data[col]
            if data_dict:
                data = pd.DataFrame(data_dict)
    
    # Clean data
    data = _clean_data(data, tickers)
    
    print(f"Successfully fetched intraday data: {data.shape[0]} data points")
    return data


def fetch_live_prices(tickers: List[str]) -> pd.Series:
    """
    Fetch current/live prices for multiple tickers (simplified version).
    
    This is a convenience function that returns just the current prices
    as a pandas Series.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols with .NS suffix for NSE
    
    Returns
    -------
    pd.Series
        Series with current prices, indexed by ticker
    
    Examples
    --------
    >>> prices = fetch_live_prices(["RELIANCE.NS", "TCS.NS", "INFY.NS"])
    >>> print(prices)
    """
    if not tickers:
        raise ValueError("Tickers list cannot be empty")
    
    current_prices = {}
    
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.fast_info
            price = getattr(info, 'lastPrice', None) or getattr(info, 'regularMarketPrice', None)
            current_prices[ticker] = price
        except Exception as e:
            print(f"Warning: Could not fetch price for {ticker}: {e}")
            current_prices[ticker] = None
    
    return pd.Series(current_prices, name='currentPrice')
