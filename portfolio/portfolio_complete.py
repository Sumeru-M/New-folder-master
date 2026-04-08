"""
portfolio_complete.py
=====================
Milestone 3, 4 & 5 — Complete Portfolio Library
CONSOLIDATED SINGLE-FILE EDITION

All portfolio modules used by run_milestone3, run_milestone4,
and run_milestone5 merged into one self-contained file.
Place this file inside your  portfolio/  folder.

Modules included
----------------
DATA LOADER         fetch_market_data, get_stock_data, load_price_data
OPTIMIZER           PortfolioOptimizer, compute_daily_returns,
                    compute_expected_returns, compute_covariance_matrix,
                    compute_ledoit_wolf_shrinkage, compute_weight_dispersion,
                    compare_covariance_methods, OptimizationResult
PLOTTING            plot_efficient_frontier, plot_correlation_heatmap,
                    plot_drawdown_chart
INVESTOR GUIDE      format_portfolio_summary, format_comparison_summary,
                    create_investor_friendly_csv, create_simple_weights_csv
FACTOR MODEL        FactorModel, FactorAnalysisResult
RISK METRICS        compute_parametric_var, compute_historical_var,
                    compute_cvar, compute_component_var,
                    compute_portfolio_risk_metrics, compute_max_drawdown,
                    compute_ulcer_index, detect_market_regime
SCENARIO ENGINE     MarketShock, ScenarioEngine, nearest_positive_definite
SECTOR DATA         NSE_SECTOR_MAP, get_sector

NOT included (M5 only — keep their original files)
---------------------------------------------------
constraints.py, optimization_engine.py, robust_optimizer.py,
risk_contribution.py, allocation_scorer.py

These five files are only used by Milestone 5 and were not available
for consolidation. Keep them as individual files in portfolio/.

Public entry points (same as before — no import changes needed)
---------------------------------------------------------------
from portfolio.portfolio_complete import (
    load_price_data, get_stock_data,
    PortfolioOptimizer, OptimizationResult,
    compute_daily_returns, compute_expected_returns,
    compute_covariance_matrix, compute_ledoit_wolf_shrinkage,
    compute_weight_dispersion, compare_covariance_methods,
    plot_efficient_frontier, plot_correlation_heatmap, plot_drawdown_chart,
    format_portfolio_summary, format_comparison_summary,
    create_investor_friendly_csv, create_simple_weights_csv,
    FactorModel, FactorAnalysisResult,
    compute_parametric_var, compute_historical_var, compute_cvar,
    compute_component_var, compute_portfolio_risk_metrics,
    compute_max_drawdown, compute_ulcer_index, detect_market_regime,
    MarketShock, ScenarioEngine, nearest_positive_definite,
    NSE_SECTOR_MAP, get_sector,
)

Dependencies: numpy, pandas, scipy, matplotlib, cvxpy, yfinance, statsmodels
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from datetime import datetime
import hashlib
import os
import re
import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from typing import Any, Dict, List, Optional, Tuple, Union
import yfinance as yf


# ══════════════════════════════════════════════════════════════════════════
# DATA LOADER — fetch_market_data, get_stock_data, load_price_data alias
# ══════════════════════════════════════════════════════════════════════════

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
    
    # Create cache directory if it doesn't exist.
    # Vercel functions are read-only except /tmp, so use /tmp there.
    cache_dir = os.getenv("DATA_CACHE_DIR")
    if not cache_dir:
        cache_dir = "/tmp/data_cache" if os.getenv("VERCEL") else "data_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
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
            # Load with MultiIndex columns (header=[0,1] reads two header rows)
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True, date_format='ISO8601', header=[0, 1])
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
        # Save with MultiIndex columns preserved (header=True saves both levels)
        data.to_csv(cache_file, header=True)
        print(f"Data cached to: {cache_file}")
    except Exception as e:
        print(f"Warning: Could not save to cache: {e}")


_NSE_NAME_TO_SYMBOL = {
    "BHARATELECTRONICSLTD": "BEL.NS",
    "BIOCONLIMITED": "BIOCON.NS",
    "COFORGELIMITED": "COFORGE.NS",
    "DRREDDYSLABORATORIES": "DRREDDY.NS",
    "HCLTECHNOLOGIESLTD": "HCLTECH.NS",
    "HDFCBANKLTD": "HDFCBANK.NS",
    "JBMAUTOLIMITED": "JBMA.NS",
    "JINDALSTAINLESSLIMITED": "JSL.NS",
    "KFINTECHNOLOGIESLIMITED": "KFINTECH.NS",
    "MOTILALOSNASDAQ100ETF": "MON100.NS",
    "NIPINDETFNIFTYBEES": "NIFTYBEES.NS",
    "SAKSOFTLIMITED": "SAKSOFT.NS",
    "STATEBANKOFINDIA": "SBIN.NS",
    "TATAMOTORSLIMITED": "TATAMOTORS.NS",
}
_YF_TICKER_SEARCH_CACHE: Dict[str, str] = {}


def _normalize_nse_ticker(raw: str) -> str:
    """
    Convert common company-name style inputs into valid NSE ticker symbols.
    """
    value = str(raw or "").strip().upper().rstrip(".")
    if not value:
        return value

    # Keep indices/explicit symbols untouched (e.g. ^NSEI)
    if value.startswith("^"):
        return value

    # If this is already a compact exchange symbol, ensure NSE suffix
    if re.fullmatch(r"[A-Z0-9_-]+(?:\.[A-Z]{1,4})?", value):
        if "." not in value:
            return f"{value}.NS"
        return value

    # Company name fallback (e.g., "HDFC BANK LTD.NS", "DR. REDDY S LABORATORIES")
    base = value[:-3] if value.endswith(".NS") else value
    compact = re.sub(r"[^A-Z0-9]+", "", base)
    mapped = _NSE_NAME_TO_SYMBOL.get(compact)
    if mapped:
        return mapped

    # Dynamic Yahoo Finance lookup for unknown names/symbol variants.
    # This keeps CSV/company-name ingestion flexible without hardcoding every name.
    search_queries = [base]
    simplified = re.sub(r"\b(LTD|LIMITED|INC|PLC|CO|COMPANY)\b", "", base).strip()
    if simplified and simplified not in search_queries:
        search_queries.append(simplified)

    for query in search_queries:
        qkey = query.upper()
        if qkey in _YF_TICKER_SEARCH_CACHE:
            cached = _YF_TICKER_SEARCH_CACHE[qkey]
            if cached:
                return cached
            continue
        try:
            search = yf.Search(query=query, max_results=8)
            quotes = getattr(search, "quotes", []) or []
            picked = ""
            for q in quotes:
                symbol = str(q.get("symbol", "")).upper().strip()
                exchange = str(q.get("exchange", "")).upper().strip()
                if not symbol:
                    continue
                if symbol.endswith(".NS"):
                    picked = symbol
                    break
                if exchange in {"NSE", "NSI", "NATIONAL STOCK EXCHANGE OF INDIA"}:
                    picked = symbol if "." in symbol else f"{symbol}.NS"
                    break
            _YF_TICKER_SEARCH_CACHE[qkey] = picked
            if picked:
                return picked
        except Exception:
            _YF_TICKER_SEARCH_CACHE[qkey] = ""

    # Last-resort cleanup for unknown names
    guessed = re.sub(r"[^A-Z0-9]+", "", base)
    return f"{guessed}.NS" if guessed else value


def _normalize_ticker_list(tickers: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for ticker in tickers:
        norm = _normalize_nse_ticker(ticker)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out


def normalize_tickers_for_market_data(tickers: List[str]) -> List[str]:
    """
    Public helper to normalize symbols/company names into yfinance-ready tickers.
    """
    return _normalize_ticker_list(tickers)


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

    original_tickers = list(tickers)
    tickers = _normalize_ticker_list(tickers)
    if not tickers:
        raise ValueError("No valid tickers found after normalization")
    if tickers != original_tickers:
        print(f"Normalized tickers: {original_tickers} -> {tickers}")
    
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
                # Check if cached data has the tickers we need
                if isinstance(cached_data.columns, pd.MultiIndex):
                    # For MultiIndex columns, get unique tickers from level 0
                    cached_tickers = list(cached_data.columns.get_level_values(0).unique())
                else:
                    # For flat columns, use column names directly
                    cached_tickers = list(cached_data.columns)
                
                # If all requested tickers are in cache, return cached data
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


# ══════════════════════════════════════════════════════════════════════════
# OPTIMIZER — PortfolioOptimizer, compute_daily_returns, compute_expected_returns, compute_covariance_matrix, compute_ledoit_wolf_shrinkage, compute_weight_dispersion, compare_covariance_methods
# ══════════════════════════════════════════════════════════════════════════

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
        # For LOG returns, use exponential annualization
        # Formula: exp(mean_log_return * 252) - 1
        # This is correct because log returns add: ln(1+r_total) = sum(ln(1+r_daily))
        mean_daily_log_return = daily_returns.mean()
        annualized_return = np.exp(mean_daily_log_return * 252) - 1
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
    pi_hat = 0.0
    for i in range(N):
        for j in range(N):
            # Compute variance of sample covariance element S_ij
            cross_prod = X_centered[:, i] * X_centered[:, j]
            pi_hat += np.var(cross_prod, ddof=1)
    
    pi_hat = pi_hat / T
    
    # Compute gamma: squared Frobenius norm of (S - F)
    gamma_hat = np.sum((S - F) ** 2)
    
    # Compute kappa (simplification - assumes constant correlation)
    kappa_hat = pi_hat / gamma_hat if gamma_hat > 0 else 1.0
    
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

# ══════════════════════════════════════════════════════════════════════════
# PLOTTING — plot_efficient_frontier, plot_correlation_heatmap, plot_drawdown_chart
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# INVESTOR GUIDE — format_portfolio_summary, format_comparison_summary, create_investor_friendly_csv, create_simple_weights_csv
# ══════════════════════════════════════════════════════════════════════════

def interpret_volatility(volatility: float) -> Dict[str, Any]:
    """
    Interpret volatility in investor-friendly terms.
    
    Parameters
    ----------
    volatility : float
        Annual volatility (e.g., 0.20 for 20%)
    
    Returns
    -------
    Dict with interpretation, risk level, and explanation
    """
    vol_pct = volatility * 100
    
    if vol_pct < 10:
        risk_level = "Very Low"
        interpretation = "Conservative"
        explanation = "Your portfolio is very stable with minimal price swings. Suitable for risk-averse investors."
    elif vol_pct < 15:
        risk_level = "Low"
        interpretation = "Moderate"
        explanation = "Your portfolio has modest price movements. Good balance of stability and growth potential."
    elif vol_pct < 25:
        risk_level = "Moderate"
        interpretation = "Balanced"
        explanation = "Your portfolio has moderate price swings. Suitable for investors comfortable with some risk."
    elif vol_pct < 35:
        risk_level = "High"
        interpretation = "Aggressive"
        explanation = "Your portfolio experiences significant price movements. Higher risk, higher potential returns."
    else:
        risk_level = "Very High"
        interpretation = "Very Aggressive"
        explanation = "Your portfolio is highly volatile with large price swings. Only for risk-tolerant investors."
    
    return {
        'risk_level': risk_level,
        'interpretation': interpretation,
        'explanation': explanation,
        'daily_range': f"±{vol_pct/np.sqrt(252):.2f}%"
    }


def interpret_sharpe_ratio(sharpe: float) -> Dict[str, Any]:
    """
    Interpret Sharpe ratio in investor-friendly terms.
    
    Parameters
    ----------
    sharpe : float
        Sharpe ratio
    
    Returns
    -------
    Dict with interpretation and explanation
    """
    if sharpe < 0:
        quality = "Poor"
        explanation = "Your portfolio is not providing adequate returns for the risk taken. Consider rebalancing."
    elif sharpe < 0.5:
        quality = "Below Average"
        explanation = "Your portfolio's risk-adjusted returns are modest. There may be better allocation options."
    elif sharpe < 1.0:
        quality = "Good"
        explanation = "Your portfolio provides decent returns relative to risk. A solid investment choice."
    elif sharpe < 2.0:
        quality = "Very Good"
        explanation = "Your portfolio offers excellent risk-adjusted returns. Well-optimized allocation."
    else:
        quality = "Excellent"
        explanation = "Your portfolio has outstanding risk-adjusted returns. Exceptional performance."
    
    return {
        'quality': quality,
        'explanation': explanation
    }


def interpret_diversification(effective_n: float, total_assets: int) -> Dict[str, Any]:
    """
    Interpret diversification level.
    
    Parameters
    ----------
    effective_n : float
        Effective number of assets
    total_assets : int
        Total number of assets in portfolio
    
    Returns
    -------
    Dict with interpretation and explanation
    """
    diversification_pct = (effective_n / total_assets) * 100
    
    if effective_n < 2:
        level = "Highly Concentrated"
        explanation = f"Your portfolio is heavily focused on just {effective_n:.1f} assets. High risk if one asset underperforms."
        recommendation = "Consider diversifying across more assets to reduce risk."
    elif effective_n < total_assets * 0.5:
        level = "Concentrated"
        explanation = f"Your portfolio is somewhat concentrated with effective diversification of {effective_n:.1f} assets."
        recommendation = "Adding more diversification could help reduce risk."
    elif effective_n < total_assets * 0.8:
        level = "Well Diversified"
        explanation = f"Your portfolio is well-diversified with effective exposure to {effective_n:.1f} assets."
        recommendation = "Good diversification level. Maintain this balance."
    else:
        level = "Highly Diversified"
        explanation = f"Your portfolio is highly diversified with effective exposure to {effective_n:.1f} assets."
        recommendation = "Excellent diversification. This helps reduce risk from individual asset movements."
    
    return {
        'level': level,
        'explanation': explanation,
        'recommendation': recommendation,
        'diversification_pct': diversification_pct
    }


def format_portfolio_summary(
    result,
    dispersion: Dict[str, float],
    portfolio_name: str = "Portfolio"
) -> str:
    """
    Create an investor-friendly portfolio summary.
    
    Parameters
    ----------
    result : OptimizationResult
        Portfolio optimization result
    dispersion : Dict
        Weight dispersion metrics
    portfolio_name : str
        Name of the portfolio
    
    Returns
    -------
    str
        Formatted summary text
    """
    vol_info = interpret_volatility(result.volatility)
    sharpe_info = interpret_sharpe_ratio(result.sharpe_ratio)
    div_info = interpret_diversification(
        dispersion['effective_n'],
        len(result.weights)
    )
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  {portfolio_name:^58}  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  📈 EXPECTED RETURNS                                           ║
║     {result.expected_return:>6.2%} per year (annualized)                    ║
║                                                                ║
║  📊 RISK LEVEL                                                 ║
║     {vol_info['risk_level']:^20} ({result.volatility:.2%} volatility)        ║
║     {vol_info['explanation']:<58} ║
║     Daily price swings: typically {vol_info['daily_range']} per day        ║
║                                                                ║
║  ⭐ RISK-ADJUSTED PERFORMANCE                                   ║
║     Sharpe Ratio: {result.sharpe_ratio:.2f} ({sharpe_info['quality']:^20})  ║
║     {sharpe_info['explanation']:<58} ║
║                                                                ║
║  🎯 DIVERSIFICATION                                            ║
║     {div_info['level']:^20} ({dispersion['effective_n']:.1f} effective assets)  ║
║     {div_info['explanation']:<58} ║
║     💡 {div_info['recommendation']:<54} ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    return summary


def format_comparison_summary(comparison, optimization_type: str) -> str:
    """
    Create investor-friendly comparison summary.
    
    Parameters
    ----------
    comparison : ComparisonResult
        Comparison result
    optimization_type : str
        Type of optimization ("min_variance" or "max_sharpe")
    
    Returns
    -------
    str
        Formatted comparison text
    """
    opt_name = "Lowest Risk" if optimization_type == "min_variance" else "Best Risk-Adjusted Returns"
    
    sample_vol = interpret_volatility(comparison.sample_result.volatility)
    shrink_vol = interpret_volatility(comparison.shrinkage_result.volatility)
    
    sample_sharpe = interpret_sharpe_ratio(comparison.sample_result.sharpe_ratio)
    shrink_sharpe = interpret_sharpe_ratio(comparison.shrinkage_result.sharpe_ratio)
    
    sample_div = interpret_diversification(
        comparison.sample_dispersion['effective_n'],
        len(comparison.sample_result.weights)
    )
    shrink_div = interpret_diversification(
        comparison.shrinkage_dispersion['effective_n'],
        len(comparison.shrinkage_result.weights)
    )
    
    # Determine which is better
    better_volatility = "Robust Method" if comparison.volatility_difference < 0 else "Standard Method"
    better_sharpe = "Robust Method" if comparison.sharpe_difference > 0 else "Standard Method"
    better_div = "Robust Method" if comparison.shrinkage_dispersion['effective_n'] > comparison.sample_dispersion['effective_n'] else "Standard Method"
    better_return = "Robust Method" if comparison.shrinkage_result.expected_return > comparison.sample_result.expected_return else "Standard Method"
    
    summary = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PORTFOLIO COMPARISON: {opt_name:^45}  ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  📊 TWO METHODS COMPARED                                                    ║
║     • Standard Method: Uses historical data directly                        ║
║     • Robust Method: Uses advanced technique to reduce estimation errors   ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  STANDARD METHOD                                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Expected Return:     {comparison.sample_result.expected_return:>6.2%} per year              ║
║  Risk Level:          {sample_vol['risk_level']:^20} ({comparison.sample_result.volatility:.2%})  ║
║  Performance Quality: {sample_sharpe['quality']:^20} (Sharpe: {comparison.sample_result.sharpe_ratio:.2f}) ║
║  Diversification:     {sample_div['level']:^20} ({comparison.sample_dispersion['effective_n']:.1f} assets) ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  ROBUST METHOD (Recommended)                                               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Expected Return:     {comparison.shrinkage_result.expected_return:>6.2%} per year              ║
║  Risk Level:          {shrink_vol['risk_level']:^20} ({comparison.shrinkage_result.volatility:.2%})  ║
║  Performance Quality: {shrink_sharpe['quality']:^20} (Sharpe: {comparison.shrinkage_result.sharpe_ratio:.2f}) ║
║  Diversification:     {shrink_div['level']:^20} ({comparison.shrinkage_dispersion['effective_n']:.1f} assets) ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  KEY DIFFERENCES                                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Return Change:       {(comparison.shrinkage_result.expected_return - comparison.sample_result.expected_return):+.4%} ({better_return:^20})  ║
║  Risk Change:         {comparison.volatility_difference:+.4%} ({better_volatility:^20})  ║
║  Performance Change:  {comparison.sharpe_difference:+.4f} ({better_sharpe:^20})  ║
║  Diversification:     {comparison.shrinkage_dispersion['effective_n'] - comparison.sample_dispersion['effective_n']:+.2f} assets ({better_div:^20})║
║                                                                              ║
║  💡 RECOMMENDATION                                                           ║
║     The Robust Method typically provides more stable and reliable results.  ║
║     It reduces the impact of estimation errors and produces more balanced  ║
║     portfolio allocations. Consider using the Robust Method weights.        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    return summary


def create_investor_friendly_csv(comparison, optimization_type: str) -> pd.DataFrame:
    """
    Create investor-friendly CSV output.
    
    Parameters
    ----------
    comparison : ComparisonResult
        Comparison result
    optimization_type : str
        Type of optimization
    
    Returns
    -------
    pd.DataFrame
        Investor-friendly comparison DataFrame
    """
    # Create simplified comparison
    data = {
        'Metric': [
            'Expected Annual Return (Standard)',
            'Expected Annual Return (Robust)',
            'Risk Level - Volatility (Standard)',
            'Risk Level - Volatility (Robust)',
            'Risk-Adjusted Performance - Sharpe (Standard)',
            'Risk-Adjusted Performance - Sharpe (Robust)',
            'Number of Effective Assets (Standard)',
            'Number of Effective Assets (Robust)',
            'Diversification Level (Standard)',
            'Diversification Level (Robust)',
            'Change in Risk (Robust vs Standard)',
            'Change in Performance (Robust vs Standard)',
            'Change in Diversification (Robust vs Standard)'
        ],
        'Value': [
            f"{comparison.sample_result.expected_return:.2%}",
            f"{comparison.shrinkage_result.expected_return:.2%}",
            f"{comparison.sample_result.volatility:.2%}",
            f"{comparison.shrinkage_result.volatility:.2%}",
            f"{comparison.sample_result.sharpe_ratio:.2f}",
            f"{comparison.shrinkage_result.sharpe_ratio:.2f}",
            f"{comparison.sample_dispersion['effective_n']:.1f}",
            f"{comparison.shrinkage_dispersion['effective_n']:.1f}",
            interpret_diversification(comparison.sample_dispersion['effective_n'], len(comparison.sample_result.weights))['level'],
            interpret_diversification(comparison.shrinkage_dispersion['effective_n'], len(comparison.shrinkage_result.weights))['level'],
            f"{comparison.volatility_difference:+.2%}",
            f"{comparison.sharpe_difference:+.2f}",
            f"{comparison.shrinkage_dispersion['effective_n'] - comparison.sample_dispersion['effective_n']:+.1f} assets"
        ],
        'What This Means': [
            'How much your portfolio is expected to grow per year',
            'How much your portfolio is expected to grow per year (more reliable estimate)',
            'How much your portfolio price can swing up or down',
            'How much your portfolio price can swing (more stable estimate)',
            'How good your returns are relative to risk taken',
            'How good your returns are relative to risk (more accurate)',
            'How many different investments you effectively have',
            'How many different investments you effectively have (more balanced)',
            'How spread out your investments are',
            'How spread out your investments are (better balanced)',
            'Lower is better - shows if robust method reduces risk',
            'Higher is better - shows if robust method improves performance',
            'Higher is better - shows if robust method improves diversification'
        ]
    }
    
    return pd.DataFrame(data)


def create_simple_weights_csv(result, method_name: str) -> pd.DataFrame:
    """
    Create simple, investor-friendly weights CSV.
    
    Parameters
    ----------
    result : OptimizationResult
        Optimization result
    method_name : str
        Name of the method (e.g., "Robust Method - Best Returns")
    
    Returns
    -------
    pd.DataFrame
        Simple weights DataFrame
    """
    df = pd.DataFrame({
        'Stock': result.weights.index,
        'Recommended Allocation (%)': (result.weights.values * 100).round(2),
        'What This Means': [
            f"Invest {weight*100:.1f}% of your portfolio in {ticker}" 
            for ticker, weight in result.weights.items()
        ]
    })
    
    return df


def interpret_var(var_amount: float, var_percent: float, portfolio_value: float, confidence: float) -> Dict[str, Any]:
    """
    Interpret Value at Risk (VaR) in investor-friendly terms.
    
    Parameters
    ----------
    var_amount : float
        VaR in INR terms
    var_percent : float
        VaR as percentage
    portfolio_value : float
        Total portfolio value
    confidence : float
        Confidence level (e.g., 0.95 for 95%)
    
    Returns
    -------
    Dict with interpretation and explanation
    """
    conf_pct = confidence * 100
    
    if var_percent < 0.01:
        risk_level = "Very Low"
        explanation = f"Your portfolio has very low risk. On {conf_pct:.0f}% of days, you might lose at most {var_percent:.2%}."
    elif var_percent < 0.02:
        risk_level = "Low"
        explanation = f"Your portfolio has low risk. On {conf_pct:.0f}% of days, you might lose at most {var_percent:.2%}."
    elif var_percent < 0.05:
        risk_level = "Moderate"
        explanation = f"Your portfolio has moderate risk. On {conf_pct:.0f}% of days, you might lose at most {var_percent:.2%}."
    elif var_percent < 0.10:
        risk_level = "High"
        explanation = f"Your portfolio has high risk. On {conf_pct:.0f}% of days, you might lose at most {var_percent:.2%}."
    else:
        risk_level = "Very High"
        explanation = f"Your portfolio has very high risk. On {conf_pct:.0f}% of days, you might lose at most {var_percent:.2%}."
    
    return {
        'risk_level': risk_level,
        'explanation': explanation,
        'daily_loss': f"₹{var_amount:,.2f}",
        'daily_loss_pct': f"{var_percent:.2%}"
    }


def format_risk_summary(var_dict: Dict, cvar_dict: Dict, portfolio_value: float, confidence: float) -> str:
    """
    Create investor-friendly risk summary.
    
    Parameters
    ----------
    var_dict : Dict
        VaR results
    cvar_dict : Dict
        CVaR results
    portfolio_value : float
        Portfolio value
    confidence : float
        Confidence level
    
    Returns
    -------
    str
        Formatted risk summary
    """
    var_info = interpret_var(
        var_dict.get('var_amount', 0),
        abs(var_dict.get('var_percent', 0)),
        portfolio_value,
        confidence
    )
    
    cvar_info = interpret_var(
        cvar_dict.get('cvar_amount', 0),
        abs(cvar_dict.get('cvar_percent', 0)),
        portfolio_value,
        confidence
    )
    
    conf_pct = confidence * 100
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  PORTFOLIO RISK ASSESSMENT                                     ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  💰 PORTFOLIO VALUE                                             ║
║     ₹{portfolio_value:,.2f}                                              ║
║                                                                ║
║  ⚠️  DAILY RISK (Worst Case - {conf_pct:.0f}% Confidence)                    ║
║     Risk Level: {var_info['risk_level']:^20}                    ║
║     Maximum Daily Loss: {var_info['daily_loss']:^15} ({var_info['daily_loss_pct']}) ║
║     {var_info['explanation']:<58} ║
║                                                                ║
║  📉 AVERAGE LOSS (If Bad Day Occurs)                           ║
║     Average Loss on Bad Days: ₹{cvar_dict.get('cvar_amount', 0):,.2f} ({abs(cvar_dict.get('cvar_percent', 0)):.2%}) ║
║     This is the average loss you'd expect if a bad day happens. ║
║                                                                ║
║  💡 WHAT THIS MEANS                                             ║
║     • On most days ({conf_pct:.0f}% of the time), your losses won't exceed the ║
║       maximum daily loss shown above.                         ║
║     • If a bad day does occur, expect losses around the       ║
║       average loss amount.                                     ║
║     • These are estimates based on historical patterns.       ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    return summary


def format_scenario_result(scenario_name: str, return_val: float, volatility: float, 
                          sharpe: float, base_sharpe: float, portfolio_value: float) -> str:
    """
    Format scenario analysis result in investor-friendly way.
    
    Parameters
    ----------
    scenario_name : str
        Name of scenario
    return_val : float
        Expected return under scenario
    volatility : float
        Volatility under scenario
    sharpe : float
        Sharpe ratio under scenario
    base_sharpe : float
        Base case Sharpe ratio
    portfolio_value : float
        Portfolio value
    
    Returns
    -------
    str
        Formatted scenario result
    """
    vol_info = interpret_volatility(volatility)
    sharpe_info = interpret_sharpe_ratio(sharpe)
    sharpe_change = sharpe - base_sharpe
    
    expected_value = portfolio_value * (1 + return_val)
    value_change = expected_value - portfolio_value
    
    change_direction = "Better" if sharpe_change > 0 else "Worse" if sharpe_change < 0 else "Similar"
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  SCENARIO ANALYSIS: {scenario_name:^40}  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  📈 EXPECTED PERFORMANCE                                      ║
║     Annual Return: {return_val:>6.2%} per year                            ║
║     Portfolio Value After 1 Year: ₹{expected_value:,.2f}              ║
║     Change: ₹{value_change:+,.2f} ({return_val:+.2%})                    ║
║                                                                ║
║  📊 RISK LEVEL                                                 ║
║     {vol_info['risk_level']:^20} ({volatility:.2%} volatility)        ║
║     {vol_info['explanation']:<58} ║
║                                                                ║
║  ⭐ PERFORMANCE QUALITY                                        ║
║     Sharpe Ratio: {sharpe:.2f} ({sharpe_info['quality']:^20})  ║
║     {sharpe_info['explanation']:<58} ║
║     Change from Normal: {sharpe_change:+.2f} ({change_direction:^20}) ║
║                                                                ║
║  💡 WHAT THIS MEANS                                            ║
║     This shows how your portfolio would perform if this       ║
║     market scenario were to occur. Use this to understand     ║
║     potential risks and opportunities.                        ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    return summary


def format_stress_test_result(scenario_name: str, result: Dict, portfolio_value: float) -> str:
    """
    Format historical stress test result in investor-friendly way.
    
    Parameters
    ----------
    scenario_name : str
        Name of historical scenario
    result : Dict
        Stress test result dictionary
    portfolio_value : float
        Portfolio value
    
    Returns
    -------
    str
        Formatted stress test result
    """
    final_value = portfolio_value * (1 + result.get('total_return', 0))
    value_change = final_value - portfolio_value
    
    vol_info = interpret_volatility(result.get('volatility', 0))
    sharpe_info = interpret_sharpe_ratio(result.get('sharpe_ratio', 0))
    
    # Determine if it was a good or bad period
    if result.get('total_return', 0) < -0.20:
        period_type = "Severe Downturn"
        period_desc = "This was a very difficult period for markets."
    elif result.get('total_return', 0) < -0.10:
        period_type = "Market Decline"
        period_desc = "This was a challenging period for markets."
    elif result.get('total_return', 0) < 0:
        period_type = "Mild Decline"
        period_desc = "This was a slightly negative period."
    elif result.get('total_return', 0) < 0.10:
        period_type = "Moderate Growth"
        period_desc = "This was a positive period for markets."
    else:
        period_type = "Strong Growth"
        period_desc = "This was an excellent period for markets."
    
    summary = f"""
╔════════════════════════════════════════════════════════════════╗
║  HISTORICAL STRESS TEST: {scenario_name:^40}  ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  📅 PERIOD TYPE                                                ║
║     {period_type:^20}                                        ║
║     {period_desc:<58} ║
║     Period: {result.get('period_label', 'N/A'):<50} ║
║                                                                ║
║  💰 PORTFOLIO PERFORMANCE                                      ║
║     Starting Value: ₹{portfolio_value:,.2f}                              ║
║     Ending Value:   ₹{final_value:,.2f}                              ║
║     Total Change:   ₹{value_change:+,.2f} ({result.get('total_return', 0):+.2%})                    ║
║     Annual Return:  {result.get('annualized_return', 0):>6.2%} per year                            ║
║                                                                ║
║  📊 RISK METRICS                                              ║
║     Maximum Loss:   {result.get('max_drawdown', 0):>6.2%} (worst decline from peak)              ║
║     Volatility:     {vol_info['risk_level']:^20} ({result.get('volatility', 0):.2%})  ║
║     Performance:    {sharpe_info['quality']:^20} (Sharpe: {result.get('sharpe_ratio', 0):.2f}) ║
║                                                                ║
║  💡 WHAT THIS MEANS                                            ║
║     This shows how your portfolio would have performed        ║
║     during this historical market event. Use this to          ║
║     understand how your portfolio handles difficult periods.  ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
"""
    return summary


def get_scenario_menu() -> Dict[str, Dict[str, Any]]:
    """
    Get menu of available scenarios for user selection.
    
    Returns
    -------
    Dict mapping scenario numbers to scenario info
    """
    from portfolio.scenario_engine import ScenarioEngine
    from portfolio.stress_testing import StressTester
    
    scenarios = {}
    
    # Hypothetical scenarios
    hypothetical = ScenarioEngine.create_standard_scenarios()
    for i, shock in enumerate(hypothetical, 1):
        scenarios[f"H{i}"] = {
            'type': 'hypothetical',
            'name': shock.name,
            'description': f"Hypothetical scenario: {shock.name}",
            'shock': shock
        }
    
    # Historical scenarios
    historical = StressTester.get_historical_scenarios()
    for i, (name, (start, end)) in enumerate(historical.items(), 1):
        scenarios[f"S{i}"] = {
            'type': 'historical',
            'name': name,
            'description': f"Historical period: {name} ({start} to {end})",
            'dates': (start, end)
        }
    
    return scenarios
    def interpret_var(var_percent: float, confidence_level: float) -> str:
    
     bad_days_pct = (1 - confidence_level) * 100
    good_days_pct = confidence_level * 100
    
    output = []
    output.append(f"\n💡 UNDERSTANDING VALUE AT RISK (VaR)")
    output.append(f"\n   Your VaR is {var_percent:.2%} at {confidence_level:.0%} confidence.")
    output.append(f"\n   What does this mean in simple terms?")
    output.append(f"\n   ✅ On {good_days_pct:.0f} out of 100 days:")
    output.append(f"      Your losses will be LESS than {var_percent:.2%}")
    output.append(f"\n   ⚠️  On about {bad_days_pct:.0f} out of 100 days:")
    output.append(f"      You might lose MORE than {var_percent:.2%}")
    output.append(f"\n   💭 Think of it as:")
    output.append(f"      '{confidence_level:.0%} of the time, I won't lose more than {var_percent:.2%}'")
    
    # Risk level assessment
    if var_percent < 0.01:
        output.append(f"\n   📊 Risk Assessment: VERY LOW")
        output.append(f"      Your portfolio is very conservative with minimal daily risk.")
    elif var_percent < 0.02:
        output.append(f"\n   📊 Risk Assessment: LOW")
        output.append(f"      Your portfolio has low daily risk. Suitable for cautious investors.")
    elif var_percent < 0.03:
        output.append(f"\n   📊 Risk Assessment: MODERATE")
        output.append(f"      Your portfolio has moderate daily risk. Balanced approach.")
    elif var_percent < 0.05:
        output.append(f"\n   📊 Risk Assessment: HIGH")
        output.append(f"      Your portfolio has high daily risk. Suitable for aggressive investors.")
    else:
        output.append(f"\n   📊 Risk Assessment: VERY HIGH")
        output.append(f"      Your portfolio has very high daily risk. Only for risk-tolerant investors.")
    
    output.append(f"\n   ⚡ Remember:")
    output.append(f"      • VaR is an estimate, not a guarantee")
    output.append(f"      • Based on historical patterns and statistical models")
    output.append(f"      • Extreme events (black swans) can exceed VaR")
    
    return "\n".join(output)



# ══════════════════════════════════════════════════════════════════════════
# FACTOR MODEL — FactorModel, FactorAnalysisResult
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class FactorAnalysisResult:
    """Stores results of factor analysis."""
    betas: pd.Series            # Factor sensitivities (Beta)
    r_squared: float            # Model R-squared
    alpha: float                # Jensen's Alpha (annualized)
    systematic_risk: float      # Portion of volatility explained by factors
    idiosyncratic_risk: float   # Portion of volatility unexplained
    total_risk: float           # Total volatility
    factor_returns: pd.Series   # Return contribution from factors

class FactorModel:
    """
    Implements factor-based risk models (CAPM, Multi-Factor).
    """
    
    def __init__(self, factor_data: pd.DataFrame):
        """
        Initialize with factor returns data.
        
        Parameters:
        - factor_data: DataFrame where columns are factors (e.g. 'Market', 'SMB', 'HML')
                       and index is datetime. Must be aligned with asset returns.
        """
        self.factors = factor_data
        self.factor_names = factor_data.columns.tolist()
        
    def fit_asset(self, asset_returns: pd.Series) -> FactorAnalysisResult:
        """
        Fit factor model to a single asset's returns.
        """
        # Align data
        df = pd.concat([asset_returns, self.factors], axis=1).dropna()
        
        if len(df) < 30:
            raise ValueError("Insufficient data for factor analysis (need > 30 points)")
            
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        
        # Extract metrics
        alpha = model.params['const'] * 252 # Annualize alpha
        betas = model.params.drop('const')
        r2 = model.rsquared
        
        # Risk Decomposition (Annualized)
        total_risk = y.std() * np.sqrt(252)
        systematic_risk = total_risk * np.sqrt(r2)
        idiosyncratic_risk = total_risk * np.sqrt(1 - r2)
        
        # Factor contributions to return
        # (Average Factor Return * Beta)
        factor_contrib = betas * self.factors.mean() * 252
        
        return FactorAnalysisResult(
            betas=betas,
            r_squared=r2,
            alpha=alpha,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            total_risk=total_risk,
            factor_returns=factor_contrib
        )
    
    def calculate_rolling_betas(self, asset_returns: pd.Series, window: int = 60) -> pd.DataFrame:
        """
        Compute rolling betas for an asset.
        """
        # Align
        df = pd.concat([asset_returns, self.factors], axis=1).dropna()
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
        
        # Rolling regression using RollingOLS
        # wrapper for statsmodels RollingOLS
        try:
            from statsmodels.regression.rolling import RollingOLS
            rolling_model = RollingOLS(y, sm.add_constant(X), window=window)
            results = rolling_model.fit()
            return results.params.drop('const', axis=1)
        except ImportError:
            # Fallback using pure numpy least squares
            betas = []
            for start in range(len(df) - window + 1):
                end = start + window
                y_sub = y.iloc[start:end].values
                X_sub = X.iloc[start:end].values
                coef, _, _, _ = np.linalg.lstsq(X_sub, y_sub, rcond=None)
                betas.append(coef)
            
            return pd.DataFrame(betas, index=df.index[window-1:], columns=X.columns)

    def decompose_portfolio_risk(self, weights: pd.Series, asset_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Decompose TOTAL PORTFOLIO risk into Factor vs Specific.
        
        Formula:
        Sigma_port = w.T * (Sigma_assets) * w
        Sigma_assets approx = B * Sigma_factors * B.T + D
        
        Where D is diagonal matrix of idiosyncratic variances.
        """
        # 1. Fit model for each asset to get Betas and Residual Vars
        betas_list = []
        resid_vars = []
        
        for asset in weights.index:
            if asset not in asset_returns.columns:
                continue
            
            res = self.fit_asset(asset_returns[asset])
            betas_list.append(res.betas)
            # Idiosyncratic variance (daily)
            resid_vars.append((res.idiosyncratic_risk / np.sqrt(252))**2)
            
        B = pd.DataFrame(betas_list, index=weights.index) # N x K
        D = np.diag(resid_vars) # N x N diagonal
        
        # Factor Covariance (Daily)
        Sigma_f = self.factors.cov()
        
        # Portfolio loadings
        # w_betas = w.T * B (1 x K)
        w = weights.values
        port_betas = np.dot(w, B)
        
        # Systemic Variance = beta_p * Sigma_f * beta_p.T
        syst_var = np.dot(np.dot(port_betas, Sigma_f), port_betas.T)
        
        # Idiosyncratic Variance = w.T * D * w
        spec_var = np.dot(np.dot(w, D), w.T)
        
        total_var = syst_var + spec_var
        
        return {
            "systematic_volatility": np.sqrt(syst_var * 252),
            "idiosyncratic_volatility": np.sqrt(spec_var * 252),
            "total_volatility_model": np.sqrt(total_var * 252),
            "r_squared": syst_var / total_var
        }

# ══════════════════════════════════════════════════════════════════════════
# RISK METRICS — compute_parametric_var, compute_historical_var, compute_cvar, compute_component_var, compute_portfolio_risk_metrics, compute_max_drawdown, compute_ulcer_index, detect_market_regime
# ══════════════════════════════════════════════════════════════════════════

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
    - % Contribution_i = Component VaR_i / Portfolio VaR  (sums to 100%)
    
    CRITICAL: Percentage contributions MUST sum to 100% (or very close).
    If they don't, there's a numerical error in the calculation.
    
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
        - % Contribution: Percentage of total portfolio VaR (sums to 100%)
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
    # CRITICAL: Use portfolio VaR (not sum of components) as denominator
    # Portfolio VaR = Z_alpha * port_std
    portfolio_var = z_score * port_std
    
    if portfolio_var > 0:
        # Percentage contribution: each component / total portfolio VaR
        # This MUST sum to 100% by Euler decomposition theorem
        percent_contribution = (component_var / portfolio_var) * 100
    else:
        percent_contribution = np.zeros(len(w))
    
    # Sanity check: contributions should sum to ~100%
    contrib_sum = np.sum(percent_contribution)
    if not np.isclose(contrib_sum, 100.0, atol=1.0):
        # If not close to 100%, normalize to ensure they sum to 100%
        if contrib_sum > 0:
            percent_contribution = (percent_contribution / contrib_sum) * 100
    
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

# ══════════════════════════════════════════════════════════════════════════
# SCENARIO ENGINE — MarketShock, ScenarioEngine
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class MarketShock:
    """
    Defines a market shock scenario.
    
    Attributes
    ----------
    name : str
        Name of the scenario (e.g., "Market Crash", "High Inflation")
    return_shock : float or Dict[str, float]
        Absolute shock to expected returns (e.g., -0.20 for -20%).
        Can be a single float (applied to all) or a dict mapping tickers to shocks.
    volatility_shock : float
        Multiplicative shock to volatility (e.g., 1.5 for 50% increase).
        Applied to standard deviations, which affects covariance matrix.
    correlation_shock : float
        Absolute shock to correlations (e.g., 0.2 to increase correlations).
        Applied to off-diagonal elements. Matrix is then repaired to ensure PSD.
    """
    name: str
    return_shock: Union[float, Dict[str, float]] = 0.0
    volatility_shock: float = 1.0
    correlation_shock: float = 0.0


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """
    Find the nearest positive semi-definite matrix to the input matrix.
    
    Uses the Higham (1988) algorithm to find the nearest correlation matrix
    in the Frobenius norm that is positive semi-definite.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix (potentially non-PSD)
        
    Returns
    -------
    np.ndarray
        Nearest positive semi-definite matrix
        
    Reference
    ---------
    Higham, N. J. (1988). Computing a nearest symmetric positive semidefinite matrix.
    Linear Algebra and its Applications, 103, 103-118.
    """
    # Symmetrize
    symmetric = (matrix + matrix.T) / 2
    
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    
    # Clamp negative eigenvalues to small positive value
    eigvals[eigvals < 0] = 1e-8
    
    # Reconstruct matrix
    result = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Ensure diagonal is exactly 1.0 for correlation matrices
    # (Only if input was a correlation matrix)
    if np.allclose(np.diag(matrix), 1.0):
        # Scale to unit diagonal
        d = np.sqrt(np.diag(result))
        result = result / np.outer(d, d)
        np.fill_diagonal(result, 1.0)
    
    return result


class ScenarioEngine:
    """
    Engine to apply market shocks to portfolio parameters.
    
    Ensures mathematical consistency by repairing covariance matrices
    to be positive semi-definite after applying shocks.
    """
    
    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ):
        """
        Initialize the Scenario Engine.
        
        Parameters
        ----------
        expected_returns : pd.Series
            Base expected annual returns for each asset
        covariance_matrix : pd.DataFrame
            Base annualized covariance matrix (must be PSD)
        """
        self.base_expected_returns = expected_returns
        self.base_covariance_matrix = covariance_matrix
        self.tickers = expected_returns.index.tolist()
        
        # Validate input covariance is PSD
        if not self._is_positive_semidefinite(covariance_matrix.values):
            raise ValueError("Input covariance matrix must be positive semi-definite")
        
    @staticmethod
    def _is_positive_semidefinite(matrix: np.ndarray, tol: float = 1e-8) -> bool:
        """Check if matrix is positive semi-definite."""
        eigvals = np.linalg.eigvalsh(matrix)
        return np.all(eigvals > -tol)
        
    def apply_scenario(self, shock: MarketShock) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Apply a defined market shock to the base parameters.
        
        The method ensures the resulting covariance matrix is positive semi-definite
        by using the nearest PSD matrix if needed.
        
        Parameters
        ----------
        shock : MarketShock
            Market shock scenario to apply
        
        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            (shocked_expected_returns, shocked_covariance_matrix)
            
        Examples
        --------
        >>> engine = ScenarioEngine(returns, cov)
        >>> crash = MarketShock("Crash", return_shock=-0.30, volatility_shock=2.0)
        >>> new_returns, new_cov = engine.apply_scenario(crash)
        """
        # 1. Apply Return Shock
        shocked_returns = self.base_expected_returns.copy()
        
        if isinstance(shock.return_shock, dict):
            # Asset-specific shocks
            for ticker, shock_val in shock.return_shock.items():
                if ticker in shocked_returns.index:
                    shocked_returns[ticker] += shock_val
        else:
            # Uniform shock to all assets
            shocked_returns += shock.return_shock
            
        # 2. Apply Volatility and Correlation Shocks
        # Decompose Covariance to Correlation and Standard Deviations
        cov_values = self.base_covariance_matrix.values
        std_devs = np.sqrt(np.diag(cov_values))
        
        # Compute correlation matrix
        outer_vols = np.outer(std_devs, std_devs)
        correlation_matrix = cov_values / outer_vols
        
        # Apply Correlation Shock
        if shock.correlation_shock != 0.0:
            # Shift off-diagonal elements
            n = correlation_matrix.shape[0]
            mask = ~np.eye(n, dtype=bool)
            correlation_matrix[mask] += shock.correlation_shock
            
            # Clip to valid correlation range
            correlation_matrix = np.clip(correlation_matrix, -0.99, 0.99)
            
            # Ensure diagonal is exactly 1.0
            np.fill_diagonal(correlation_matrix, 1.0)
            
            # Repair matrix to ensure it's positive semi-definite
            if not self._is_positive_semidefinite(correlation_matrix):
                correlation_matrix = nearest_positive_definite(correlation_matrix)
        
        # Apply Volatility Shock (multiplicative to standard deviations)
        new_std_devs = std_devs * shock.volatility_shock
        
        # Reconstruct Covariance Matrix
        new_outer_vols = np.outer(new_std_devs, new_std_devs)
        shocked_cov_values = correlation_matrix * new_outer_vols
        
        # Final check and repair if needed
        if not self._is_positive_semidefinite(shocked_cov_values):
            shocked_cov_values = nearest_positive_definite(shocked_cov_values)
        
        shocked_covariance = pd.DataFrame(
            shocked_cov_values,
            index=self.base_covariance_matrix.index,
            columns=self.base_covariance_matrix.columns
        )
        
        return shocked_returns, shocked_covariance

    @staticmethod
    def create_standard_scenarios() -> List[MarketShock]:
        """
        Create a list of standard stress scenarios.
        
        These scenarios are based on historical market events and
        common stress testing practices.
        
        Returns
        -------
        List[MarketShock]
            List of predefined market shock scenarios
        """
        return [
            MarketShock(
                name="2008-Style Financial Crisis",
                return_shock=-0.40,  # 40% return drop
                volatility_shock=2.0,  # Volatility doubles
                correlation_shock=0.3  # Correlations increase (flight to quality)
            ),
            MarketShock(
                name="Rate Hike Shock",
                return_shock=-0.15,  # General market drag
                volatility_shock=1.2,  # 20% vol increase
                correlation_shock=0.1  # Correlations tighten slightly
            ),
            MarketShock(
                name="Volatility Regime Shift",
                return_shock=0.0,    # No directional bias assumed
                volatility_shock=1.5, # 50% Higher volatility regime
                correlation_shock=0.0 # Correlations unchanged
            ),
            MarketShock(
                name="Tech Sector Crash",
                return_shock=-0.25,  # 25% return drop
                volatility_shock=1.5,  # 50% volatility increase
                correlation_shock=0.15  # Moderate correlation increase
            ),
            MarketShock(
                name="Inflation Spike",
                return_shock=-0.15,  # 15% return drop
                volatility_shock=1.3,  # 30% volatility increase
                correlation_shock=0.1  # Slight correlation increase
            ),
            MarketShock(
                name="Mild Recession",
                return_shock=-0.10,  # 10% return drop
                volatility_shock=1.2,  # 20% volatility increase
                correlation_shock=0.05  # Small correlation increase
            ),
            MarketShock(
                name="Market Boom",
                return_shock=0.20,  # 20% return increase
                volatility_shock=0.8,  # 20% volatility decrease
                correlation_shock=-0.1  # Correlations decrease (diversification returns)
            )
        ]
    
    def apply_multiple_scenarios(
        self,
        scenarios: Optional[List[MarketShock]] = None
    ) -> Dict[str, Tuple[pd.Series, pd.DataFrame]]:
        """
        Apply multiple scenarios and return all results.
        
        Parameters
        ----------
        scenarios : List[MarketShock], optional
            List of scenarios to apply. If None, uses standard scenarios.
            
        Returns
        -------
        Dict[str, Tuple[pd.Series, pd.DataFrame]]
            Dictionary mapping scenario names to (returns, covariance) tuples
        """
        if scenarios is None:
            scenarios = self.create_standard_scenarios()
            
        results = {}
        for scenario in scenarios:
            results[scenario.name] = self.apply_scenario(scenario)
            
        return results

# ══════════════════════════════════════════════════════════════════════════════
# ALIASES & SECTOR DATA — used by run_milestone3, run_milestone4, run_milestone5
# ══════════════════════════════════════════════════════════════════════════════

def load_price_data(
    tickers,
    period: str = "1y",
    start_date=None,
    end_date=None,
    use_cache: bool = True,
) -> "pd.DataFrame":
    """
    Alias for get_stock_data() — satisfies the import used in run_milestone3,
    run_milestone4, and run_milestone5.

    Returns close prices DataFrame (tickers as columns, dates as index).
    """
    return get_stock_data(
        tickers,
        period=period,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_cache,
    )


# NSE sector mapping — used by run_milestone5 via get_sector()
NSE_SECTOR_MAP = {
    "RELIANCE.NS":   "Energy",
    "ONGC.NS":       "Energy",
    "POWERGRID.NS":  "Energy",
    "NTPC.NS":       "Energy",
    "TCS.NS":        "Technology",
    "INFY.NS":       "Technology",
    "WIPRO.NS":      "Technology",
    "HCLTECH.NS":    "Technology",
    "TECHM.NS":      "Technology",
    "LTIM.NS":       "Technology",
    "HDFCBANK.NS":   "Financials",
    "ICICIBANK.NS":  "Financials",
    "AXISBANK.NS":   "Financials",
    "KOTAKBANK.NS":  "Financials",
    "SBIN.NS":       "Financials",
    "BAJFINANCE.NS": "Financials",
    "BAJAJFINSV.NS": "Financials",
    "INDUSINDBK.NS": "Financials",
    "HINDUNILVR.NS": "Consumer Staples",
    "ITC.NS":        "Consumer Staples",
    "NESTLEIND.NS":  "Consumer Staples",
    "BRITANNIA.NS":  "Consumer Staples",
    "MARUTI.NS":     "Consumer Discretionary",
    "TATAMOTORS.NS": "Consumer Discretionary",
    "M&M.NS":        "Consumer Discretionary",
    "ASIANPAINT.NS": "Consumer Discretionary",
    "TITAN.NS":      "Consumer Discretionary",
    "BAJAJ-AUTO.NS": "Consumer Discretionary",
    "EICHERMOT.NS":  "Consumer Discretionary",
    "HEROMOTOCO.NS": "Consumer Discretionary",
    "TATASTEEL.NS":  "Materials",
    "JSWSTEEL.NS":   "Materials",
    "HINDALCO.NS":   "Materials",
    "ULTRACEMCO.NS": "Materials",
    "GRASIM.NS":     "Materials",
    "SUNPHARMA.NS":  "Healthcare",
    "DRREDDY.NS":    "Healthcare",
    "CIPLA.NS":      "Healthcare",
    "DIVISLAB.NS":   "Healthcare",
    "APOLLOHOSP.NS": "Healthcare",
    "LT.NS":         "Industrials",
    "ADANIPORTS.NS": "Industrials",
    "BHARTIARTL.NS": "Communication",
}


def get_sector(ticker: str) -> str:
    """Return the NSE sector for a ticker, or 'Other' if not mapped."""
    return NSE_SECTOR_MAP.get(ticker, "Other")
