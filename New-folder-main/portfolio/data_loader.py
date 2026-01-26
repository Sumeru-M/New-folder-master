"""
Portfolio Data Loader Module

This module loads preprocessed price data for portfolio optimization.
It wraps the existing data_loader from src/ to provide a clean interface.
"""

from typing import List, Optional
import pandas as pd
import sys
import os

# Add src to path to import existing data loader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import get_stock_data, fetch_market_data, get_close_prices


def load_price_data(
    tickers: List[str],
    period: str = "1y",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Load preprocessed price data for portfolio optimization.
    
    This function wraps the existing get_stock_data() function to provide
    a clean interface for portfolio optimization workflows.
    
    Parameters
    ----------
    tickers : List[str]
        List of ticker symbols with .NS suffix for NSE (e.g., ["RELIANCE.NS", "TCS.NS"])
    period : str, default="1y"
        Period of data to fetch. Valid values: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'
    start_date : Optional[str], default=None
        Start date in 'YYYY-MM-DD' format. If provided, overrides period.
    end_date : Optional[str], default=None
        End date in 'YYYY-MM-DD' format. If None, uses today's date.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with close prices. Columns are tickers, index is datetime.
        Example:
                    RELIANCE.NS  TCS.NS  INFY.NS
        2023-01-01     2450.50  3456.75  1523.25
        ...
    
    Examples
    --------
    >>> from portfolio.data_loader import load_price_data
    >>> 
    >>> tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
    >>> prices = load_price_data(tickers, period="2y")
    """
    return get_stock_data(
        tickers=tickers,
        period=period,
        start_date=start_date,
        end_date=end_date,
        use_cache=True
    )
