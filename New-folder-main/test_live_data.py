"""
Test Live Data Fetching Script

This script demonstrates how to fetch live/real-time market data
for Indian stocks using the data_loader module.

Usage:
    python test_live_data.py
"""

from src.data_loader import (
    fetch_live_quotes,
    fetch_intraday_data,
    fetch_live_prices,
    get_close_prices
)
import pandas as pd
from datetime import datetime


def main():
    """Main test function for live data."""
    print("=" * 80)
    print("Live Data Fetching Test - Indian Markets (NSE)")
    print("=" * 80)
    print()
    
    # Define test tickers
    tickers = [
        "RELIANCE.NS",
        "TCS.NS",
        "INFY.NS",
        "HDFCBANK.NS",
        "ICICIBANK.NS"
    ]
    
    print(f"Test Tickers: {', '.join(tickers)}")
    print()
    
    # Test 1: Fetch Live Quotes
    print("-" * 80)
    print("Test 1: Fetching Live Quotes (Current Market Data)")
    print("-" * 80)
    print()
    
    try:
        quotes = fetch_live_quotes(tickers)
        print("Live Quotes:")
        print(quotes.to_string())
        print()
    except Exception as e:
        print(f"✗ Error fetching live quotes: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test 2: Fetch Current Prices (Simplified)
    print("-" * 80)
    print("Test 2: Fetching Current Prices (Simplified)")
    print("-" * 80)
    print()
    
    try:
        current_prices = fetch_live_prices(tickers)
        print("Current Prices:")
        for ticker, price in current_prices.items():
            if price is not None:
                print(f"  {ticker:20s}: ₹{price:,.2f}")
            else:
                print(f"  {ticker:20s}: N/A")
        print()
    except Exception as e:
        print(f"✗ Error fetching current prices: {e}")
        print()
    
    # Test 3: Fetch Intraday Data
    print("-" * 80)
    print("Test 3: Fetching Intraday Data (5-minute intervals, last 1 day)")
    print("-" * 80)
    print()
    
    try:
        intraday = fetch_intraday_data(tickers, interval="5m", period="1d")
        
        if len(intraday) > 0:
            print(f"✓ Intraday data fetched: {intraday.shape[0]} data points")
            print(f"  Date range: {intraday.index[0]} to {intraday.index[-1]}")
            print()
            
            # Extract close prices
            intraday_prices = get_close_prices(intraday)
            print("Latest Intraday Prices (Last 5 data points):")
            print(intraday_prices.tail().to_string())
            print()
        else:
            print("⚠ No intraday data available (market may be closed)")
            print()
    except Exception as e:
        print(f"✗ Error fetching intraday data: {e}")
        import traceback
        traceback.print_exc()
        print()
    
    # Test 4: Fetch 1-minute data (if available)
    print("-" * 80)
    print("Test 4: Fetching 1-Minute Intraday Data (Last 1 day)")
    print("Note: 1-minute data is only available for last 7 days")
    print("-" * 80)
    print()
    
    try:
        # Try with just one ticker for 1-minute data (more reliable)
        intraday_1m = fetch_intraday_data([tickers[0]], interval="1m", period="1d")
        
        if len(intraday_1m) > 0:
            print(f"✓ 1-minute data fetched: {intraday_1m.shape[0]} data points")
            print(f"  Date range: {intraday_1m.index[0]} to {intraday_1m.index[-1]}")
            
            # Extract close prices
            prices_1m = get_close_prices(intraday_1m)
            print(f"\nLatest 1-minute prices for {tickers[0]} (Last 10 data points):")
            print(prices_1m.tail(10).to_string())
            print()
        else:
            print("⚠ No 1-minute data available")
            print()
    except Exception as e:
        print(f"✗ Error fetching 1-minute data: {e}")
        print("  (This is normal if market is closed or data is unavailable)")
        print()
    
    print("=" * 80)
    print("Live Data Test Completed!")
    print("=" * 80)
    print()
    print("Note: yfinance provides delayed data (typically 15-20 minutes delay).")
    print("For true real-time data, consider using paid API services like:")
    print("  - Alpha Vantage")
    print("  - IEX Cloud")
    print("  - Polygon.io")
    print("  - Indian market-specific APIs (NSE, BSE official APIs)")


if __name__ == "__main__":
    main()
