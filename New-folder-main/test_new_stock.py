"""
Test Script for Fetching Data for a New Stock

This script allows you to test if you can fetch data for any Indian stock (NSE).

Usage:
    python test_new_stock.py
"""

from src.data_loader import get_stock_data
import pandas as pd
import numpy as np


def test_stock_data(ticker: str, period: str = "1y"):
    """
    Test fetching data for a single stock.
    
    Parameters
    ----------
    ticker : str
        Stock ticker with .NS suffix (e.g., "RELIANCE.NS")
    period : str, default="1y"
        Period of data to fetch
    """
    print("=" * 80)
    print(f"Testing Data Fetch for: {ticker}")
    print("=" * 80)
    print()
    
    try:
        # Fetch data using yfinance API (via get_stock_data which uses fetch_market_data)
        print(f"Fetching data for {ticker} (period: {period})...")
        print(f"  Using yfinance API to download from Yahoo Finance...")
        prices = get_stock_data(ticker, period=period, use_cache=True)
        
        print(f"[OK] Data fetched successfully!")
        print(f"  Shape: {prices.shape}")
        print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"  Number of trading days: {len(prices)}")
        print()
        
        # Show first few rows
        print("First 5 rows:")
        print(prices.head().to_string())
        print()
        
        # Show last few rows
        print("Last 5 rows:")
        print(prices.tail().to_string())
        print()
        
        # Show statistics
        print("Price Statistics:")
        stats = prices.describe().round(2)
        print(stats)
        print()

        # Compute Sharpe ratio from historical data (via data_cache through get_stock_data)
        # 1. Compute daily percentage returns
        daily_returns = prices.pct_change().dropna()

        # 2. Compute annualized return and volatility
        mean_daily_return = daily_returns.iloc[:, 0].mean()
        std_daily_return = daily_returns.iloc[:, 0].std()

        annualized_return = mean_daily_return * 252
        annualized_volatility = std_daily_return * np.sqrt(252)

        # 3. Assume risk-free rate (can adjust if needed)
        risk_free_rate = 0.0  # 0% risk-free rate for testing

        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        else:
            sharpe_ratio = np.nan

        print("Return & Risk Metrics (derived from historical data):")
        print(f"  Annualized Return:   {annualized_return:8.2%}")
        print(f"  Annualized Volatility: {annualized_volatility:8.2%}")
        if pd.notna(sharpe_ratio):
            print(f"  Sharpe Ratio:        {sharpe_ratio:8.3f}")
        else:
            print("  Sharpe Ratio:        N/A (insufficient data)")
        print()
        
        # Show latest price
        latest_price = prices.iloc[-1, 0]
        print(f"Latest Price: Rs. {latest_price:,.2f}")
        print()
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to fetch data for {ticker}")
        print(f"  Error: {e}")
        print()
        print("Common issues:")
        print("  1. Ticker symbol might be incorrect")
        print("  2. Stock might not be listed on NSE")
        print("  3. Make sure to use .NS suffix for NSE stocks")
        print("  4. Check if the ticker exists on Yahoo Finance")
        print()
        return False


def get_user_tickers():
    """
    Get stock tickers from user input.
    
    Returns
    -------
    List[str]
        List of ticker symbols entered by user
    """
    print("=" * 80)
    print("Stock Ticker Input")
    print("=" * 80)
    print()
    print("Enter stock ticker symbols (NSE-listed stocks with .NS suffix)")
    print("You can enter:")
    print("  - Single ticker: RELIANCE.NS")
    print("  - Multiple tickers (comma-separated): RELIANCE.NS, TCS.NS, INFY.NS")
    print()
    print("Examples:")
    print("  RELIANCE.NS")
    print("  TCS.NS, INFY.NS, HDFCBANK.NS")
    print()
    
    try:
        user_input = input("Enter ticker(s): ").strip()
        
        if not user_input:
            print("No input provided. Exiting.")
            return []
        
        # Split by comma and clean up
        tickers = [ticker.strip().upper() for ticker in user_input.split(',')]
        
        # Remove empty strings
        tickers = [t for t in tickers if t]
        
        if not tickers:
            print("No valid tickers found. Exiting.")
            return []
        
        # Ensure .NS suffix for all tickers
        processed_tickers = []
        for ticker in tickers:
            if not ticker.endswith('.NS'):
                ticker = ticker + '.NS'
            processed_tickers.append(ticker)
        
        print()
        print(f"Processing {len(processed_tickers)} ticker(s):")
        for ticker in processed_tickers:
            print(f"  - {ticker}")
        print()
        
        return processed_tickers
        
    except (KeyboardInterrupt, EOFError):
        print("\n\nInput cancelled or no input available. Exiting.")
        return []
    except Exception as e:
        print(f"Error reading input: {e}")
        return []


def get_period_input():
    """
    Get data period from user input.
    
    Returns
    -------
    str
        Period string (e.g., '1y', '6mo', '2y')
    """
    print("Select data period:")
    print("  1. 1 day (1d)")
    print("  2. 5 days (5d)")
    print("  3. 1 month (1mo)")
    print("  4. 3 months (3mo)")
    print("  5. 6 months (6mo) - Default")
    print("  6. 1 year (1y)")
    print("  7. 2 years (2y)")
    print("  8. 5 years (5y)")
    print()
    
    period_map = {
        '1': '1d',
        '2': '5d',
        '3': '1mo',
        '4': '3mo',
        '5': '6mo',
        '6': '1y',
        '7': '2y',
        '8': '5y'
    }
    
    while True:
        try:
            choice = input("Enter choice (1-8) or press Enter for default (6mo): ").strip()
            
            if not choice:
                return '6mo'  # Default
            
            if choice in period_map:
                return period_map[choice]
            else:
                print("Invalid choice. Please enter a number between 1-8.")
                continue
                
        except KeyboardInterrupt:
            print("\nUsing default period: 6mo")
            return '6mo'
        except Exception as e:
            print(f"Error: {e}. Using default period: 6mo")
            return '6mo'


def main():
    """Main function."""
    print("Stock Data Fetching Test")
    print("=" * 80)
    print()
    print("This script fetches stock data using yfinance API for Indian stocks (NSE).")
    print("Data is fetched from Yahoo Finance and cached locally for faster access.")
    print()
    
    # Get tickers from user
    test_tickers = get_user_tickers()
    
    if not test_tickers:
        print("No tickers provided. Exiting.")
        return
    
    # Get period from user
    period = get_period_input()
    print()
    
    print("=" * 80)
    print(f"Fetching data for {len(test_tickers)} stock(s) using yfinance API")
    print(f"Period: {period}")
    print("=" * 80)
    print()
    
    success_count = 0
    
    for ticker in test_tickers:
        if test_stock_data(ticker, period=period):
            success_count += 1
        print()
    
    print("=" * 80)
    print(f"Results: {success_count}/{len(test_tickers)} stocks fetched successfully")
    print("=" * 80)
    print()
    
    # Option to test more stocks
    try:
        more = input("\nWould you like to test more stocks? (y/n): ").strip().lower()
        if more in ['y', 'yes']:
            print()
            new_tickers = get_user_tickers()
            if new_tickers:
                period = get_period_input()
                print()
                for ticker in new_tickers:
                    test_stock_data(ticker, period=period)
                    print()
    except (KeyboardInterrupt, EOFError):
        print("\n\nExiting.")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
