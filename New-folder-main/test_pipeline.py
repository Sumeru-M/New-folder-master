"""
Test Pipeline Script (user-provided tickers)

This script tests the data loading and portfolio engine pipeline
for Indian market stocks (NSE) using user input tickers.

Usage examples:
    python test_pipeline.py --tickers RELIANCE.NS,TCS.NS,INFY.NS --period 1y
    python test_pipeline.py --tickers RELIANCE.NS,TCS.NS --start_date 2023-01-01 --end_date 2023-12-31
"""

from src.data_loader import fetch_market_data, get_close_prices
from src.portfolio_engine import compute_portfolio_metrics
import pandas as pd
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline test with user-provided tickers.")
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated tickers (e.g., RELIANCE.NS,TCS.NS,INFY.NS)"
    )
    parser.add_argument(
        "--period",
        type=str,
        default="1y",
        help="Data period if start/end not provided (e.g., 6mo,1y,2y)"
    )
    parser.add_argument("--start_date", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end_date", type=str, default=None, help="End date YYYY-MM-DD")
    return parser.parse_args()


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
            print("No input provided.")
            return []
        
        # Split by comma and clean up
        tickers = [ticker.strip().upper() for ticker in user_input.split(',')]
        
        # Remove empty strings
        tickers = [t for t in tickers if t]
        
        if not tickers:
            print("No valid tickers found.")
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
        print("\n\nInput cancelled. Exiting.")
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
            choice = input("Enter choice (1-8) or press Enter for default (1y): ").strip()
            
            if not choice:
                return '1y'  # Default
            
            if choice in period_map:
                return period_map[choice]
            else:
                print("Invalid choice. Please enter a number between 1-8.")
                continue
                
        except KeyboardInterrupt:
            print("\nUsing default period: 1y")
            return '1y'
        except Exception as e:
            print(f"Error: {e}. Using default period: 1y")
            return '1y'


def main():
    """Main test function."""
    args = parse_args()
    
    # Get tickers from arguments or user input
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        # Ensure .NS suffix
        tickers = [t if t.endswith('.NS') else t + '.NS' for t in tickers]
    else:
        tickers = get_user_tickers()
    
    if not tickers:
        print("[ERROR] No tickers provided.")
        return
    
    period = args.period
    start_date = args.start_date
    end_date = args.end_date
    
    # If no CLI args, prompt for period
    if not args.tickers:
        period = get_period_input()
        print()

    print("=" * 80)
    print("Portfolio Data Pipeline Test - Indian Markets (NSE)")
    print("=" * 80)
    print()
    print(f"Test Tickers: {', '.join(tickers)}")
    if start_date and end_date:
        print(f"Date range: {start_date} to {end_date}")
    else:
        print(f"Period: {period}")
    print()
    
    # Step 1: Fetch market data
    print("-" * 80)
    print("Step 1: Fetching Market Data")
    print("-" * 80)
    
    try:
        print(f"Fetching data for {len(tickers)} stock(s)...")
        print(f"  Using yfinance API to download from Yahoo Finance...")
        
        # Fetch data
        ohlcv_data = fetch_market_data(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            use_cache=True
        )
        
        print("[OK] Data fetched successfully")
        print(f"  Shape: {ohlcv_data.shape}")
        print(f"  Date range: {ohlcv_data.index[0].date()} to {ohlcv_data.index[-1].date()}")
        print(f"  Number of trading days: {len(ohlcv_data)}")
        print()
        
        # Extract close prices
        prices = get_close_prices(ohlcv_data)
        print("[OK] Close prices extracted")
        print(f"  Price data shape: {prices.shape}")
        print(f"  Columns: {list(prices.columns)}")
        print()
        
    except Exception as e:
        print(f"[ERROR] Error fetching data: {e}")
        print()
        print("Common issues:")
        print("  1. Ticker symbol might be incorrect")
        print("  2. Stock might not be listed on NSE")
        print("  3. Make sure to use .NS suffix for NSE stocks")
        print("  4. Check if the ticker exists on Yahoo Finance")
        print()
        return
    
    # Step 2: Compute portfolio metrics
    print("-" * 80)
    print("Step 2: Computing Portfolio Metrics")
    print("-" * 80)
    
    try:
        metrics = compute_portfolio_metrics(prices, risk_free_rate=0.0)
        print("[OK] Portfolio metrics computed successfully")
        print()
    except Exception as e:
        print(f"[ERROR] Error computing metrics: {e}")
        import traceback
        traceback.print_exc()
        return
    
       # Step 3: Display results
    print("-" * 80)
    print("Step 3: Results Summary")
    print("-" * 80)
    print()
    
    # Annualized Returns
    print("Annualized Returns:")
    print("-" * 40)
    if isinstance(metrics['annualized_returns'], dict):
        for ticker, ret in metrics['annualized_returns'].items():
            print(f"  {ticker:20s}: {ret:8.2%}")
    else:
        for ticker in prices.columns:
            ret = metrics['annualized_returns'][ticker] if ticker in metrics['annualized_returns'] else metrics['annualized_returns'].get(ticker, None)
            if ret is not None and pd.notna(ret):
                print(f"  {ticker:20s}: {ret:8.2%}")
    print()
    
    # Annualized Volatility
    print("Annualized Volatility:")
    print("-" * 40)
    if isinstance(metrics['annualized_volatility'], dict):
        for ticker, vol in metrics['annualized_volatility'].items():
            print(f"  {ticker:20s}: {vol:8.2%}")
    else:
        for ticker in prices.columns:
            vol = metrics['annualized_volatility'][ticker] if ticker in metrics['annualized_volatility'] else metrics['annualized_volatility'].get(ticker, None)
            if vol is not None and pd.notna(vol):
                print(f"  {ticker:20s}: {vol:8.2%}")
    print()
    
    # Sharpe Ratio
    print("Sharpe Ratio (risk-free rate = 0%):")
    print("-" * 40)
    if isinstance(metrics['sharpe_ratio'], dict):
        for ticker, sharpe in metrics['sharpe_ratio'].items():
            if pd.notna(sharpe):
                print(f"  {ticker:20s}: {sharpe:8.2f}")
            else:
                print(f"  {ticker:20s}: N/A")
    else:
        for ticker in prices.columns:
            sharpe = metrics['sharpe_ratio'][ticker] if ticker in metrics['sharpe_ratio'] else metrics['sharpe_ratio'].get(ticker, None)
            if sharpe is not None and pd.notna(sharpe):
                print(f"  {ticker:20s}: {sharpe:8.2f}")
            else:
                print(f"  {ticker:20s}: N/A")
    print()
    
    # Maximum Drawdown
    print("Maximum Drawdown:")
    print("-" * 40)
    if isinstance(metrics['max_drawdown'], dict):
        for ticker, mdd in metrics['max_drawdown'].items():
            if pd.notna(mdd):
                print(f"  {ticker:20s}: {mdd:8.2%}")
            else:
                print(f"  {ticker:20s}: N/A")
    else:
        for ticker in prices.columns:
            mdd = metrics['max_drawdown'][ticker] if ticker in metrics['max_drawdown'] else metrics['max_drawdown'].get(ticker, None)
            if mdd is not None and pd.notna(mdd):
                print(f"  {ticker:20s}: {mdd:8.2%}")
            else:
                print(f"  {ticker:20s}: N/A")
    print()
    
    # Correlation Matrix
    print("Correlation Matrix:")
    print("-" * 40)
    if 'correlation_matrix' in metrics and metrics['correlation_matrix'] is not None:
        corr_matrix = metrics['correlation_matrix']
        if isinstance(corr_matrix, pd.DataFrame):
            print(corr_matrix.round(3))
        else:
            print("Correlation matrix not available for single ticker")
    else:
        print("Correlation matrix not available")
    print()
    
    # Covariance Matrix (diagonal elements)
    print("Covariance Matrix (Diagonal - Variances):")
    print("-" * 40)
    if 'covariance_matrix' in metrics and metrics['covariance_matrix'] is not None:
        cov_matrix = metrics['covariance_matrix']
        if isinstance(cov_matrix, pd.DataFrame):
            for ticker in cov_matrix.index:
                variance = cov_matrix.loc[ticker, ticker]
                print(f"  {ticker:20s}: {variance:12.6f}")
        else:
            print("Covariance matrix not available for single ticker")
    else:
        print("Covariance matrix not available")
    print()
    
    # Rolling Volatility (last 5 values)
    print("Rolling 30-Day Volatility (Last 5 Days):")
    print("-" * 40)
    if 'rolling_volatility_30d' in metrics and metrics['rolling_volatility_30d'] is not None:
        rolling_vol = metrics['rolling_volatility_30d']
        if len(rolling_vol) > 0:
            last_5 = rolling_vol.tail(5)
            print(last_5.round(4).to_string())
        else:
            print("  Insufficient data for rolling volatility")
    else:
        print("  Rolling volatility not available")
    print()
    
    # Data Statistics
    print("Price Data Statistics:")
    print("-" * 40)
    print(f"  Total trading days: {len(prices)}")
    print(f"  Assets: {len(prices.columns)}")
    print(f"  Missing values: {prices.isna().sum().sum()}")
    print()
    
    print("=" * 80)
    print("Pipeline test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()