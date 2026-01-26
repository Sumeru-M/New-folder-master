# Live Data API Guide

This guide explains how to fetch live/real-time market data for Indian stocks using the data_loader module.

## Overview

The `data_loader.py` module now supports three types of live data fetching:

1. **Live Quotes** - Current market prices, bid/ask, volume, etc.
2. **Intraday Data** - Minute-level price data (1m, 5m, 15m, etc.)
3. **Current Prices** - Simple function to get just current prices

## Important Note About Data Delay

⚠️ **yfinance provides delayed data** (typically 15-20 minutes delay for free tier). This is normal for free market data APIs.

For **true real-time data**, you would need:
- Paid API services (Alpha Vantage, IEX Cloud, Polygon.io)
- Indian market-specific APIs (NSE/BSE official APIs)
- Broker APIs (Zerodha Kite, Upstox, etc.)

## Functions Available

### 1. `fetch_live_quotes(tickers)`

Fetches comprehensive live market quotes including current price, bid/ask, volume, etc.

```python
from src.data_loader import fetch_live_quotes

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
quotes = fetch_live_quotes(tickers)

print(quotes)
# Returns DataFrame with columns:
# - currentPrice: Current/last traded price
# - previousClose: Previous day's close
# - open: Day's open price
# - dayHigh: Day's high
# - dayLow: Day's low
# - volume: Current day volume
# - bid: Bid price
# - ask: Ask price
# - marketCap: Market capitalization
```

### 2. `fetch_intraday_data(tickers, interval, period)`

Fetches intraday price data with minute-level granularity.

```python
from src.data_loader import fetch_intraday_data, get_close_prices

# Fetch 5-minute data for last 1 day
intraday = fetch_intraday_data(
    tickers=["RELIANCE.NS", "TCS.NS"],
    interval="5m",  # Options: '1m', '2m', '5m', '15m', '30m', '60m', '1h'
    period="1d"    # Options: '1d', '5d', '1mo', etc.
)

# Extract close prices
prices = get_close_prices(intraday)
print(prices.tail())  # Last few data points
```

**Available Intervals:**
- `'1m'` - 1 minute (max 7 days of data)
- `'2m'` - 2 minutes
- `'5m'` - 5 minutes
- `'15m'` - 15 minutes
- `'30m'` - 30 minutes
- `'60m'` or `'1h'` - 1 hour
- `'1d'` - Daily

### 3. `fetch_live_prices(tickers)`

Simple function to get just current prices.

```python
from src.data_loader import fetch_live_prices

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
prices = fetch_live_prices(tickers)

print(prices)
# Returns Series:
# RELIANCE.NS    2450.50
# TCS.NS         3456.75
# INFY.NS        1523.25
```

## Usage Examples

### Example 1: Monitor Current Prices

```python
from src.data_loader import fetch_live_prices

# Your portfolio tickers
portfolio = ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

# Get current prices
current_prices = fetch_live_prices(portfolio)

print("Current Portfolio Prices:")
for ticker, price in current_prices.items():
    print(f"{ticker}: ₹{price:,.2f}")
```

### Example 2: Get Detailed Live Quotes

```python
from src.data_loader import fetch_live_quotes

tickers = ["RELIANCE.NS", "TCS.NS"]
quotes = fetch_live_quotes(tickers)

# Access specific data
for ticker in tickers:
    if ticker in quotes.index:
        quote = quotes.loc[ticker]
        print(f"\n{ticker}:")
        print(f"  Current Price: ₹{quote.get('currentPrice', 'N/A'):,.2f}")
        print(f"  Day High: ₹{quote.get('dayHigh', 'N/A'):,.2f}")
        print(f"  Day Low: ₹{quote.get('dayLow', 'N/A'):,.2f}")
        print(f"  Volume: {quote.get('volume', 'N/A'):,}")
```

### Example 3: Real-time Monitoring Loop

```python
import time
from src.data_loader import fetch_live_prices

tickers = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]

print("Starting real-time price monitoring (Press Ctrl+C to stop)...")
try:
    while True:
        prices = fetch_live_prices(tickers)
        print(f"\n[{time.strftime('%H:%M:%S')}] Current Prices:")
        for ticker, price in prices.items():
            if price:
                print(f"  {ticker}: ₹{price:,.2f}")
        
        # Update every 30 seconds
        time.sleep(30)
except KeyboardInterrupt:
    print("\nMonitoring stopped.")
```

### Example 4: Intraday Analysis

```python
from src.data_loader import fetch_intraday_data, get_close_prices
from src.portfolio_engine import compute_portfolio_metrics

# Fetch today's 5-minute data
intraday = fetch_intraday_data(
    tickers=["RELIANCE.NS", "TCS.NS"],
    interval="5m",
    period="1d"
)

# Extract close prices
prices = get_close_prices(intraday)

# Compute metrics (if enough data)
if len(prices) > 30:  # Need at least 30 data points
    metrics = compute_portfolio_metrics(prices)
    print("Intraday Volatility:")
    print(metrics['annualized_volatility'])
```

## Integration with Portfolio Engine

You can use live data with the portfolio engine:

```python
from src.data_loader import fetch_intraday_data, get_close_prices
from src.portfolio_engine import compute_portfolio_metrics

# Fetch intraday data
intraday = fetch_intraday_data(
    tickers=["RELIANCE.NS", "TCS.NS", "INFY.NS"],
    interval="15m",
    period="5d"  # Last 5 days
)

# Get close prices
prices = get_close_prices(intraday)

# Compute portfolio metrics
metrics = compute_portfolio_metrics(prices)

# Access results
print("Correlation Matrix:")
print(metrics['correlation_matrix'])
print("\nRolling Volatility:")
print(metrics['rolling_volatility_30d'].tail())
```

## Testing Live Data

Run the test script to see live data in action:

```bash
python test_live_data.py
```

This will:
1. Fetch live quotes for your tickers
2. Get current prices
3. Fetch intraday 5-minute data
4. Fetch 1-minute data (if available)

## Alternative APIs for True Real-Time Data

If you need true real-time data (no delay), consider these options:

### 1. **Zerodha Kite API** (Indian Market)
- Official API from Zerodha broker
- Real-time data for NSE/BSE
- Requires trading account
- Documentation: https://kite.trade/docs/

### 2. **Upstox API** (Indian Market)
- Official API from Upstox broker
- Real-time market data
- Requires trading account
- Documentation: https://upstox.com/developer/

### 3. **Alpha Vantage**
- Paid API service
- Global market data including India
- Website: https://www.alphavantage.co/

### 4. **NSE/BSE Official APIs**
- Direct from exchanges
- May require registration
- Check NSE/BSE official websites

### 5. **Polygon.io**
- Paid API service
- Real-time and historical data
- Website: https://polygon.io/

## Error Handling

All functions include error handling. If a ticker fails to fetch:

```python
from src.data_loader import fetch_live_prices

tickers = ["RELIANCE.NS", "INVALID.NS", "TCS.NS"]
prices = fetch_live_prices(tickers)

# Invalid tickers will show None
print(prices)
# RELIANCE.NS    2450.50
# INVALID.NS     None
# TCS.NS         3456.75
```

## Best Practices

1. **Rate Limiting**: Don't fetch too frequently (every few seconds is fine)
2. **Error Handling**: Always check for None values
3. **Market Hours**: Data may not be available when market is closed
4. **Caching**: Live data functions don't use cache (by design)
5. **Data Validation**: Verify data before using in calculations

## Summary

- Use `fetch_live_quotes()` for comprehensive market data
- Use `fetch_intraday_data()` for minute-level price history
- Use `fetch_live_prices()` for simple current price lookup
- Remember: yfinance has 15-20 minute delay (free tier)
- For true real-time: consider paid APIs or broker APIs
