# Data Flow Explanation

## Where Stock Names Are Input

### 1. **Test Scripts** (User Input Points)

#### `test_new_stock.py` (Line 116-126)
```python
test_tickers = [
    "RELIANCE.NS",      # ← Stock names input here
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    # ... more stocks
]
```

#### `test_any_stock.py` (Line 21)
```python
TICKER = "RELIANCE.NS"  # ← Single stock input here
```

### 2. **Direct API Calls** (In Your Code)
```python
from src.data_loader import get_stock_data

# Stock names input here
prices = get_stock_data(["RELIANCE.NS", "TCS.NS"], period="1y")
```

---

## Data Flow: From Input to Cache

```
┌─────────────────────────────────────────────────────────────┐
│ 1. USER INPUT (Stock Names)                                 │
│    - test_new_stock.py: test_tickers = ["RELIANCE.NS", ...] │
│    - test_any_stock.py: TICKER = "RELIANCE.NS"              │
│    - Your code: get_stock_data(["RELIANCE.NS"])             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. get_stock_data() in src/data_loader.py (Line 310)        │
│    - Receives ticker list                                    │
│    - Calls fetch_market_data()                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. fetch_market_data() in src/data_loader.py (Line 102)     │
│    - Generates cache filename (Line 157)                    │
│    - Checks if cache exists (Line 160-166)                  │
└────────────────────┬────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐   ┌──────────────────────┐
│ CACHE EXISTS?    │   │ CACHE MISS           │
│ (Line 161)       │   │                      │
└────────┬─────────┘   └──────────┬───────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐   ┌──────────────────────┐
│ 4a. Load from    │   │ 4b. Download from    │
│     Cache File   │   │     yfinance API     │
│                  │   │                      │
│ _load_from_cache │   │ yf.download()        │
│ (Line 59)        │   │ (Line 172 or 176)    │
│                  │   │                      │
│ Reads CSV from:  │   │ Downloads from:      │
│ data_cache/      │   │ Yahoo Finance        │
│ market_data_*.csv│   │ (Internet)            │
└────────┬─────────┘   └──────────┬───────────┘
         │                        │
         │                        ▼
         │              ┌──────────────────────┐
         │              │ 5. Save to Cache     │
         │              │    _save_to_cache()  │
         │              │    (Line 84)         │
         │              │                      │
         │              │ Saves CSV to:        │
         │              │ data_cache/          │
         │              │ market_data_*.csv    │
         │              └──────────┬───────────┘
         │                        │
         └────────────┬───────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 6. Return Price Data                                        │
│    - Cleaned and normalized                                 │
│    - Ready for analysis                                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Which Files Store/Provide Data

### **Cache Files** (Derived Data Storage)
**Location:** `data_cache/` directory

**Files:** 
- `market_data_2bdae18b.csv` - RELIANCE.NS (6mo period)
- `market_data_2ae30a81.csv` - TCS.NS (6mo period)
- `market_data_fe7619c8.csv` - INFY.NS (6mo period)
- `market_data_69388a86.csv` - Multiple stocks (5 tickers, 2y period)
- ... and more

**How cache filename is generated:**
```python
# In src/data_loader.py, line 24-56
def _generate_cache_filename(tickers, start_date, end_date):
    # Creates hash from: tickers + date range
    # Example: "RELIANCE.NS_None_None" → hash → "market_data_2bdae18b.csv"
```

**Cache file structure:**
- CSV format
- Index: Dates (DateTimeIndex)
- Columns: MultiIndex (Ticker, OHLCV) - Open, High, Low, Close, Volume

### **Source of Data** (When Cache Doesn't Exist)

**Primary Source:** Yahoo Finance API (via `yfinance` library)
- **Function:** `yf.download()` in `src/data_loader.py` (Line 172 or 176)
- **Data Source:** Internet (Yahoo Finance servers)
- **Format:** OHLCV (Open, High, Low, Close, Volume) data

---

## Complete Flow Example

### Example: Testing "RELIANCE.NS"

1. **Input:** `test_new_stock.py` line 117: `"RELIANCE.NS"`

2. **Function Call Chain:**
   ```python
   test_stock_data("RELIANCE.NS", period="6mo")
     ↓
   get_stock_data("RELIANCE.NS", period="6mo")  # Line 310
     ↓
   fetch_market_data(["RELIANCE.NS"], period="6mo")  # Line 102
   ```

3. **Cache Check:**
   - Generates filename: `data_cache/market_data_2bdae18b.csv`
   - Checks if file exists
   - **If exists:** Loads from CSV (Line 75)
   - **If not:** Downloads from yfinance (Line 172)

4. **Data Processing:**
   - Cleans data (removes NaN, sorts dates)
   - Extracts Close prices
   - Returns DataFrame

5. **Sharpe Ratio Calculation:**
   - Uses the cached/loaded price data
   - Computes daily returns
   - Calculates annualized return, volatility, Sharpe ratio

---

## Key Files Summary

| File | Purpose | Line Numbers |
|------|---------|--------------|
| `test_new_stock.py` | **Input:** Stock ticker list | 116-126 |
| `test_any_stock.py` | **Input:** Single ticker | 21 |
| `src/data_loader.py` | **Core logic:** Fetch/cache data | 102-590 |
| `data_cache/*.csv` | **Storage:** Cached historical data | N/A |
| Yahoo Finance API | **Source:** Live market data | Internet |

---

## How to Change Stock Names

### Option 1: Edit Test Scripts
```python
# In test_new_stock.py, line 116
test_tickers = [
    "YOURSTOCK.NS",  # ← Add your stock here
    "ANOTHER.NS",
]
```

### Option 2: Use Direct API
```python
from src.data_loader import get_stock_data

# Your stock names here
prices = get_stock_data(["YOURSTOCK.NS", "ANOTHER.NS"])
```

### Option 3: Create Your Own Script
```python
from src.data_loader import get_stock_data

my_stocks = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
prices = get_stock_data(my_stocks, period="1y")
```

---

## Cache File Naming Convention

Cache files are named using MD5 hash of:
- **Ticker names** (sorted alphabetically)
- **Date range** (start_date, end_date, or period)

Example:
- Input: `["RELIANCE.NS"]` with `period="6mo"`
- Hash input: `"RELIANCE.NS_None_None"`
- Hash output: `2bdae18b`
- Cache file: `data_cache/market_data_2bdae18b.csv`

This ensures:
- Same tickers + same period = same cache file
- Different tickers/periods = different cache files
- Automatic cache reuse for faster subsequent loads
