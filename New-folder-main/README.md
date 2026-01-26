# Portfolio Optimization and Risk Management System

A comprehensive finance project for portfolio optimization and risk management with scenario-based stress testing and Monte Carlo simulations.

## Project Structure

```
.
├── src/                    # Modular Python code (Milestones 1 & 2)
│   ├── __init__.py
│   ├── data_ingestion.py  # Data download and processing functions
│   ├── data_loader.py     # Data loader with caching
│   └── portfolio_engine.py # Portfolio metrics computation
├── portfolio/              # Portfolio optimization module (Milestone 3)
│   ├── __init__.py
│   ├── data_loader.py     # Load preprocessed price data
│   ├── optimizer.py        # Markowitz mean-variance optimization
│   └── plotting.py         # Efficient frontier visualization
├── notebooks/              # Jupyter notebooks for analysis
│   └── 01_data_exploration.ipynb
├── examples/               # Example scripts
│   └── run_milestone3.py  # Milestone 3 example
├── artifacts/              # Output files
│   └── milestone3/         # Milestone 3 outputs
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Launch Jupyter Notebook:
```bash
jupyter notebook
```

3. Open `notebooks/01_data_exploration.ipynb` to start the analysis.

Testing with user-provided tickers:
- Quick fetch & metrics: `python test_get_stock_data.py --tickers RELIANCE.NS,TCS.NS --period 1y`
- Single/multiple fetch: `python test_any_stock.py --tickers RELIANCE.NS,TCS.NS --period 6mo`
- Pipeline test: `python test_pipeline.py --tickers RELIANCE.NS,TCS.NS,INFY.NS --period 1y`
- Optimization example: `python examples/run_milestone3.py --tickers RELIANCE.NS,TCS.NS,INFY.NS --period 2y`

## Milestone 1: Data Layer

### Features
- Download historical data for Indian market instruments (NIFTYBEES, BANKBEES, GOLDBEES, RELIANCE, INFY) using yfinance
- Compute daily log returns
- Compute monthly returns
- Calculate annualized volatility
- Generate correlation matrix
- Visualize volatility and correlation heatmaps

### Usage

```python
from src.data_ingestion import (
    download_etf_data,
    compute_log_returns,
    compute_volatility,
    compute_correlation_matrix
)

# Download Indian market data (use .NS suffix for NSE-listed instruments)
prices = download_etf_data(['NIFTYBEES.NS', 'BANKBEES.NS', 'GOLDBEES.NS', 'RELIANCE.NS', 'INFY.NS'], period='5y')

# Compute metrics
returns = compute_log_returns(prices)
volatility = compute_volatility(returns)
correlation = compute_correlation_matrix(returns)
```

## Milestone 2: Data Engine (Completed)

### Features
- Download historical OHLCV data for Indian stocks (NSE) using yfinance
- CSV-based caching for faster subsequent loads
- Compute portfolio metrics: returns, volatility, covariance, correlation, drawdowns
- Simple API interface: `get_stock_data()` for multiple tickers

### Usage

```python
from src.data_loader import get_stock_data
from src.portfolio_engine import compute_portfolio_metrics

# Fetch data for multiple tickers
prices = get_stock_data(["RELIANCE.NS", "TCS.NS", "INFY.NS"], period="2y")

# Compute portfolio metrics
metrics = compute_portfolio_metrics(prices)
print(metrics['correlation_matrix'])
```

## Milestone 3: Portfolio Optimization Engine (Completed)

### Features
- Markowitz mean-variance portfolio optimization using cvxpy
- Two optimization modes:
  - **Minimum Variance Portfolio**: Minimizes portfolio risk
  - **Maximum Sharpe Ratio Portfolio**: Maximizes risk-adjusted returns
- Efficient frontier computation and visualization
- Computes daily percentage returns, expected annual returns, and covariance matrix

### Usage

```python
from portfolio.data_loader import load_price_data
from portfolio.optimizer import (
    PortfolioOptimizer,
    compute_daily_returns,
    compute_expected_returns,
    compute_covariance_matrix
)
from portfolio.plotting import plot_efficient_frontier

# Load data
prices = load_price_data(["RELIANCE.NS", "TCS.NS", "INFY.NS"], period="2y")

# Compute returns and metrics
daily_returns = compute_daily_returns(prices)
expected_returns = compute_expected_returns(daily_returns)
covariance_matrix = compute_covariance_matrix(daily_returns)

# Initialize optimizer
optimizer = PortfolioOptimizer(
    expected_returns=expected_returns,
    covariance_matrix=covariance_matrix,
    risk_free_rate=0.05
)

# Optimize portfolios
min_var_portfolio = optimizer.optimize_min_variance()
max_sharpe_portfolio = optimizer.optimize_max_sharpe()

# Compute efficient frontier
ef_returns, ef_volatilities, ef_sharpe = optimizer.compute_efficient_frontier(n_points=50)

# Plot results
plot_efficient_frontier(
    returns=ef_returns,
    volatilities=ef_volatilities,
    min_var_result=min_var_portfolio,
    max_sharpe_result=max_sharpe_portfolio
)
```

### Run Example

```bash
python examples/run_milestone3.py
```

This will:
1. Load price data for 5 Indian stocks
2. Compute daily returns, expected returns, and covariance matrix
3. Optimize for minimum variance and maximum Sharpe ratio portfolios
4. Compute and plot the efficient frontier
5. Save results to `artifacts/milestone3/`

### Output Files

Results are saved to `artifacts/milestone3/`:
- `efficient_frontier.png` - Visualization of efficient frontier
- `efficient_frontier.csv` - Efficient frontier data points
- `min_variance_portfolio.csv` - Minimum variance portfolio weights
- `max_sharpe_portfolio.csv` - Maximum Sharpe ratio portfolio weights

## Future Milestones

- **Milestone 4**: Risk Management (VaR, CVaR)
- **Milestone 5**: Stress Testing
- **Milestone 6**: Monte Carlo Simulations
