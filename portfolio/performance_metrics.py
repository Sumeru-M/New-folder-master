"""
Performance Metrics Module

Comprehensive portfolio performance analytics including:
- Portfolio & per-stock CAGR
- XIRR for irregular cash flows
- Return attribution
- Future value projections
- Absolute return calculations

All calculations use log returns for consistency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
from scipy.optimize import newton

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from portfolio.constants import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE_ANNUAL
from portfolio.types import PerformanceMetrics, ProjectedValue


class PerformanceAnalyzer:
    """
    Comprehensive performance metrics calculation.
    
    Handles portfolio-level and per-stock performance analytics including
    CAGR, XIRR, projections, and return attribution.
    """
    
    def __init__(
        self,
        prices: pd.DataFrame,
        holdings: Dict[str, float],
        purchase_history: Optional[Dict[str, List[Dict]]] = None,
        risk_free_rate: float = RISK_FREE_RATE_ANNUAL
    ):
        """
        Initialize performance analyzer.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (Close prices).
            Columns are tickers, index is datetime.
        holdings : Dict[str, float]
            Current holdings as {ticker: shares}
        purchase_history : Dict[str, List[Dict]], optional
            Purchase history for XIRR calculation.
            Format: {ticker: [{"date": "2020-01-01", "shares": 10, "price": 100}, ...]}
        risk_free_rate : float, default=0.07
            Annual risk-free rate for Sharpe calculations
        """
        self.prices = prices
        self.holdings = holdings
        self.purchase_history = purchase_history or {}
        self.risk_free_rate = risk_free_rate
        
        # Compute returns (log returns)
        self.returns = self._compute_returns()
        
        # Align holdings with available data
        self.tickers = [t for t in holdings.keys() if t in prices.columns]
        
        # Current prices
        self.current_prices = prices.iloc[-1]
        
        # Time period
        self.start_date = prices.index[0]
        self.end_date = prices.index[-1]
        self.n_days = len(prices)
        self.n_years = self.n_days / TRADING_DAYS_PER_YEAR
    
    def _compute_returns(self) -> pd.DataFrame:
        """Compute daily log returns from prices."""
        return np.log(self.prices / self.prices.shift(1)).dropna()
    
    # ========================================================================
    # CAGR CALCULATIONS
    # ========================================================================
    
    def compute_cagr(
        self,
        returns: pd.Series,
        n_years: float = None
    ) -> float:
        """
        Compute Compound Annual Growth Rate from log returns.
        
        Parameters
        ----------
        returns : pd.Series
            Daily log returns
        n_years : float, optional
            Number of years. If None, computed from data length.
        
        Returns
        -------
        float
            Annualized CAGR
        """
        if returns.empty:
            return 0.0
        
        if n_years is None:
            n_years = len(returns) / TRADING_DAYS_PER_YEAR
        
        if n_years <= 0:
            return 0.0
        
        # For log returns: sum then exponentiate
        total_log_return = returns.sum()
        total_return = np.exp(total_log_return) - 1
        
        # CAGR formula
        cagr = (1 + total_return) ** (1 / n_years) - 1
        
        return cagr
    
    def compute_portfolio_cagr(self) -> float:
        """
        Compute portfolio CAGR based on holdings weights.
        
        Returns
        -------
        float
            Portfolio CAGR
        """
        weights = self._compute_weights()
        
        # Portfolio returns = weighted sum
        portfolio_returns = self.returns[self.tickers].dot(weights[self.tickers])
        
        return self.compute_cagr(portfolio_returns, self.n_years)
    
    def compute_stock_cagrs(self) -> pd.Series:
        """
        Compute CAGR for each individual stock.
        
        Returns
        -------
        pd.Series
            CAGR for each stock
        """
        cagrs = {}
        for ticker in self.tickers:
            cagrs[ticker] = self.compute_cagr(
                self.returns[ticker],
                self.n_years
            )
        
        return pd.Series(cagrs)
    
    # ========================================================================
    # XIRR CALCULATIONS
    # ========================================================================
    
    def compute_xirr(
        self,
        cash_flows: List[Tuple[datetime, float]],
        guess: float = 0.1,
        max_iterations: int = 100,
        tolerance: float = 1e-6
    ) -> float:
        """
        Compute XIRR (Extended Internal Rate of Return) for irregular cash flows.
        
        Uses Newton-Raphson method to solve:
        NPV = sum(CF_i / (1+r)^(t_i)) = 0
        
        Parameters
        ----------
        cash_flows : List[Tuple[datetime, float]]
            List of (date, amount) tuples.
            Negative for investments, positive for returns.
        guess : float, default=0.1
            Initial guess for IRR
        max_iterations : int, default=100
            Maximum iterations
        tolerance : float, default=1e-6
            Convergence tolerance
        
        Returns
        -------
        float
            Annualized XIRR, or np.nan if doesn't converge
        """
        if len(cash_flows) < 2:
            return 0.0
        
        # Sort by date
        cash_flows = sorted(cash_flows, key=lambda x: x[0])
        
        # Convert dates to years from first date
        first_date = cash_flows[0][0]
        years = np.array([
            (date - first_date).days / 365.25
            for date, _ in cash_flows
        ])
        cf = np.array([amount for _, amount in cash_flows])
        
        # Newton-Raphson iteration
        rate = guess
        
        for i in range(max_iterations):
            # NPV and its derivative
            powers = np.power(1 + rate, years)
            npv = np.sum(cf / powers)
            dnpv = np.sum(-cf * years / np.power(1 + rate, years + 1))
            
            if abs(dnpv) < tolerance:
                return np.nan
            
            # Update
            new_rate = rate - npv / dnpv
            
            # Check convergence
            if abs(new_rate - rate) < tolerance:
                return new_rate
            
            rate = new_rate
        
        return np.nan
    
    def compute_portfolio_xirr(self) -> float:
        """
        Compute portfolio-level XIRR from purchase history.
        
        Returns
        -------
        float
            Portfolio XIRR, or np.nan if insufficient data
        """
        if not self.purchase_history:
            return np.nan
        
        cash_flows = []
        
        # Add all purchases (negative cash flows)
        for ticker, purchases in self.purchase_history.items():
            if ticker not in self.tickers:
                continue
            
            for purchase in purchases:
                date = pd.to_datetime(purchase["date"])
                shares = purchase["shares"]
                price = purchase["price"]
                amount = -shares * price  # Negative = outflow
                
                cash_flows.append((date, amount))
        
        # Add current value (positive cash flow)
        current_value = self.compute_current_portfolio_value()
        cash_flows.append((self.end_date, current_value))
        
        if len(cash_flows) < 2:
            return np.nan
        
        return self.compute_xirr(cash_flows)
    
    def compute_stock_xirrs(self) -> pd.Series:
        """
        Compute XIRR for each stock with purchase history.
        
        Returns
        -------
        pd.Series
            XIRR for each stock
        """
        xirrs = {}
        
        for ticker in self.tickers:
            if ticker not in self.purchase_history:
                xirrs[ticker] = np.nan
                continue
            
            cash_flows = []
            
            # Purchases
            for purchase in self.purchase_history[ticker]:
                date = pd.to_datetime(purchase["date"])
                shares = purchase["shares"]
                price = purchase["price"]
                amount = -shares * price
                
                cash_flows.append((date, amount))
            
            # Current value
            current_value = self.holdings[ticker] * self.current_prices[ticker]
            cash_flows.append((self.end_date, current_value))
            
            xirrs[ticker] = self.compute_xirr(cash_flows)
        
        return pd.Series(xirrs)
    
    # ========================================================================
    # RETURN ATTRIBUTION
    # ========================================================================
    
    def compute_absolute_returns(self) -> Dict:
        """
        Compute absolute return metrics (invested vs current value).
        
        Returns
        -------
        Dict
            invested_value: Total invested
            current_value: Current portfolio value
            absolute_return: Difference in INR
            total_return_pct: Percentage return
        """
        # If we have purchase history, use it
        if self.purchase_history:
            invested_value = 0.0
            for ticker, purchases in self.purchase_history.items():
                for purchase in purchases:
                    invested_value += purchase["shares"] * purchase["price"]
        else:
            # Estimate: assume bought at start of period
            invested_value = 0.0
            for ticker in self.tickers:
                if ticker in self.prices.columns:
                    initial_price = self.prices[ticker].iloc[0]
                    invested_value += self.holdings[ticker] * initial_price
        
        current_value = self.compute_current_portfolio_value()
        absolute_return = current_value - invested_value
        total_return_pct = (current_value / invested_value - 1) if invested_value > 0 else 0.0
        
        return {
            "invested_value": invested_value,
            "current_value": current_value,
            "absolute_return": absolute_return,
            "total_return_pct": total_return_pct
        }
    
    def compute_contribution_to_return(self) -> pd.Series:
        """
        Compute each stock's contribution to portfolio return.
        
        Returns
        -------
        pd.Series
            Contribution to total return for each stock
        """
        weights = self._compute_weights()
        stock_returns = self.compute_stock_total_returns()
        
        contributions = weights * stock_returns
        
        return contributions
    
    def compute_stock_total_returns(self) -> pd.Series:
        """
        Compute total return for each stock over the period.
        
        Returns
        -------
        pd.Series
            Total return for each stock
        """
        total_returns = {}
        
        for ticker in self.tickers:
            if ticker not in self.prices.columns:
                total_returns[ticker] = 0.0
                continue
            
            # Total log return
            total_log_return = self.returns[ticker].sum()
            total_return = np.exp(total_log_return) - 1
            total_returns[ticker] = total_return
        
        return pd.Series(total_returns)
    
    # ========================================================================
    # PROJECTIONS
    # ========================================================================
    
    def project_portfolio_value(
        self,
        horizons: List[int] = [1, 3, 5, 10],
        method: str = "historical_cagr",
        n_simulations: int = 1000
    ) -> Dict[int, ProjectedValue]:
        """
        Project portfolio value at future horizons.
        
        Parameters
        ----------
        horizons : List[int]
            Projection horizons in years
        method : str
            "historical_cagr" or "monte_carlo"
        n_simulations : int
            Number of simulations for monte_carlo method
        
        Returns
        -------
        Dict[int, ProjectedValue]
            Projections for each horizon
        """
        current_value = self.compute_current_portfolio_value()
        
        projections = {}
        
        for horizon in horizons:
            if method == "historical_cagr":
                projections[horizon] = self._project_cagr(
                    current_value, horizon
                )
            elif method == "monte_carlo":
                projections[horizon] = self._project_monte_carlo(
                    current_value, horizon, n_simulations
                )
        
        return projections
    
    def _project_cagr(
        self,
        current_value: float,
        horizon_years: int
    ) -> ProjectedValue:
        """Project value using historical CAGR (deterministic)."""
        cagr = self.compute_portfolio_cagr()
        
        # Expected value
        expected_value = current_value * ((1 + cagr) ** horizon_years)
        
        # Estimate volatility-based confidence intervals
        weights = self._compute_weights()
        portfolio_returns = self.returns[self.tickers].dot(weights[self.tickers])
        volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Approximate percentiles using normal distribution
        # This is simplistic - monte carlo gives better estimates
        annual_std = volatility * np.sqrt(horizon_years)
        
        percentiles = {
            10: expected_value * np.exp(-1.28 * annual_std),
            25: expected_value * np.exp(-0.67 * annual_std),
            50: expected_value,
            75: expected_value * np.exp(0.67 * annual_std),
            90: expected_value * np.exp(1.28 * annual_std),
        }
        
        return ProjectedValue(
            horizon_years=horizon_years,
            expected_value=expected_value,
            median_value=expected_value,
            std_dev=current_value * annual_std,
            percentiles=percentiles,
            probability_of_loss=0.0,  # Simplified
            probability_of_double=1.0 if expected_value >= 2 * current_value else 0.0
        )
    
    def _project_monte_carlo(
        self,
        current_value: float,
        horizon_years: int,
        n_simulations: int
    ) -> ProjectedValue:
        """Project value using Monte Carlo simulation."""
        # This is a simplified version - full MC in monte_carlo.py
        weights = self._compute_weights()
        portfolio_returns = self.returns[self.tickers].dot(weights[self.tickers])
        
        # Estimate parameters
        mu = portfolio_returns.mean() * TRADING_DAYS_PER_YEAR
        sigma = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        
        # Simulate
        n_steps = horizon_years * TRADING_DAYS_PER_YEAR
        final_values = []
        
        for _ in range(n_simulations):
            # Geometric Brownian Motion
            returns = np.random.normal(mu/TRADING_DAYS_PER_YEAR, sigma/np.sqrt(TRADING_DAYS_PER_YEAR), int(n_steps))
            final_value = current_value * np.exp(returns.sum())
            final_values.append(final_value)
        
        final_values = np.array(final_values)
        
        percentiles = {
            int(p): np.percentile(final_values, p)
            for p in [10, 25, 50, 75, 90]
        }
        
        return ProjectedValue(
            horizon_years=horizon_years,
            expected_value=final_values.mean(),
            median_value=np.median(final_values),
            std_dev=final_values.std(),
            percentiles=percentiles,
            probability_of_loss=(final_values < current_value).mean(),
            probability_of_double=(final_values >= 2 * current_value).mean()
        )
    
    # ========================================================================
    # COMPREHENSIVE METRICS
    # ========================================================================
    
    def compute_all_metrics(self) -> PerformanceMetrics:
        """
        Compute comprehensive performance metrics.
        
        Returns
        -------
        PerformanceMetrics
            Complete performance analytics
        """
        # Portfolio-level metrics
        portfolio_cagr = self.compute_portfolio_cagr()
        portfolio_xirr = self.compute_portfolio_xirr()
        absolute_metrics = self.compute_absolute_returns()
        
        # Per-stock metrics
        stock_cagrs = self.compute_stock_cagrs()
        stock_xirrs = self.compute_stock_xirrs()
        contributions = self.compute_contribution_to_return()
        
        # Risk-adjusted metrics (simplified - full version in risk_metrics.py)
        weights = self._compute_weights()
        portfolio_returns = self.returns[self.tickers].dot(weights[self.tickers])
        
        volatility = portfolio_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR)
        sharpe_ratio = (portfolio_cagr - self.risk_free_rate) / volatility if volatility > 0 else 0.0
        
        # Downside metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else volatility
        sortino_ratio = (portfolio_cagr - self.risk_free_rate) / downside_vol if downside_vol > 0 else 0.0
        
        # Max drawdown (simplified)
        cumulative_returns = np.exp(portfolio_returns.cumsum())
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        calmar_ratio = portfolio_cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0
        
        return PerformanceMetrics(
            cagr=portfolio_cagr,
            total_return=absolute_metrics["total_return_pct"],
            annualized_return=portfolio_cagr,
            annualized_volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            max_drawdown=max_drawdown,
            start_date=self.start_date,
            end_date=self.end_date,
            n_days=self.n_days,
            n_years=self.n_years,
            invested_value=absolute_metrics["invested_value"],
            current_value=absolute_metrics["current_value"],
            absolute_return=absolute_metrics["absolute_return"],
            xirr=portfolio_xirr,
            per_stock_cagrs=stock_cagrs.to_dict(),
            contribution_to_return=contributions.to_dict()
        )
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _compute_weights(self) -> pd.Series:
        """Compute portfolio weights based on current holdings and prices."""
        current_value = self.compute_current_portfolio_value()
        
        if current_value == 0:
            # Equal weights if no value
            return pd.Series(
                1.0 / len(self.tickers),
                index=self.tickers
            )
        
        weights = {}
        for ticker in self.tickers:
            if ticker in self.current_prices.index:
                position_value = self.holdings[ticker] * self.current_prices[ticker]
                weights[ticker] = position_value / current_value
            else:
                weights[ticker] = 0.0
        
        return pd.Series(weights)
    
    def compute_current_portfolio_value(self) -> float:
        """Compute current portfolio value in INR."""
        total_value = 0.0
        
        for ticker in self.tickers:
            if ticker in self.current_prices.index:
                total_value += self.holdings[ticker] * self.current_prices[ticker]
        
        return total_value


# ============================================================================
# STANDALONE FUNCTIONS
# ============================================================================

def compute_simple_cagr(
    start_value: float,
    end_value: float,
    n_years: float
) -> float:
    """
    Compute CAGR from start and end values.
    
    Parameters
    ----------
    start_value : float
        Initial value
    end_value : float
        Final value
    n_years : float
        Number of years
    
    Returns
    -------
    float
        CAGR
    """
    if start_value <= 0 or n_years <= 0:
        return 0.0
    
    return (end_value / start_value) ** (1 / n_years) - 1
