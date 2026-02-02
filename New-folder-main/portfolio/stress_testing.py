"""
Stress Testing Module

This module enables historical stress testing by replaying portfolio performance
over specific historical periods (e.g., 2008 Financial Crisis, 2020 COVID Crash).

Uses log returns for consistency with other portfolio modules.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

def compute_drawdown(series: pd.Series) -> pd.Series:
    """
    Compute drawdown series from cumulative returns or price series.
    
    Drawdown is the percentage decline from the running maximum.
    Formula: DD_t = (Value_t - RunningMax_t) / RunningMax_t
    
    Parameters
    ----------
    series : pd.Series
        Time series of prices or cumulative portfolio value
        
    Returns
    -------
    pd.Series
        Drawdown series (negative values, e.g., -0.05 for 5% drawdown)
    """
    running_max = series.cummax()
    drawdown = (series - running_max) / running_max
    return drawdown


def compute_max_drawdown(series: pd.Series) -> float:
    """
    Compute Maximum Drawdown (MDD) from a series.
    
    Returns the largest peak-to-trough decline.
    
    Parameters
    ----------
    series : pd.Series
        Price or value series
        
    Returns
    -------
    float
        Maximum drawdown (negative value)
    """
    return compute_drawdown(series).min()


class StressTester:
    """
    Engine for historical stress testing using log returns.
    
    This class enables backtesting portfolio performance over historical
    periods to understand how the portfolio would have performed during
    various market conditions.
    """
    
    def __init__(self, prices: pd.DataFrame):
        """
        Initialize Stress Tester.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Historical price data (Close prices).
            Columns are tickers, index is datetime.
            Should cover the periods intended for testing.
        """
        self.prices = prices
        # Use log returns for consistency with other modules
        self.daily_returns = np.log(prices / prices.shift(1)).dropna()
        
    def replay_period(
        self,
        weights: pd.Series,
        start_date: str,
        end_date: str,
        initial_investment: float = 1_00_000.0,
        risk_free_rate: float = 0.07
    ) -> Dict:
        """
        Replay portfolio performance over a specific historical period.
        
        Uses log returns for calculations and assumes daily rebalancing
        to fixed weights (simplified approach for stress testing).
        
        Parameters
        ----------
        weights : pd.Series
            Portfolio weights (index must match tickers in prices)
            Weights should sum to 1.0
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        initial_investment : float, default=1_00_000.0
            Starting portfolio value in INR (₹1,00,000 = 1 lakh)
        risk_free_rate : float, default=0.07
            Annual risk-free rate for Sharpe ratio calculation (7% default for India)
                
            Returns
            -------
            Dict
                Results containing:
                - period_label: Description of the period
                - portfolio_value: Time series of portfolio value
                - daily_returns: Daily portfolio returns (log returns)
                - drawdown: Drawdown series
                - total_return: Total return over period
                - annualized_return: Annualized return (CAGR)
                - max_drawdown: Maximum drawdown
                - volatility: Annualized volatility
                - sharpe_ratio: Sharpe ratio (using provided risk-free rate)
            """
        # Check if requested dates are within available data range
        data_start = self.daily_returns.index.min()
        data_end = self.daily_returns.index.max()
        
        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)
        
        # Check if period is completely outside available data
        if requested_end < data_start or requested_start > data_end:
            return {
                "error": f"No data found for period {start_date} to {end_date} (data available: {data_start.date()} to {data_end.date()})"
            }
        
        # Adjust dates to available data range
        actual_start = max(requested_start, data_start)
        actual_end = min(requested_end, data_end)
        
        # Slice returns for the period using adjusted dates
        try:
            period_returns = self.daily_returns.loc[actual_start:actual_end]
        except KeyError:
            return {
                "error": f"Invalid date range: {start_date} to {end_date}"
            }
        
        if period_returns.empty or len(period_returns) < 5:
            return {
                "error": f"Insufficient data for period {start_date} to {end_date} (found {len(period_returns)} days)"
            }
            
        # Filter weights for available assets
        valid_tickers = [t for t in weights.index if t in period_returns.columns]
        if not valid_tickers:
            return {"error": "No matching tickers found in price data"}
             
        valid_weights = weights[valid_tickers]
        
        # Renormalize weights if some assets are missing
        if not np.isclose(valid_weights.sum(), 1.0):
            valid_weights = valid_weights / valid_weights.sum()
        
        # Calculate portfolio daily log returns: R_p = sum(w_i * R_i)
        # Note: For log returns, portfolio return is weighted sum
        portfolio_daily_returns = period_returns[valid_tickers].dot(valid_weights)
        
        # Compute Portfolio Value Path
        # For log returns: cumulative_value = initial * exp(sum of log returns)
        cumulative_log_returns = portfolio_daily_returns.cumsum()
        portfolio_value = initial_investment * np.exp(cumulative_log_returns)
        
        # Metrics
        # Total return: (Final Value / Initial Value) - 1
        total_return = (portfolio_value.iloc[-1] / initial_investment) - 1
        
        # Annualized return (CAGR)
        n_days = len(portfolio_daily_returns)
        years = n_days / 252
        if years > 0:
            annualized_return = (1 + total_return) ** (1 / years) - 1
        else:
            annualized_return = 0.0
        
        # Drawdown
        drawdown_series = compute_drawdown(portfolio_value)
        max_drawdown = drawdown_series.min()
        
        # Volatility (annualized)
        volatility = portfolio_daily_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (using provided risk-free rate)
        sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0.0
        
        return {
            "period_label": f"{start_date} to {end_date}",
            "portfolio_value": portfolio_value,
            "daily_returns": portfolio_daily_returns,
            "drawdown": drawdown_series,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "n_days": n_days
        }

    def replay_multiple_periods(
        self,
        weights: pd.Series,
        periods: Optional[Dict[str, Tuple[str, str]]] = None,
        initial_investment: float = 1_00_000.0,
        risk_free_rate: float = 0.07
    ) -> Dict[str, Dict]:
        """
        Replay portfolio across multiple historical periods.
        
        Parameters
        ----------
        weights : pd.Series
            Portfolio weights
        periods : Dict[str, Tuple[str, str]], optional
            Dictionary mapping period names to (start_date, end_date) tuples.
            If None, uses standard historical scenarios.
        initial_investment : float, default=1_00_000.0
            Starting portfolio value in INR for each period (₹1,00,000 = 1 lakh)
        risk_free_rate : float, default=0.07
            Annual risk-free rate for Sharpe ratio calculation
                
            Returns
            -------
            Dict[str, Dict]
                Dictionary mapping period names to results
            """
        if periods is None:
            periods = self.get_historical_scenarios()
            
        results = {}
        for name, (start, end) in periods.items():
            results[name] = self.replay_period(
                weights, start, end, initial_investment, risk_free_rate
            )
            
        return results

    @staticmethod
    def get_historical_scenarios() -> Dict[str, Tuple[str, str]]:
        """
        Return common historical stress periods.
        
        These periods represent major market events that are useful
        for stress testing portfolios.
        
        Returns
        -------
        Dict[str, Tuple[str, str]]
            Dictionary mapping scenario names to (start_date, end_date) tuples
        """
        return {
            "2008 Financial Crisis": ("2008-01-01", "2009-03-09"),
            "2020 COVID Crash": ("2020-02-19", "2020-03-23"),
            "2022 Tech Bear Market": ("2022-01-01", "2022-12-31"),
            "2000 Dotcom Bubble Burst": ("2000-03-10", "2002-10-09"),
            "2011 European Debt Crisis": ("2011-07-01", "2011-12-31"),
            "2015 China Slowdown": ("2015-06-01", "2015-09-30")
        }
    
    def compute_stress_summary(
        self,
        weights: pd.Series,
        scenarios: Optional[Dict[str, Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """
        Compute a summary table of stress test results across scenarios.
        
        Parameters
        ----------
        weights : pd.Series
            Portfolio weights
        scenarios : Dict[str, Tuple[str, str]], optional
            Historical periods to test. If None, uses standard scenarios.
            
        Returns
        -------
        pd.DataFrame
            Summary table with metrics for each scenario
        """
        results = self.replay_multiple_periods(weights, scenarios)
        
        summary_data = []
        for name, result in results.items():
            if "error" in result:
                continue
                
            summary_data.append({
                "Scenario": name,
                "Total Return": result["total_return"],
                "Annualized Return": result["annualized_return"],
                "Max Drawdown": result["max_drawdown"],
                "Volatility": result["volatility"],
                "Sharpe Ratio": result["sharpe_ratio"],
                "Days": result["n_days"]
            })
        
        return pd.DataFrame(summary_data)