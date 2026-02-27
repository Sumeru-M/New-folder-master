"""
Portfolio State Module

This module defines the PortfolioState dataclass, which serves as a unified
container for all portfolio-related data (weights, returns, value) at any
point in time. This simplifies passing data between the risk, scenario,
and visualization engines.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List
import pandas as pd
import numpy as np

@dataclass
class PortfolioState:
    """
    Represents the state of a portfolio at a specific point in time or
    under a specific scenario.
    """
    weights: pd.Series
    expected_returns: pd.Series
    covariance_matrix: pd.DataFrame
    daily_returns: pd.DataFrame  # Historical returns of assets
    total_value: float = 1_000_000.0  # Default 10 Lakhs
    risk_free_rate: float = 0.07      # Default 7%
    name: str = "Base Portfolio"
    
    # Computed properties (lazy loaded or pre-computed)
    _portfolio_return: Optional[float] = None
    _portfolio_volatility: Optional[float] = None
    _sharpe_ratio: Optional[float] = None
    
    def __post_init__(self):
        """Validate inputs after initialization."""
        # Ensure weights sum to 1 (approx)
        if not np.isclose(self.weights.sum(), 1.0, atol=0.01):
            # Normalize if close, warn if far
            if abs(self.weights.sum() - 1.0) < 0.05:
                self.weights = self.weights / self.weights.sum()
            else:
                pass  # Allow non-unitary weights for specific stress scenarios (e.g. valid withdrawals)

    @property
    def assets(self) -> List[str]:
        """Get list of asset tickers."""
        return self.weights.index.tolist()
        
    @property
    def portfolio_return(self) -> float:
        """Annualized expected portfolio return."""
        if self._portfolio_return is None:
            self._portfolio_return = float(np.dot(self.weights.values, self.expected_returns.values))
        return self._portfolio_return
    
    @property
    def portfolio_volatility(self) -> float:
        """Annualized portfolio volatility."""
        if self._portfolio_volatility is None:
            w = self.weights.values
            sigma = self.covariance_matrix.values
            self._portfolio_volatility = float(np.sqrt(np.dot(w.T, np.dot(sigma, w))))
        return self._portfolio_volatility
        
    @property
    def sharpe_ratio(self) -> float:
        """Annualized Sharpe Ratio."""
        if self._sharpe_ratio is None:
            if self.portfolio_volatility > 0:
                self._sharpe_ratio = (self.portfolio_return - self.risk_free_rate) / self.portfolio_volatility
            else:
                self._sharpe_ratio = 0.0
        return self._sharpe_ratio

    def get_value_at_risk(self, confidence: float = 0.95) -> float:
        """Get Parametric VaR amount."""
        # Simple parametric VaR for quick access
        from scipy.stats import norm
        z_score = abs(norm.ppf(1 - confidence))
        daily_vol = self.portfolio_volatility / np.sqrt(252)
        return z_score * daily_vol * self.total_value

    def copy_with_shocks(self, shock_name: str, return_shock: float = 0.0, 
                         vol_shock: float = 0.0, correlation_shock: float = 0.0) -> 'PortfolioState':
        """
        Create a new PortfolioState with applied shocks.
        
        Parameters:
        - return_shock: Additive shock to expected returns (e.g. -0.05 for -5%)
        - vol_shock: Multiplicative shock to volatility (e.g. 1.2 for +20% vol)
        - correlation_shock: Additive shock to correlations (e.g. 0.1 for +0.1 corr)
        """
        # 1. Shock Expected Returns
        new_returns = self.expected_returns + return_shock
        
        # 2. Shock Covariance Matrix
        # Decompose cov to corr and vol
        stds = np.sqrt(np.diag(self.covariance_matrix))
        outer_vol = np.outer(stds, stds)
        corr_matrix = self.covariance_matrix / outer_vol
        
        # Apply shocks
        new_stds = stds * (1 + vol_shock)
        new_corr = corr_matrix.copy()
        # Apply shock to off-diagonal elements only
        mask = ~np.eye(len(stds), dtype=bool)
        new_corr.values[mask] += correlation_shock
        new_corr = new_corr.clip(-1, 1)
        np.fill_diagonal(new_corr.values, 1.0)
        
        # Reconstruct Covariance
        new_outer_vol = np.outer(new_stds, new_stds)
        new_cov = new_corr * new_outer_vol
        
        return PortfolioState(
            weights=self.weights.copy(),
            expected_returns=new_returns,
            covariance_matrix=new_cov,
            daily_returns=self.daily_returns.copy(), # Historical not changed, only projected params
            total_value=self.total_value, # Start value same, end value will differ
            risk_free_rate=self.risk_free_rate,
            name=f"{self.name} ({shock_name})"
        )
