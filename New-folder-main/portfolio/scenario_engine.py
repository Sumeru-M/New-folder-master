"""
Scenario Engine Module

This module defines hypothetical market shocks and applies them to portfolio input parameters
(expected returns, covariance matrices). It allows for stress testing portfolios against
various "what-if" scenarios.

Ensures covariance matrices remain positive semi-definite after shocks.
"""

from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class MarketShock:
    """
    Defines a market shock scenario.
    
    Attributes
    ----------
    name : str
        Name of the scenario (e.g., "Market Crash", "High Inflation")
    return_shock : float or Dict[str, float]
        Absolute shock to expected returns (e.g., -0.20 for -20%).
        Can be a single float (applied to all) or a dict mapping tickers to shocks.
    volatility_shock : float
        Multiplicative shock to volatility (e.g., 1.5 for 50% increase).
        Applied to standard deviations, which affects covariance matrix.
    correlation_shock : float
        Absolute shock to correlations (e.g., 0.2 to increase correlations).
        Applied to off-diagonal elements. Matrix is then repaired to ensure PSD.
    """
    name: str
    return_shock: Union[float, Dict[str, float]] = 0.0
    volatility_shock: float = 1.0
    correlation_shock: float = 0.0


def nearest_positive_definite(matrix: np.ndarray) -> np.ndarray:
    """
    Find the nearest positive semi-definite matrix to the input matrix.
    
    Uses the Higham (1988) algorithm to find the nearest correlation matrix
    in the Frobenius norm that is positive semi-definite.
    
    Parameters
    ----------
    matrix : np.ndarray
        Input matrix (potentially non-PSD)
        
    Returns
    -------
    np.ndarray
        Nearest positive semi-definite matrix
        
    Reference
    ---------
    Higham, N. J. (1988). Computing a nearest symmetric positive semidefinite matrix.
    Linear Algebra and its Applications, 103, 103-118.
    """
    # Symmetrize
    symmetric = (matrix + matrix.T) / 2
    
    # Eigenvalue decomposition
    eigvals, eigvecs = np.linalg.eigh(symmetric)
    
    # Clamp negative eigenvalues to small positive value
    eigvals[eigvals < 0] = 1e-8
    
    # Reconstruct matrix
    result = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    # Ensure diagonal is exactly 1.0 for correlation matrices
    # (Only if input was a correlation matrix)
    if np.allclose(np.diag(matrix), 1.0):
        # Scale to unit diagonal
        d = np.sqrt(np.diag(result))
        result = result / np.outer(d, d)
        np.fill_diagonal(result, 1.0)
    
    return result


class ScenarioEngine:
    """
    Engine to apply market shocks to portfolio parameters.
    
    Ensures mathematical consistency by repairing covariance matrices
    to be positive semi-definite after applying shocks.
    """
    
    def __init__(
        self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame
    ):
        """
        Initialize the Scenario Engine.
        
        Parameters
        ----------
        expected_returns : pd.Series
            Base expected annual returns for each asset
        covariance_matrix : pd.DataFrame
            Base annualized covariance matrix (must be PSD)
        """
        self.base_expected_returns = expected_returns
        self.base_covariance_matrix = covariance_matrix
        self.tickers = expected_returns.index.tolist()
        
        # Validate input covariance is PSD
        if not self._is_positive_semidefinite(covariance_matrix.values):
            raise ValueError("Input covariance matrix must be positive semi-definite")
        
    @staticmethod
    def _is_positive_semidefinite(matrix: np.ndarray, tol: float = 1e-8) -> bool:
        """Check if matrix is positive semi-definite."""
        eigvals = np.linalg.eigvalsh(matrix)
        return np.all(eigvals > -tol)
        
    def apply_scenario(self, shock: MarketShock) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Apply a defined market shock to the base parameters.
        
        The method ensures the resulting covariance matrix is positive semi-definite
        by using the nearest PSD matrix if needed.
        
        Parameters
        ----------
        shock : MarketShock
            Market shock scenario to apply
        
        Returns
        -------
        Tuple[pd.Series, pd.DataFrame]
            (shocked_expected_returns, shocked_covariance_matrix)
            
        Examples
        --------
        >>> engine = ScenarioEngine(returns, cov)
        >>> crash = MarketShock("Crash", return_shock=-0.30, volatility_shock=2.0)
        >>> new_returns, new_cov = engine.apply_scenario(crash)
        """
        # 1. Apply Return Shock
        shocked_returns = self.base_expected_returns.copy()
        
        if isinstance(shock.return_shock, dict):
            # Asset-specific shocks
            for ticker, shock_val in shock.return_shock.items():
                if ticker in shocked_returns.index:
                    shocked_returns[ticker] += shock_val
        else:
            # Uniform shock to all assets
            shocked_returns += shock.return_shock
            
        # 2. Apply Volatility and Correlation Shocks
        # Decompose Covariance to Correlation and Standard Deviations
        cov_values = self.base_covariance_matrix.values
        std_devs = np.sqrt(np.diag(cov_values))
        
        # Compute correlation matrix
        outer_vols = np.outer(std_devs, std_devs)
        correlation_matrix = cov_values / outer_vols
        
        # Apply Correlation Shock
        if shock.correlation_shock != 0.0:
            # Shift off-diagonal elements
            n = correlation_matrix.shape[0]
            mask = ~np.eye(n, dtype=bool)
            correlation_matrix[mask] += shock.correlation_shock
            
            # Clip to valid correlation range
            correlation_matrix = np.clip(correlation_matrix, -0.99, 0.99)
            
            # Ensure diagonal is exactly 1.0
            np.fill_diagonal(correlation_matrix, 1.0)
            
            # Repair matrix to ensure it's positive semi-definite
            if not self._is_positive_semidefinite(correlation_matrix):
                correlation_matrix = nearest_positive_definite(correlation_matrix)
        
        # Apply Volatility Shock (multiplicative to standard deviations)
        new_std_devs = std_devs * shock.volatility_shock
        
        # Reconstruct Covariance Matrix
        new_outer_vols = np.outer(new_std_devs, new_std_devs)
        shocked_cov_values = correlation_matrix * new_outer_vols
        
        # Final check and repair if needed
        if not self._is_positive_semidefinite(shocked_cov_values):
            shocked_cov_values = nearest_positive_definite(shocked_cov_values)
        
        shocked_covariance = pd.DataFrame(
            shocked_cov_values,
            index=self.base_covariance_matrix.index,
            columns=self.base_covariance_matrix.columns
        )
        
        return shocked_returns, shocked_covariance

    @staticmethod
    def create_standard_scenarios() -> List[MarketShock]:
        """
        Create a list of standard stress scenarios.
        
        These scenarios are based on historical market events and
        common stress testing practices.
        
        Returns
        -------
        List[MarketShock]
            List of predefined market shock scenarios
        """
        return [
            MarketShock(
                name="2008-Style Financial Crisis",
                return_shock=-0.40,  # 40% return drop
                volatility_shock=2.0,  # Volatility doubles
                correlation_shock=0.3  # Correlations increase (flight to quality)
            ),
            MarketShock(
                name="Tech Sector Crash",
                return_shock=-0.25,  # 25% return drop
                volatility_shock=1.5,  # 50% volatility increase
                correlation_shock=0.15  # Moderate correlation increase
            ),
            MarketShock(
                name="Inflation Spike",
                return_shock=-0.15,  # 15% return drop
                volatility_shock=1.3,  # 30% volatility increase
                correlation_shock=0.1  # Slight correlation increase
            ),
            MarketShock(
                name="Mild Recession",
                return_shock=-0.10,  # 10% return drop
                volatility_shock=1.2,  # 20% volatility increase
                correlation_shock=0.05  # Small correlation increase
            ),
            MarketShock(
                name="Market Boom",
                return_shock=0.20,  # 20% return increase
                volatility_shock=0.8,  # 20% volatility decrease
                correlation_shock=-0.1  # Correlations decrease (diversification returns)
            )
        ]
    
    def apply_multiple_scenarios(
        self,
        scenarios: Optional[List[MarketShock]] = None
    ) -> Dict[str, Tuple[pd.Series, pd.DataFrame]]:
        """
        Apply multiple scenarios and return all results.
        
        Parameters
        ----------
        scenarios : List[MarketShock], optional
            List of scenarios to apply. If None, uses standard scenarios.
            
        Returns
        -------
        Dict[str, Tuple[pd.Series, pd.DataFrame]]
            Dictionary mapping scenario names to (returns, covariance) tuples
        """
        if scenarios is None:
            scenarios = self.create_standard_scenarios()
            
        results = {}
        for scenario in scenarios:
            results[scenario.name] = self.apply_scenario(scenario)
            
        return results