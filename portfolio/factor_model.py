"""
Factor Model Module

Decomposes portfolio risk into systematic (factor) risk and idiosyncratic (specific) risk
using regression analysis. Supports rolling beta calculations.
"""

from typing import Dict, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass
import statsmodels.api as sm

@dataclass
class FactorAnalysisResult:
    """Stores results of factor analysis."""
    betas: pd.Series            # Factor sensitivities (Beta)
    r_squared: float            # Model R-squared
    alpha: float                # Jensen's Alpha (annualized)
    systematic_risk: float      # Portion of volatility explained by factors
    idiosyncratic_risk: float   # Portion of volatility unexplained
    total_risk: float           # Total volatility
    factor_returns: pd.Series   # Return contribution from factors

class FactorModel:
    """
    Implements factor-based risk models (CAPM, Multi-Factor).
    """
    
    def __init__(self, factor_data: pd.DataFrame):
        """
        Initialize with factor returns data.
        
        Parameters:
        - factor_data: DataFrame where columns are factors (e.g. 'Market', 'SMB', 'HML')
                       and index is datetime. Must be aligned with asset returns.
        """
        self.factors = factor_data
        self.factor_names = factor_data.columns.tolist()
        
    def fit_asset(self, asset_returns: pd.Series) -> FactorAnalysisResult:
        """
        Fit factor model to a single asset's returns.
        """
        # Align data
        df = pd.concat([asset_returns, self.factors], axis=1).dropna()
        
        if len(df) < 30:
            raise ValueError("Insufficient data for factor analysis (need > 30 points)")
            
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
        X = sm.add_constant(X)
        
        model = sm.OLS(y, X).fit()
        
        # Extract metrics
        alpha = model.params['const'] * 252 # Annualize alpha
        betas = model.params.drop('const')
        r2 = model.rsquared
        
        # Risk Decomposition (Annualized)
        total_risk = y.std() * np.sqrt(252)
        systematic_risk = total_risk * np.sqrt(r2)
        idiosyncratic_risk = total_risk * np.sqrt(1 - r2)
        
        # Factor contributions to return
        # (Average Factor Return * Beta)
        factor_contrib = betas * self.factors.mean() * 252
        
        return FactorAnalysisResult(
            betas=betas,
            r_squared=r2,
            alpha=alpha,
            systematic_risk=systematic_risk,
            idiosyncratic_risk=idiosyncratic_risk,
            total_risk=total_risk,
            factor_returns=factor_contrib
        )
    
    def calculate_rolling_betas(self, asset_returns: pd.Series, window: int = 60) -> pd.DataFrame:
        """
        Compute rolling betas for an asset.
        """
        # Align
        df = pd.concat([asset_returns, self.factors], axis=1).dropna()
        y = df.iloc[:, 0]
        X = df.iloc[:, 1:]
        
        # Rolling regression using RollingOLS
        # wrapper for statsmodels RollingOLS
        try:
            from statsmodels.regression.rolling import RollingOLS
            rolling_model = RollingOLS(y, sm.add_constant(X), window=window)
            results = rolling_model.fit()
            return results.params.drop('const', axis=1)
        except ImportError:
            # Fallback using pure numpy least squares
            betas = []
            for start in range(len(df) - window + 1):
                end = start + window
                y_sub = y.iloc[start:end].values
                X_sub = X.iloc[start:end].values
                coef, _, _, _ = np.linalg.lstsq(X_sub, y_sub, rcond=None)
                betas.append(coef)
            
            return pd.DataFrame(betas, index=df.index[window-1:], columns=X.columns)

    def decompose_portfolio_risk(self, weights: pd.Series, asset_returns: pd.DataFrame) -> Dict[str, float]:
        """
        Decompose TOTAL PORTFOLIO risk into Factor vs Specific.
        
        Formula:
        Sigma_port = w.T * (Sigma_assets) * w
        Sigma_assets approx = B * Sigma_factors * B.T + D
        
        Where D is diagonal matrix of idiosyncratic variances.
        """
        # 1. Fit model for each asset to get Betas and Residual Vars
        betas_list = []
        resid_vars = []
        
        for asset in weights.index:
            if asset not in asset_returns.columns:
                continue
            
            res = self.fit_asset(asset_returns[asset])
            betas_list.append(res.betas)
            # Idiosyncratic variance (daily)
            resid_vars.append((res.idiosyncratic_risk / np.sqrt(252))**2)
            
        B = pd.DataFrame(betas_list, index=weights.index) # N x K
        D = np.diag(resid_vars) # N x N diagonal
        
        # Factor Covariance (Daily)
        Sigma_f = self.factors.cov()
        
        # Portfolio loadings
        # w_betas = w.T * B (1 x K)
        w = weights.values
        port_betas = np.dot(w, B)
        
        # Systemic Variance = beta_p * Sigma_f * beta_p.T
        syst_var = np.dot(np.dot(port_betas, Sigma_f), port_betas.T)
        
        # Idiosyncratic Variance = w.T * D * w
        spec_var = np.dot(np.dot(w, D), w.T)
        
        total_var = syst_var + spec_var
        
        return {
            "systematic_volatility": np.sqrt(syst_var * 252),
            "idiosyncratic_volatility": np.sqrt(spec_var * 252),
            "total_volatility_model": np.sqrt(total_var * 252),
            "r_squared": syst_var / total_var
        }