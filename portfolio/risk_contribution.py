"""
risk_contribution.py — Risk & Factor Contribution Engine

Computes:
- Marginal risk contribution per asset
- Percentage risk contribution per asset
- Factor risk contribution (systematic vs idiosyncratic, per factor)
- Diversification ratio

Wraps existing risk_metrics.compute_component_var and factor_model.FactorModel.
All outputs are JSON-serialisable dicts keyed by ticker.

Mathematical basis
------------------
For a portfolio with weight vector w and covariance Σ:

    σ_p = sqrt(wᵀΣw)

Marginal risk contribution (MRC):
    MRC_i = (Σw)_i / σ_p

Component risk contribution (CRC):
    CRC_i = w_i × MRC_i

Percentage risk contribution (%RC):
    %RC_i = CRC_i / σ_p  (sums to 1)

Factor risk contribution (Barra decomposition):
    Σ ≈ B Σ_f Bᵀ + D
    Factor component : B Σ_f Bᵀ
    Specific component: D
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Asset-level risk contribution
# ---------------------------------------------------------------------------

def compute_risk_contributions(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
    annualised: bool = True,
) -> pd.DataFrame:
    """
    Compute marginal, component, and percentage risk contributions per asset.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights indexed by ticker. Must sum to ~1.
    covariance_matrix : pd.DataFrame
        Annualised covariance matrix (must share index with weights).
    annualised : bool, default True
        If False, covariance is assumed daily; multiply output by sqrt(252).

    Returns
    -------
    pd.DataFrame
        Columns: weight, marginal_rc, component_rc, pct_rc
        Indexed by ticker.
    """
    # Align
    tickers = weights.index.tolist()
    w = weights.reindex(tickers).values
    Sigma = covariance_matrix.reindex(index=tickers, columns=tickers).values

    port_var = w @ Sigma @ w
    if port_var <= 0:
        raise ValueError("Portfolio variance is non-positive — check covariance matrix.")
    port_std = np.sqrt(port_var)

    # Marginal risk contribution: ∂σ_p/∂w_i = (Σw)_i / σ_p
    Sigma_w = Sigma @ w
    marginal_rc = Sigma_w / port_std          # shape (N,)

    # Component risk contribution: w_i × MRC_i
    component_rc = w * marginal_rc            # shape (N,)

    # Percentage risk contribution: CRC_i / σ_p  (sums to 1)
    pct_rc = component_rc / port_std          # shape (N,)

    return pd.DataFrame(
        {
            "weight": w,
            "marginal_rc": marginal_rc,
            "component_rc": component_rc,
            "pct_rc": pct_rc,
        },
        index=tickers,
    )


def compute_risk_contribution_dict(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> Dict[str, float]:
    """
    Return percentage risk contribution as a plain dict (JSON-ready).

    Keys are tickers, values are fractional contributions (sum to 1.0).
    """
    rc = compute_risk_contributions(weights, covariance_matrix)
    return rc["pct_rc"].to_dict()


# ---------------------------------------------------------------------------
# Factor-level risk contribution
# ---------------------------------------------------------------------------

def compute_factor_risk_contributions(
    weights: pd.Series,
    factor_betas: pd.DataFrame,
    factor_covariance: pd.DataFrame,
    idiosyncratic_vols: pd.Series,
) -> Dict[str, float]:
    """
    Decompose portfolio risk into per-factor and idiosyncratic contributions.

    Uses Barra-style decomposition:
        Σ_assets ≈ B Σ_f Bᵀ + D
        where B = (N × K) factor loading matrix
              Σ_f = (K × K) factor covariance (annualised)
              D = diag(σ_ε²) idiosyncratic variance matrix

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights (N,).
    factor_betas : pd.DataFrame
        Shape (N × K). Rows = assets, columns = factors.
        Produced by FactorModel.fit_asset(). Index must match weights.
    factor_covariance : pd.DataFrame
        Shape (K × K). Annualised factor return covariance.
        Pass self.factors.cov() * 252 from FactorModel.
    idiosyncratic_vols : pd.Series
        Annualised idiosyncratic volatility per asset.
        Pass FactorAnalysisResult.idiosyncratic_risk for each asset.

    Returns
    -------
    Dict[str, float]
        Keys: factor names + "Idiosyncratic"
        Values: fraction of total portfolio variance explained (sum ≈ 1).
    """
    tickers = weights.index.tolist()
    w = weights.reindex(tickers).values

    # Align betas
    B = factor_betas.reindex(index=tickers).values            # (N, K)
    Sigma_f = factor_covariance.values                         # (K, K)
    sigma_eps = idiosyncratic_vols.reindex(tickers).values     # (N,)

    # Idiosyncratic variance matrix (diagonal)
    D = np.diag(sigma_eps ** 2)                                # (N, N)

    # Portfolio factor loadings: h = Bᵀ w  (K,)
    h = B.T @ w

    # Systematic variance per factor: h_k² × Σ_f_kk (diagonal contribution)
    # Full systematic variance = hᵀ Σ_f h
    # Factor-specific contribution = h_k × (Σ_f h)_k
    Sigma_f_h = Sigma_f @ h                                    # (K,)
    factor_variances = h * Sigma_f_h                           # (K,) — sum = hᵀΣ_f h

    # Idiosyncratic variance: wᵀ D w
    idio_variance = w @ D @ w

    total_variance = np.sum(factor_variances) + idio_variance

    if total_variance <= 0:
        return {"Idiosyncratic": 1.0}

    result: Dict[str, float] = {}
    for k, factor_name in enumerate(factor_covariance.columns):
        result[factor_name] = float(factor_variances[k] / total_variance)

    result["Idiosyncratic"] = float(idio_variance / total_variance)
    return result


# ---------------------------------------------------------------------------
# Diversification ratio
# ---------------------------------------------------------------------------

def compute_diversification_ratio(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> float:
    """
    Diversification Ratio = (wᵀ σ) / σ_p

    Numerator: weighted average of individual asset volatilities.
    Denominator: portfolio volatility.

    DR = 1 for a single-asset portfolio (no diversification).
    DR > 1 indicates diversification benefit.
    DR near sqrt(N) for equal-weighted uncorrelated assets.

    Parameters
    ----------
    weights : pd.Series
    covariance_matrix : pd.DataFrame

    Returns
    -------
    float
        Diversification ratio (>= 1).
    """
    tickers = weights.index.tolist()
    w = weights.reindex(tickers).values
    Sigma = covariance_matrix.reindex(index=tickers, columns=tickers).values

    # Individual volatilities
    sigma_i = np.sqrt(np.diag(Sigma))

    # Weighted average vol (numerator)
    weighted_avg_vol = w @ sigma_i

    # Portfolio vol (denominator)
    port_vol = np.sqrt(w @ Sigma @ w)

    if port_vol <= 0:
        return 1.0

    return float(weighted_avg_vol / port_vol)


# ---------------------------------------------------------------------------
# Concentration metrics
# ---------------------------------------------------------------------------

def compute_concentration_metrics(weights: pd.Series) -> Dict[str, float]:
    """
    Compute portfolio concentration metrics.

    Returns
    -------
    Dict with:
        herfindahl_index : sum of squared weights (0 = perfect spread, 1 = single asset)
        effective_n      : 1 / HHI — effective number of positions
        entropy          : Shannon entropy of weight distribution
        max_weight       : largest single position
        gini_coefficient : Gini coefficient of weight distribution
    """
    w = weights.values
    w = np.maximum(w, 0)
    if w.sum() == 0:
        return {}

    w = w / w.sum()  # normalise just in case

    hhi = float(np.sum(w ** 2))
    effective_n = 1.0 / hhi if hhi > 0 else float(len(w))

    # Shannon entropy (nats)
    eps = 1e-12
    w_pos = w[w > eps]
    entropy = float(-np.sum(w_pos * np.log(w_pos)))
    max_entropy = np.log(len(w))  # uniform distribution
    normalised_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    # Gini coefficient
    w_sorted = np.sort(w)
    n = len(w_sorted)
    cumw = np.cumsum(w_sorted)
    gini = float(1 - 2 * np.sum(cumw) / (n * np.sum(w_sorted)) + 1 / n)

    return {
        "herfindahl_index": hhi,
        "effective_n": effective_n,
        "entropy": entropy,
        "normalised_entropy": normalised_entropy,
        "max_weight": float(np.max(w)),
        "gini_coefficient": max(0.0, gini),
    }


# ---------------------------------------------------------------------------
# Full risk attribution report
# ---------------------------------------------------------------------------

def build_risk_attribution_report(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
    factor_betas: Optional[pd.DataFrame] = None,
    factor_covariance: Optional[pd.DataFrame] = None,
    idiosyncratic_vols: Optional[pd.Series] = None,
) -> Dict:
    """
    Consolidated risk attribution report.

    Returns
    -------
    Dict with:
        risk_contribution   : {ticker: pct_rc}  — asset-level % risk contribution
        factor_exposure     : {factor: pct}       — factor-level % risk (if provided)
        diversification_ratio : float
        concentration       : dict of concentration metrics
    """
    # Asset-level
    rc_df = compute_risk_contributions(weights, covariance_matrix)
    risk_contribution = rc_df["pct_rc"].to_dict()

    # Factor-level (optional)
    factor_exposure: Dict = {}
    if (
        factor_betas is not None
        and factor_covariance is not None
        and idiosyncratic_vols is not None
    ):
        factor_exposure = compute_factor_risk_contributions(
            weights, factor_betas, factor_covariance, idiosyncratic_vols
        )

    dr = compute_diversification_ratio(weights, covariance_matrix)
    concentration = compute_concentration_metrics(weights)

    return {
        "risk_contribution": risk_contribution,
        "factor_exposure": factor_exposure,
        "diversification_ratio": dr,
        "concentration": concentration,
    }
