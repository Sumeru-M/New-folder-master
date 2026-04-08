"""
allocation_scorer.py — Allocation Intelligence Layer

Takes an AllocationResult (from any optimizer) and enriches it with:

1. Allocation Health Score (0–100)
   Composite score across 6 dimensions:
     - Diversification (Effective N vs theoretical max)
     - Concentration (HHI penalty)
     - Risk Balance (how evenly risk is distributed)
     - Factor Imbalance (excessive single-factor loading)
     - Tracking Error vs benchmark (if applicable)
     - Liquidity (weight vs ADV cap)

2. Overweight / Underweight Flags
   Compare optimised weights vs benchmark or equal-weight reference.
   Flag significant deviations.

3. Rebalance Actions (in Rupees)
   Given current holdings and optimised target weights:
       Buy/Sell rupee amounts = (target_weight - current_weight) × portfolio_value

4. Diagnostics
   - Concentration risk score
   - Diversification ratio
   - Factor imbalance warnings
   - Comparison: current portfolio vs optimised

All outputs are JSON-ready and feed directly into the frontend payload.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from .optimization_engine import AllocationResult
from .risk_contribution import (
    compute_risk_contributions,
    compute_diversification_ratio,
    compute_concentration_metrics,
    build_risk_attribution_report,
)


# ---------------------------------------------------------------------------
# Scoring sub-functions (each returns 0–100)
# ---------------------------------------------------------------------------

def _score_diversification(weights: pd.Series, covariance_matrix: pd.DataFrame) -> float:
    """
    Score based on diversification ratio.
    DR_min=1 (single asset) → 0 points
    DR_max≈sqrt(N) (uncorrelated equal-weight) → 100 points
    """
    N = len(weights)
    dr = compute_diversification_ratio(weights, covariance_matrix)
    dr_max = np.sqrt(N)  # theoretical upper bound (uncorrelated equal-weight)
    dr_min = 1.0
    if dr_max <= dr_min:
        return 50.0
    score = (dr - dr_min) / (dr_max - dr_min) * 100
    return float(np.clip(score, 0, 100))


def _score_concentration(weights: pd.Series) -> float:
    """
    Score penalising concentration (HHI).
    HHI=1/N (equal weight) → 100 points
    HHI=1 (single asset)   → 0 points
    """
    N = len(weights)
    hhi = float(np.sum(weights.values ** 2))
    hhi_equal = 1.0 / N
    # Linear penalty from equal-weight HHI to 1
    score = (1.0 - hhi) / (1.0 - hhi_equal + 1e-9) * 100
    return float(np.clip(score, 0, 100))


def _score_risk_balance(weights: pd.Series, covariance_matrix: pd.DataFrame) -> float:
    """
    Score based on how evenly risk is distributed.
    Equal %RC (1/N each) → 100 points.
    Single asset carries all risk → 0 points.

    Measure: 1 - (std of %RC) / (std of uniform distribution).
    """
    N = len(weights)
    try:
        rc_df = compute_risk_contributions(weights, covariance_matrix)
        pct_rc = rc_df["pct_rc"].values
    except Exception:
        return 50.0

    # Std of a uniform distribution on N assets where each is 1/N
    # = 0 (all equal). Real std relative to maximum possible std.
    std_rc = float(np.std(pct_rc))
    # Maximum std: one asset carries all risk → std = sqrt((N-1)/N²)
    max_std = float(np.sqrt((N - 1) / N ** 2)) if N > 1 else 1.0
    if max_std == 0:
        return 100.0
    score = (1.0 - std_rc / max_std) * 100
    return float(np.clip(score, 0, 100))


def _score_factor_balance(
    factor_exposure: Dict[str, float],
    max_single_factor_share: float = 0.50,
) -> float:
    """
    Score penalising excessive single-factor concentration.
    If any factor explains > max_single_factor_share of risk → low score.
    """
    if not factor_exposure:
        return 75.0  # neutral when no factor data

    # Exclude idiosyncratic
    systematic = {k: v for k, v in factor_exposure.items() if k != "Idiosyncratic"}
    if not systematic:
        return 75.0

    max_factor_share = max(systematic.values())
    # Score 100 if max_factor_share <= 1/K (equal), 0 if >= max_single_factor_share
    K = len(systematic)
    ideal = 1.0 / K
    if max_single_factor_share <= ideal:
        return 100.0
    score = (1.0 - (max_factor_share - ideal) / (max_single_factor_share - ideal)) * 100
    return float(np.clip(score, 0, 100))


def _score_liquidity(
    weights: pd.Series,
    liquidity_caps: Optional[np.ndarray] = None,
) -> float:
    """
    Score based on how close weights are to their liquidity caps.
    All weights << caps → 100.
    Any weight at cap → reduced score.
    """
    if liquidity_caps is None:
        return 100.0

    w = weights.values
    caps = np.minimum(liquidity_caps, 1.0)
    # Ratio of weight to cap for each asset
    utilisation = w / np.maximum(caps, 1e-9)
    # Mean utilisation: 0=perfectly liquid, 1=at cap
    avg_util = float(np.mean(utilisation))
    score = (1.0 - avg_util) * 100
    return float(np.clip(score, 0, 100))


def _score_tracking_error(
    te: Optional[float] = None,
    te_target: float = 0.05,
) -> float:
    """
    Score based on tracking error vs target.
    TE = 0 → 100 (index hugger).
    TE at target → 70 (acceptable active management).
    TE = 2× target → 0.
    No TE provided → neutral 80.
    """
    if te is None:
        return 80.0
    if te <= te_target:
        return float(100 - (te / te_target) * 30)
    excess = te - te_target
    score = 70.0 - (excess / te_target) * 70.0
    return float(np.clip(score, 0, 100))


# ---------------------------------------------------------------------------
# Composite health score
# ---------------------------------------------------------------------------

SCORE_WEIGHTS = {
    "diversification":  0.25,
    "concentration":    0.20,
    "risk_balance":     0.25,
    "factor_balance":   0.15,
    "liquidity":        0.10,
    "tracking_error":   0.05,
}


def compute_health_score(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_contribution: Dict[str, float],
    factor_exposure: Dict[str, float],
    liquidity_caps: Optional[np.ndarray] = None,
    tracking_error: Optional[float] = None,
    te_target: float = 0.05,
) -> Tuple[int, Dict[str, float]]:
    """
    Compute composite allocation health score (0–100).

    Returns
    -------
    (health_score: int, sub_scores: Dict[str, float])
        health_score: integer 0–100
        sub_scores: individual dimension scores for diagnostics
    """
    sub_scores = {
        "diversification": _score_diversification(weights, covariance_matrix),
        "concentration":   _score_concentration(weights),
        "risk_balance":    _score_risk_balance(weights, covariance_matrix),
        "factor_balance":  _score_factor_balance(factor_exposure),
        "liquidity":       _score_liquidity(weights, liquidity_caps),
        "tracking_error":  _score_tracking_error(tracking_error, te_target),
    }

    composite = sum(
        SCORE_WEIGHTS[k] * v for k, v in sub_scores.items()
    )
    return int(round(composite)), sub_scores


# ---------------------------------------------------------------------------
# Overweight / Underweight flags
# ---------------------------------------------------------------------------

def compute_overweight_flags(
    weights: pd.Series,
    reference_weights: Optional[pd.Series] = None,
    ow_threshold: float = 0.05,
    uw_threshold: float = 0.03,
) -> Dict[str, str]:
    """
    Flag assets as overweight (OW), underweight (UW), or neutral.

    Compares optimised weights against a reference (benchmark or equal-weight).

    Parameters
    ----------
    weights : pd.Series
        Optimised weights.
    reference_weights : pd.Series, optional
        Benchmark or previous weights. If None, uses equal-weight.
    ow_threshold : float
        Deviation above reference to flag as OW (e.g. 0.05 = 5pp).
    uw_threshold : float
        Deviation below reference to flag as UW (e.g. 0.03 = 3pp).

    Returns
    -------
    Dict[str, str]  — {ticker: "OW" | "UW" | "Neutral"}
    """
    if reference_weights is None:
        N = len(weights)
        reference_weights = pd.Series(1.0 / N, index=weights.index)

    flags: Dict[str, str] = {}
    for ticker in weights.index:
        w_opt = weights.get(ticker, 0.0)
        w_ref = reference_weights.get(ticker, 0.0)
        diff = w_opt - w_ref

        if diff >= ow_threshold:
            flags[ticker] = f"OW +{diff:.1%}"
        elif diff <= -uw_threshold:
            flags[ticker] = f"UW {diff:.1%}"
        else:
            flags[ticker] = "Neutral"

    return flags


# ---------------------------------------------------------------------------
# Rebalance actions
# ---------------------------------------------------------------------------

def compute_rebalance_actions(
    current_weights: pd.Series,
    target_weights: pd.Series,
    portfolio_value: float,
) -> Dict[str, float]:
    """
    Compute buy/sell amounts in rupees.

    rebalance_action_i = (target_weight_i - current_weight_i) × portfolio_value

    Positive = BUY, Negative = SELL.

    Parameters
    ----------
    current_weights : pd.Series
        Current portfolio weights (before rebalance).
    target_weights : pd.Series
        Optimised target weights.
    portfolio_value : float
        Total portfolio value in INR.

    Returns
    -------
    Dict[str, float]  — {ticker: rupee_amount}
    """
    all_tickers = target_weights.index.union(current_weights.index)
    actions: Dict[str, float] = {}

    for ticker in all_tickers:
        target = target_weights.get(ticker, 0.0)
        current = current_weights.get(ticker, 0.0)
        delta = (target - current) * portfolio_value
        if abs(delta) > 1.0:  # ignore sub-rupee rounding noise
            actions[ticker] = round(delta, 2)

    return actions


# ---------------------------------------------------------------------------
# Portfolio comparison diagnostics
# ---------------------------------------------------------------------------

def compare_portfolios(
    current_weights: pd.Series,
    optimised_weights: pd.Series,
    covariance_matrix: pd.DataFrame,
    mu: pd.Series,
    rf: float = 0.07,
) -> Dict:
    """
    Side-by-side comparison of current vs optimised portfolio.

    Returns dict with metrics for both portfolios and the improvement.
    """
    def _metrics(w: pd.Series) -> Dict:
        w_arr = w.reindex(mu.index).fillna(0).values
        mu_arr = mu.values
        Sigma_arr = covariance_matrix.values

        ret = float(w_arr @ mu_arr)
        vol = float(np.sqrt(w_arr @ Sigma_arr @ w_arr))
        sharpe = (ret - rf) / vol if vol > 0 else 0.0

        conc = compute_concentration_metrics(w)
        dr = compute_diversification_ratio(w, covariance_matrix)
        return {
            "expected_return": round(ret, 4),
            "volatility": round(vol, 4),
            "sharpe_ratio": round(sharpe, 4),
            "herfindahl_index": round(conc.get("herfindahl_index", 0), 4),
            "effective_n": round(conc.get("effective_n", 0), 2),
            "diversification_ratio": round(dr, 4),
        }

    current_metrics = _metrics(current_weights)
    optimised_metrics = _metrics(optimised_weights)

    improvement = {
        k: round(optimised_metrics[k] - current_metrics[k], 4)
        for k in current_metrics
    }

    # Turnover
    all_tickers = current_weights.index.union(optimised_weights.index)
    turnover = float(sum(
        abs(optimised_weights.get(t, 0) - current_weights.get(t, 0))
        for t in all_tickers
    ))

    return {
        "current": current_metrics,
        "optimised": optimised_metrics,
        "improvement": improvement,
        "one_way_turnover": round(turnover, 4),
    }


# ---------------------------------------------------------------------------
# Main enrichment function
# ---------------------------------------------------------------------------

def enrich_allocation_result(
    result: AllocationResult,
    covariance_matrix: pd.DataFrame,
    mu: pd.Series,
    current_weights: Optional[pd.Series] = None,
    portfolio_value: float = 1_000_000.0,
    benchmark_weights: Optional[pd.Series] = None,
    factor_betas: Optional[pd.DataFrame] = None,
    factor_covariance: Optional[pd.DataFrame] = None,
    idiosyncratic_vols: Optional[pd.Series] = None,
    liquidity_caps: Optional[np.ndarray] = None,
    tracking_error: Optional[float] = None,
    ow_threshold: float = 0.05,
    uw_threshold: float = 0.03,
    rf: float = 0.07,
) -> AllocationResult:
    """
    Enrich an AllocationResult with the full intelligence layer output.

    This is the single function that converts a raw optimizer output
    into the complete frontend-ready JSON payload.

    Parameters
    ----------
    result : AllocationResult
        Raw output from any optimizer.
    covariance_matrix : pd.DataFrame
        Annualised covariance (aligned with result.weights).
    mu : pd.Series
        Expected returns (aligned with result.weights).
    current_weights : pd.Series, optional
        Current holdings weights (for rebalance actions + comparison).
    portfolio_value : float
        Total portfolio value in INR.
    benchmark_weights : pd.Series, optional
        Benchmark weights for OW/UW flags. Uses equal-weight if None.
    factor_betas, factor_covariance, idiosyncratic_vols : optional
        Required for factor risk attribution.
    liquidity_caps : np.ndarray, optional
        Per-asset max weight from liquidity constraint.
    tracking_error : float, optional
        Pre-computed ex-ante TE (if available).
    ow_threshold, uw_threshold : float
        Deviation thresholds for OW/UW flagging.

    Returns
    -------
    AllocationResult
        Same object, with all intelligence fields populated in-place.
    """
    weights = result.weights
    tickers = weights.index.tolist()

    # 1. Risk attribution
    risk_report = build_risk_attribution_report(
        weights=weights,
        covariance_matrix=covariance_matrix,
        factor_betas=factor_betas,
        factor_covariance=factor_covariance,
        idiosyncratic_vols=idiosyncratic_vols,
    )
    result.risk_contribution = risk_report["risk_contribution"]
    result.factor_exposure = risk_report["factor_exposure"]

    # 2. Health score
    health_score, sub_scores = compute_health_score(
        weights=weights,
        covariance_matrix=covariance_matrix,
        risk_contribution=result.risk_contribution,
        factor_exposure=result.factor_exposure,
        liquidity_caps=liquidity_caps,
        tracking_error=tracking_error,
    )
    result.allocation_health_score = health_score

    # 3. OW/UW flags
    result.overweight_underweight_flags = compute_overweight_flags(
        weights=weights,
        reference_weights=benchmark_weights,
        ow_threshold=ow_threshold,
        uw_threshold=uw_threshold,
    )

    # 4. Rebalance actions
    if current_weights is not None:
        result.rebalance_actions_rupees = compute_rebalance_actions(
            current_weights=current_weights,
            target_weights=weights,
            portfolio_value=portfolio_value,
        )

    # 5. Diagnostics
    comparison = {}
    if current_weights is not None:
        comparison = compare_portfolios(
            current_weights=current_weights,
            optimised_weights=weights,
            covariance_matrix=covariance_matrix,
            mu=mu,
            rf=rf,
        )

    result.diagnostics = {
        "health_score_breakdown": sub_scores,
        "concentration": risk_report["concentration"],
        "diversification_ratio": risk_report["diversification_ratio"],
        "factor_imbalance_warning": _factor_imbalance_warning(result.factor_exposure),
        "portfolio_comparison": comparison,
    }

    return result


def _factor_imbalance_warning(factor_exposure: Dict[str, float]) -> Optional[str]:
    """Return a warning string if any single factor dominates risk."""
    if not factor_exposure:
        return None

    systematic = {k: v for k, v in factor_exposure.items() if k != "Idiosyncratic"}
    if not systematic:
        return None

    dominant_factor = max(systematic, key=systematic.get)
    dominant_share = systematic[dominant_factor]

    if dominant_share > 0.40:
        return (
            f"WARNING: '{dominant_factor}' factor explains {dominant_share:.1%} of "
            f"portfolio risk. Consider reducing exposure to this factor."
        )
    return None
