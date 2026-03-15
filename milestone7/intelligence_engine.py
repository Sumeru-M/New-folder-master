"""
intelligence_engine.py — Regime-Switching Adaptive Allocation Master Engine
============================================================================

This module is the single public entry point for Milestone 7.

Orchestration sequence
----------------------

1. REGIME DETECTION (regime_model.py)
   - Build 4-feature observation matrix from daily returns
   - Fit K=4 Gaussian HMM via Baum–Welch EM (3 random restarts)
   - Decode Viterbi path; compute filtered + smoothed state probabilities
   - Output: current regime, γ_T, transition matrix A

2. VOLATILITY MODELLING (state_space_filter.py)
   - Fit GARCH(1,1) to equal-weight portfolio returns
   - Compute 30-day h-step variance forecast (mean-reverting)
   - Bootstrap 95% confidence intervals
   - Output: σ_T (current conditional vol), σ_forecast[1:30]

3. TRANSITION ANALYSIS (transition_matrix.py)
   - Compute stationary distribution π via principal eigenvector
   - Compute A^h for h ∈ {1, 5, 10, 21, 63}
   - Compute expected durations, mixing time, row entropy
   - Output: forward regime probabilities P(S_{T+h})

4. FORWARD DISTRIBUTION (forward_risk_forecast.py)
   - Build Gaussian mixture distribution at H ∈ {21, 63} days
   - Compute E[r_H], Var[r_H] (law of total variance)
   - Compute VaR/CVaR via numerical root-finding on mixture CDF
   - Apply GARCH volatility adjustment
   - Output: ForwardRiskReport at 1M and 3M horizons

5. ADAPTIVE PARAMETERS (adaptive_allocator.py)
   - Blend regime-specific parameter sets via p_k weights
   - Apply entropy-based shrinkage toward ergodic params
   - Compute position scale from Kelly vol targeting
   - Output: AdaptiveParameters (λ, max_weight, rf, etc.)

6. OPTIMIZATION (optimization_engine.run_optimizer) [M5 black-box]
   - Build ConstraintBuilder with adapted bounds
   - Call run_optimizer(method, mu, Sigma, tickers, cb, **adaptive_kwargs)
   - Output: AllocationResult (weights, Sharpe, CVaR, etc.)

7. UNCERTAINTY QUANTIFICATION (uncertainty_quantifier.py)
   - Bootstrap CI on regime probabilities (B=300)
   - Transition matrix asymptotic standard errors
   - Forecast return confidence intervals (delta method)
   - Weight sensitivity to regime shifts (finite differences)
   - Effective sample size
   - Output: UncertaintyReport

8. ASSEMBLE FULL REPORT
   - All outputs merged into a structured dict
   - No look-ahead: only filtered (causal) quantities used for decisions

Isolation from M1–M6
--------------------
- optimization_engine.run_optimizer() called as black-box (import only)
- constraints.ConstraintBuilder used as black-box
- No modification of any M1–M6 file
- All M7 state is local to this module's return value
- M6 (crypto/virtual trade) is entirely independent; not called here

Design notes
------------
- All numpy; no heuristics; every parameter choice is model-derived
- Graceful degradation: if optimization engine unavailable, returns
  analytical equal-risk-parity weights with a warning
- Verbosity controls: quiet=False prints structured progress log
"""

from __future__ import annotations

import sys
import os
import time
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Internal imports (M7 modules only)
# ---------------------------------------------------------------------------
from milestone7.regime_model import (
    run_hmm, RegimeOutput, STATE_LABELS,
)
from milestone7.state_space_filter import (
    fit_portfolio_garch, VolatilityForecast,
)
from milestone7.transition_matrix import (
    analyse_transitions, TransitionAnalysis,
)
from milestone7.forward_risk_forecast import (
    compute_forward_risk, ForwardRiskReport,
)
from milestone7.adaptive_allocator import (
    compute_adaptive_parameters, build_adapted_constraints, AdaptiveParameters,
)
from milestone7.uncertainty_quantifier import (
    quantify_uncertainty, UncertaintyReport,
)


# ---------------------------------------------------------------------------
# Data class: full intelligence report
# ---------------------------------------------------------------------------

@dataclass
class IntelligenceReport:
    """
    Complete output of run_adaptive_intelligence().

    All sub-reports are structured dicts (serialisable to JSON).
    The .weights property returns the final recommended portfolio weights.
    """
    regime_output:        RegimeOutput
    garch_forecast:       VolatilityForecast
    transition_analysis:  TransitionAnalysis
    forward_report:       ForwardRiskReport
    adaptive_params:      AdaptiveParameters
    uncertainty_report:   UncertaintyReport
    allocation_result:    Optional[Any]   # AllocationResult from M5
    tickers:              List[str]
    elapsed_seconds:      float

    @property
    def weights(self) -> np.ndarray:
        if self.allocation_result is not None:
            return self.allocation_result.weights.values
        return np.ones(len(self.tickers)) / len(self.tickers)

    def to_dict(self) -> Dict:
        """
        Full structured output matching the specification:
        {
          "regime_probabilities":      {...},
          "transition_matrix":         {...},
          "forward_return_forecast":   {...},
          "volatility_forecast":       {...},
          "adaptive_parameter_shift":  {...},
          "uncertainty_metrics":       {...},
          "optimal_weights":           {...},
          "meta":                      {...},
        }
        """
        from milestone7.regime_model import STATE_LABELS

        alloc_dict = {}
        if self.allocation_result is not None:
            try:
                alloc_dict = {
                    "weights": {
                        t: round(float(w), 6)
                        for t, w in self.allocation_result.weights.items()
                    },
                    "expected_return_ann":  round(float(self.allocation_result.expected_return), 6),
                    "volatility_ann":       round(float(self.allocation_result.volatility), 6),
                    "sharpe_ratio":         round(float(self.allocation_result.sharpe_ratio), 6),
                    "cvar_95":              round(float(self.allocation_result.cvar_95), 6),
                    "optimization_type":    self.allocation_result.optimization_type,
                    "solve_status":         self.allocation_result.solve_status,
                    "position_scale":       round(float(self.adaptive_params.position_scale), 6),
                }
            except Exception as e:
                alloc_dict = {"error": str(e)}

        return {
            "regime_probabilities":    self.regime_output.to_dict(),
            "transition_matrix":       self.transition_analysis.to_dict(),
            "forward_return_forecast": self.forward_report.to_dict(),
            "volatility_forecast":     self.garch_forecast.to_dict(),
            "adaptive_parameter_shift":self.adaptive_params.to_dict(),
            "uncertainty_metrics":     self.uncertainty_report.to_dict(),
            "optimal_allocation":      alloc_dict,
            "meta": {
                "tickers":                  self.tickers,
                "n_assets":                 len(self.tickers),
                "elapsed_seconds":          round(self.elapsed_seconds, 2),
                "milestone":                7,
                "model":                    "HMM(K=4) + GARCH(1,1) + Gaussian Mixture Forecast",
                "look_ahead_bias":          False,
                "optimizer_isolation":      "M5 optimization_engine called as black-box",
                "m1_m6_isolation":          "No M1-M6 files modified",
            },
        }


# ---------------------------------------------------------------------------
# Helper: load M5 optimizer (black-box import)
# ---------------------------------------------------------------------------

def _load_optimizer():
    """
    Attempt to import M5 optimization_engine.run_optimizer as a black box.
    Returns (run_optimizer, ConstraintBuilder) or (None, None) on failure.
    """
    try:
        m5_path = os.path.join(os.path.dirname(__file__), "..")
        if m5_path not in sys.path:
            sys.path.insert(0, m5_path)
        from optimization_engine import run_optimizer
        return run_optimizer
    except Exception as e:
        warnings.warn(f"M5 optimizer unavailable: {e}. Using equal-weight fallback.")
        return None


def _fallback_equal_risk_parity(
    mu:     np.ndarray,
    Sigma:  np.ndarray,
    tickers: List[str],
) -> Any:
    """
    Analytical equal-risk-parity fallback when M5 optimizer is unavailable.

    w ∝ 1/σ_i  (inverse-volatility weighting)
    """
    sig     = np.sqrt(np.diag(Sigma))
    sig     = np.maximum(sig, 1e-6)
    w       = (1.0 / sig)
    w      /= w.sum()

    class _FR:
        def __init__(self, w_, mu_, Sigma_, tickers_):
            import pandas as pd
            self.weights          = pd.Series(w_, index=tickers_)
            self.expected_return  = float(w_ @ mu_)
            self.volatility       = float(np.sqrt(w_ @ Sigma_ @ w_))
            self.sharpe_ratio     = (self.expected_return - 0.07) / max(self.volatility, 1e-6)
            self.cvar_95          = float(1.645 * self.volatility / np.sqrt(252))
            self.optimization_type = "equal_risk_parity_fallback"
            self.solve_status      = "fallback"

    return _FR(w, mu, Sigma, tickers)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _prepare_returns(
    prices_or_returns: pd.DataFrame,
    is_returns:        bool = False,
) -> pd.DataFrame:
    """
    Ensure we have a log-return DataFrame.
    If given prices, computes log(P_t / P_{t-1}).
    Drops NaN rows.
    """
    if is_returns:
        returns = prices_or_returns.astype(float).dropna()
    else:
        returns = np.log(prices_or_returns / prices_or_returns.shift(1)).dropna()
    return returns.astype(float)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def run_adaptive_intelligence(
    prices_or_returns:   pd.DataFrame,
    tickers:             Optional[List[str]]   = None,
    is_returns:          bool                  = False,
    horizons:            List[int]             = [21, 63],
    rf_base:             float                 = 0.07,
    sector_map:          Optional[Dict]        = None,
    existing_weights:    Optional[np.ndarray]  = None,
    hmm_restarts:        int                   = 3,
    hmm_max_iter:        int                   = 200,
    garch_n_sim:         int                   = 500,
    uncertainty_n_boot:  int                   = 300,
    quiet:               bool                  = False,
) -> IntelligenceReport:
    """
    Regime-Switching Adaptive Allocation Engine — Master Orchestrator.

    Parameters
    ----------
    prices_or_returns : (T, N) DataFrame of daily prices or log-returns.
    tickers           : asset names (default: DataFrame column names).
    is_returns        : True if DataFrame already contains log-returns.
    horizons          : forward distribution horizons (trading days).
    rf_base           : base annual risk-free rate (e.g. 0.07 for NSE).
    sector_map        : {ticker: sector} for sector cap constraints.
    existing_weights  : (N,) current weights for turnover constraint.
    hmm_restarts      : Baum–Welch random restarts.
    hmm_max_iter      : maximum EM iterations per restart.
    garch_n_sim       : bootstrap samples for GARCH forecast CI.
    uncertainty_n_boot: bootstrap samples for regime probability CI.
    quiet             : suppress progress output.

    Returns
    -------
    IntelligenceReport  (contains .to_dict() for JSON-serialisable output)
    """
    t_start = time.time()

    def _log(msg: str):
        if not quiet:
            print(f"  [M7] {msg}")

    # ── Prepare data ─────────────────────────────────────────────────────────
    _log("Preparing return data...")
    daily_returns = _prepare_returns(prices_or_returns, is_returns=is_returns)
    if tickers is None:
        tickers = list(daily_returns.columns)
    N = len(tickers)
    T = len(daily_returns)
    _log(f"  {T} trading days × {N} assets")

    if T < 60:
        raise ValueError(f"Insufficient history: {T} days. Minimum 60 required.")

    # ── Compute μ, Σ (for optimizer input) ───────────────────────────────────
    _log("Computing expected returns and covariance...")
    log_ret = daily_returns.values                                # (T, N)
    mu_daily = log_ret.mean(axis=0)
    mu_ann   = (np.exp(mu_daily * 252) - 1)                      # (N,) annualised
    Sigma_daily = np.cov(log_ret.T) if N > 1 else np.array([[log_ret.var()]])
    Sigma_ann   = Sigma_daily * 252

    # ── Step 1: HMM Regime Detection ─────────────────────────────────────────
    _log("Fitting Hidden Markov Model (K=4 regimes)...")
    t1 = time.time()
    regime_output = run_hmm(
        daily_returns = daily_returns,
        K             = 4,
        window        = 21,
        max_iter      = hmm_max_iter,
        tol           = 1e-6,
        n_restarts    = hmm_restarts,
    )
    _log(f"  HMM converged: loglik={regime_output.log_likelihood:.2f} "
         f"in {time.time()-t1:.1f}s")
    _log(f"  Current regime: {STATE_LABELS[regime_output.current_regime]} "
         f"(p={regime_output.current_probs[regime_output.current_regime]:.3f})")

    # ── Step 2: GARCH(1,1) Portfolio Volatility ───────────────────────────────
    _log("Fitting GARCH(1,1) to equal-weight portfolio returns...")
    eq_weights    = np.ones(N) / N
    garch_params, garch_forecast = fit_portfolio_garch(daily_returns, eq_weights)
    _log(f"  GARCH: α={garch_params.alpha:.4f}, β={garch_params.beta:.4f}, "
         f"persistence={garch_params.persistence:.4f}")
    _log(f"  Current vol: {garch_forecast.current_vol_ann:.2%}  "
         f"30d forecast: {garch_forecast.vol_path_ann[29]:.2%}")

    # ── Step 3: Transition Analysis ───────────────────────────────────────────
    _log("Analysing regime transition matrix...")
    trans_analysis = analyse_transitions(
        A             = regime_output.params.A,
        current_probs = regime_output.current_probs,
        horizons      = horizons + [1, 5, 10],
        crisis_state  = 2,
    )
    _log(f"  Mixing time: ~{trans_analysis.mixing_time} days")
    _log(f"  Stationary dist: " +
         " | ".join(f"{STATE_LABELS[k]}={trans_analysis.stationary_dist[k]:.3f}"
                    for k in range(4)))

    # ── Step 4: Forward Distribution ──────────────────────────────────────────
    _log("Computing forward return distributions...")
    forward_report = compute_forward_risk(
        regime_output        = regime_output,
        transition_analysis  = trans_analysis,
        garch_forecast       = garch_forecast,
        horizons             = horizons,
    )
    for fd in forward_report.horizons:
        _log(f"  {fd.horizon_days}d: E[r]={fd.expected_return:.2%}  "
             f"σ={fd.std:.4f}  CVaR95={fd.cvar_95:.4f}  "
             f"skew={fd.skewness:.2f}  kurt={fd.kurtosis:.2f}")

    # ── Step 5: Adaptive Parameters ───────────────────────────────────────────
    _log("Computing adaptive optimizer parameters...")
    adaptive_params = compute_adaptive_parameters(
        regime_probs      = regime_output.current_probs,
        stationary_dist   = trans_analysis.stationary_dist,
        garch_vol_current = garch_forecast.current_vol_ann,
        rf_base           = rf_base,
    )
    _log(f"  Dominant: {STATE_LABELS[adaptive_params.dominant_regime]}  "
         f"entropy={adaptive_params.regime_entropy:.3f}  "
         f"shrinkage={adaptive_params.shrinkage:.3f}")
    _log(f"  λ_return={adaptive_params.lam_return:.3f}  "
         f"λ_vol={adaptive_params.lam_vol:.3f}  "
         f"λ_cvar={adaptive_params.lam_cvar:.3f}  "
         f"max_w={adaptive_params.max_weight:.3f}")
    _log(f"  Method: {adaptive_params.optimization_method}  "
         f"pos_scale={adaptive_params.position_scale:.3f}")
    for note in adaptive_params.notes:
        _log(f"  ↳ {note}")

    # ── Step 6: Run M5 Optimizer (black-box) ──────────────────────────────────
    _log("Calling M5 optimization engine (black-box)...")
    run_optimizer = _load_optimizer()
    allocation_result = None

    if run_optimizer is not None:
        try:
            cb = build_adapted_constraints(
                adaptive_params  = adaptive_params,
                tickers          = tickers,
                sector_map       = sector_map,
                existing_weights = existing_weights,
            )
            opt_kwargs = adaptive_params.to_optimizer_kwargs()
            opt_kwargs["returns_history"] = log_ret   # for CVaR

            allocation_result = run_optimizer(
                method              = adaptive_params.optimization_method,
                mu                  = mu_ann,
                Sigma               = Sigma_ann,
                tickers             = tickers,
                constraint_builder  = cb,
                **opt_kwargs,
            )
            _log(f"  Solved: {allocation_result.solve_status}  "
                 f"Sharpe={allocation_result.sharpe_ratio:.4f}  "
                 f"vol={allocation_result.volatility:.2%}")
        except Exception as e:
            warnings.warn(f"Optimizer failed: {e}. Using fallback.")
            allocation_result = _fallback_equal_risk_parity(mu_ann, Sigma_ann, tickers)
            _log(f"  Fallback weights applied: {allocation_result.optimization_type}")
    else:
        allocation_result = _fallback_equal_risk_parity(mu_ann, Sigma_ann, tickers)
        _log(f"  Fallback: {allocation_result.optimization_type}")

    # ── Step 7: Uncertainty Quantification ────────────────────────────────────
    _log("Quantifying model uncertainty...")
    port_returns = log_ret @ eq_weights
    from milestone7.regime_model import build_observation_matrix
    obs = build_observation_matrix(daily_returns, window=21)

    uncertainty = quantify_uncertainty(
        obs               = obs,
        regime_output     = regime_output,
        forward_report    = forward_report,
        adaptive_params   = adaptive_params,
        tickers           = tickers,
        mu                = mu_ann,
        Sigma             = Sigma_ann,
        portfolio_returns = port_returns,
        n_boot            = uncertainty_n_boot,
        seed              = 42,
    )
    _log(f"  ESS={uncertainty.effective_sample_size}  "
         f"Regime CI widths: " +
         " | ".join(
             f"{STATE_LABELS[k]}=±{list(uncertainty.regime_prob_ci.values())[k]['half_width']:.3f}"
             for k in range(4)
         ))

    elapsed = time.time() - t_start
    _log(f"Intelligence engine complete in {elapsed:.1f}s")

    return IntelligenceReport(
        regime_output        = regime_output,
        garch_forecast       = garch_forecast,
        transition_analysis  = trans_analysis,
        forward_report       = forward_report,
        adaptive_params      = adaptive_params,
        uncertainty_report   = uncertainty,
        allocation_result    = allocation_result,
        tickers              = tickers,
        elapsed_seconds      = elapsed,
    )
