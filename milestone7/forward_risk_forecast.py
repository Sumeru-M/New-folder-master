"""
forward_risk_forecast.py — Regime-Conditioned Forward Return Distribution
=========================================================================

Mathematical Framework
----------------------

Let S_t ∈ {0,...,K-1} be the latent regime at time t, and let:

    γ_t(k) = P(S_t = k | O_{1:t})          filtered probabilities (causal)
    A^(h)_{ij} = P(S_{t+h} = j | S_t = i)  h-step transition matrix

**Regime-weighted forward moments**

The marginal distribution of returns at horizon h, integrating over
regime uncertainty:

    P(S_{t+h} = k) = Σ_i γ_t(i) × [A^h]_{ik}   := p_k^(h)

Within each regime k, daily returns are modelled as:

    r | S = k  ~  N(μ_k, σ_k²)

where μ_k and σ_k are the regime emission parameters.

Over an H-day horizon with daily returns approximately i.i.d. within regime
(first-order approximation valid for short horizons vs. persistence time):

    r_H | S = k  ~  N(H·μ_k, H·σ_k²)        [independent increments]

**Mixture distribution**

The marginal H-day return distribution is a Gaussian mixture:

    r_H  ~  Σ_k p_k^(h) × N(H·μ_k, H·σ_k²)

**Forward expected return**

    E[r_H] = H × Σ_k p_k^(h) × μ_k

**Forward variance** (law of total variance):

    Var[r_H] = Σ_k p_k^(h) × (H·σ_k² + H²·μ_k²)  -  (E[r_H])²
             = H × Σ_k p_k^(h) × σ_k²   +   H² × (Σ_k p_k^(h)·μ_k² - (Σ_k p_k^(h)·μ_k)²)
                 ↑ within-regime term              ↑ between-regime term

The between-regime term captures excess variance from regime uncertainty —
this is the econometric source of fat tails in empirical return distributions.

**Regime-weighted CVaR**

For a mixture distribution, the CVaR at level α is:

    CVaR_α = -E[r_H | r_H ≤ VaR_α]

For a K-component Gaussian mixture:
    VaR_α solved numerically from:  Σ_k p_k^(h) Φ((VaR_α - H·μ_k)/(√H·σ_k)) = α

    CVaR_α = -(1/α) Σ_k p_k^(h) [H·μ_k·Φ((VaR_α - H·μ_k)/(√H·σ_k))
                                   - √H·σ_k·φ((VaR_α - H·μ_k)/(√H·σ_k))]

where φ(·) is the standard normal PDF and Φ(·) is the CDF.

**Tail probabilities**

    P(r_H < -x%) = Σ_k p_k^(h) × Φ((-x/100 - H·μ_k) / (√H·σ_k))

**GARCH-adjustment**

When a GARCH(1,1) σ_T² estimate is available, the current-period
regime volatility is scaled to match the GARCH forecast:

    σ_k^* = σ_k × (σ_garch,T / σ_baseline)

where σ_baseline = Σ_k γ_T(k) × σ_k  is the regime-weighted baseline.

This ensures the forward distribution uses time-varying volatility
information from GARCH while preserving the regime mixture structure.

Assumptions
-----------
A1. Within-regime returns are approximately Gaussian with parameters
    estimated from the HMM emission model.
A2. Independent increments within regime over the forecast horizon.
    (First-order approximation; autocorrelation is neglected.)
A3. Regime transitions governed by the time-homogeneous A matrix.
A4. No parameter uncertainty propagated into forward moments here —
    that is handled in uncertainty_quantifier.py.
A5. GARCH adjustment is applied at the portfolio level using portfolio
    log-returns, not per-asset (for computational tractability).
A6. Annualised parameters from emission means/covariances are scaled
    back to daily using:  μ_daily = μ_ann / 252, σ²_daily = σ²_ann / 252.

References
----------
- Hamilton, J.D. (1989). A new approach to the economic analysis of
  nonstationary time series and the business cycle. Econometrica.
- Ang, A. & Bekaert, G. (2002). Regime switches in interest rates.
  Journal of Business and Economic Statistics.
- Rockafellar, R.T. & Uryasev, S. (2000). Optimization of CVaR.
  Journal of Risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ForwardDistribution:
    """
    Regime-conditioned forward return distribution at a given horizon.

    Attributes
    ----------
    horizon_days    : int     forecast horizon H
    mixture_weights : (K,)   regime probabilities at horizon p_k^(h)
    mixture_means   : (K,)   H × μ_k  (H-day expected returns per regime)
    mixture_stds    : (K,)   √H × σ_k  (H-day std per regime)
    expected_return : float  E[r_H]  (annualised equivalent)
    expected_return_H : float E[r_H] over H days
    variance        : float  Var[r_H]  (H-day)
    std             : float  √Var[r_H]  (H-day)
    skewness        : float  third standardised central moment
    kurtosis        : float  fourth standardised central moment (excess)
    var_95          : float  5% VaR  (as loss, positive)
    var_99          : float  1% VaR  (as loss, positive)
    cvar_95         : float  95% CVaR  (expected shortfall)
    cvar_99         : float  99% CVaR
    tail_prob_5pct  : float  P(r_H < -5%)
    tail_prob_10pct : float  P(r_H < -10%)
    tail_prob_20pct : float  P(r_H < -20%)
    regime_contrib  : dict   per-regime contribution to variance
    """
    horizon_days:      int
    mixture_weights:   np.ndarray
    mixture_means:     np.ndarray
    mixture_stds:      np.ndarray
    expected_return:   float    # annualised
    expected_return_H: float    # over horizon
    variance:          float
    std:               float
    skewness:          float
    kurtosis:          float
    var_95:            float
    var_99:            float
    cvar_95:           float
    cvar_99:           float
    tail_prob_5pct:    float
    tail_prob_10pct:   float
    tail_prob_20pct:   float
    regime_contrib:    Dict

    def to_dict(self) -> Dict:
        from milestone7.regime_model import STATE_LABELS
        K = len(self.mixture_weights)
        return {
            "horizon_days":           self.horizon_days,
            "expected_return_ann":    round(self.expected_return, 6),
            "expected_return_H_day":  round(self.expected_return_H, 6),
            "std_H_day":              round(self.std, 6),
            "annualised_vol":         round(self.std * np.sqrt(252 / self.horizon_days), 6),
            "skewness":               round(self.skewness, 4),
            "excess_kurtosis":        round(self.kurtosis, 4),
            "var_95_percent":         round(self.var_95, 6),
            "var_99_percent":         round(self.var_99, 6),
            "cvar_95_percent":        round(self.cvar_95, 6),
            "cvar_99_percent":        round(self.cvar_99, 6),
            "tail_prob_minus5pct":    round(self.tail_prob_5pct, 6),
            "tail_prob_minus10pct":   round(self.tail_prob_10pct, 6),
            "tail_prob_minus20pct":   round(self.tail_prob_20pct, 6),
            "mixture_components":     {
                STATE_LABELS.get(k, str(k)): {
                    "weight":    round(float(self.mixture_weights[k]), 6),
                    "mean_H":    round(float(self.mixture_means[k]), 6),
                    "std_H":     round(float(self.mixture_stds[k]), 6),
                }
                for k in range(K)
            },
            "regime_variance_contribution": self.regime_contrib,
        }


@dataclass
class ForwardRiskReport:
    """
    Complete forward risk report across multiple horizons.

    Attributes
    ----------
    horizons          : list of ForwardDistribution objects
    dominant_regime   : int   most probable current regime
    regime_weights_now: (K,) γ_T (current filtered probabilities)
    garch_adjusted    : bool  whether GARCH adjustment was applied
    garch_vol_current : float current GARCH volatility (annualised)
    """
    horizons:           List[ForwardDistribution]
    dominant_regime:    int
    regime_weights_now: np.ndarray
    garch_adjusted:     bool
    garch_vol_current:  Optional[float]

    def to_dict(self) -> Dict:
        from milestone7.regime_model import STATE_LABELS
        return {
            "dominant_regime":     STATE_LABELS.get(self.dominant_regime,
                                                     str(self.dominant_regime)),
            "regime_weights_now":  {
                STATE_LABELS.get(k, str(k)): round(float(v), 6)
                for k, v in enumerate(self.regime_weights_now)
            },
            "garch_adjusted":      self.garch_adjusted,
            "garch_vol_current":   round(self.garch_vol_current, 6)
                                   if self.garch_vol_current else None,
            "forward_distributions": {
                f"{fd.horizon_days}d": fd.to_dict()
                for fd in self.horizons
            },
        }


# ---------------------------------------------------------------------------
# Mixture distribution utilities
# ---------------------------------------------------------------------------

def _mixture_cdf(x: float, weights: np.ndarray,
                 means: np.ndarray, stds: np.ndarray) -> float:
    """
    CDF of a Gaussian mixture:  F(x) = Σ_k w_k Φ((x - μ_k)/σ_k)
    """
    val = 0.0
    for w, mu, sig in zip(weights, means, stds):
        if sig > 1e-10:
            val += w * norm.cdf((x - mu) / sig)
    return float(val)


def _mixture_var(
    alpha:   float,
    weights: np.ndarray,
    means:   np.ndarray,
    stds:    np.ndarray,
    bracket: Tuple[float, float] = (-2.0, 0.5),
) -> float:
    """
    Solve for VaR_α: smallest x such that F(x) = α.

    Uses Brent's method on the mixture CDF.
    Falls back to normal approximation if root-finding fails.
    """
    f     = lambda x: _mixture_cdf(x, weights, means, stds) - alpha
    lo, hi = bracket
    # Expand bracket if needed
    while f(lo) > 0:
        lo *= 2
    while f(hi) < 0:
        hi = hi * 2 + 0.1
    try:
        return float(brentq(f, lo, hi, xtol=1e-8, maxiter=200))
    except Exception:
        # Normal approximation fallback
        mu_mix = float(np.dot(weights, means))
        var_mix = float(np.dot(weights, stds ** 2 + means ** 2) - mu_mix ** 2)
        return float(mu_mix + norm.ppf(alpha) * np.sqrt(max(var_mix, 1e-10)))


def _mixture_cvar(
    alpha:   float,
    weights: np.ndarray,
    means:   np.ndarray,
    stds:    np.ndarray,
) -> float:
    """
    CVaR_α of a Gaussian mixture (expected shortfall at level α):

        CVaR_α = -(1/α) Σ_k w_k [μ_k Φ(z_k) - σ_k φ(z_k)]

    where z_k = (VaR_α - μ_k) / σ_k,  VaR_α = _mixture_var(α, ...) < 0.

    Returns positive value (loss convention).
    """
    var_val = _mixture_var(alpha, weights, means, stds)  # negative
    cvar    = 0.0
    for w, mu, sig in zip(weights, means, stds):
        if sig < 1e-10:
            cvar += w * min(mu, var_val)
            continue
        z     = (var_val - mu) / sig
        cvar += w * (mu * norm.cdf(z) - sig * norm.pdf(z))
    return float(-cvar / max(alpha, 1e-10))


def _mixture_moment(
    order:   int,
    weights: np.ndarray,
    means:   np.ndarray,
    stds:    np.ndarray,
) -> float:
    """
    Compute raw moments of a Gaussian mixture numerically via Gaussian quadrature.

    For order ≤ 4, uses analytical Gaussian moment formulas:
        E[X^1] = Σ w_k μ_k
        E[X^2] = Σ w_k (μ_k² + σ_k²)
        E[X^3] = Σ w_k (μ_k³ + 3μ_k σ_k²)
        E[X^4] = Σ w_k (μ_k⁴ + 6μ_k² σ_k² + 3σ_k⁴)
    """
    m = 0.0
    for w, mu, sig in zip(weights, means, stds):
        s2 = sig ** 2
        if order == 1:
            m += w * mu
        elif order == 2:
            m += w * (mu ** 2 + s2)
        elif order == 3:
            m += w * (mu ** 3 + 3 * mu * s2)
        elif order == 4:
            m += w * (mu ** 4 + 6 * mu ** 2 * s2 + 3 * s2 ** 2)
    return float(m)


def _mixture_skewness_kurtosis(
    weights: np.ndarray,
    means:   np.ndarray,
    stds:    np.ndarray,
) -> Tuple[float, float]:
    """
    Skewness and excess kurtosis of a Gaussian mixture.

    Skewness  = (μ₃ - 3μ₁σ² - μ₁³) / σ³   where σ² = μ₂ - μ₁²
    Kurtosis  = (μ₄ - 4μ₁μ₃ + 6μ₁²μ₂ - 3μ₁⁴) / σ⁴ - 3

    Uses raw moments of the mixture.
    """
    m1 = _mixture_moment(1, weights, means, stds)
    m2 = _mixture_moment(2, weights, means, stds)
    m3 = _mixture_moment(3, weights, means, stds)
    m4 = _mixture_moment(4, weights, means, stds)

    var     = m2 - m1 ** 2
    std     = max(np.sqrt(var), 1e-10)
    # Central moments
    mu3_c   = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4_c   = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    skew    = mu3_c / std ** 3
    kurt    = mu4_c / std ** 4 - 3.0     # excess kurtosis
    return float(skew), float(kurt)


# ---------------------------------------------------------------------------
# Core forward distribution computation
# ---------------------------------------------------------------------------

def compute_forward_distribution(
    horizon_days:   int,
    regime_probs_h: np.ndarray,
    emission_means: np.ndarray,
    emission_stds:  np.ndarray,
    garch_adj_factor: float = 1.0,
) -> ForwardDistribution:
    """
    Compute the H-day forward return distribution under regime mixture.

    Parameters
    ----------
    horizon_days    : H  (forecast horizon in trading days)
    regime_probs_h  : (K,) P(S_{T+H} = k)  — h-step forward regime probs
    emission_means  : (K,) daily μ_k (from HMM emission parameters)
    emission_stds   : (K,) daily σ_k
    garch_adj_factor: scalar ≥ 0, multiplied into all σ_k (GARCH adjustment)

    Returns
    -------
    ForwardDistribution
    """
    H   = horizon_days
    K   = len(regime_probs_h)

    w   = regime_probs_h / (regime_probs_h.sum() + 1e-300)

    # Scale emissions to H-day horizon
    mu_H  = emission_means * H                                  # (K,)
    std_H = emission_stds * np.sqrt(H) * garch_adj_factor       # (K,)
    std_H = np.maximum(std_H, 1e-8)

    # Mixture moments
    E_r    = float(np.dot(w, mu_H))
    E_r2   = float(np.dot(w, std_H ** 2 + mu_H ** 2))
    Var_r  = E_r2 - E_r ** 2
    Var_r  = max(Var_r, 1e-10)
    Std_r  = np.sqrt(Var_r)

    skew, kurt = _mixture_skewness_kurtosis(w, mu_H, std_H)

    # VaR and CVaR
    var_95  = -_mixture_var(0.05, w, mu_H, std_H)   # positive loss
    var_99  = -_mixture_var(0.01, w, mu_H, std_H)
    cvar_95 = _mixture_cvar(0.05, w, mu_H, std_H)
    cvar_99 = _mixture_cvar(0.01, w, mu_H, std_H)

    # Tail probabilities
    p5  = _mixture_cdf(-0.05, w, mu_H, std_H)
    p10 = _mixture_cdf(-0.10, w, mu_H, std_H)
    p20 = _mixture_cdf(-0.20, w, mu_H, std_H)

    # Per-regime variance contributions (law of total variance decomposition)
    from milestone7.regime_model import STATE_LABELS
    mu_mix     = E_r
    regime_contrib = {}
    for k in range(K):
        within_k  = float(w[k] * std_H[k] ** 2)             # w_k σ_k²
        between_k = float(w[k] * (mu_H[k] - mu_mix) ** 2)   # w_k (μ_k - μ̄)²
        regime_contrib[STATE_LABELS.get(k, str(k))] = {
            "within_regime_var":   round(within_k, 8),
            "between_regime_var":  round(between_k, 8),
        }

    # Annualised expected return
    E_r_ann = E_r * (252 / H)

    return ForwardDistribution(
        horizon_days      = H,
        mixture_weights   = w,
        mixture_means     = mu_H,
        mixture_stds      = std_H,
        expected_return   = E_r_ann,
        expected_return_H = E_r,
        variance          = Var_r,
        std               = Std_r,
        skewness          = skew,
        kurtosis          = kurt,
        var_95            = var_95,
        var_99            = var_99,
        cvar_95           = cvar_95,
        cvar_99           = cvar_99,
        tail_prob_5pct    = p5,
        tail_prob_10pct   = p10,
        tail_prob_20pct   = p20,
        regime_contrib    = regime_contrib,
    )


# ---------------------------------------------------------------------------
# Full forward risk report
# ---------------------------------------------------------------------------

def compute_forward_risk(
    regime_output:   "RegimeOutput",
    transition_analysis: "TransitionAnalysis",
    garch_forecast:  Optional["VolatilityForecast"] = None,
    horizons:        List[int] = [21, 63],
) -> ForwardRiskReport:
    """
    Compute regime-conditioned forward risk across multiple horizons.

    1. Extract emission parameters (μ_k, σ_k) from HMM.
       Emission means[k, 0] = mean daily return for regime k.
       Emission covs[k, 0, 0] = daily return variance for regime k.
       Both are in daily units (the HMM was fit on daily observations).

    2. For each horizon H:
       a. Look up p_k^(h) from TransitionAnalysis.forward_probs[H].
       b. Apply optional GARCH volatility adjustment.
       c. Compute full mixture distribution.

    3. GARCH adjustment:
       σ_k^* = σ_k × (σ_garch_current / σ_baseline)
       where σ_baseline = Σ_k γ_T(k) σ_k.
       This scales regime volatilities to match the current GARCH estimate
       without changing the mixture structure.

    Parameters
    ----------
    regime_output        : RegimeOutput from regime_model.run_hmm()
    transition_analysis  : TransitionAnalysis from transition_matrix.analyse_transitions()
    garch_forecast       : VolatilityForecast (optional); used for vol adjustment
    horizons             : list of forecast horizons in trading days
    """
    params = regime_output.params
    K      = params.K

    # Extract daily emission parameters
    # emission means[:, 0] = mean portfolio return (daily)
    # emission covs[:, 0, 0] = variance of portfolio return (daily)
    mu_k   = params.means[:, 0].copy()          # (K,) daily mean returns
    var_k  = params.covs[:, 0, 0].copy()        # (K,) daily variances
    var_k  = np.maximum(var_k, 1e-8)
    sig_k  = np.sqrt(var_k)                     # (K,) daily std devs

    # GARCH adjustment factor
    garch_adj    = 1.0
    garch_vol_c  = None
    garch_applied = False
    if garch_forecast is not None:
        garch_vol_daily = garch_forecast.current_vol_ann / np.sqrt(252)
        baseline_vol    = float(np.dot(regime_output.current_probs, sig_k))
        if baseline_vol > 1e-8:
            garch_adj = garch_vol_daily / baseline_vol
        garch_vol_c   = garch_forecast.current_vol_ann
        garch_applied = True

    # Compute forward distributions
    fwd_dists = []
    for H in horizons:
        # Get h-step forward regime probs from pre-computed transition analysis
        if H in transition_analysis.forward_probs:
            p_h = transition_analysis.forward_probs[H]
        else:
            from milestone7.transition_matrix import compute_matrix_power
            Ah  = compute_matrix_power(params.A, H)
            p_h = regime_output.current_probs @ Ah

        fd = compute_forward_distribution(
            horizon_days    = H,
            regime_probs_h  = p_h,
            emission_means  = mu_k,
            emission_stds   = sig_k,
            garch_adj_factor = garch_adj,
        )
        fwd_dists.append(fd)

    return ForwardRiskReport(
        horizons           = fwd_dists,
        dominant_regime    = regime_output.current_regime,
        regime_weights_now = regime_output.current_probs,
        garch_adjusted     = garch_applied,
        garch_vol_current  = garch_vol_c,
    )
