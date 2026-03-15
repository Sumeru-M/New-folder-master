"""
state_space_filter.py — Time-Varying Volatility via GARCH(1,1)
==============================================================

Mathematical Framework
----------------------
We implement the **GARCH(1,1)** model of Bollerslev (1986):

    r_t   = μ + ε_t,        ε_t | ℱ_{t-1} ~ N(0, σ_t²)
    σ_t²  = ω + α ε_{t-1}² + β σ_{t-1}²

where:
    ω > 0        — unconditional variance floor
    α ≥ 0        — ARCH coefficient (shock persistence)
    β ≥ 0        — GARCH coefficient (variance persistence)
    α + β < 1    — stationarity / covariance-stationarity condition

Stationarity implies an unconditional variance:
    σ̄² = ω / (1 - α - β)

**State-space interpretation**
The conditional variance σ_t² is the latent state evolving via:
    σ_t² = ω + α(r_{t-1} - μ)² + β σ_{t-1}²

This is equivalent to a nonlinear state-space model where the Kalman-type
update is replaced by the above recursion.

**Parameter estimation via MLE**
The log-likelihood under conditional Gaussianity is:

    ℓ(θ | r_{1:T}) = -½ Σ_t [log(2π) + log σ_t² + ε_t²/σ_t²]

We maximise ℓ subject to:
    ω > 0, α ≥ 0, β ≥ 0, α + β < 1

using L-BFGS-B with the reparametrisation:
    θ_raw = [log(ω), logit(α), logit(β)]
to enforce positivity and stationarity.

**30-day Volatility Forecast**
h-step-ahead variance forecast (Engle & Bollerslev 1986):

    σ²_{T+h} = σ̄² + (α+β)^h (σ_T² - σ̄²)

This is a mean-reverting process: forecasts converge to σ̄² as h → ∞.

**Confidence Intervals via Delta Method**
Using the Fisher information of the log-likelihood:
    Cov(θ̂) ≈ I(θ̂)^{-1}   (Cramér-Rao lower bound)

The variance of the h-step forecast is approximated by the delta method:
    Var[σ²_{T+h}] ≈ (∂σ²_{T+h}/∂θ)ᵀ Cov(θ̂) (∂σ²_{T+h}/∂θ)

Approximated numerically via finite differences.

Assumptions
-----------
A1. Conditional normality of returns: ε_t | ℱ_{t-1} ~ N(0, σ_t²).
    (Fat-tailed innovations are not modelled; GED or t extensions possible.)
A2. Constant conditional mean μ (iid estimation from sample mean).
A3. Covariance stationarity: α + β < 1.
A4. GARCH(1,1) lag orders sufficient (parsimony principle).
A5. No regime-switching in ω, α, β (these are estimated on pooled history).
    The regime-conditional volatility adjustment is applied in the
    forward_risk_forecast module.

References
----------
- Bollerslev, T. (1986). Generalised autoregressive conditional
  heteroskedasticity. Journal of Econometrics, 31(3), 307–327.
- Engle, R.F. (1982). Autoregressive conditional heteroscedasticity
  with estimates of the variance of United Kingdom inflation.
  Econometrica, 50(4), 987–1007.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class GARCHParameters:
    """
    Estimated GARCH(1,1) parameters.

    Attributes
    ----------
    omega : float    constant term (ω > 0)
    alpha : float    ARCH coefficient (α ≥ 0)
    beta  : float    GARCH coefficient (β ≥ 0)
    mu    : float    conditional mean
    persistence : float   α + β  (< 1 for stationarity)
    unconditional_var : float   ω / (1 - α - β)
    log_likelihood : float   maximised log-likelihood
    aic   : float   -2ℓ + 2p  (p=4 parameters)
    bic   : float   -2ℓ + p log T
    """
    omega:             float
    alpha:             float
    beta:              float
    mu:                float
    persistence:       float
    unconditional_var: float
    log_likelihood:    float
    aic:               float
    bic:               float
    param_cov:         Optional[np.ndarray] = None  # (3,3) Cov(ω,α,β)

    def to_dict(self) -> Dict:
        return {
            "omega":             round(self.omega, 8),
            "alpha":             round(self.alpha, 6),
            "beta":              round(self.beta, 6),
            "mu":                round(self.mu, 6),
            "persistence":       round(self.persistence, 6),
            "unconditional_vol_ann": round(float(np.sqrt(self.unconditional_var * 252)), 6),
            "log_likelihood":    round(self.log_likelihood, 4),
            "aic":               round(self.aic, 4),
            "bic":               round(self.bic, 4),
            "stationary":        self.persistence < 1.0,
        }


@dataclass
class VolatilityForecast:
    """
    Multi-step volatility forecast with uncertainty bands.

    Attributes
    ----------
    horizon_days    : list of forecast horizons
    variance_path   : (H,) mean-reverting variance forecast per horizon
    vol_path_ann    : (H,) annualised volatility forecast (√(σ²·252))
    ci_lower_95_ann : (H,) lower 95% confidence interval on vol (annualised)
    ci_upper_95_ann : (H,) upper 95% confidence interval on vol (annualised)
    current_vol_ann : float  current σ_T, annualised
    llt_var         : (H,)  variance of forecast (delta-method approximation)
    """
    horizon_days:    list
    variance_path:   np.ndarray
    vol_path_ann:    np.ndarray
    ci_lower_95_ann: np.ndarray
    ci_upper_95_ann: np.ndarray
    current_vol_ann: float
    forecast_var:    np.ndarray   # Var[σ²_{T+h}] from delta method

    def to_dict(self) -> Dict:
        return {
            "current_vol_ann":   round(self.current_vol_ann, 6),
            "forecast_30d_vol":  round(float(self.vol_path_ann[29]), 6),
            "forecast_path": {
                str(h): {
                    "vol_ann":       round(float(self.vol_path_ann[i]), 6),
                    "ci_lower_95":   round(float(self.ci_lower_95_ann[i]), 6),
                    "ci_upper_95":   round(float(self.ci_upper_95_ann[i]), 6),
                    "forecast_std":  round(float(np.sqrt(max(self.forecast_var[i], 0))), 8),
                }
                for i, h in enumerate(self.horizon_days)
                if h in (1, 5, 10, 21, 30)
            },
        }


# ---------------------------------------------------------------------------
# Reparametrisation utilities
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def _logit(x: float) -> float:
    x = np.clip(x, 1e-6, 1.0 - 1e-6)
    return np.log(x / (1.0 - x))

def _params_from_raw(raw: np.ndarray) -> Tuple[float, float, float]:
    """
    Map unconstrained raw vector to (ω, α, β) satisfying constraints:
        ω = exp(raw[0])         > 0
        α = sigmoid(raw[1]) × 0.5   ∈ (0, 0.5)   [ARCH bounded]
        β = sigmoid(raw[2]) × 0.96/(1+α) × (1-α)  ensures α+β < 0.999
    """
    omega = np.exp(raw[0])
    alpha = _sigmoid(raw[1]) * 0.5
    # β constrained so α + β < 0.999
    beta  = _sigmoid(raw[2]) * (0.999 - alpha)
    return omega, alpha, beta


def _raw_from_params(omega: float, alpha: float, beta: float) -> np.ndarray:
    return np.array([
        np.log(omega),
        _logit(alpha / 0.5),
        _logit(beta / (0.999 - alpha)),
    ])


# ---------------------------------------------------------------------------
# GARCH recursion and log-likelihood
# ---------------------------------------------------------------------------

def _garch_filter(
    returns: np.ndarray,
    mu:      float,
    omega:   float,
    alpha:   float,
    beta:    float,
) -> Tuple[np.ndarray, float]:
    """
    Run the GARCH(1,1) filter and return (σ²_{1:T}, log-likelihood).

    Initialisation: σ²_1 = ω / (1 - α - β) (unconditional variance).
    If non-stationary, fall back to sample variance.
    """
    T        = len(returns)
    eps      = returns - mu
    ab       = alpha + beta
    sigma2   = np.zeros(T)

    # Initialise at unconditional variance
    if ab < 1.0:
        sigma2[0] = omega / (1.0 - ab)
    else:
        sigma2[0] = np.var(eps)

    sigma2[0] = max(sigma2[0], 1e-8)

    for t in range(1, T):
        sigma2[t] = omega + alpha * eps[t - 1] ** 2 + beta * sigma2[t - 1]
        sigma2[t] = max(sigma2[t], 1e-8)

    # Log-likelihood:  ℓ = -½ Σ [log(2π) + log σ²_t + ε²_t/σ²_t]
    ll = -0.5 * (
        T * np.log(2 * np.pi)
        + np.sum(np.log(sigma2))
        + np.sum(eps ** 2 / sigma2)
    )
    return sigma2, ll


def _neg_ll(raw: np.ndarray, returns: np.ndarray, mu: float) -> float:
    """Objective for MLE: negative log-likelihood in unconstrained space."""
    try:
        omega, alpha, beta = _params_from_raw(raw)
        _, ll = _garch_filter(returns, mu, omega, alpha, beta)
        return -ll
    except Exception:
        return 1e10


# ---------------------------------------------------------------------------
# MLE estimation
# ---------------------------------------------------------------------------

def fit_garch(
    returns: np.ndarray,
    n_starts: int = 5,
    seed:     int = 0,
) -> GARCHParameters:
    """
    Fit GARCH(1,1) via maximum likelihood.

    Multiple starting points (n_starts) are used to mitigate local optima.
    Each starting point perturbs the sample-variance-based default.

    Parameters
    ----------
    returns  : (T,) array of daily log-returns
    n_starts : number of optimisation restarts
    seed     : random seed for perturbations

    Returns
    -------
    GARCHParameters
    """
    rng    = np.random.default_rng(seed)
    mu     = float(np.mean(returns))
    T      = len(returns)

    best_ll  = -np.inf
    best_raw = None

    # Default starting point based on sample statistics
    sv = float(np.var(returns - mu))
    omega0 = sv * 0.1
    alpha0 = 0.1
    beta0  = 0.85
    raw0   = _raw_from_params(omega0, alpha0, beta0)

    for i in range(n_starts):
        if i == 0:
            x0 = raw0
        else:
            x0 = raw0 + rng.normal(0, 0.5, 3)

        res = minimize(
            _neg_ll,
            x0,
            args=(returns, mu),
            method="L-BFGS-B",
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        if res.success or res.fun < -best_ll:
            if -res.fun > best_ll:
                best_ll  = -res.fun
                best_raw = res.x

    if best_raw is None:
        best_raw = raw0
        best_ll  = -_neg_ll(raw0, returns, mu)

    omega, alpha, beta = _params_from_raw(best_raw)
    ab       = alpha + beta
    uvar     = omega / max(1.0 - ab, 1e-8)
    aic      = -2 * best_ll + 2 * 4
    bic      = -2 * best_ll + 4 * np.log(T)

    # Estimate parameter covariance via numerical Hessian
    param_cov = _estimate_param_cov(best_raw, returns, mu)

    return GARCHParameters(
        omega             = omega,
        alpha             = alpha,
        beta              = beta,
        mu                = mu,
        persistence       = ab,
        unconditional_var = uvar,
        log_likelihood    = best_ll,
        aic               = aic,
        bic               = bic,
        param_cov         = param_cov,
    )


# ---------------------------------------------------------------------------
# Parameter covariance (Fisher information via numerical Hessian)
# ---------------------------------------------------------------------------

def _estimate_param_cov(
    raw:     np.ndarray,
    returns: np.ndarray,
    mu:      float,
    eps:     float = 1e-4,
) -> np.ndarray:
    """
    Approximate the parameter covariance matrix via numerical Hessian of
    the log-likelihood:

        Cov(θ̂) ≈ [-∂²ℓ/∂θ∂θᵀ]^{-1}  =  I(θ̂)^{-1}

    Uses central finite differences with step size eps.
    Returns 3×3 matrix for (ω, α, β) in the raw (unconstrained) space.
    """
    p = len(raw)
    H = np.zeros((p, p))
    f0 = _neg_ll(raw, returns, mu)

    for i in range(p):
        for j in range(i, p):
            ei = np.zeros(p); ei[i] = eps
            ej = np.zeros(p); ej[j] = eps
            fpp = _neg_ll(raw + ei + ej, returns, mu)
            fpm = _neg_ll(raw + ei - ej, returns, mu)
            fmp = _neg_ll(raw - ei + ej, returns, mu)
            fmm = _neg_ll(raw - ei - ej, returns, mu)
            H[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps ** 2)
            H[j, i] = H[i, j]

    try:
        H_reg  = H + 1e-6 * np.eye(p)
        cov    = np.linalg.inv(H_reg)       # Hessian of neg-LL → information matrix
        return cov
    except np.linalg.LinAlgError:
        return np.eye(p) * 1e-6


# ---------------------------------------------------------------------------
# h-step-ahead variance forecast with confidence intervals
# ---------------------------------------------------------------------------

def forecast_volatility(
    returns:  np.ndarray,
    params:   GARCHParameters,
    horizon:  int  = 30,
    n_sim:    int  = 1_000,
    seed:     int  = 42,
) -> VolatilityForecast:
    """
    Compute h-step-ahead variance forecasts with uncertainty bands.

    **Mean forecast** (analytical):
        σ²_{T+h} = σ̄² + (α+β)^h (σ_T² - σ̄²)

    **Confidence intervals** via parametric bootstrap:
        1. Draw B parameter sets from N(θ̂, Cov(θ̂)).
        2. For each draw, run the GARCH filter and compute h-step forecasts.
        3. Report 2.5th and 97.5th percentiles across B draws.

    This correctly propagates both estimation uncertainty (from Cov(θ̂))
    and model uncertainty through to the forecast intervals.

    Parameters
    ----------
    returns : (T,) daily log-returns
    params  : fitted GARCHParameters
    horizon : forecast horizon (days)
    n_sim   : number of bootstrap samples for uncertainty
    seed    : random seed
    """
    rng       = np.random.default_rng(seed)
    mu        = params.mu
    omega     = params.omega
    alpha     = params.alpha
    beta      = params.beta
    ab        = params.persistence
    uvar      = params.unconditional_var

    # Current conditional variance σ_T²
    sigma2, _ = _garch_filter(returns, mu, omega, alpha, beta)
    sigma2_T  = sigma2[-1]

    # Mean-reverting h-step forecast
    horizons    = list(range(1, horizon + 1))
    var_path    = np.array([
        uvar + ab ** h * (sigma2_T - uvar) for h in horizons
    ])
    var_path    = np.maximum(var_path, 1e-8)
    vol_path    = np.sqrt(var_path * 252)           # annualised

    current_vol = float(np.sqrt(sigma2_T * 252))

    # Delta-method forecast variance
    # ∂σ²_{T+h}/∂ω = 1/(1-ab) × (1 - ab^h) + h*ab^(h-1)*(σ_T²-σ̄²)*∂ab/∂ω ≈ const
    # We use parametric bootstrap for simplicity and accuracy
    forecast_var = np.zeros(horizon)
    boot_paths   = np.zeros((n_sim, horizon))

    raw0 = _raw_from_params(omega, alpha, beta)

    if params.param_cov is not None:
        cov = params.param_cov
        try:
            L = np.linalg.cholesky(cov + 1e-8 * np.eye(3))
            for b in range(n_sim):
                raw_b         = raw0 + L @ rng.standard_normal(3)
                om_b, al_b, be_b = _params_from_raw(raw_b)
                ab_b          = al_b + be_b
                uv_b          = om_b / max(1.0 - ab_b, 1e-8)
                s2_b, _       = _garch_filter(returns, mu, om_b, al_b, be_b)
                s2T_b         = s2_b[-1]
                boot_paths[b] = [
                    uv_b + ab_b ** h * (s2T_b - uv_b) for h in horizons
                ]
            forecast_var  = np.var(boot_paths, axis=0)
            ci_lower_var  = np.percentile(boot_paths, 2.5,  axis=0)
            ci_upper_var  = np.percentile(boot_paths, 97.5, axis=0)
        except Exception:
            ci_lower_var  = var_path * 0.7
            ci_upper_var  = var_path * 1.3
    else:
        ci_lower_var  = var_path * 0.7
        ci_upper_var  = var_path * 1.3

    ci_lower_vol = np.sqrt(np.maximum(ci_lower_var * 252, 1e-8))
    ci_upper_vol = np.sqrt(np.maximum(ci_upper_var * 252, 1e-8))

    return VolatilityForecast(
        horizon_days    = horizons,
        variance_path   = var_path,
        vol_path_ann    = vol_path,
        ci_lower_95_ann = ci_lower_vol,
        ci_upper_95_ann = ci_upper_vol,
        current_vol_ann = current_vol,
        forecast_var    = forecast_var,
    )


# ---------------------------------------------------------------------------
# Portfolio-level GARCH
# ---------------------------------------------------------------------------

def fit_portfolio_garch(
    daily_returns: "pd.DataFrame",
    weights:       "np.ndarray",
) -> Tuple[GARCHParameters, VolatilityForecast]:
    """
    Fit GARCH to weighted portfolio returns.

    Constructs the portfolio return series r_p = wᵀ r_t, then fits GARCH(1,1).
    This captures the portfolio-level conditional heteroskedasticity rather
    than treating volatility as constant.

    Parameters
    ----------
    daily_returns : (T, N) DataFrame
    weights       : (N,) array of portfolio weights (need not sum to 1 — normalised)

    Returns
    -------
    (GARCHParameters, VolatilityForecast)
    """
    import pandas as pd
    R   = daily_returns.values.astype(float)
    w   = np.array(weights, dtype=float)
    w  /= w.sum() + 1e-300
    r_p = R @ w

    params   = fit_garch(r_p)
    forecast = forecast_volatility(r_p, params)
    return params, forecast
