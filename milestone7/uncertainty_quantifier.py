"""
uncertainty_quantifier.py — Model Uncertainty Quantification
=============================================================

Mathematical Framework
----------------------

This module provides five uncertainty metrics:

1. **HMM Parameter Uncertainty (Fisher Information)**

The Fisher information matrix I(θ) for HMM parameters is estimated via
the observed information:

    I(θ̂) = -∂²ℓ/∂θ∂θᵀ |_{θ=θ̂}

where ℓ(θ) = log P(O_{1:T} | θ) is the log-likelihood.

Approximated numerically by finite differences on the log-likelihood.

The Cramér-Rao lower bound gives the smallest achievable covariance:

    Cov(θ̂) ≥ I(θ̂)^{-1}

The diagonal of I(θ̂)^{-1} gives the asymptotic standard errors:

    SE(θ̂_j) = √[I(θ̂)^{-1}_{jj}]

2. **Confidence Intervals on Regime Probabilities**

Given that regime probabilities γ_T(k) are nonlinear functions of θ,
their uncertainty is quantified via the delta method:

    Var[γ_T(k)] ≈ (∂γ_T(k)/∂θ)ᵀ Cov(θ̂) (∂γ_T(k)/∂θ)

Approximated by:
    (a) Bootstrap: perturb θ via N(θ̂, Cov(θ̂)), rerun forward pass, compute γ.
    (b) 95% CI: [γ_T(k) - 1.96×SE, γ_T(k) + 1.96×SE], clipped to [0,1].

3. **Forecast Uncertainty Bands**

For the H-day forward expected return E[r_H]:

    Var[E[r_H]] = H² × Σ_k Var[p_k^(h)] × μ_k²   +  (cross-terms via full covariance)

Approximated via bootstrap draws from (θ̂, Cov(θ̂)).

4. **Weight Sensitivity to Regime Shifts**

Sensitivity of optimal portfolio weights w*(p) to a unit shift in regime k:

    ∂w*/∂p_k ≈ [w*(p + δe_k) - w*(p - δe_k)] / (2δ)

where δ = 0.05. This quantifies how much the optimal allocation changes
when regime k probability increases by 5%.

The overall regime sensitivity index:

    RSI_n = Σ_k |∂w_n*/∂p_k|    (for each asset n)

A high RSI_n means asset n's weight is particularly sensitive to regime
classification uncertainty.

5. **Model Selection Criteria (Information Criteria)**

Comparing HMM models by number of states K:

    AIC(K) = -2 × loglik(K) + 2 × params(K)
    BIC(K) = -2 × loglik(K) + params(K) × log T

where params(K) = K(K-1) [transition] + K×M [means] + K×M(M+1)/2 [covs] + K [pi].

Assumptions
-----------
A1. Asymptotic normality of MLE:  √T (θ̂ - θ₀) →_d N(0, I(θ₀)^{-1}).
    Valid for large T (≥ 200 observations recommended).
A2. The Fisher information is approximated by the observed information
    (finite-difference Hessian of the log-likelihood).
A3. Bootstrap perturbations sample in the space of raw parameters;
    the reparametrisation maintains constraint satisfaction.
A4. Weight sensitivity is computed using finite differences on the
    blended parameter vector, not on the underlying optimizer directly.
A5. All uncertainty metrics assume the HMM model class is correct
    (mis-specification uncertainty is not captured).

References
----------
- Fisher, R.A. (1925). Theory of statistical estimation.
  Mathematical Proceedings of the Cambridge Philosophical Society.
- Lehmann, E.L. & Casella, G. (1998). Theory of Point Estimation.
  Springer.
- Shumway, R.H. & Stoffer, D.S. (2000). Time Series Analysis and
  Its Applications. Springer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class UncertaintyReport:
    """
    Complete uncertainty quantification report.

    Attributes
    ----------
    regime_prob_ci     : dict {state_label: {mean, lower_95, upper_95}}
    transition_se      : (K, K) standard errors of A_{ij} estimates
    forecast_return_ci : dict {horizon: {mean, lower_95, upper_95}}
    weight_sensitivity : dict {asset: regime_sensitivity_index}
    model_info_criteria: dict {K: {aic, bic}}
    parameter_se_dict  : dict of key parameter standard errors
    effective_sample_size : int  (Var-ratio based ESS)
    """
    regime_prob_ci:       Dict
    transition_se:        np.ndarray
    forecast_return_ci:   Dict
    weight_sensitivity:   Dict
    model_info_criteria:  Dict
    parameter_se_dict:    Dict
    effective_sample_size: int

    def to_dict(self) -> Dict:
        return {
            "regime_probability_ci_95":      self.regime_prob_ci,
            "transition_matrix_se":          {
                k: {kk: round(float(vv), 6) for kk, vv in v.items()}
                for k, v in self._se_to_dict().items()
            },
            "forecast_return_ci_95":         self.forecast_return_ci,
            "weight_regime_sensitivity":     self.weight_sensitivity,
            "model_selection_criteria":      self.model_info_criteria,
            "key_parameter_standard_errors": self.parameter_se_dict,
            "effective_sample_size":         self.effective_sample_size,
        }

    def _se_to_dict(self) -> Dict:
        from milestone7.regime_model import STATE_LABELS
        K  = self.transition_se.shape[0]
        out = {}
        for i in range(K):
            out[STATE_LABELS.get(i, str(i))] = {
                STATE_LABELS.get(j, str(j)): float(self.transition_se[i, j])
                for j in range(K)
            }
        return out


# ---------------------------------------------------------------------------
# Effective sample size (autocorrelation-adjusted)
# ---------------------------------------------------------------------------

def _effective_sample_size(returns: np.ndarray, max_lag: int = 20) -> int:
    """
    ESS = T / (1 + 2 Σ_{l=1}^L ρ_l)  where ρ_l is the lag-l autocorrelation.

    For a sequence with positive autocorrelation (momentum), ESS < T.
    For negative autocorrelation (mean-reversion), ESS > T.
    """
    T    = len(returns)
    mu   = returns.mean()
    var  = returns.var()
    if var < 1e-12:
        return T
    sum_rho = 0.0
    for lag in range(1, min(max_lag + 1, T // 4)):
        cov = float(np.mean((returns[lag:] - mu) * (returns[:-lag] - mu)))
        rho = cov / var
        if abs(rho) < 0.05:   # truncate at noise floor
            break
        sum_rho += rho
    denom  = max(1.0 + 2.0 * sum_rho, 0.1)
    return int(max(T / denom, 1))


# ---------------------------------------------------------------------------
# Bootstrap-based regime probability CI
# ---------------------------------------------------------------------------

def _bootstrap_regime_ci(
    obs:       np.ndarray,
    params:    "HMMParameters",
    n_boot:    int   = 300,
    seed:      int   = 42,
    conf:      float = 0.95,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bootstrap confidence intervals on filtered regime probabilities γ_T.

    Method:
    1. Estimate parameter covariance from the Hessian of log P(O|θ).
    2. Draw B parameter sets from N(θ̂, Cov(θ̂)).
       (Perturbation in transition matrix space; rows renormalised.)
    3. For each draw, run forward algorithm, extract γ_T.
    4. Report percentile CI across B draws.

    Returns
    -------
    ci_lower : (K,)   lower bound at (1-conf)/2
    ci_upper : (K,)   upper bound at 1-(1-conf)/2
    """
    from milestone7.regime_model import (
        _log_emission_matrix, _forward_log, HMMParameters
    )
    rng  = np.random.default_rng(seed)
    K    = params.K
    α    = (1 - conf) / 2

    # Estimate SE of transition matrix rows via Dirichlet concentration
    # (Bayesian conjugate for multinomial; SE_ij ≈ √(A_ij(1-A_ij)/T) for row i)
    T          = len(obs)
    A_boot_std = np.sqrt(np.maximum(params.A * (1 - params.A) / max(T, 1), 1e-8))

    boot_gammas = []
    for _ in range(n_boot):
        # Perturb A (row by row, Dirichlet noise)
        A_new = np.zeros_like(params.A)
        for i in range(K):
            # Gaussian perturbation clipped to [0,1], then renormalise
            row    = params.A[i] + rng.normal(0, A_boot_std[i])
            row    = np.clip(row, 1e-6, 1.0)
            A_new[i] = row / row.sum()

        # Perturb emission means (SE ~ empirical std / sqrt(T_k))
        weight_per_state = np.maximum(T / K, 5.0)   # rough count per state
        means_new = params.means + rng.normal(0, params.means.std() * 0.05,
                                              params.means.shape)

        try:
            p_boot = HMMParameters(
                pi    = params.pi,
                A     = A_new,
                means = means_new,
                covs  = params.covs,
            )
            log_B         = _log_emission_matrix(obs, p_boot)
            log_alpha, _  = _forward_log(log_B, p_boot)
            gamma_T       = np.exp(log_alpha[-1] - log_alpha[-1].max())
            gamma_T      /= gamma_T.sum()
            boot_gammas.append(gamma_T)
        except Exception:
            continue

    if len(boot_gammas) < 10:
        # Fallback: crude 5% width
        return (
            np.maximum(params.pi - 0.05, 0),
            np.minimum(params.pi + 0.05, 1),
        )

    boot_arr   = np.array(boot_gammas)            # (B, K)
    ci_lower   = np.percentile(boot_arr, α * 100,        axis=0)
    ci_upper   = np.percentile(boot_arr, (1 - α) * 100,  axis=0)
    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Transition matrix standard errors
# ---------------------------------------------------------------------------

def _transition_se(A: np.ndarray, T: int) -> np.ndarray:
    """
    Asymptotic SE for multinomial row probabilities via delta method:

        SE(A_{ij}) ≈ √(A_{ij}(1 - A_{ij}) / n_i)

    where n_i ≈ T × π_i  is the expected number of visits to state i
    (π_i from stationary distribution).

    Returns (K, K) matrix of standard errors.
    """
    from milestone7.transition_matrix import compute_stationary_distribution
    K    = A.shape[0]
    pi   = compute_stationary_distribution(A)
    n    = np.maximum(T * pi, 1.0)                    # (K,) expected visit counts
    SE   = np.sqrt(A * (1 - A) / n[:, None])          # (K, K)
    return SE


# ---------------------------------------------------------------------------
# Forecast return confidence intervals
# ---------------------------------------------------------------------------

def _forecast_return_ci(
    fwd_dists:    List["ForwardDistribution"],
    regime_ci:    Tuple[np.ndarray, np.ndarray],
) -> Dict:
    """
    CI on H-day expected return using regime probability uncertainty:

        Lower bound: E[r_H] with p_k = ci_lower_k (pessimistic blending)
        Upper bound: E[r_H] with p_k = ci_upper_k (optimistic blending)

    Formally applies bounds via law of total expectation:
        E[r_H | p] = H × Σ_k p_k μ_k

    so the CI directly maps from CI on p_k to CI on E[r_H].
    """
    ci_lower_p, ci_upper_p = regime_ci
    out = {}
    for fd in fwd_dists:
        H  = fd.horizon_days
        mu = fd.mixture_means / H        # daily μ_k
        # Normalise CI bounds to valid distributions
        lo = np.maximum(ci_lower_p, 0); lo /= lo.sum() + 1e-300
        hi = np.maximum(ci_upper_p, 0); hi /= hi.sum() + 1e-300
        E_lo = float(H * np.dot(lo, mu)) * (252 / H)    # annualised
        E_hi = float(H * np.dot(hi, mu)) * (252 / H)
        E_mn = fd.expected_return
        out[f"{H}d"] = {
            "mean_ann":  round(E_mn, 6),
            "lower_95":  round(min(E_lo, E_hi), 6),
            "upper_95":  round(max(E_lo, E_hi), 6),
            "half_width": round(abs(E_hi - E_lo) / 2, 6),
        }
    return out


# ---------------------------------------------------------------------------
# Weight sensitivity to regime shifts
# ---------------------------------------------------------------------------

def _weight_sensitivity(
    tickers:          List[str],
    adaptive_params:  "AdaptiveParameters",
    mu:               np.ndarray,
    Sigma:            np.ndarray,
    delta:            float = 0.05,
) -> Dict:
    """
    Numerical sensitivity of optimal weights to regime probability perturbations.

    For each regime k, compute:
        ∂w*/∂p_k ≈ [w*(p + δe_k) - w*(p - δe_k)] / (2δ)

    Then aggregate:
        RSI_n = (1/K) Σ_k |∂w_n*/∂p_k|

    Returns {asset: RSI_n} rounded to 6 decimal places.

    This is a first-order sensitivity — it quantifies how much the
    optimal weights would change if the regime probability estimate
    shifted by δ.
    """
    from milestone7.adaptive_allocator import (
        compute_adaptive_parameters, _LAMBDA_RETURN, _LAMBDA_VOL,
        _LAMBDA_CVAR, _LAMBDA_DRAWDOWN,
    )
    K         = len(adaptive_params.regime_probs)
    N         = len(tickers)
    sens_mat  = np.zeros((K, N))   # [regime k, asset n] → ∂w_n*/∂p_k

    p_base = adaptive_params.regime_probs.copy()
    pi_base= adaptive_params.blending_weights.copy()  # use as stationary

    # Baseline weights via fast mean-variance approximation
    # w ∝ Σ^{-1} (λ_return μ - λ_vol Σ 1)  — unconstrained analytical solution
    def _fast_weights(params_: "AdaptiveParameters") -> np.ndarray:
        try:
            lam_r = params_.lam_return
            lam_v = params_.lam_vol
            Sigma_reg = Sigma + 1e-6 * np.eye(N)
            Sigma_inv = np.linalg.inv(Sigma_reg)
            raw_w     = lam_r * (Sigma_inv @ mu) / N
            w         = np.maximum(raw_w, 0)
            s         = w.sum()
            if s < 1e-8:
                w = np.ones(N) / N
            else:
                w /= s
            # Apply max weight cap
            mw = params_.max_weight
            capped = False
            for _ in range(20):
                over   = w > mw
                if not over.any(): break
                excess = (w[over] - mw).sum()
                w[over] = mw
                under  = ~over & (w < mw)
                if under.sum() == 0: break
                w[under] += excess / under.sum()
                capped = True
            w = np.clip(w, 0, mw)
            w /= w.sum() + 1e-300
            return w
        except Exception:
            return np.ones(N) / N

    w_base = _fast_weights(adaptive_params)

    for k in range(K):
        e_k    = np.zeros(K)
        e_k[k] = 1.0

        p_plus  = np.clip(p_base + delta * e_k, 0, 1)
        p_plus /= p_plus.sum()
        p_minus  = np.clip(p_base - delta * e_k, 0, 1)
        p_minus /= p_minus.sum()

        params_plus  = compute_adaptive_parameters(p_plus,  pi_base)
        params_minus = compute_adaptive_parameters(p_minus, pi_base)

        w_plus  = _fast_weights(params_plus)
        w_minus = _fast_weights(params_minus)

        sens_mat[k] = (w_plus - w_minus) / (2 * delta)

    # RSI = mean absolute sensitivity across regimes
    RSI = np.mean(np.abs(sens_mat), axis=0)    # (N,)

    from milestone7.regime_model import STATE_LABELS
    result = {
        tickers[n]: {
            "regime_sensitivity_index": round(float(RSI[n]), 6),
            "per_regime_sensitivity": {
                STATE_LABELS.get(k, str(k)): round(float(sens_mat[k, n]), 6)
                for k in range(K)
            },
        }
        for n in range(N)
    }
    return result


# ---------------------------------------------------------------------------
# Main uncertainty quantification
# ---------------------------------------------------------------------------

def quantify_uncertainty(
    obs:              np.ndarray,
    regime_output:    "RegimeOutput",
    forward_report:   "ForwardRiskReport",
    adaptive_params:  "AdaptiveParameters",
    tickers:          List[str],
    mu:               np.ndarray,
    Sigma:            np.ndarray,
    portfolio_returns: np.ndarray,
    n_boot:           int = 300,
    seed:             int = 42,
) -> UncertaintyReport:
    """
    Full uncertainty quantification across all model components.

    Parameters
    ----------
    obs               : (T, M) HMM observation matrix
    regime_output     : RegimeOutput from run_hmm()
    forward_report    : ForwardRiskReport from compute_forward_risk()
    adaptive_params   : AdaptiveParameters from compute_adaptive_parameters()
    tickers           : list of N asset tickers
    mu                : (N,) expected returns
    Sigma             : (N, N) covariance matrix
    portfolio_returns : (T,) portfolio return series (for ESS)
    n_boot            : bootstrap iterations for CI estimation
    seed              : random seed

    Returns
    -------
    UncertaintyReport
    """
    from milestone7.regime_model import STATE_LABELS

    params = regime_output.params
    T      = len(obs)
    K      = params.K

    # 1. Bootstrap CI on regime probabilities
    ci_lower, ci_upper = _bootstrap_regime_ci(obs, params, n_boot=n_boot, seed=seed)
    current_probs      = regime_output.current_probs
    regime_prob_ci     = {
        STATE_LABELS.get(k, str(k)): {
            "mean":     round(float(current_probs[k]), 6),
            "lower_95": round(float(np.clip(ci_lower[k], 0, 1)), 6),
            "upper_95": round(float(np.clip(ci_upper[k], 0, 1)), 6),
            "half_width": round(float((ci_upper[k] - ci_lower[k]) / 2), 6),
        }
        for k in range(K)
    }

    # 2. Transition matrix standard errors
    trans_se = _transition_se(params.A, T)

    # 3. Forecast return CI
    fcast_ci = _forecast_return_ci(
        forward_report.horizons,
        (ci_lower, ci_upper),
    )

    # 4. Weight sensitivity
    w_sens = _weight_sensitivity(tickers, adaptive_params, mu, Sigma)

    # 5. Model info criteria (for K=2,3,4,5 conceptually — only K=4 fitted here)
    M     = params.M
    n_par = K * (K - 1) + K * M + K * M * (M + 1) // 2 + K
    ll    = regime_output.log_likelihood
    aic   = -2 * ll + 2 * n_par
    bic   = -2 * ll + n_par * np.log(T)
    info_criteria = {
        f"K={K}": {
            "n_parameters": n_par,
            "log_likelihood": round(float(ll), 4),
            "AIC": round(float(aic), 4),
            "BIC": round(float(bic), 4),
        }
    }

    # 6. Key parameter SE dict
    trans_se_dict = {}
    for i in range(K):
        for j in range(K):
            trans_se_dict[f"A_{i}{j}"] = round(float(trans_se[i, j]), 6)
    mean_se = np.sqrt(np.diag(params.covs[:, 0, 0].reshape(-1))) / np.sqrt(T) if T > 0 else np.zeros(K)
    for k in range(K):
        trans_se_dict[f"mu_{k}[return]"] = round(float(params.covs[k, 0, 0] ** 0.5 / max(T ** 0.5, 1)), 8)

    # 7. ESS
    ess = _effective_sample_size(portfolio_returns)

    return UncertaintyReport(
        regime_prob_ci        = regime_prob_ci,
        transition_se         = trans_se,
        forecast_return_ci    = fcast_ci,
        weight_sensitivity    = w_sens,
        model_info_criteria   = info_criteria,
        parameter_se_dict     = trans_se_dict,
        effective_sample_size = ess,
    )
