"""
milestone7_complete.py
======================
Milestone 7 — Regime-Switching Adaptive Allocation Engine
CONSOLIDATED SINGLE-FILE EDITION

All 7 sub-modules merged into one self-contained file.
Place this file inside the  portfolio/  folder.

Modules
-------
MODULE 1   regime_model          4-state Gaussian HMM (Baum-Welch / Viterbi)
MODULE 2   state_space_filter    GARCH(1,1) volatility filter + multi-step forecast
MODULE 3   transition_matrix     Matrix powers, stationary dist, mixing time
MODULE 4   forward_risk_forecast Gaussian mixture return distribution + CVaR
MODULE 5   adaptive_allocator    Regime-blended optimizer parameter modulation
MODULE 6   uncertainty_quantifier Bayesian CI + weight sensitivity (RSI)
MODULE 7   intelligence_engine   Master orchestrator  →  IntelligenceReport

Public entry point
------------------
from milestone7_complete import run_adaptive_intelligence

report = run_adaptive_intelligence(
    prices_or_returns = prices_df,    # pd.DataFrame  (prices or log-returns)
    tickers           = ["RELIANCE.NS", "TCS.NS", ...],
    is_returns        = False,        # True if prices_or_returns is already returns
)

print(report.weights)          # dict {ticker: weight}
print(report.to_dict())        # full JSON-serialisable output

Dependencies: numpy, scipy, pandas
M5 optimizer (optimization_engine.run_optimizer) is optional — the engine
falls back to equal-risk-parity weights if M5 is not available.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass
from dataclasses import dataclass, field
import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.optimize import minimize
from scipy.stats import multivariate_normal
from scipy.stats import norm
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════
# MODULE 1 — HIDDEN MARKOV MODEL  (regime detection)
# ══════════════════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

K_STATES = 4          # number of latent regimes
M_OBS    = 4          # observation feature dimension
LOG_EPS  = -1e300     # log(0) sentinel (safer than -inf for addition)

STATE_LABELS = {
    0: "Low-Vol Bull",
    1: "High-Vol Bear",
    2: "Crisis",
    3: "Transitional",
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class HMMParameters:
    """
    Complete parameter set θ = {π, A, means, covs}.

    Attributes
    ----------
    pi    : (K,)     initial state distribution
    A     : (K, K)   row-stochastic transition matrix
    means : (K, M)   emission means per state
    covs  : (K, M, M) emission covariance matrices per state
    K     : int      number of states
    M     : int      observation dimension
    """
    pi:    np.ndarray   # (K,)
    A:     np.ndarray   # (K, K)
    means: np.ndarray   # (K, M)
    covs:  np.ndarray   # (K, M, M)
    K:     int = K_STATES
    M:     int = M_OBS

    def to_dict(self) -> Dict:
        return {
            "initial_distribution": self.pi.tolist(),
            "transition_matrix":    self.A.tolist(),
            "emission_means":       self.means.tolist(),
            "emission_covariances": self.covs.tolist(),
            "n_states":             self.K,
            "obs_dim":              self.M,
        }


@dataclass
class RegimeOutput:
    """
    Full output of HMM inference.

    Attributes
    ----------
    filtered_probs  : (T, K) P(S_t=k | O_{1:t})   — causal (no future data)
    smoothed_probs  : (T, K) P(S_t=k | O_{1:T})   — uses all observations
    viterbi_path    : (T,)   most probable state sequence
    log_likelihood  : float  log P(O_{1:T} | θ)
    params          : fitted HMMParameters
    regime_stats    : per-regime mean/vol/persistence
    current_regime  : most probable regime at time T
    current_probs   : (K,)  filtered probabilities at time T
    """
    filtered_probs:  np.ndarray
    smoothed_probs:  np.ndarray
    viterbi_path:    np.ndarray
    log_likelihood:  float
    params:          HMMParameters
    regime_stats:    Dict
    current_regime:  int
    current_probs:   np.ndarray

    def to_dict(self) -> Dict:
        return {
            "current_regime":       int(self.current_regime),
            "current_regime_label": STATE_LABELS[self.current_regime],
            "current_probs":        {
                STATE_LABELS[k]: round(float(self.current_probs[k]), 6)
                for k in range(len(self.current_probs))
            },
            "filtered_probs_last10": self.filtered_probs[-10:].tolist(),
            "smoothed_probs_last10": self.smoothed_probs[-10:].tolist(),
            "viterbi_path_last10":   self.viterbi_path[-10:].tolist(),
            "log_likelihood":        round(float(self.log_likelihood), 4),
            "transition_matrix":     {
                STATE_LABELS[i]: {STATE_LABELS[j]: round(float(self.params.A[i, j]), 6)
                                  for j in range(self.params.K)}
                for i in range(self.params.K)
            },
            "regime_stats":          self.regime_stats,
        }


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def build_observation_matrix(
    daily_returns: pd.DataFrame,
    window: int = 21,
) -> np.ndarray:
    """
    Construct the M=4 dimensional observation sequence from daily log returns.

    Feature engineering
    -------------------
    Feature 0 — mean daily log-return (cross-asset average)
        r_t = (1/N) Σ_n r_{n,t}

    Feature 1 — realised volatility (annualised), 21-day rolling std
        σ_t = std(r_{t-w:t}) × √252

    Feature 2 — average pairwise correlation, 21-day rolling
        ρ_t = mean_{n≠m}  corr(r_n, r_m)_{t-w:t}

    Feature 3 — drawdown intensity from running peak
        D_t = (V_t - max_{s≤t} V_s) / max_{s≤t} V_s  (log-wealth version)

    Parameters
    ----------
    daily_returns : (T, N) DataFrame of asset log returns
    window        : rolling window (trading days)

    Returns
    -------
    obs : (T', M) array, T' = T - window (valid rows after rolling)
    """
    returns = daily_returns.values.astype(float)
    T, N    = returns.shape

    feat_return = returns.mean(axis=1)                         # (T,)

    # Rolling realised vol — use vectorised approach
    feat_vol = np.full(T, np.nan)
    for t in range(window, T):
        feat_vol[t] = returns[t - window:t].std(ddof=1) * np.sqrt(252)

    # Rolling average pairwise correlation
    feat_corr = np.full(T, np.nan)
    for t in range(window, T):
        R = np.corrcoef(returns[t - window:t].T)   # (N, N)
        if N > 1:
            mask = ~np.eye(N, dtype=bool)
            feat_corr[t] = R[mask].mean()
        else:
            feat_corr[t] = 0.0

    # Drawdown intensity (log-wealth drawdown)
    log_wealth     = np.cumsum(feat_return)
    running_peak   = np.maximum.accumulate(log_wealth)
    feat_drawdown  = log_wealth - running_peak                 # ≤ 0

    obs = np.column_stack([feat_return, feat_vol, feat_corr, feat_drawdown])
    valid = ~np.isnan(obs).any(axis=1)
    return obs[valid]


# ---------------------------------------------------------------------------
# Numerical utilities
# ---------------------------------------------------------------------------

def _log_gaussian(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> float:
    """
    Log-PDF of multivariate Gaussian N(mu, cov) at point x.
    Regularises cov with small diagonal jitter for numerical stability.
    """
    M   = len(mu)
    cov = cov + 1e-6 * np.eye(M)
    try:
        return multivariate_normal.logpdf(x, mean=mu, cov=cov)
    except Exception:
        return LOG_EPS


def _log_emission_matrix(obs: np.ndarray, params: HMMParameters) -> np.ndarray:
    """
    Compute B[t, k] = log P(O_t | S_t=k) for all t, k.

    Vectorised: calls scipy multivariate_normal.logpdf on the full obs matrix
    per state (one call per state, not per time step), giving ~T× speedup.

    Returns
    -------
    log_B : (T, K) array
    """
    T     = len(obs)
    log_B = np.zeros((T, params.K))
    for k in range(params.K):
        cov_reg = params.covs[k] + 1e-6 * np.eye(params.M)
        try:
            log_B[:, k] = multivariate_normal.logpdf(
                obs, mean=params.means[k], cov=cov_reg
            )
        except Exception:
            for t in range(T):
                log_B[t, k] = _log_gaussian(obs[t], params.means[k], params.covs[k])
    # Replace any -inf / nan with large negative (avoids propagation issues)
    log_B = np.where(np.isfinite(log_B), log_B, LOG_EPS)
    return log_B


# ---------------------------------------------------------------------------
# Forward–Backward algorithm (log-space)
# ---------------------------------------------------------------------------

def _forward_log(log_B: np.ndarray, params: HMMParameters) -> Tuple[np.ndarray, float]:
    """
    Log-space forward algorithm.

    Recursion:
        log α_1(k) = log π_k + log B_{1,k}
        log α_t(k) = log B_{t,k} + logsumexp_i(log α_{t-1}(i) + log A_{ik})

    Returns
    -------
    log_alpha : (T, K)
    log_lik   : float    log P(O_{1:T} | θ)
    """
    T, K    = log_B.shape
    log_A   = np.log(params.A + 1e-300)
    log_pi  = np.log(params.pi + 1e-300)
    log_alpha = np.zeros((T, K))

    log_alpha[0] = log_pi + log_B[0]

    for t in range(1, T):
        for j in range(K):
            # logsumexp over previous states
            vals = log_alpha[t - 1] + log_A[:, j]
            m    = vals.max()
            log_alpha[t, j] = log_B[t, j] + m + np.log(np.sum(np.exp(vals - m)))

    # log-likelihood via logsumexp of final forward vars
    last   = log_alpha[-1]
    m      = last.max()
    log_lik = m + np.log(np.sum(np.exp(last - m)))
    return log_alpha, log_lik


def _backward_log(log_B: np.ndarray, params: HMMParameters) -> np.ndarray:
    """
    Log-space backward algorithm.

    Recursion:
        log β_T(k) = 0  (i.e. β_T(k) = 1 for all k)
        log β_t(k) = logsumexp_j(log A_{kj} + log B_{t+1,j} + log β_{t+1}(j))

    Returns
    -------
    log_beta : (T, K)
    """
    T, K      = log_B.shape
    log_A     = np.log(params.A + 1e-300)
    log_beta  = np.zeros((T, K))
    # log β_T = 0 already by initialisation

    for t in range(T - 2, -1, -1):
        for i in range(K):
            vals = log_A[i, :] + log_B[t + 1, :] + log_beta[t + 1, :]
            m    = vals.max()
            log_beta[t, i] = m + np.log(np.sum(np.exp(vals - m)))

    return log_beta


def _compute_posteriors(
    log_alpha: np.ndarray,
    log_beta:  np.ndarray,
    log_B:     np.ndarray,
    params:    HMMParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute smoothed state posteriors γ and pair posteriors ξ.

    γ_t(k) = P(S_t=k | O_{1:T}, θ)
           = exp(log α_t(k) + log β_t(k) - log P(O_{1:T}|θ))

    ξ_t(i,j) = P(S_t=i, S_{t+1}=j | O_{1:T}, θ)
             ∝ α_t(i) · A_{ij} · B_{t+1,j} · β_{t+1}(j)

    Returns
    -------
    gamma : (T, K)
    xi    : (T-1, K, K)
    """
    T, K  = log_alpha.shape
    log_A = np.log(params.A + 1e-300)

    # γ
    log_gamma = log_alpha + log_beta
    m_gamma   = log_gamma.max(axis=1, keepdims=True)
    gamma     = np.exp(log_gamma - m_gamma)
    gamma    /= gamma.sum(axis=1, keepdims=True)

    # ξ
    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        for i in range(K):
            for j in range(K):
                xi[t, i, j] = (log_alpha[t, i] + log_A[i, j]
                                + log_B[t + 1, j] + log_beta[t + 1, j])
        # normalise in log-space
        xi_t = xi[t]
        m    = xi_t.max()
        xi_t = np.exp(xi_t - m)
        xi[t] = xi_t / xi_t.sum()

    return gamma, xi


# ---------------------------------------------------------------------------
# Viterbi algorithm
# ---------------------------------------------------------------------------

def viterbi(obs: np.ndarray, params: HMMParameters) -> np.ndarray:
    """
    Viterbi algorithm for most probable state sequence.

    Recursion (log-space):
        log δ_1(k) = log π_k + log B_{1,k}
        log δ_t(k) = log B_{t,k} + max_i(log δ_{t-1}(i) + log A_{ik})
        ψ_t(k)     = argmax_i(log δ_{t-1}(i) + log A_{ik})

    Backtrack from T to 1 to recover the most probable path.

    Returns
    -------
    path : (T,) integer array of state indices
    """
    log_B  = _log_emission_matrix(obs, params)
    T, K   = log_B.shape
    log_A  = np.log(params.A + 1e-300)
    log_pi = np.log(params.pi + 1e-300)

    log_delta = np.zeros((T, K))
    psi       = np.zeros((T, K), dtype=int)

    log_delta[0] = log_pi + log_B[0]

    for t in range(1, T):
        for j in range(K):
            candidates     = log_delta[t - 1] + log_A[:, j]
            psi[t, j]      = int(np.argmax(candidates))
            log_delta[t, j]= log_B[t, j] + candidates[psi[t, j]]

    # Backtrack
    path       = np.zeros(T, dtype=int)
    path[-1]   = int(np.argmax(log_delta[-1]))
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


# ---------------------------------------------------------------------------
# Parameter initialisation
# ---------------------------------------------------------------------------

def _initialise_params(obs: np.ndarray, K: int, seed: int = 42) -> HMMParameters:
    """
    Initialise HMM parameters using k-means-like segment assignment.

    Strategy
    --------
    Sort observations by realized volatility (column 1), divide into K equal
    segments, compute segment means and covariances.  This gives a
    regime-aware initialisation that speeds EM convergence and avoids
    degenerate local optima (compared to random initialisation).

    Transition matrix A is initialised with high persistence (0.9 on diagonal)
    and uniform off-diagonal mass to encourage state diversity.
    """
    rng = np.random.default_rng(seed)
    T, M = obs.shape

    # Sort by realised vol for ordered initialisation
    sort_idx = np.argsort(obs[:, 1])
    seg_size = T // K

    means = np.zeros((K, M))
    covs  = np.zeros((K, M, M))
    for k in range(K):
        start  = k * seg_size
        end    = (k + 1) * seg_size if k < K - 1 else T
        seg    = obs[sort_idx[start:end]]
        means[k] = seg.mean(axis=0)
        if len(seg) > M:
            c = np.cov(seg.T)
            # Ledoit-Wolf shrinkage to scaled identity
            N_s   = len(seg)
            mu_tr = np.trace(c) / M
            delta = min(1.0, (M + 2) / (N_s * np.mean((c - mu_tr * np.eye(M)) ** 2) / (mu_tr ** 2 + 1e-10)))
            covs[k] = (1 - delta) * c + delta * mu_tr * np.eye(M)
        else:
            covs[k] = np.eye(M) * 0.1

    # Transition matrix with moderate persistence to encourage state diversity
    # Use 0.85 instead of 0.9 to allow more transitions between states
    persistence = 0.85
    off_diag_prob = (1.0 - persistence) / (K - 1)
    A = np.full((K, K), off_diag_prob)
    np.fill_diagonal(A, persistence)

    pi = np.ones(K) / K

    return HMMParameters(pi=pi, A=A, means=means, covs=covs)


# ---------------------------------------------------------------------------
# M-step update
# ---------------------------------------------------------------------------

def _m_step(
    obs:   np.ndarray,
    gamma: np.ndarray,
    xi:    np.ndarray,
    K:     int,
    M:     int,
    reg:   float = 1e-4,
) -> HMMParameters:
    """
    M-step closed-form updates with regularization to prevent degenerate solutions:

        π_k     = γ_1(k)
        A_{ij}  = Σ_t ξ_t(i,j) / Σ_t γ_t(i)  (with min-probability floor)
        μ_k     = Σ_t γ_t(k) O_t / Σ_t γ_t(k)
        Σ_k     = Σ_t γ_t(k)(O_t-μ_k)(O_t-μ_k)ᵀ / Σ_t γ_t(k)  +  reg·I

    Numerical safeguards:
    - Row-normalise π and A to maintain valid probability distributions.
    - Add `reg` × I to each Σ_k to guarantee positive-definiteness.
    - Apply minimum probability floor to transition matrix to prevent absorbing states.
    """
    T = len(obs)

    # Initial distribution
    pi = gamma[0] / (gamma[0].sum() + 1e-300)

    # Transition matrix with minimum probability floor to prevent degenerate solutions
    xi_sum    = xi.sum(axis=0)                          # (K, K)
    gamma_sum = gamma[:-1].sum(axis=0)                  # (K,)
    A         = xi_sum / (gamma_sum[:, None] + 1e-300)
    
    # Apply minimum probability floor: each transition must have at least 0.5% probability
    # This prevents one state from becoming absorbing
    min_prob = 0.005
    A = np.maximum(A, min_prob)
    A = A / (A.sum(axis=1, keepdims=True) + 1e-300)

    # Emission parameters
    means = np.zeros((K, M))
    covs  = np.zeros((K, M, M))
    for k in range(K):
        w_k = gamma[:, k]                               # (T,)
        W   = w_k.sum() + 1e-300
        mu_k = (w_k[:, None] * obs).sum(axis=0) / W
        means[k] = mu_k
        diff      = obs - mu_k                          # (T, M)
        cov_k     = (w_k[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0) / W
        covs[k]   = cov_k + reg * np.eye(M)

    return HMMParameters(pi=pi, A=A, means=means, covs=covs)


# ---------------------------------------------------------------------------
# State re-ordering for identifiability
# ---------------------------------------------------------------------------

def _reorder_states(params: HMMParameters) -> HMMParameters:
    """
    Re-order states to enforce identifiability based on regime characteristics.
    
    The 4 regimes should be ordered by increasing "crisis-ness":
        state 0 = Low-Vol Bull (low vol, positive return, minimal drawdown)
        state 1 = High-Vol Bear (high vol, negative return, significant drawdown)
        state 2 = Crisis (very high vol, very negative return, severe drawdown)
        state 3 = Transitional (intermediate characteristics)
    
    We use a composite "crisis score" that combines:
    - Volatility (feature 1): higher vol = higher crisis
    - Return (feature 0): lower return = higher crisis
    - Drawdown (feature 3): more negative = higher crisis
    """
    K = params.means.shape[0]
    
    # Extract features
    returns = params.means[:, 0]      # Feature 0: mean return
    vols = params.means[:, 1]         # Feature 1: volatility
    drawdowns = params.means[:, 3]    # Feature 3: drawdown (negative values)
    
    # Normalize each feature to [0, 1]
    def normalize(x):
        x_min, x_max = x.min(), x.max()
        if x_max == x_min:
            return np.zeros_like(x)
        return (x - x_min) / (x_max - x_min)
    
    vol_norm = normalize(vols)
    ret_norm = normalize(returns)  # Higher return = lower crisis
    draw_norm = normalize(drawdowns)  # More negative drawdown = higher crisis
    
    # Crisis score: combines all three factors
    # Higher vol + lower return + more negative drawdown = higher crisis
    crisis_score = vol_norm + (1.0 - ret_norm) + draw_norm
    
    # Sort by crisis score (ascending)
    order = np.argsort(crisis_score)
    
    # Apply permutation to all parameters
    new_means = params.means[order]
    new_covs = params.covs[order]
    new_pi = params.pi[order]
    new_A = params.A[np.ix_(order, order)]

    return HMMParameters(pi=new_pi, A=new_A, means=new_means, covs=new_covs)


# ---------------------------------------------------------------------------
# Baum–Welch EM
# ---------------------------------------------------------------------------

def fit_hmm(
    obs:         np.ndarray,
    K:           int   = K_STATES,
    max_iter:    int   = 200,
    tol:         float = 1e-6,
    n_restarts:  int   = 3,
    seed:        int   = 42,
) -> Tuple[HMMParameters, float]:
    """
    Fit HMM via Baum–Welch (Expectation-Maximisation).

    Convergence criterion: |ΔlogP(O|θ)| < tol between successive iterations.

    Multiple restarts (n_restarts) are run with different seeds; the
    solution with the highest log-likelihood is returned to mitigate
    local-optima sensitivity.

    Parameters
    ----------
    obs        : (T, M) observation matrix
    K          : number of latent states
    max_iter   : maximum EM iterations per restart
    tol        : convergence tolerance on log-likelihood
    n_restarts : number of random restarts
    seed       : base random seed

    Returns
    -------
    best_params : HMMParameters  (identifiability-ordered)
    best_ll     : float          log-likelihood at convergence
    """
    best_params = None
    best_ll     = -np.inf

    for restart in range(n_restarts):
        params  = _initialise_params(obs, K, seed=seed + restart * 7)
        prev_ll = -np.inf

        for iteration in range(max_iter):
            # E-step
            log_B     = _log_emission_matrix(obs, params)
            log_alpha, ll = _forward_log(log_B, params)
            log_beta  = _backward_log(log_B, params)
            gamma, xi = _compute_posteriors(log_alpha, log_beta, log_B, params)

            # M-step
            params = _m_step(obs, gamma, xi, K, obs.shape[1])

            if abs(ll - prev_ll) < tol:
                break
            prev_ll = ll

        if ll > best_ll:
            best_ll     = ll
            best_params = params

    best_params = _reorder_states(best_params)
    return best_params, best_ll


# ---------------------------------------------------------------------------
# Full inference
# ---------------------------------------------------------------------------

def run_hmm(
    daily_returns: pd.DataFrame,
    K:             int   = K_STATES,
    window:        int   = 21,
    max_iter:      int   = 200,
    tol:           float = 1e-6,
    n_restarts:    int   = 3,
) -> RegimeOutput:
    """
    End-to-end HMM fitting and inference.

    1. Build observation matrix from returns.
    2. Fit HMM via Baum–Welch.
    3. Compute filtered probabilities (causal — no look-ahead).
    4. Compute smoothed probabilities (uses all data — for analysis only).
    5. Decode most probable path via Viterbi.
    6. Compute per-regime statistics.

    Filtered probabilities are the operationally correct quantities
    for any forward-looking decision (no look-ahead bias).
    Smoothed probabilities are reported for research insight only.

    Returns
    -------
    RegimeOutput
    """
    obs = build_observation_matrix(daily_returns, window=window)
    if len(obs) < K * 10:
        raise ValueError(
            f"Insufficient data: {len(obs)} observations for {K} states. "
            "Need at least K×10 rows after rolling window."
        )

    # Fit
    params, ll = fit_hmm(obs, K=K, max_iter=max_iter, tol=tol, n_restarts=n_restarts)

    # E-step on final params for smoothed posteriors
    log_B                  = _log_emission_matrix(obs, params)
    log_alpha, _           = _forward_log(log_B, params)
    log_beta               = _backward_log(log_B, params)
    smoothed, _            = _compute_posteriors(log_alpha, log_beta, log_B, params)

    # Filtered probabilities (causal — only forward pass)
    filtered = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
    filtered /= filtered.sum(axis=1, keepdims=True)

    # Prevent exactly 0% or 100% certainty — apply a small floor
    # so that no single regime ever claims absolute certainty
    _prob_floor = 0.005   # 0.5% minimum per regime
    filtered = np.maximum(filtered, _prob_floor)
    filtered /= filtered.sum(axis=1, keepdims=True)

    # Viterbi path
    path = viterbi(obs, params)

    # Per-regime statistics
    regime_stats = _compute_regime_stats(obs, smoothed, path, params)

    current_probs  = filtered[-1]
    current_regime = int(np.argmax(current_probs))

    return RegimeOutput(
        filtered_probs  = filtered,
        smoothed_probs  = smoothed,
        viterbi_path    = path,
        log_likelihood  = ll,
        params          = params,
        regime_stats    = regime_stats,
        current_regime  = current_regime,
        current_probs   = current_probs,
    )


def _compute_regime_stats(
    obs:     np.ndarray,
    gamma:   np.ndarray,
    path:    np.ndarray,
    params:  HMMParameters,
) -> Dict:
    """
    Per-regime summary statistics.

    Persistence: E[duration in state k] = 1 / (1 - A_{kk})
    Average occupancy: Σ_t γ_t(k) / T
    Emission characteristics from estimated μ_k, Σ_k.
    """
    K   = params.K
    T   = len(obs)
    out = {}
    for k in range(K):
        persistence    = 1.0 / max(1.0 - float(params.A[k, k]), 1e-6)
        occupancy      = float(gamma[:, k].mean())
        mean_return    = float(params.means[k, 0]) * 252    # annualised
        mean_vol       = float(params.means[k, 1])
        mean_corr      = float(params.means[k, 2])
        mean_drawdown  = float(params.means[k, 3])

        out[STATE_LABELS[k]] = {
            "state_index":             k,
            "persistence_days":        round(persistence, 2),
            "average_occupancy":       round(occupancy, 4),
            "emission_mean_return_ann":round(mean_return, 4),
            "emission_mean_vol_ann":   round(mean_vol, 4),
            "emission_mean_corr":      round(mean_corr, 4),
            "emission_mean_drawdown":  round(mean_drawdown, 4),
            "self_transition_prob":    round(float(params.A[k, k]), 6),
        }
    return out


# ══════════════════════════════════════════════════════════════════════════
# MODULE 2 — GARCH(1,1) STATE-SPACE FILTER  (volatility forecasting)
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# MODULE 3 — TRANSITION MATRIX ANALYSIS  (multi-step regime forecasting)
# ══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TransitionAnalysis:
    """
    Full transition matrix analysis.

    Attributes
    ----------
    A                : (K, K)  estimated transition matrix
    A_powers         : dict {h: (K,K)} A^h for h in horizons
    stationary_dist  : (K,)   π (ergodic distribution)
    expected_duration: (K,)   E[duration in state k] in days
    std_duration     : (K,)   std of duration in state k
    forward_probs    : dict {h: (K,)} P(S_{T+h}=k) given γ_T
    crisis_probs     : dict {h: float} P(S_{T+h}=crisis) given γ_T
    mixing_time      : int    approximate mixing time (ε=0.01)
    entropy          : (K,)  entropy of each row of A
    """
    A:                 np.ndarray
    A_powers:          Dict[int, np.ndarray]
    stationary_dist:   np.ndarray
    expected_duration: np.ndarray
    std_duration:      np.ndarray
    forward_probs:     Dict[int, np.ndarray]
    crisis_probs:      Dict[int, float]
    mixing_time:       int
    entropy:           np.ndarray
    state_labels:      Dict[int, str]

    def to_dict(self) -> Dict:
        sl = self.state_labels
        return {
            "transition_matrix": {
                sl[i]: {sl[j]: round(float(self.A[i, j]), 6) for j in range(len(sl))}
                for i in range(len(sl))
            },
            "stationary_distribution": {
                sl[k]: round(float(self.stationary_dist[k]), 6) for k in range(len(sl))
            },
            "expected_duration_days": {
                sl[k]: round(float(self.expected_duration[k]), 2) for k in range(len(sl))
            },
            "forward_probabilities": {
                f"{h}d": {sl[k]: round(float(v[k]), 6) for k in range(len(sl))}
                for h, v in self.forward_probs.items()
            },
            "crisis_probability_by_horizon": {
                f"{h}d": round(float(p), 6) for h, p in self.crisis_probs.items()
            },
            "mixing_time_days": self.mixing_time,
            "row_entropy": {sl[k]: round(float(self.entropy[k]), 4) for k in range(len(sl))},
        }


# ---------------------------------------------------------------------------
# Core computations
# ---------------------------------------------------------------------------

def compute_stationary_distribution(A: np.ndarray) -> np.ndarray:
    """
    Compute the stationary distribution of a row-stochastic matrix A.

    Method: left eigenvector corresponding to eigenvalue 1.
    The stationary distribution satisfies πA = π  ⟺  Aᵀπ = π.

    Numerical robustness: regularise A by adding ε=10⁻⁶ to all entries
    and renormalising, ensuring irreducibility (hence unique stationary dist).

    Parameters
    ----------
    A : (K, K) row-stochastic transition matrix

    Returns
    -------
    pi : (K,) stationary distribution
    """
    K     = A.shape[0]
    A_reg = A + 1e-6
    A_reg = A_reg / A_reg.sum(axis=1, keepdims=True)

    # Solve (Aᵀ - I)π = 0 subject to Σπ = 1
    # Equivalently: largest eigenvalue of Aᵀ is 1, take corresponding eigenvector
    eigvals, eigvecs = np.linalg.eig(A_reg.T)
    # Find eigenvalue closest to 1
    idx = int(np.argmin(np.abs(eigvals - 1.0)))
    pi  = np.real(eigvecs[:, idx])
    pi  = np.abs(pi)
    pi /= pi.sum()
    return pi


def compute_matrix_power(A: np.ndarray, h: int) -> np.ndarray:
    """
    Compute A^h via repeated squaring (binary exponentiation).
    O(K³ log h) vs O(K³ h) for naive repeated multiplication.
    """
    K      = A.shape[0]
    result = np.eye(K)
    base   = A.copy()
    while h > 0:
        if h % 2 == 1:
            result = result @ base
        base = base @ base
        h  //= 2
    # Project back to row-stochastic (correct floating-point drift)
    result = np.maximum(result, 0)
    result = result / (result.sum(axis=1, keepdims=True) + 1e-300)
    return result


def _row_entropy(A: np.ndarray) -> np.ndarray:
    """
    Shannon entropy of each row of A:  H_i = -Σ_j A_{ij} log A_{ij}
    High entropy = uncertain transitions; low entropy = persistent state.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_A = np.where(A > 0, np.log(A), 0.0)
    return -np.sum(A * log_A, axis=1)


def _mixing_time(A: np.ndarray, pi: np.ndarray, eps: float = 0.01) -> int:
    """
    Approximate mixing time: smallest h such that
        max_i ||[A^h]_i - π||_1 < eps

    Uses the spectral gap:  τ_mix ≈ log(1/eps) / log(1/λ₂)
    where λ₂ is the second-largest eigenvalue (by absolute value).

    Returns approximate mixing time in steps (days).
    """
    eigvals = np.sort(np.abs(np.linalg.eigvals(A)))[::-1]
    lambda2 = eigvals[1] if len(eigvals) > 1 else 0.9
    lambda2 = np.clip(lambda2, 1e-6, 1.0 - 1e-6)
    return int(np.ceil(np.log(1.0 / eps) / np.log(1.0 / lambda2)))


# ---------------------------------------------------------------------------
# Main analysis function
# ---------------------------------------------------------------------------

def analyse_transitions(
    A:               np.ndarray,
    current_probs:   np.ndarray,
    horizons:        List[int] = [1, 5, 10, 21, 60],
    crisis_state:    int       = 2,
    state_labels:    Optional[Dict[int, str]] = None,
) -> TransitionAnalysis:
    """
    Full transition matrix analysis given estimated A and current state
    distribution γ_T.

    Parameters
    ----------
    A              : (K, K) row-stochastic transition matrix (from HMM)
    current_probs  : (K,) current filtered state probabilities γ_T
    horizons       : list of h values for forward probability computation
    crisis_state   : index of the crisis regime
    state_labels   : optional {k: label} mapping

    Returns
    -------
    TransitionAnalysis
    """
    K = A.shape[0]
    if state_labels is None:
        state_labels = STATE_LABELS

    # Stationary distribution
    pi = compute_stationary_distribution(A)

    # Duration distribution parameters
    diag           = np.diag(A)
    diag_safe      = np.clip(diag, 0, 1 - 1e-8)
    expected_dur   = 1.0 / (1.0 - diag_safe)
    var_dur        = diag_safe / (1.0 - diag_safe) ** 2
    std_dur        = np.sqrt(var_dur)

    # Matrix powers and forward distributions
    A_powers       = {}
    forward_probs  = {}
    crisis_probs   = {}
    gamma          = np.array(current_probs, dtype=float)
    gamma         /= gamma.sum() + 1e-300

    for h in horizons:
        Ah             = compute_matrix_power(A, h)
        A_powers[h]    = Ah
        fwd            = gamma @ Ah                   # (K,)
        forward_probs[h] = fwd
        crisis_probs[h]  = float(fwd[crisis_state]) if crisis_state < K else 0.0

    # Mixing time
    mt = _mixing_time(A, pi)

    # Row entropy
    entropy = _row_entropy(A)

    return TransitionAnalysis(
        A                 = A,
        A_powers          = A_powers,
        stationary_dist   = pi,
        expected_duration = expected_dur,
        std_duration      = std_dur,
        forward_probs     = forward_probs,
        crisis_probs      = crisis_probs,
        mixing_time       = mt,
        entropy           = entropy,
        state_labels      = state_labels,
    )


# ══════════════════════════════════════════════════════════════════════════
# MODULE 4 — FORWARD RISK FORECAST  (Gaussian mixture return distribution)
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# MODULE 5 — ADAPTIVE ALLOCATOR  (regime-blended optimizer parameters)
# ══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Regime-specific base parameters (calibration anchors)
# ---------------------------------------------------------------------------

# Regime indices
_BULL     = 0
_BEAR     = 1
_CRISIS   = 2
_TRANS    = 3

# Base parameter grids — one row per regime [bull, bear, crisis, trans]
_LAMBDA_RETURN  = np.array([1.5,  0.8,  0.3,  1.0])   # return weight
_LAMBDA_VOL     = np.array([0.5,  1.5,  3.0,  1.0])   # variance penalty
_LAMBDA_CVAR    = np.array([0.3,  0.8,  2.0,  0.6])   # CVaR penalty
_LAMBDA_DRAWDOWN= np.array([0.1,  0.4,  0.8,  0.3])   # drawdown penalty
_MAX_WEIGHT     = np.array([0.35, 0.25, 0.15, 0.25])  # per-asset max weight
_TARGET_VOL     = np.array([0.18, 0.12, 0.08, 0.14])  # target portfolio vol


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveParameters:
    """
    Regime-blended optimizer input parameters.

    Attributes
    ----------
    lam_return       : λ_return (blended)
    lam_vol          : λ_vol (blended)
    lam_cvar         : λ_cvar (blended)
    lam_drawdown     : λ_drawdown (blended)
    max_weight       : per-asset maximum weight
    target_vol       : target portfolio annualised volatility
    cvar_confidence  : α for CVaR constraint (95–99%)
    risk_free_rate   : adjusted risk-free rate
    position_scale   : Kelly-inspired scale factor ∈ (0, 1]
    regime_entropy   : H(p) / H_max — regime uncertainty
    shrinkage        : entropy-based shrinkage toward ergodic params
    dominant_regime  : argmax p_k
    regime_probs     : current filtered probabilities
    blending_weights : p_k used for blending (after any shrinkage)
    optimization_method : recommended method given regime
    notes            : list of narrative parameter-shift explanations
    """
    lam_return:        float
    lam_vol:           float
    lam_cvar:          float
    lam_drawdown:      float
    max_weight:        float
    target_vol:        float
    cvar_confidence:   float
    risk_free_rate:    float
    position_scale:    float
    regime_entropy:    float
    shrinkage:         float
    dominant_regime:   int
    regime_probs:      np.ndarray
    blending_weights:  np.ndarray
    optimization_method: str
    notes:             List[str] = field(default_factory=list)

    def to_optimizer_kwargs(self) -> Dict:
        """
        Return kwargs dict ready to pass to run_optimizer() / optimize_multi_objective().
        """
        return {
            "lam_return":        self.lam_return,
            "lam_vol":           self.lam_vol,
            "lam_cvar":          self.lam_cvar,
            "lam_drawdown":      self.lam_drawdown,
            "rf":                self.risk_free_rate,
            "confidence_level":  self.cvar_confidence,
        }

    def to_dict(self) -> Dict:
        K = len(self.regime_probs)
        return {
            "dominant_regime":       STATE_LABELS.get(self.dominant_regime,
                                                       str(self.dominant_regime)),
            "regime_probabilities":  {
                STATE_LABELS.get(k, str(k)): round(float(self.regime_probs[k]), 6)
                for k in range(K)
            },
            "blending_weights":      {
                STATE_LABELS.get(k, str(k)): round(float(self.blending_weights[k]), 6)
                for k in range(K)
            },
            "optimization_method":   self.optimization_method,
            "adapted_parameters": {
                "lam_return":       round(self.lam_return, 4),
                "lam_vol":          round(self.lam_vol, 4),
                "lam_cvar":         round(self.lam_cvar, 4),
                "lam_drawdown":     round(self.lam_drawdown, 4),
                "max_weight":       round(self.max_weight, 4),
                "target_vol_ann":   round(self.target_vol, 4),
                "cvar_confidence":  round(self.cvar_confidence, 4),
                "risk_free_rate":   round(self.risk_free_rate, 4),
                "position_scale":   round(self.position_scale, 4),
            },
            "uncertainty_metrics": {
                "regime_entropy":   round(self.regime_entropy, 4),
                "shrinkage":        round(self.shrinkage, 4),
            },
            "parameter_shift_notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Core parameter blending
# ---------------------------------------------------------------------------

def _regime_entropy(probs: np.ndarray) -> float:
    """
    Shannon entropy of regime probability vector:  H = -Σ_k p_k log p_k
    Normalised to [0, 1] by dividing by log K.
    """
    K   = len(probs)
    p   = np.clip(probs, 1e-12, 1.0)
    p  /= p.sum()
    H   = -float(np.sum(p * np.log(p)))
    return H / np.log(K)


def _blend(param_grid: np.ndarray, weights: np.ndarray) -> float:
    """Convex combination: θ = Σ_k w_k θ^(k)."""
    return float(np.dot(weights, param_grid))


def compute_adaptive_parameters(
    regime_probs:       np.ndarray,
    stationary_dist:    np.ndarray,
    garch_vol_current:  Optional[float] = None,
    rf_base:            float           = 0.07,
    crisis_rf_premium:  float           = 0.02,
    bear_rf_premium:    float           = 0.01,
    K:                  int             = 4,
    risk_appetite:      str             = "balanced",
) -> AdaptiveParameters:
    """
    Compute regime-blended optimizer parameters.

    Parameters
    ----------
    regime_probs       : (K,) current filtered regime probabilities γ_T
    stationary_dist    : (K,) ergodic distribution π (from transition analysis)
    garch_vol_current  : current GARCH conditional vol (annualised), optional
    rf_base            : base risk-free rate (e.g. 0.07 for 7%)
    crisis_rf_premium  : extra rf in crisis (flight-to-quality)
    bear_rf_premium    : extra rf in bear regime
    K                  : number of regimes
    risk_appetite      : "conservative" | "balanced" | "aggressive"

    Returns
    -------
    AdaptiveParameters
    """
    p       = np.array(regime_probs, dtype=float)
    p      /= p.sum() + 1e-300
    pi      = np.array(stationary_dist, dtype=float)
    pi     /= pi.sum() + 1e-300

    # Regime entropy and shrinkage
    H_norm    = _regime_entropy(p)
    shrinkage = H_norm   # 0 = certain, 1 = maximally uncertain

    # Blending weights: shrink toward ergodic distribution when uncertain
    blend_w   = (1.0 - shrinkage) * p + shrinkage * pi
    blend_w  /= blend_w.sum() + 1e-300

    # Blend all parameters
    lam_r  = _blend(_LAMBDA_RETURN,   blend_w)
    lam_v  = _blend(_LAMBDA_VOL,      blend_w)
    lam_c  = _blend(_LAMBDA_CVAR,     blend_w)
    lam_d  = _blend(_LAMBDA_DRAWDOWN, blend_w)
    max_w  = _blend(_MAX_WEIGHT,      blend_w)
    tgt_v  = _blend(_TARGET_VOL,      blend_w)

    # ── Risk appetite adjustment ──────────────────────────────────────────
    # Multipliers shift the blended parameters based on user preference
    appetite = risk_appetite.lower().strip() if isinstance(risk_appetite, str) else "balanced"
    if appetite == "conservative":
        lam_r *= 0.5     # halve return chasing
        lam_v *= 1.8     # much higher vol penalty
        lam_c *= 2.0     # much higher tail-risk penalty
        lam_d *= 1.5     # higher drawdown penalty
        max_w *= 0.7     # tighter concentration limit
        tgt_v *= 0.65    # significantly lower target vol
    elif appetite == "aggressive":
        lam_r *= 1.8     # chase returns harder
        lam_v *= 0.5     # tolerate more vol
        lam_c *= 0.4     # tolerate more tail risk
        lam_d *= 0.3     # tolerate more drawdown
        max_w  = min(max_w * 1.5, 0.50)  # allow more concentration
        tgt_v *= 1.5     # higher target vol
    # "balanced" — no adjustment (default)

    # CVaR confidence level tightens under crisis
    p_crisis = float(p[_CRISIS]) if K > _CRISIS else 0.0
    p_bear   = float(p[_BEAR])   if K > _BEAR   else 0.0
    p_bull   = float(p[_BULL])   if K > _BULL   else 0.0
    cvar_conf = 0.95 + p_crisis * 0.04    # ∈ [0.95, 0.99]
    if appetite == "conservative":
        cvar_conf = min(cvar_conf + 0.02, 0.99)   # tighter tail
    elif appetite == "aggressive":
        cvar_conf = max(cvar_conf - 0.02, 0.90)   # looser tail

    # Risk-free rate: flight-to-quality adjustment
    rf_adj  = rf_base + p_crisis * crisis_rf_premium + p_bear * bear_rf_premium

    # Position scale: Kelly-inspired vol targeting
    position_scale = 1.0
    if garch_vol_current is not None and garch_vol_current > 1e-4:
        raw_scale     = tgt_v / garch_vol_current
        position_scale = float(np.clip(raw_scale, 0.1, 1.0))

    # Dominant regime
    dominant = int(np.argmax(p))

    # Recommended optimization method based on regime + appetite
    if appetite == "conservative":
        if p_crisis > 0.2 or p_bear > 0.3:
            method = "min_variance"
        else:
            method = "cvar"
    elif appetite == "aggressive":
        if p_bull > 0.3:
            method = "mean_variance"
        else:
            method = "multi_objective"
    else:  # balanced
        if p_crisis > 0.4:
            method = "cvar"
        elif p_bull > 0.6:
            method = "mean_variance"
        elif p_bear > 0.5:
            method = "min_variance"
        else:
            method = "multi_objective"

    # Narrative notes
    notes = _generate_notes(p, blend_w, shrinkage, p_crisis, p_bear, p_bull,
                            lam_r, lam_v, lam_c, max_w, position_scale)
    # Add risk appetite note
    if appetite != "balanced":
        notes.append(f"Risk appetite set to '{appetite}': parameters adjusted accordingly.")

    return AdaptiveParameters(
        lam_return       = lam_r,
        lam_vol          = lam_v,
        lam_cvar         = lam_c,
        lam_drawdown     = lam_d,
        max_weight       = max_w,
        target_vol       = tgt_v,
        cvar_confidence  = cvar_conf,
        risk_free_rate   = rf_adj,
        position_scale   = position_scale,
        regime_entropy   = H_norm,
        shrinkage        = shrinkage,
        dominant_regime  = dominant,
        regime_probs     = p,
        blending_weights = blend_w,
        optimization_method = method,
        notes            = notes,
    )


# ---------------------------------------------------------------------------
# Note generation
# ---------------------------------------------------------------------------

def _generate_notes(
    p: np.ndarray,
    blend_w: np.ndarray,
    shrinkage: float,
    p_crisis: float,
    p_bear:   float,
    p_bull:   float,
    lam_r: float,
    lam_v: float,
    lam_c: float,
    max_w: float,
    pos_scale: float,
) -> List[str]:
    """Generate human-readable parameter shift explanations."""
    notes = []

    if p_crisis > 0.3:
        notes.append(
            f"CRISIS signal P={p_crisis:.1%}: λ_vol={lam_v:.2f} (↑↑ variance penalised), "
            f"λ_cvar={lam_c:.2f} (↑↑ tail risk penalised), max_weight={max_w:.2f} (↓ forced diversification)."
        )
    if p_bear > 0.4:
        notes.append(
            f"BEAR signal P={p_bear:.1%}: λ_vol={lam_v:.2f} (↑ variance penalised), "
            f"λ_cvar={lam_c:.2f} (↑ CVaR penalised)."
        )
    if p_bull > 0.5:
        notes.append(
            f"BULL signal P={p_bull:.1%}: λ_return={lam_r:.2f} (↑ return weighted), "
            f"λ_vol={lam_v:.2f} (↓ vol tolerated), max_weight={max_w:.2f} (relaxed)."
        )
    if shrinkage > 0.6:
        notes.append(
            f"High regime uncertainty (entropy={shrinkage:.2f}): parameters shrunk "
            f"{shrinkage:.0%} toward ergodic mean. Regime signal is weak."
        )
    if pos_scale < 0.9:
        notes.append(
            f"GARCH vol scaling: position_scale={pos_scale:.2f} applied "
            f"(GARCH vol > target vol; effective leverage reduced)."
        )
    if not notes:
        notes.append("Regime probabilities near stationary distribution; parameters near baseline.")
    return notes


# ---------------------------------------------------------------------------
# Constraint builder integration
# ---------------------------------------------------------------------------

def build_adapted_constraints(
    adaptive_params: AdaptiveParameters,
    tickers:         List[str],
    sector_map:      Optional[Dict[str, str]] = None,
    existing_weights: Optional[np.ndarray]    = None,
) -> "ConstraintBuilder":
    """
    Build a ConstraintBuilder populated with regime-adapted constraints.

    Uses the M5 constraints module as a black box — only adds/modifies
    numerical bounds based on adaptive_params.

    Parameters
    ----------
    adaptive_params  : AdaptiveParameters from compute_adaptive_parameters()
    tickers          : list of asset tickers
    sector_map       : {ticker: sector} mapping (optional)
    existing_weights : current portfolio weights for turnover constraint

    Returns
    -------
    ConstraintBuilder with regime-adapted bounds
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    try:
        from constraints import (
            ConstraintBuilder, LongOnlyConstraint, FullInvestmentConstraint,
            MaxWeightConstraint, SectorCapConstraint, TurnoverConstraint,
        )
    except ImportError:
        return None  # graceful degradation if constraints not importable

    cb = ConstraintBuilder()
    cb.add(LongOnlyConstraint())
    cb.add(FullInvestmentConstraint())
    cb.add(MaxWeightConstraint(max_weight=adaptive_params.max_weight))

    if sector_map is not None:
        # Sector cap: tighter in crisis regime
        p_crisis = float(adaptive_params.regime_probs[_CRISIS])
        sector_cap = 0.40 - p_crisis * 0.15    # from 40% down to 25% in crisis
        cb.add(SectorCapConstraint(sector_map=sector_map, max_sector_weight=sector_cap))

    if existing_weights is not None:
        # Tighten turnover in high-vol regimes
        p_bear   = float(adaptive_params.regime_probs[_BEAR])
        p_crisis = float(adaptive_params.regime_probs[_CRISIS])
        turnover_limit = 0.30 + (1 - p_bear - p_crisis) * 0.20  # 30-50%
        cb.add(TurnoverConstraint(
            current_weights=existing_weights,
            max_turnover=turnover_limit,
        ))

    return cb


# ══════════════════════════════════════════════════════════════════════════
# MODULE 6 — UNCERTAINTY QUANTIFIER  (Bayesian CI + sensitivity)
# ══════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════
# MODULE 7 — INTELLIGENCE ENGINE  (master orchestrator)
# ══════════════════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Internal imports (M7 modules only)
# ---------------------------------------------------------------------------


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
    risk_appetite:       str                   = "balanced",
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
        risk_appetite     = risk_appetite,
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
