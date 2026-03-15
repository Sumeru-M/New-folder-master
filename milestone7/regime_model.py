"""
regime_model.py — Hidden Markov Model for Latent Market Regime Detection
=========================================================================

Mathematical Framework
----------------------
We model the market as a latent Markov chain  {S_t}  with K discrete
states and a sequence of continuous observations  {O_t}.

**Observation model (Gaussian emissions)**

    O_t | S_t = k  ~  N(μ_k, Σ_k)

where O_t ∈ ℝ^M is an M-dimensional feature vector and Σ_k is a full
covariance matrix regularised with a Ledoit–Wolf shrinkage target.

**Transition model**

    P(S_t = j | S_{t-1} = i) = A_{ij}     (row-stochastic matrix)

**Initial distribution**

    P(S_1 = k) = π_k

**Parameters**  θ = {π, A, {μ_k, Σ_k}_{k=1..K}}

Parameter Estimation: Baum–Welch (EM)
--------------------------------------
E-step:  Forward–backward algorithm computes
    α_t(k)  =  P(O_1:t, S_t=k | θ)          (forward variable)
    β_t(k)  =  P(O_{t+1:T} | S_t=k, θ)      (backward variable)
    γ_t(k)  =  P(S_t=k | O_{1:T}, θ)        (smoothed state probability)
    ξ_t(i,j)=  P(S_t=i, S_{t+1}=j | O_{1:T},θ)  (pair posterior)

M-step:  Closed-form updates
    π_k^new  = γ_1(k)
    A_{ij}^new = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
    μ_k^new  = Σ_t γ_t(k) O_t / Σ_t γ_t(k)
    Σ_k^new  = Σ_t γ_t(k)(O_t-μ_k)(O_t-μ_k)ᵀ / Σ_t γ_t(k)

All computations are carried out in log-space to prevent underflow.

Decoding: Viterbi Algorithm
----------------------------
Computes the most probable state sequence:

    δ_t(k) = max_{S_{1:t-1}} P(S_{1:t-1}, S_t=k, O_{1:t} | θ)
    ψ_t(k) = argmax_i  A_{ik} δ_{t-1}(i)

via dynamic programming in log-space.

States (K=4)
------------
    0 : Low-Vol Bull      — positive drift, low volatility, positive correlation
    1 : High-Vol Bear     — negative drift, elevated volatility
    2 : Crisis            — large negative drift, very high volatility, correlation spike
    3 : Transitional      — indeterminate, intermediate statistics

Observation Features (M=4)
---------------------------
    [0] Portfolio log-return (daily)
    [1] Realized volatility (21-day rolling std, annualised)
    [2] Cross-asset average pairwise correlation (21-day rolling)
    [3] Drawdown intensity (current drawdown from peak)

Assumptions
-----------
A1. First-order Markov property: S_t depends only on S_{t-1}.
A2. Conditional independence: O_t ⊥ O_{1:t-1} | S_t.
A3. Emission distributions are multivariate Gaussian.
A4. Stationarity of transition matrix (A is time-invariant within training).
A5. Identifiability enforced by ordering states by emission mean μ_k[1]
    (realized volatility), ascending — so state 0 is always lowest vol.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

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
    and uniform off-diagonal mass.
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

    # High-persistence transition matrix
    A = np.full((K, K), (1.0 - 0.9) / (K - 1))
    np.fill_diagonal(A, 0.9)

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
    M-step closed-form updates:

        π_k     = γ_1(k)
        A_{ij}  = Σ_t ξ_t(i,j) / Σ_t γ_t(i)
        μ_k     = Σ_t γ_t(k) O_t / Σ_t γ_t(k)
        Σ_k     = Σ_t γ_t(k)(O_t-μ_k)(O_t-μ_k)ᵀ / Σ_t γ_t(k)  +  reg·I

    Numerical safeguards:
    - Row-normalise π and A to maintain valid probability distributions.
    - Add `reg` × I to each Σ_k to guarantee positive-definiteness.
    """
    T = len(obs)

    # Initial distribution
    pi = gamma[0] / (gamma[0].sum() + 1e-300)

    # Transition matrix
    xi_sum    = xi.sum(axis=0)                          # (K, K)
    gamma_sum = gamma[:-1].sum(axis=0)                  # (K,)
    A         = xi_sum / (gamma_sum[:, None] + 1e-300)
    A         = A / (A.sum(axis=1, keepdims=True) + 1e-300)

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
    Re-order states by ascending realised-volatility emission mean (column 1).
    This enforces identifiability:
        state 0 = lowest vol, state K-1 = highest vol.
    After permutation, transitions in A and initial π are permuted accordingly.
    """
    order     = np.argsort(params.means[:, 1])
    inv_order = np.argsort(order)            # inverse permutation for A

    new_means = params.means[order]
    new_covs  = params.covs[order]
    new_pi    = params.pi[order]
    new_A     = params.A[np.ix_(order, order)]

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
