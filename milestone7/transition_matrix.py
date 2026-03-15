"""
transition_matrix.py — Regime Transition Analysis & Multi-Step Forecasting
===========================================================================

Mathematical Framework
----------------------

**Transition matrix A**
    A_{ij} = P(S_t = j | S_{t-1} = i)    (estimated via Baum–Welch)

A is row-stochastic: each row sums to 1, A_{ij} ≥ 0.

**Multi-step transition probabilities**
The h-step transition matrix is the h-th matrix power:

    A^(h) = A^h

so  P(S_{t+h} = j | S_t = i) = [A^h]_{ij}.

**Stationary distribution π**
The stationary (ergodic) distribution satisfies:

    π A = π,    Σ_k π_k = 1

Computed as the left eigenvector of A corresponding to eigenvalue 1.

**Regime duration distribution**
Under the Markov assumption, the duration in state k follows a geometric
distribution with success probability 1 - A_{kk}:

    P(duration = d | regime = k) = A_{kk}^{d-1} (1 - A_{kk})
    E[duration | k] = 1 / (1 - A_{kk})
    Var[duration | k] = A_{kk} / (1 - A_{kk})²

**Conditional forward distribution**
Given current filtered state probabilities γ_T = (γ_T(1), ..., γ_T(K)):

    P(S_{T+h} = j) = Σ_i γ_T(i) × [A^h]_{ij}

This is the h-step-ahead marginal distribution over regimes,
accounting for all possible current-state uncertainty.

**Regime transition risk metrics**
Crisis-transition probability at horizon h:
    P_crisis(h) = Σ_i γ_T(i) × [A^h]_{i, crisis}

Assumptions
-----------
A1. Time-homogeneous transition matrix (A constant across time).
A2. Markov property (transition depends only on current state).
A3. Ergodicity (A is irreducible and aperiodic — guaranteed by adding
    a small epsilon to all entries and renormalising).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


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
    from milestone7.regime_model import STATE_LABELS
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
