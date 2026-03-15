"""
adaptive_allocator.py — Regime-Driven Optimizer Parameter Modulation
=====================================================================

Mathematical Framework
----------------------

This module translates regime probabilities into modified optimizer inputs.
It does NOT rewrite or replace the optimizer — it only adjusts the
parameters that flow into `optimization_engine.run_optimizer()`.

**Design principle: Parameter Lifting**

The optimizer solves the general problem:

    max_w  U(w; θ)   s.t.  w ∈ C(θ_C)

where θ = {λ_return, λ_vol, λ_cvar, ...} are objective weights and
C(θ_C) = constraints parametrised by {max_weight, target_vol, ...}.

Regime probabilities p = (p_bull, p_bear, p_crisis, p_trans) define a
probability simplex. We map p → (θ, θ_C) via **convex interpolation**
between regime-specific parameter sets:

    θ^regime = Σ_k p_k × θ^(k)

where θ^(k) is the parameter vector designed for pure regime k.

**Regime-specific parameter regimes**

Regime 0 (Low-Vol Bull):
    - High return penalty (λ_return ↑): exploit low-vol environment
    - Low variance penalty (λ_vol ↓):  volatility is benign
    - Low CVaR penalty (λ_cvar ↓):     tail risk is low
    - Relaxed max weight:               allow concentration

Regime 1 (High-Vol Bear):
    - Balanced return/risk tradeoff
    - Elevated variance penalty (λ_vol ↑)
    - Elevated CVaR penalty (λ_cvar ↑)
    - Tighter max weight

Regime 2 (Crisis):
    - Minimum return pursuit (λ_return ↓): capital preservation
    - Maximum variance penalty (λ_vol ↑↑)
    - Maximum CVaR penalty (λ_cvar ↑↑)
    - Tightest max weight: force diversification

Regime 3 (Transitional):
    - Parameters near long-run average (ergodic distribution weighted mean)

**Formal blending**

Let p = (p_0, p_1, p_2, p_3) be the current filtered regime probabilities.

    λ_return^blend  = Σ_k p_k × λ_return^(k)
    λ_vol^blend     = Σ_k p_k × λ_vol^(k)
    λ_cvar^blend    = Σ_k p_k × λ_cvar^(k)
    max_weight^blend= Σ_k p_k × max_weight^(k)

**CVaR confidence level adjustment**

In crisis regimes, we tighten the CVaR confidence level:

    α^blend = 0.95 + p_crisis × 0.04    ∈ [0.95, 0.99]

This makes the CVaR constraint progressively more conservative as
crisis probability rises.

**Risk-free rate adjustment**

The effective risk-free rate used in Sharpe-like objectives is adjusted
upward in bear/crisis regimes to reflect the premium on safe assets:

    rf^blend = rf_base + p_crisis × Δrf_crisis + p_bear × Δrf_bear

where Δrf values encode the flight-to-quality premium.

**Sensitivity to regime uncertainty**

When regime probabilities are diffuse (high entropy), parameters revert
toward their ergodic-weighted mean. This avoids overreaction when the
HMM is uncertain:

    H(p) = -Σ_k p_k log p_k   ∈ [0, log K]   (regime entropy)
    H_max = log K               (maximum entropy = uniform)

    shrinkage = H(p) / H_max    ∈ [0, 1]

    θ^final = (1 - shrinkage) × θ^blend + shrinkage × θ^ergodic

where θ^ergodic = Σ_k π_k × θ^(k) uses the stationary distribution π.

**Leverage / position sizing**

Target volatility scaling (Kelly-inspired):

    position_scale = min(1, target_vol / σ_forecast)

where σ_forecast is the current GARCH conditional volatility.

Assumptions
-----------
A1. Convex interpolation is the principled, model-consistent way to
    combine regime-specific parameter sets (linear in probability space).
A2. Each regime's parameter set encodes a risk preference that is
    explicitly calibrated to that regime's empirical characteristics.
A3. The optimizer itself is treated as a black box; only its inputs change.
A4. No look-ahead: only current filtered (causal) probabilities are used.
A5. Monotone Crisis Response: as p_crisis increases, constraints
    tighten monotonically (enforced by design of θ^(k) vectors).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


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
        from milestone7.regime_model import STATE_LABELS
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

    # CVaR confidence level tightens under crisis
    p_crisis = float(p[_CRISIS]) if K > _CRISIS else 0.0
    p_bear   = float(p[_BEAR])   if K > _BEAR   else 0.0
    p_bull   = float(p[_BULL])   if K > _BULL   else 0.0
    cvar_conf = 0.95 + p_crisis * 0.04    # ∈ [0.95, 0.99]

    # Risk-free rate: flight-to-quality adjustment
    rf_adj  = rf_base + p_crisis * crisis_rf_premium + p_bear * bear_rf_premium

    # Position scale: Kelly-inspired vol targeting
    position_scale = 1.0
    if garch_vol_current is not None and garch_vol_current > 1e-4:
        raw_scale     = tgt_v / garch_vol_current
        position_scale = float(np.clip(raw_scale, 0.1, 1.0))

    # Dominant regime
    dominant = int(np.argmax(p))

    # Recommended optimization method based on regime
    if p_crisis > 0.4:
        method = "cvar"           # tail-risk focused in crisis
    elif p_bull > 0.6:
        method = "mean_variance"  # return/risk tradeoff in bull
    elif p_bear > 0.5:
        method = "min_variance"   # capital preservation in bear
    else:
        method = "multi_objective"  # balanced in uncertainty

    # Narrative notes
    notes = _generate_notes(p, blend_w, shrinkage, p_crisis, p_bear, p_bull,
                            lam_r, lam_v, lam_c, max_w, position_scale)

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
    from milestone7.regime_model import STATE_LABELS
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
