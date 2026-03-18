"""
milestone8_complete.py
======================
Milestone 8 — Autonomous Decision Intelligence Layer
CONSOLIDATED SINGLE-FILE EDITION

Consumes outputs from M4 (risk), M5 (optimisation), M6/M7 (security and
regime intelligence) and produces actionable, explainable portfolio
decisions. Nothing in M1-M7 is modified.

Modules
-------
MODULE 1   decision_engine        Probabilistic signal model → FinancialDecision
MODULE 2   strategy_adapter       Regime-aware parameter modulation → AdaptedStrategy
MODULE 3   security_aware_logic   Threat integration + action override → SecurityConstraints
MODULE 4   explainability         Human-readable decision reasoning → DecisionExplanation
MODULE 5   recommendation_api     Master pipeline + audit logging → SystemRecommendation

Public entry point
------------------
from milestone8_complete import RecommendationEngine, SystemState

engine = RecommendationEngine()
rec    = engine.get_recommendation(
             m7_dict  = intelligence_report.to_dict(),
             m4_risk  = {"volatility": 0.18, "var_95": 0.025,
                         "cvar_95": 0.04, "max_drawdown": 0.12},
             m5_result= {"expected_return_ann": 0.14, "sharpe_ratio": 0.72,
                         "weights": {"RELIANCE.NS": 0.4, "TCS.NS": 0.6}},
             security = {"pqc": pqc_report.to_dict(),
                         "bayesian": bay_report.to_dict()},
             scenario = None,
         )
print(rec.to_dict())

Dependencies: numpy (only)
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import time
import uuid
import numpy as np
from typing import Any, Dict, List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════════════
# MODULE 1 — DECISION ENGINE  (core probabilistic signal model)
# ══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Thresholds (calibrated for NSE equity portfolios)
# ---------------------------------------------------------------------------

VOL_THRESHOLD    = 0.20    # annualised volatility alarm level
VAR_THRESHOLD    = 0.03    # 95% daily VaR alarm level (fraction of portfolio)
CVAR_THRESHOLD   = 0.05    # 95% daily CVaR alarm level
DD_THRESHOLD     = 0.15    # max drawdown alarm level
RETURN_THRESHOLD = 0.08    # minimum acceptable expected return (risk-free proxy)

# Signal weights (must sum to 1.0)
SIGNAL_WEIGHTS = {
    "regime":  0.30,
    "vol":     0.20,
    "var":     0.10,
    "cvar":    0.10,
    "dd":      0.10,
    "ret":     0.10,
    "sharpe":  0.10,
}

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9

# Action thresholds
ACTION_THRESHOLDS = {
    "INCREASE_EXPOSURE": 0.30,
    "HOLD":             -0.10,
    "REDUCE_RISK":      -0.50,
    # below -0.50 → REBALANCE
}


# ---------------------------------------------------------------------------
# Input data class
# ---------------------------------------------------------------------------

@dataclass
class SystemState:
    """
    Consolidated snapshot of all module outputs consumed by M8.

    All fields are optional with safe defaults so the decision engine
    degrades gracefully when some modules have not been run.

    Sources:
        M4 / M5  →  risk and optimisation fields
        M7       →  regime and forward forecast fields
        M6 / M7 security  →  threat fields
    """
    # ── M4 / M5 Risk & Optimisation ──────────────────────────────────────
    volatility_ann:    float = 0.20     # annualised portfolio volatility
    var_95:            float = 0.03     # 95% VaR as fraction
    cvar_95:           float = 0.05     # 95% CVaR as fraction
    max_drawdown:      float = 0.10     # historical max drawdown (positive)
    expected_return:   float = 0.10     # expected annual return
    sharpe_ratio:      float = 0.50     # Sharpe ratio
    optimal_weights:   Dict[str, float] = field(default_factory=dict)
    tickers:           List[str]        = field(default_factory=list)

    # ── M7 Regime ─────────────────────────────────────────────────────────
    current_regime:    str   = "Transitional"
    regime_probs:      Dict[str, float] = field(default_factory=lambda: {
        "Low-Vol Bull": 0.25, "High-Vol Bear": 0.25,
        "Crisis": 0.25, "Transitional": 0.25
    })
    forward_return_21d:  float = 0.0
    forward_vol_21d:     float = 0.20
    forward_cvar_21d:    float = 0.05
    garch_vol_current:   float = 0.20
    garch_vol_30d:       float = 0.20
    mixing_time_days:    int   = 500
    regime_entropy:      float = 0.5    # 0 = certain, 1 = uncertain

    # ── M6 / M7 Security ──────────────────────────────────────────────────
    pqc_threat_level:        str   = "low"      # "low" | "medium" | "high"
    pqc_anomaly_score:       float = 0.0
    bayesian_posterior:      float = 0.05       # P(threat | evidence)
    bayesian_threat_level:   str   = "SAFE"     # SAFE/MONITOR/ELEVATED_RISK/CRITICAL_THREAT
    transaction_quarantined: bool  = False
    key_rotation_active:     bool  = False

    # ── Scenario context ─────────────────────────────────────────────────
    scenario:          Optional[str] = None     # "market_crash" | "high_inflation" | etc.

    # ── Metadata ──────────────────────────────────────────────────────────
    timestamp:         float = field(default_factory=time.time)

    @classmethod
    def from_m7_dict(cls, m7: Dict, m4_risk: Optional[Dict] = None,
                     m5_result: Optional[Dict] = None,
                     security: Optional[Dict] = None) -> "SystemState":
        """
        Convenience constructor: build SystemState from raw M7/M6 JSON dicts.

        Parameters
        ----------
        m7         : output of IntelligenceReport.to_dict()
        m4_risk    : {"volatility": f, "var_95": f, "cvar_95": f, "max_drawdown": f}
        m5_result  : AllocationResult-like dict with expected_return, volatility, etc.
        security   : combined dict with pqc and bayesian sub-keys
        """
        # Regime
        rp     = m7.get("regime_probabilities", {})
        probs  = rp.get("current_probs", {})
        regime = rp.get("current_regime_label", "Transitional")

        # Volatility + forward forecasts
        vf     = m7.get("volatility_forecast", {})
        frf    = m7.get("forward_return_forecast", {}).get("forward_distributions", {})
        fd21   = frf.get("21d", {})
        ap     = m7.get("adaptive_parameter_shift", {})
        ue     = ap.get("uncertainty_metrics", {})

        # Allocation
        oa     = m7.get("optimal_allocation", {})
        tickers= m7.get("meta", {}).get("tickers", [])

        # Risk — prefer M4/M5 if provided, fall back to M7 forward estimates
        if m4_risk:
            vol = float(m4_risk.get("volatility", vf.get("current_vol_ann", 0.20)))
            var = float(m4_risk.get("var_95", 0.03))
            cvar= float(m4_risk.get("cvar_95", fd21.get("cvar_95_percent", 0.05)))
            mdd = float(m4_risk.get("max_drawdown", 0.10))
        else:
            vol = float(vf.get("current_vol_ann", 0.20))
            var = float(fd21.get("var_95_percent", 0.03))
            cvar= float(fd21.get("cvar_95_percent", 0.05))
            mdd = 0.10

        if m5_result:
            exp_ret= float(m5_result.get("expected_return_ann",
                           oa.get("expected_return_ann", 0.10)))
            sharpe = float(m5_result.get("sharpe_ratio",
                           oa.get("sharpe_ratio", 0.5)))
            weights= m5_result.get("weights", oa.get("weights", {}))
        else:
            exp_ret= float(oa.get("expected_return_ann", fd21.get("expected_return_ann", 0.0)))
            sharpe = float(oa.get("sharpe_ratio", 0.5))
            weights= oa.get("weights", {})

        # Security
        pqc_level     = "low"
        pqc_anomaly   = 0.0
        bay_posterior = 0.05
        bay_level     = "SAFE"
        quarantined   = False
        key_rot       = False

        if security:
            pqc           = security.get("pqc", {})
            bay           = security.get("bayesian", {})
            pqc_level     = pqc.get("threat_level", "low")
            pqc_anomaly   = float(pqc.get("anomaly_score", 0.0))
            bay_posterior = float(bay.get("posterior_probability", 0.05))
            bay_level     = bay.get("threat_level", "SAFE")
            quarantined   = bool(pqc.get("quarantine_status") == "QUARANTINED"
                                 or bay.get("quarantine_status") == "QUARANTINED")
            key_rot       = bool(pqc.get("key_rotation", False)
                                 or bay.get("key_rotation_signal", False))

        return cls(
            volatility_ann    = vol,
            var_95            = var,
            cvar_95           = cvar,
            max_drawdown      = mdd,
            expected_return   = exp_ret,
            sharpe_ratio      = sharpe,
            optimal_weights   = weights,
            tickers           = tickers,
            current_regime    = regime,
            regime_probs      = probs,
            forward_return_21d= float(fd21.get("expected_return_ann", 0.0)),
            forward_vol_21d   = float(fd21.get("annualised_vol", 0.20)),
            forward_cvar_21d  = float(fd21.get("cvar_95_percent", 0.05)),
            garch_vol_current = float(vf.get("current_vol_ann", 0.20)),
            garch_vol_30d     = float(vf.get("forecast_30d_vol", 0.20)),
            mixing_time_days  = int(m7.get("transition_matrix", {})
                                    .get("mixing_time_days", 500)),
            regime_entropy    = float(ue.get("regime_entropy", 0.5)),
            pqc_threat_level  = pqc_level,
            pqc_anomaly_score = pqc_anomaly,
            bayesian_posterior= bay_posterior,
            bayesian_threat_level = bay_level,
            transaction_quarantined = quarantined,
            key_rotation_active     = key_rot,
        )


# ---------------------------------------------------------------------------
# Decision output
# ---------------------------------------------------------------------------

@dataclass
class FinancialDecision:
    """
    Core financial decision output from the decision engine.

    Attributes
    ----------
    action      : recommended portfolio action
    confidence  : |composite_score| ∈ [0, 1]
    priority    : urgency level
    composite_score : raw score ∈ [-1, +1]
    signal_breakdown : per-signal contribution
    regime_context   : plain-English regime summary
    triggered_rules  : list of rule names that fired
    """
    action:           str
    confidence:       float
    priority:         str
    composite_score:  float
    signal_breakdown: Dict[str, float]
    regime_context:   str
    triggered_rules:  List[str]

    def to_dict(self) -> Dict:
        return {
            "action":           self.action,
            "confidence":       round(self.confidence, 4),
            "priority":         self.priority,
            "composite_score":  round(self.composite_score, 4),
            "signal_breakdown": {k: round(v, 4)
                                 for k, v in self.signal_breakdown.items()},
            "regime_context":   self.regime_context,
            "triggered_rules":  self.triggered_rules,
        }


# ---------------------------------------------------------------------------
# Signal computation
# ---------------------------------------------------------------------------

def _clip(x: float) -> float:
    return float(np.clip(x, -1.0, 1.0))


def _compute_signals(state: SystemState) -> Dict[str, float]:
    """
    Map SystemState to normalised signal vector ∈ [-1, +1] per dimension.
    """
    probs = state.regime_probs
    p_bull  = float(probs.get("Low-Vol Bull",  0.0))
    p_bear  = float(probs.get("High-Vol Bear", 0.0))
    p_cris  = float(probs.get("Crisis",        0.0))
    p_trans = float(probs.get("Transitional",  0.0))

    # 1. Regime pressure
    s_regime = _clip(p_bull - p_cris - 0.6 * p_bear)

    # 2. Volatility pressure
    s_vol = _clip(-(state.volatility_ann - VOL_THRESHOLD) / VOL_THRESHOLD)

    # 3. VaR breach
    s_var = -1.0 if state.var_95 > VAR_THRESHOLD else 0.0

    # 4. CVaR breach
    s_cvar = -1.0 if state.cvar_95 > CVAR_THRESHOLD else 0.0

    # 5. Drawdown pressure
    s_dd = _clip(-state.max_drawdown / DD_THRESHOLD)
    s_dd = min(s_dd, 0.0)   # only negative contribution

    # 6. Return outlook
    s_ret = _clip((state.expected_return - RETURN_THRESHOLD) / RETURN_THRESHOLD)

    # 7. Sharpe quality (normalised around 0; Sharpe of 1.0 = fully positive)
    s_sharpe = _clip(state.sharpe_ratio / 1.0)

    return {
        "regime":  s_regime,
        "vol":     s_vol,
        "var":     s_var,
        "cvar":    s_cvar,
        "dd":      s_dd,
        "ret":     s_ret,
        "sharpe":  s_sharpe,
    }


def _scenario_overlay(state: SystemState, signals: Dict[str, float]) -> Dict[str, float]:
    """
    Apply scenario-specific signal adjustments.

    Scenarios override or amplify base signals to reflect hypothetical
    forward-looking market conditions.
    """
    s = dict(signals)
    sc = state.scenario

    if sc == "market_crash":
        s["regime"]  = min(s["regime"],  -0.9)
        s["vol"]     = min(s["vol"],     -0.9)
        s["dd"]      = min(s["dd"],      -1.0)
        s["var"]     = -1.0
        s["cvar"]    = -1.0

    elif sc == "high_inflation":
        # Inflation erodes real returns; reduce return outlook and Sharpe
        s["ret"]     = min(s["ret"] - 0.3, -0.5)
        s["sharpe"]  = min(s["sharpe"] - 0.2, -0.3)

    elif sc == "liquidity_crisis":
        s["regime"]  = min(s["regime"],  -0.8)
        s["vol"]     = min(s["vol"],     -0.8)
        s["cvar"]    = -1.0

    elif sc == "rate_shock":
        s["ret"]     = min(s["ret"] - 0.4, -0.4)
        s["vol"]     = min(s["vol"] - 0.2, -0.6)

    elif sc == "commodity_boom":
        s["ret"]     = max(s["ret"] + 0.2,  0.4)
        s["regime"]  = max(s["regime"] + 0.2, s["regime"])

    elif sc == "geopolitical_risk":
        s["regime"]  = min(s["regime"],  -0.5)
        s["vol"]     = min(s["vol"],     -0.5)

    return {k: _clip(v) for k, v in s.items()}


def _action_from_score(score: float) -> str:
    if score >= ACTION_THRESHOLDS["INCREASE_EXPOSURE"]:
        return "INCREASE_EXPOSURE"
    elif score >= ACTION_THRESHOLDS["HOLD"]:
        return "HOLD"
    elif score >= ACTION_THRESHOLDS["REDUCE_RISK"]:
        return "REDUCE_RISK"
    else:
        return "REBALANCE"


def _priority_from_confidence(confidence: float) -> str:
    if confidence >= 0.70:
        return "high"
    elif confidence >= 0.35:
        return "medium"
    return "low"


def _regime_context(state: SystemState) -> str:
    regime = state.current_regime
    p = state.regime_probs.get(regime, 0.0)
    templates = {
        "Low-Vol Bull":  f"Market is in a calm bull phase (P={p:.0%}). Conditions favour equity exposure.",
        "High-Vol Bear": f"Bear market with elevated volatility (P={p:.0%}). Defensive positioning recommended.",
        "Crisis":        f"Crisis regime detected (P={p:.0%}). Capital preservation is the primary objective.",
        "Transitional":  f"Market is in a transitional phase (P={p:.0%}). Mixed signals — balanced approach advised.",
    }
    return templates.get(regime, f"Regime: {regime} (P={p:.0%}).")


def _triggered_rules(state: SystemState, signals: Dict) -> List[str]:
    rules = []
    if signals["regime"] < -0.5:
        rules.append("REGIME_RISK: bear/crisis regime dominates")
    if state.volatility_ann > VOL_THRESHOLD:
        rules.append(f"VOL_BREACH: vol={state.volatility_ann:.1%} > threshold={VOL_THRESHOLD:.0%}")
    if state.var_95 > VAR_THRESHOLD:
        rules.append(f"VAR_BREACH: VaR={state.var_95:.2%} > threshold={VAR_THRESHOLD:.2%}")
    if state.cvar_95 > CVAR_THRESHOLD:
        rules.append(f"CVAR_BREACH: CVaR={state.cvar_95:.2%} > threshold={CVAR_THRESHOLD:.2%}")
    if state.max_drawdown > DD_THRESHOLD:
        rules.append(f"DRAWDOWN_BREACH: dd={state.max_drawdown:.1%} > threshold={DD_THRESHOLD:.0%}")
    if state.sharpe_ratio < 0:
        rules.append(f"NEGATIVE_SHARPE: Sharpe={state.sharpe_ratio:.2f}")
    if state.expected_return < 0:
        rules.append(f"NEGATIVE_EXPECTED_RETURN: E[r]={state.expected_return:.2%}")
    if state.scenario:
        rules.append(f"SCENARIO_OVERRIDE: {state.scenario}")
    return rules


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_decision(state: SystemState) -> FinancialDecision:
    """
    Generate a core financial decision from the current system state.

    Pipeline:
        1. Compute per-signal scores from SystemState
        2. Apply scenario overlay (if any)
        3. Compute weighted composite score
        4. Map score to action + priority
        5. Identify triggered rules

    Parameters
    ----------
    state : SystemState — consolidated module outputs

    Returns
    -------
    FinancialDecision
    """
    # 1. Base signals
    signals = _compute_signals(state)

    # 2. Scenario overlay
    if state.scenario:
        signals = _scenario_overlay(state, signals)

    # 3. Weighted composite score
    composite = sum(SIGNAL_WEIGHTS[k] * v for k, v in signals.items())
    composite = float(np.clip(composite, -1.0, 1.0))

    # 4. Action + priority
    action     = _action_from_score(composite)
    confidence = abs(composite)
    priority   = _priority_from_confidence(confidence)

    # 5. Per-signal weighted contributions for explainability
    breakdown = {k: round(SIGNAL_WEIGHTS[k] * v, 4) for k, v in signals.items()}

    return FinancialDecision(
        action           = action,
        confidence       = confidence,
        priority         = priority,
        composite_score  = composite,
        signal_breakdown = breakdown,
        regime_context   = _regime_context(state),
        triggered_rules  = _triggered_rules(state, signals),
    )


# ══════════════════════════════════════════════════════════════════════════
# MODULE 2 — STRATEGY ADAPTER  (regime-aware parameter modulation)
# ══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Regime parameter anchors (consistent with M7 adaptive_allocator.py)
# ---------------------------------------------------------------------------

_REGIME_PARAMS: Dict[str, Dict] = {
    "Low-Vol Bull": {
        "lam_return":  1.3,
        "lam_vol":     0.6,
        "lam_cvar":    0.5,
        "lam_drawdown":0.1,
        "max_weight":  0.40,
        "target_vol":  0.18,
        "min_assets":  2,
    },
    "High-Vol Bear": {
        "lam_return":  0.7,
        "lam_vol":     1.6,
        "lam_cvar":    1.2,
        "lam_drawdown":0.4,
        "max_weight":  0.25,
        "target_vol":  0.12,
        "min_assets":  4,
    },
    "Crisis": {
        "lam_return":  0.3,
        "lam_vol":     2.5,
        "lam_cvar":    2.0,
        "lam_drawdown":0.8,
        "max_weight":  0.15,
        "target_vol":  0.08,
        "min_assets":  5,
    },
    "Transitional": {
        "lam_return":  1.0,
        "lam_vol":     1.0,
        "lam_cvar":    1.0,
        "lam_drawdown":0.3,
        "max_weight":  0.30,
        "target_vol":  0.14,
        "min_assets":  3,
    },
}

# Neutral (baseline) params used for blending under low confidence
_NEUTRAL_PARAMS: Dict = {
    "lam_return":  1.0,
    "lam_vol":     1.0,
    "lam_cvar":    1.0,
    "lam_drawdown":0.3,
    "max_weight":  0.30,
    "target_vol":  0.15,
    "min_assets":  3,
}

# Action-level leverage modifiers
_ACTION_LEVERAGE: Dict[str, float] = {
    "INCREASE_EXPOSURE": 1.0,      # full position
    "HOLD":              0.90,     # slight precaution
    "REDUCE_RISK":       0.70,     # reduce exposure
    "REBALANCE":         0.55,     # significantly de-risk
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdaptedStrategy:
    """
    Concrete optimization parameters adjusted for current regime and decision.

    These parameters can be passed directly to M5's run_optimizer() to
    produce regime-conditioned portfolio weights.

    Attributes
    ----------
    lam_return      : return objective weight
    lam_vol         : variance penalty weight
    lam_cvar        : CVaR penalty weight
    lam_drawdown    : drawdown penalty weight
    max_weight      : per-asset weight cap
    target_vol      : target annualised portfolio volatility
    leverage_scalar : effective position size (0–1)
    min_assets      : minimum number of assets to hold
    sector_cap      : maximum sector concentration
    confidence_blend: fraction of regime vs neutral params used
    regime_applied  : which regime anchor was used
    action_applied  : which financial decision drove scaling
    rebalance_urgency: "immediate" | "next_session" | "routine"
    notes           : list of plain-English strategy notes
    """
    lam_return:       float
    lam_vol:          float
    lam_cvar:         float
    lam_drawdown:     float
    max_weight:       float
    target_vol:       float
    leverage_scalar:  float
    min_assets:       int
    sector_cap:       float
    confidence_blend: float
    regime_applied:   str
    action_applied:   str
    rebalance_urgency:str
    notes:            List[str]

    def to_optimizer_kwargs(self) -> Dict:
        """
        Return keyword arguments ready to pass to M5 run_optimizer().
        """
        return {
            "lam_return":   self.lam_return,
            "lam_vol":      self.lam_vol,
            "lam_cvar":     self.lam_cvar,
            "lam_drawdown": self.lam_drawdown,
            "max_weight":   self.max_weight,
            "target_vol":   self.target_vol,
        }

    def to_dict(self) -> Dict:
        return {
            "optimizer_params": self.to_optimizer_kwargs(),
            "leverage_scalar":   round(self.leverage_scalar, 4),
            "min_assets":        self.min_assets,
            "sector_cap":        round(self.sector_cap, 4),
            "confidence_blend":  round(self.confidence_blend, 4),
            "regime_applied":    self.regime_applied,
            "action_applied":    self.action_applied,
            "rebalance_urgency": self.rebalance_urgency,
            "notes":             self.notes,
        }


# ---------------------------------------------------------------------------
# Core adapter function
# ---------------------------------------------------------------------------

def adapt_strategy(
    state:    SystemState,
    decision: FinancialDecision,
) -> AdaptedStrategy:
    """
    Produce regime-adapted optimization parameters.

    Blending rule (confidence-weighted):
        θ_adapted = blend × θ_regime + (1 - blend) × θ_neutral

    where blend ∈ {1.0, 0.70, 0.40} for high / medium / low confidence.

    Parameters
    ----------
    state    : SystemState from decision_engine
    decision : FinancialDecision from generate_decision()

    Returns
    -------
    AdaptedStrategy
    """
    # Confidence blend weight
    blend_map = {"high": 1.0, "medium": 0.70, "low": 0.40}
    blend = blend_map.get(decision.priority, 0.70)

    # Regime anchor
    regime  = state.current_regime
    r_params = _REGIME_PARAMS.get(regime, _NEUTRAL_PARAMS)

    # Blend regime params with neutral
    def blend_param(key: str) -> float:
        return blend * r_params[key] + (1 - blend) * _NEUTRAL_PARAMS[key]

    lam_return   = blend_param("lam_return")
    lam_vol      = blend_param("lam_vol")
    lam_cvar     = blend_param("lam_cvar")
    lam_drawdown = blend_param("lam_drawdown")
    max_weight   = blend_param("max_weight")
    target_vol   = blend_param("target_vol")
    min_assets   = max(2, int(round(
        blend * r_params["min_assets"] + (1 - blend) * _NEUTRAL_PARAMS["min_assets"]
    )))

    # Action-level leverage scalar
    action_lev = _ACTION_LEVERAGE.get(decision.action, 0.80)

    # GARCH vol scaling (consistent with M7 formula)
    garch_scale = 1.0
    if state.garch_vol_current > 1e-4:
        garch_scale = float(np.clip(target_vol / state.garch_vol_current, 0.1, 1.0))

    leverage_scalar = float(np.clip(action_lev * garch_scale, 0.1, 1.0))

    # Sector cap: tighten in crisis/bear
    sector_cap_base = 0.40
    if regime == "Crisis":
        sector_cap = 0.25
    elif regime == "High-Vol Bear":
        sector_cap = 0.30
    else:
        sector_cap = sector_cap_base

    # Rebalance urgency
    if decision.action == "REBALANCE" and decision.priority == "high":
        urgency = "immediate"
    elif decision.action in ("REBALANCE", "REDUCE_RISK"):
        urgency = "next_session"
    else:
        urgency = "routine"

    # Plain-English notes
    notes = _generate_notes(
        regime, decision.action, decision.priority, blend,
        lam_vol, lam_cvar, max_weight, leverage_scalar, garch_scale
    )

    return AdaptedStrategy(
        lam_return       = round(lam_return,    4),
        lam_vol          = round(lam_vol,        4),
        lam_cvar         = round(lam_cvar,       4),
        lam_drawdown     = round(lam_drawdown,   4),
        max_weight       = round(max_weight,     4),
        target_vol       = round(target_vol,     4),
        leverage_scalar  = round(leverage_scalar,4),
        min_assets       = min_assets,
        sector_cap       = round(sector_cap,     4),
        confidence_blend = round(blend,          4),
        regime_applied   = regime,
        action_applied   = decision.action,
        rebalance_urgency= urgency,
        notes            = notes,
    )


def _generate_notes(
    regime: str, action: str, priority: str, blend: float,
    lam_vol: float, lam_cvar: float, max_weight: float,
    leverage: float, garch_scale: float,
) -> List[str]:
    notes = []
    if regime == "Crisis":
        notes.append(
            f"CRISIS regime: variance penalty raised to {lam_vol:.2f}, "
            f"CVaR penalty raised to {lam_cvar:.2f}, "
            f"max weight tightened to {max_weight:.0%}."
        )
    elif regime == "High-Vol Bear":
        notes.append(
            f"BEAR regime: risk penalties elevated "
            f"(λ_vol={lam_vol:.2f}, λ_cvar={lam_cvar:.2f}), "
            f"max weight capped at {max_weight:.0%}."
        )
    elif regime == "Low-Vol Bull":
        notes.append(
            f"BULL regime: return emphasis increased, "
            f"risk constraints relaxed (max weight={max_weight:.0%})."
        )

    if garch_scale < 0.90:
        notes.append(
            f"GARCH scaling: effective exposure reduced to {leverage:.0%} "
            f"because current volatility exceeds the target."
        )

    if priority == "low":
        notes.append(
            f"Low signal confidence (blend={blend:.0%}): "
            "parameters partially shrunk toward neutral baseline."
        )

    if action == "REBALANCE":
        notes.append(
            "Full rebalance triggered: portfolio weights should be "
            "realigned to the adapted target allocation."
        )
    elif action == "REDUCE_RISK":
        notes.append(
            "Risk reduction mode: consider trimming high-beta positions "
            "and increasing cash or low-vol assets."
        )

    return notes


# ══════════════════════════════════════════════════════════════════════════
# MODULE 3 — SECURITY-AWARE LOGIC  (threat integration + action override)
# ══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PQC_SCORE_MAP: Dict[str, float] = {
    "low":    0.10,
    "medium": 0.50,
    "high":   1.00,
}

BAYESIAN_LEVEL_SCORE: Dict[str, float] = {
    "SAFE":           0.05,
    "MONITOR":        0.30,
    "ELEVATED_RISK":  0.60,
    "CRITICAL_THREAT":0.95,
}

W_PQC = 0.40
W_BAY = 0.60

LARGE_TX_THRESHOLD = 0.10   # 10% of portfolio

# Tier boundaries
TIER_THRESHOLDS = {
    "CLEAR":    (0.00, 0.20),
    "MONITOR":  (0.20, 0.40),
    "ELEVATED": (0.40, 0.60),
    "HIGH":     (0.60, 0.80),
    "CRITICAL": (0.80, 1.01),
}

# Action downgrade map when security tier is HIGH
ACTION_DOWNGRADE: Dict[str, str] = {
    "INCREASE_EXPOSURE": "HOLD",
    "HOLD":              "REDUCE_RISK",
    "REDUCE_RISK":       "REBALANCE",
    "REBALANCE":         "REBALANCE",
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class SecurityConstraints:
    """
    Security-driven constraints on portfolio and transaction activity.

    Attributes
    ----------
    threat_composite     : weighted composite threat score T ∈ [0, 1]
    security_tier        : CLEAR | MONITOR | ELEVATED | HIGH | CRITICAL
    security_action      : plain-English primary security action
    restrictions         : ordered list of active restrictions
    large_tx_blocked     : True if large transactions are blocked
    large_tx_threshold   : fraction of portfolio that defines "large"
    multi_step_required  : True if multi-step transaction verification required
    max_trades_per_session: maximum trades allowed this session
    account_flagged      : True if account investigation triggered
    key_rotation_required: True if key rotation should be executed
    overridden_action    : if security overrides financial action, new action here
    action_was_overridden: True if financial action was changed
    reasoning            : list of security reasoning statements
    """
    threat_composite:        float
    security_tier:           str
    security_action:         str
    restrictions:            List[str]
    large_tx_blocked:        bool
    large_tx_threshold:      float
    multi_step_required:     bool
    max_trades_per_session:  int
    account_flagged:         bool
    key_rotation_required:   bool
    overridden_action:       Optional[str]
    action_was_overridden:   bool
    reasoning:               List[str]

    def to_dict(self) -> Dict:
        return {
            "threat_composite":         round(self.threat_composite, 4),
            "security_tier":            self.security_tier,
            "security_action":          self.security_action,
            "restrictions":             self.restrictions,
            "large_tx_blocked":         self.large_tx_blocked,
            "large_tx_threshold":       round(self.large_tx_threshold, 4),
            "multi_step_required":      self.multi_step_required,
            "max_trades_per_session":   self.max_trades_per_session,
            "account_flagged":          self.account_flagged,
            "key_rotation_required":    self.key_rotation_required,
            "overridden_action":        self.overridden_action,
            "action_was_overridden":    self.action_was_overridden,
            "reasoning":                self.reasoning,
        }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def apply_security_constraints(
    state:    SystemState,
    decision: FinancialDecision,
) -> SecurityConstraints:
    """
    Evaluate security threat level and generate trading restrictions.

    The financial decision's action may be overridden if threat is HIGH
    or CRITICAL.

    Parameters
    ----------
    state    : SystemState (includes PQC and Bayesian security signals)
    decision : FinancialDecision (may be modified by security override)

    Returns
    -------
    SecurityConstraints
    """
    # ── 1. Threat composite score ─────────────────────────────────────────
    pqc_score = PQC_SCORE_MAP.get(state.pqc_threat_level.lower(), 0.10)

    # Use the higher of posterior probability and level-implied score
    bay_level_score = BAYESIAN_LEVEL_SCORE.get(
        state.bayesian_threat_level, 0.05
    )
    bay_score = max(state.bayesian_posterior, bay_level_score)

    T = float(np.clip(W_PQC * pqc_score + W_BAY * bay_score, 0.0, 1.0))

    # ── 2. Security tier ──────────────────────────────────────────────────
    tier = "CLEAR"
    for t_name, (lo, hi) in TIER_THRESHOLDS.items():
        if lo <= T < hi:
            tier = t_name
            break

    # Hard overrides from existing security reports
    if state.transaction_quarantined:
        tier = "CRITICAL"
    if state.key_rotation_active and tier not in ("HIGH", "CRITICAL"):
        tier = "HIGH"

    # ── 3. Build restrictions ─────────────────────────────────────────────
    restrictions  = []
    large_blocked = False
    multi_step    = False
    acct_flagged  = False
    key_rot_req   = False
    max_trades    = 10  # unlimited in normal circumstances

    if tier == "MONITOR":
        restrictions.append("AUDIT_FLAG: all transactions logged for review")

    elif tier == "ELEVATED":
        restrictions.append("AUDIT_FLAG: all transactions logged for review")
        restrictions.append("REDUCED_FREQUENCY: trading frequency capped")
        max_trades = max(1, int(10 * (1 - T)))

    elif tier == "HIGH":
        restrictions.append("AUDIT_FLAG: all transactions logged for review")
        restrictions.append("REDUCED_FREQUENCY: trading frequency capped")
        restrictions.append(
            f"LARGE_TX_BLOCKED: transactions > {LARGE_TX_THRESHOLD:.0%} of "
            "portfolio value are blocked"
        )
        restrictions.append("MULTI_STEP_VERIFICATION: secondary confirmation required")
        large_blocked = True
        multi_step    = True
        max_trades    = max(1, int(5 * (1 - T)))

    elif tier == "CRITICAL":
        restrictions.append("FULL_QUARANTINE: all new transactions blocked")
        restrictions.append("KEY_ROTATION: cryptographic key rotation triggered")
        restrictions.append("ACCOUNT_INVESTIGATION: account flagged for review")
        restrictions.append("MULTI_STEP_VERIFICATION: secondary confirmation required")
        large_blocked = True
        multi_step    = True
        acct_flagged  = True
        key_rot_req   = True
        max_trades    = 0

    # ── 4. Financial action override ──────────────────────────────────────
    overridden_action    = None
    action_was_overridden= False

    if tier == "CRITICAL":
        overridden_action     = "HOLD"
        action_was_overridden = (decision.action != "HOLD")

    elif tier == "HIGH" and decision.action in ACTION_DOWNGRADE:
        new_action = ACTION_DOWNGRADE[decision.action]
        if new_action != decision.action:
            overridden_action     = new_action
            action_was_overridden = True

    # ── 5. Security action label ──────────────────────────────────────────
    tier_actions = {
        "CLEAR":    "ALLOW — no security restrictions active",
        "MONITOR":  "MONITOR — transactions logged for audit",
        "ELEVATED": "RESTRICT — reduced frequency, secondary confirmation recommended",
        "HIGH":     "BLOCK_LARGE — large transactions blocked, multi-step verification required",
        "CRITICAL": "FULL_QUARANTINE — all transactions suspended pending investigation",
    }
    security_action = tier_actions.get(tier, "ALLOW")

    # ── 6. Reasoning ──────────────────────────────────────────────────────
    reasoning = _build_reasoning(state, T, tier, pqc_score, bay_score,
                                 overridden_action, decision.action)

    return SecurityConstraints(
        threat_composite        = T,
        security_tier           = tier,
        security_action         = security_action,
        restrictions            = restrictions,
        large_tx_blocked        = large_blocked,
        large_tx_threshold      = LARGE_TX_THRESHOLD,
        multi_step_required     = multi_step,
        max_trades_per_session  = max_trades,
        account_flagged         = acct_flagged,
        key_rotation_required   = key_rot_req,
        overridden_action       = overridden_action,
        action_was_overridden   = action_was_overridden,
        reasoning               = reasoning,
    )


def _build_reasoning(
    state:       SystemState,
    T:           float,
    tier:        str,
    pqc_score:   float,
    bay_score:   float,
    override:    Optional[str],
    original:    str,
) -> List[str]:
    reasons = [
        f"PQC threat level: {state.pqc_threat_level} "
        f"(score={pqc_score:.2f}, anomaly={state.pqc_anomaly_score:.2f})",
        f"Bayesian threat probability: {state.bayesian_posterior:.1%} "
        f"[{state.bayesian_threat_level}] (effective score={bay_score:.2f})",
        f"Composite threat score T = {T:.3f} → tier = {tier}",
    ]
    if state.transaction_quarantined:
        reasons.append("Existing quarantine flag detected — escalated to CRITICAL.")
    if state.key_rotation_active:
        reasons.append("Active key rotation detected — elevated to HIGH minimum.")
    if override:
        reasons.append(
            f"Financial action overridden: {original} → {override} "
            "due to security constraints."
        )
    return reasons


# ══════════════════════════════════════════════════════════════════════════
# MODULE 4 — EXPLAINABILITY ENGINE  (human-readable decision reasoning)
# ══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Factor confidence thresholds
# ---------------------------------------------------------------------------

PRIMARY_THRESHOLD     = 0.15
CONTRIBUTING_THRESHOLD= 0.07

# Human-readable signal names
SIGNAL_LABELS: Dict[str, str] = {
    "regime":  "Market Regime",
    "vol":     "Portfolio Volatility",
    "var":     "Value at Risk (VaR)",
    "cvar":    "Conditional VaR (CVaR)",
    "dd":      "Maximum Drawdown",
    "ret":     "Expected Return",
    "sharpe":  "Sharpe Ratio",
}

# Action descriptions in plain English
ACTION_DESCRIPTIONS: Dict[str, str] = {
    "INCREASE_EXPOSURE": "Increase portfolio exposure — conditions are favourable.",
    "HOLD":              "Hold current positions — no significant change warranted.",
    "REDUCE_RISK":       "Reduce portfolio risk — conditions are deteriorating.",
    "REBALANCE":         "Full rebalance required — current allocation is misaligned with risk tolerance.",
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class DecisionExplanation:
    """
    Complete human-readable and machine-readable explanation of an M8 decision.

    Attributes
    ----------
    summary         : one-sentence plain English decision summary
    action_rationale: why this specific action was chosen
    factors         : ordered list of contributing factors with tier labels
    narrative       : multi-paragraph plain-English narrative
    risk_assessment : summary of risk signals that contributed
    security_summary: plain-English security assessment
    confidence_statement: plain-English confidence and priority explanation
    raw_signals     : raw signal values for audit purposes
    timestamp       : Unix time of explanation generation
    """
    summary:             str
    action_rationale:    str
    factors:             List[Dict[str, Any]]
    narrative:           str
    risk_assessment:     str
    security_summary:    str
    confidence_statement:str
    raw_signals:         Dict[str, float]
    timestamp:           float

    def to_dict(self) -> Dict:
        return {
            "summary":              self.summary,
            "action_rationale":     self.action_rationale,
            "factors":              self.factors,
            "narrative":            self.narrative,
            "risk_assessment":      self.risk_assessment,
            "security_summary":     self.security_summary,
            "confidence_statement": self.confidence_statement,
            "raw_signals":          {k: round(v, 4) for k, v in self.raw_signals.items()},
            "timestamp":            self.timestamp,
        }


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_explanation(
    state:       SystemState,
    decision:    FinancialDecision,
    strategy:    AdaptedStrategy,
    security:    SecurityConstraints,
) -> DecisionExplanation:
    """
    Generate a complete explanation for an M8 decision.

    Parameters
    ----------
    state    : SystemState — raw module inputs
    decision : FinancialDecision — core financial decision
    strategy : AdaptedStrategy — regime-adapted parameters
    security : SecurityConstraints — security overlay

    Returns
    -------
    DecisionExplanation
    """
    # Determine the effective action (security may have overridden)
    effective_action = (
        security.overridden_action
        if security.action_was_overridden else decision.action
    )

    # ── Summary ───────────────────────────────────────────────────────────
    action_desc = ACTION_DESCRIPTIONS.get(effective_action, effective_action)
    summary = (
        f"Decision: {effective_action} — {action_desc} "
        f"[{decision.priority.upper()} priority, "
        f"confidence={decision.confidence:.0%}]"
    )

    # ── Action rationale ──────────────────────────────────────────────────
    rationale = _build_rationale(decision, security, state)

    # ── Factor list ───────────────────────────────────────────────────────
    factors = _build_factor_list(decision, state, security)

    # ── Narrative ─────────────────────────────────────────────────────────
    narrative = _build_narrative(state, decision, strategy, security, effective_action)

    # ── Risk assessment ───────────────────────────────────────────────────
    risk_assessment = _build_risk_assessment(state)

    # ── Security summary ──────────────────────────────────────────────────
    security_summary = _build_security_summary(state, security)

    # ── Confidence statement ──────────────────────────────────────────────
    confidence_stmt = _build_confidence_statement(decision, strategy)

    return DecisionExplanation(
        summary              = summary,
        action_rationale     = rationale,
        factors              = factors,
        narrative            = narrative,
        risk_assessment      = risk_assessment,
        security_summary     = security_summary,
        confidence_statement = confidence_stmt,
        raw_signals          = dict(decision.signal_breakdown),
        timestamp            = time.time(),
    )


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _build_rationale(
    decision: FinancialDecision,
    security: SecurityConstraints,
    state:    SystemState,
) -> str:
    parts = [decision.regime_context]

    top_rules = decision.triggered_rules[:3]
    if top_rules:
        parts.append("Key signals that drove this decision: " +
                     "; ".join(top_rules) + ".")

    if security.action_was_overridden:
        parts.append(
            f"Note: the financial analysis suggested {decision.action}, "
            f"but this was overridden to {security.overridden_action} "
            f"due to an active security threat (tier: {security.security_tier})."
        )

    return " ".join(parts)


def _build_factor_list(
    decision: FinancialDecision,
    state:    SystemState,
    security: SecurityConstraints,
) -> List[Dict[str, Any]]:
    """
    Build an ordered list of contributing factors sorted by |contribution|.
    """
    factors = []

    for sig_key, contribution in decision.signal_breakdown.items():
        abs_c = abs(contribution)
        if abs_c >= PRIMARY_THRESHOLD:
            tier = "primary driver"
        elif abs_c >= CONTRIBUTING_THRESHOLD:
            tier = "contributing factor"
        else:
            tier = "minor factor"

        direction = "↑ bullish" if contribution > 0 else "↓ bearish" if contribution < 0 else "→ neutral"

        # Human-readable value string
        val_str = _signal_value_str(sig_key, state)

        factors.append({
            "factor":       SIGNAL_LABELS.get(sig_key, sig_key),
            "contribution": round(contribution, 4),
            "direction":    direction,
            "tier":         tier,
            "current_value":val_str,
        })

    # Add security as a factor if elevated
    if security.threat_composite > 0.20:
        tier = "primary driver" if security.threat_composite > 0.60 else "contributing factor"
        factors.append({
            "factor":       "Security Threat",
            "contribution": round(-security.threat_composite * 0.20, 4),
            "direction":    "↓ bearish",
            "tier":         tier,
            "current_value":
                f"T={security.threat_composite:.2f}, tier={security.security_tier}",
        })

    # Sort: primary first, then by |contribution|
    tier_order = {"primary driver": 0, "contributing factor": 1, "minor factor": 2}
    factors.sort(key=lambda x: (tier_order[x["tier"]], -abs(x["contribution"])))

    return factors


def _signal_value_str(sig_key: str, state: SystemState) -> str:
    """Return a concise human-readable current value for each signal."""
    m = {
        "regime":  f"{state.current_regime} "
                   f"(P={state.regime_probs.get(state.current_regime, 0):.0%})",
        "vol":     f"{state.volatility_ann:.1%} annualised",
        "var":     f"{state.var_95:.2%} daily at 95%",
        "cvar":    f"{state.cvar_95:.2%} daily CVaR 95%",
        "dd":      f"{state.max_drawdown:.1%} max drawdown",
        "ret":     f"{state.expected_return:.1%} expected annual return",
        "sharpe":  f"{state.sharpe_ratio:.2f} Sharpe ratio",
    }
    return m.get(sig_key, "N/A")


def _build_narrative(
    state:    SystemState,
    decision: FinancialDecision,
    strategy: AdaptedStrategy,
    security: SecurityConstraints,
    effective_action: str,
) -> str:
    lines = []

    # Paragraph 1: market environment
    lines.append(
        f"Market Environment: The portfolio is currently operating in a "
        f"{state.current_regime} regime "
        f"(probability {state.regime_probs.get(state.current_regime, 0):.0%}). "
        f"{decision.regime_context}"
    )

    # Paragraph 2: risk snapshot
    lines.append(
        f"Risk Snapshot: Portfolio volatility is {state.volatility_ann:.1%} per year. "
        f"In the worst 5% of trading days, the daily loss could reach {state.var_95:.2%} "
        f"(VaR 95%) with an average of {state.cvar_95:.2%} on those bad days (CVaR 95%). "
        f"The maximum historical drawdown in the analysis period was {state.max_drawdown:.1%}."
    )

    # Paragraph 3: forward outlook
    if state.forward_return_21d != 0:
        direction = "gain" if state.forward_return_21d > 0 else "loss"
        lines.append(
            f"21-Day Outlook: The regime-weighted model expects {state.forward_return_21d:.1%} "
            f"annualised return ({direction}) over the next month, with forward volatility of "
            f"{state.forward_vol_21d:.1%} and a forward CVaR of {state.forward_cvar_21d:.2%}. "
            f"GARCH-estimated current volatility is {state.garch_vol_current:.1%} per year."
        )

    # Paragraph 4: strategy adaptation
    lines.append(
        f"Strategy Adaptation: The engine has calibrated optimization parameters for the "
        f"{strategy.regime_applied} regime — variance penalty λ_vol={strategy.lam_vol:.2f}, "
        f"CVaR penalty λ_cvar={strategy.lam_cvar:.2f}, max weight per asset "
        f"{strategy.max_weight:.0%}, effective leverage {strategy.leverage_scalar:.0%}. "
        f"Rebalance urgency is {strategy.rebalance_urgency}."
    )

    # Paragraph 5: security
    if security.security_tier != "CLEAR":
        lines.append(
            f"Security Assessment: The security layer detected a threat composite score of "
            f"{security.threat_composite:.2f} ({security.security_tier} tier). "
            f"Active restrictions: {'; '.join(security.restrictions[:2]) if security.restrictions else 'none'}."
        )
        if security.action_was_overridden:
            lines.append(
                f"As a result, the financial action was overridden from "
                f"{decision.action} to {effective_action}."
            )
    else:
        lines.append("Security Assessment: No active security threats detected. All transactions clear.")

    # Paragraph 6: final recommendation
    lines.append(
        f"Final Recommendation: {effective_action} — "
        f"{ACTION_DESCRIPTIONS.get(effective_action, effective_action)} "
        f"This is a {decision.priority}-priority signal with {decision.confidence:.0%} confidence."
    )

    return "\n\n".join(lines)


def _build_risk_assessment(state: SystemState) -> str:
    alerts = []
    if state.volatility_ann > VOL_THRESHOLD:
        alerts.append(
            f"Volatility {state.volatility_ann:.1%} exceeds threshold {VOL_THRESHOLD:.0%}"
        )
    if state.var_95 > VAR_THRESHOLD:
        alerts.append(
            f"VaR {state.var_95:.2%} exceeds threshold {VAR_THRESHOLD:.2%}"
        )
    if state.cvar_95 > CVAR_THRESHOLD:
        alerts.append(
            f"CVaR {state.cvar_95:.2%} exceeds threshold {CVAR_THRESHOLD:.2%}"
        )
    if state.max_drawdown > DD_THRESHOLD:
        alerts.append(
            f"Drawdown {state.max_drawdown:.1%} exceeds threshold {DD_THRESHOLD:.0%}"
        )
    if not alerts:
        return "All risk metrics within normal bounds. No breaches detected."
    return "Risk alerts: " + "; ".join(alerts) + "."


def _build_security_summary(state: SystemState, security: SecurityConstraints) -> str:
    parts = [
        f"PQC immune engine: {state.pqc_threat_level} threat "
        f"(anomaly score={state.pqc_anomaly_score:.2f}).",
        f"Bayesian engine: {state.bayesian_posterior:.1%} threat probability "
        f"[{state.bayesian_threat_level}].",
        f"Combined tier: {security.security_tier}. "
        f"Action: {security.security_action}.",
    ]
    if security.account_flagged:
        parts.append("Account has been flagged for investigation.")
    if security.key_rotation_required:
        parts.append("Cryptographic key rotation is required.")
    return " ".join(parts)


def _build_confidence_statement(
    decision: FinancialDecision,
    strategy: AdaptedStrategy,
) -> str:
    blend_pct = strategy.confidence_blend * 100
    return (
        f"Signal confidence is {decision.confidence:.0%} ({decision.priority} priority). "
        f"The regime anchor was applied at {blend_pct:.0f}% weight "
        f"(remaining {100-blend_pct:.0f}% blended toward neutral baseline). "
        f"Higher confidence means the regime signal is clearer and the "
        f"adapted parameters are closer to the pure regime anchor."
    )


# ══════════════════════════════════════════════════════════════════════════
# MODULE 5 — RECOMMENDATION API  (master pipeline + audit logging)
# ══════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("milestone8")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [M8-DECISION] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Fused decision dataclass
# ---------------------------------------------------------------------------

@dataclass
class FusedDecision:
    """
    Fusion of financial, strategy, and security decisions.

    The final_action resolves any conflict between the financial decision
    and security constraints:
        - If security overrides: final_action = security.overridden_action
        - Otherwise:             final_action = financial_decision.action

    portfolio_adjustments contains the concrete optimizer parameters.
    security_constraints contains the active trading restrictions.
    confidence is the financial confidence reduced by security uncertainty:

        fused_confidence = financial_confidence × (1 - 0.5 × threat_composite)

    This ensures high threat automatically reduces the reported confidence
    in the combined recommendation.
    """
    final_action:          str
    financial_action:      str
    security_override:     bool
    portfolio_adjustments: Dict
    security_constraints:  Dict
    fused_confidence:      float
    priority:              str

    def to_dict(self) -> Dict:
        return {
            "final_action":          self.final_action,
            "financial_action":      self.financial_action,
            "security_override":     self.security_override,
            "portfolio_adjustments": self.portfolio_adjustments,
            "security_constraints":  self.security_constraints,
            "fused_confidence":      round(self.fused_confidence, 4),
            "priority":              self.priority,
        }


def _fuse_decisions(
    financial: FinancialDecision,
    strategy:  AdaptedStrategy,
    security:  SecurityConstraints,
) -> FusedDecision:
    final_action = (
        security.overridden_action
        if security.action_was_overridden else financial.action
    )
    fused_conf = financial.confidence * (1 - 0.5 * security.threat_composite)
    fused_conf = float(np.clip(fused_conf, 0.0, 1.0))

    return FusedDecision(
        final_action          = final_action,
        financial_action      = financial.action,
        security_override     = security.action_was_overridden,
        portfolio_adjustments = strategy.to_dict(),
        security_constraints  = security.to_dict(),
        fused_confidence      = fused_conf,
        priority              = financial.priority,
    )


# ---------------------------------------------------------------------------
# Full recommendation output
# ---------------------------------------------------------------------------

@dataclass
class SystemRecommendation:
    """
    Complete M8 output — all decisions, adaptations, and explanations.

    Attributes
    ----------
    recommendation_id : unique identifier for this recommendation
    final_action      : the effective portfolio action
    fused_confidence  : confidence after security adjustment
    priority          : urgency level
    portfolio_adjustments : regime-adapted optimizer parameters
    security_actions  : active security constraints and restrictions
    explanation       : full structured explanation
    decision_detail   : raw financial decision signals
    processing_ms     : total pipeline latency
    """
    recommendation_id:    str
    final_action:         str
    fused_confidence:     float
    priority:             str
    portfolio_adjustments:Dict
    security_actions:     Dict
    explanation:          Dict
    decision_detail:      Dict
    processing_ms:        float
    timestamp:            float

    def to_dict(self) -> Dict:
        return {
            "recommendation_id":    self.recommendation_id,
            "final_action":         self.final_action,
            "fused_confidence":     round(self.fused_confidence, 4),
            "priority":             self.priority,
            "portfolio_adjustments":self.portfolio_adjustments,
            "security_actions":     self.security_actions,
            "explanation":          self.explanation,
            "decision_detail":      self.decision_detail,
            "processing_ms":        round(self.processing_ms, 3),
            "timestamp":            self.timestamp,
            "milestone":            8,
        }


# ---------------------------------------------------------------------------
# Recommendation Engine
# ---------------------------------------------------------------------------

class RecommendationEngine:
    """
    M8 Autonomous Decision Intelligence Engine.

    Maintains a decision log for auditability.
    One instance should be shared across the application lifetime.

    Parameters
    ----------
    log_level : Python logging level (default INFO)
    max_log   : maximum number of decisions to keep in memory
    """

    def __init__(self, log_level: int = logging.INFO, max_log: int = 1000):
        self._decision_log: List[Dict] = []
        self._total_processed = 0
        self._max_log = max_log
        logger.setLevel(log_level)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def get_recommendation(
        self,
        m7_dict:    Dict[str, Any],
        m4_risk:    Optional[Dict[str, float]] = None,
        m5_result:  Optional[Dict[str, Any]]   = None,
        security:   Optional[Dict[str, Any]]   = None,
        scenario:   Optional[str]              = None,
    ) -> SystemRecommendation:
        """
        Run the full M8 decision pipeline.

        Parameters
        ----------
        m7_dict   : IntelligenceReport.to_dict() from Milestone 7
        m4_risk   : risk metrics from M4
                    {"volatility": float, "var_95": float,
                     "cvar_95": float, "max_drawdown": float}
        m5_result : optimization result from M5
                    {"expected_return_ann": float, "sharpe_ratio": float,
                     "weights": {ticker: weight}, ...}
        security  : security reports from M6/M7
                    {"pqc": SecurityReport.to_dict(),
                     "bayesian": BayesianSecurityReport.to_dict()}
        scenario  : optional scenario name
                    "market_crash" | "high_inflation" | "liquidity_crisis" |
                    "rate_shock" | "commodity_boom" | "geopolitical_risk"

        Returns
        -------
        SystemRecommendation
        """
        t0 = time.perf_counter()
        self._total_processed += 1
        rec_id = str(uuid.uuid4())[:12]

        logger.info("Processing recommendation %s (scenario=%s)", rec_id, scenario)

        # ── Step 1: Build SystemState ──────────────────────────────────────
        state = SystemState.from_m7_dict(
            m7=m7_dict, m4_risk=m4_risk, m5_result=m5_result, security=security
        )
        if scenario:
            state.scenario = scenario

        # ── Step 2: Financial decision ─────────────────────────────────────
        financial_decision = generate_decision(state)
        logger.info(
            "rec=%s  action=%s  priority=%s  confidence=%.2f  score=%.3f",
            rec_id, financial_decision.action, financial_decision.priority,
            financial_decision.confidence, financial_decision.composite_score
        )

        # ── Step 3: Strategy adaptation ────────────────────────────────────
        adapted_strategy = adapt_strategy(state, financial_decision)

        # ── Step 4: Security constraints ───────────────────────────────────
        security_constraints = apply_security_constraints(state, financial_decision)

        if security_constraints.security_tier not in ("CLEAR", "MONITOR"):
            logger.warning(
                "rec=%s  security_tier=%s  threat=%.3f  override=%s",
                rec_id, security_constraints.security_tier,
                security_constraints.threat_composite,
                security_constraints.overridden_action,
            )

        # ── Step 5: Fuse decisions ─────────────────────────────────────────
        fused = _fuse_decisions(financial_decision, adapted_strategy, security_constraints)

        # ── Step 6: Explanation ────────────────────────────────────────────
        explanation = generate_explanation(
            state, financial_decision, adapted_strategy, security_constraints
        )

        # ── Step 7: Assemble recommendation ───────────────────────────────
        t1 = time.perf_counter()
        processing_ms = (t1 - t0) * 1000.0

        rec = SystemRecommendation(
            recommendation_id     = rec_id,
            final_action          = fused.final_action,
            fused_confidence      = fused.fused_confidence,
            priority              = fused.priority,
            portfolio_adjustments = fused.portfolio_adjustments,
            security_actions      = security_constraints.to_dict(),
            explanation           = explanation.to_dict(),
            decision_detail       = financial_decision.to_dict(),
            processing_ms         = processing_ms,
            timestamp             = time.time(),
        )

        # ── Step 8: Log ────────────────────────────────────────────────────
        self._log_decision(rec_id, state, rec, processing_ms)

        logger.info(
            "rec=%s  COMPLETE — final_action=%s  confidence=%.2f  "
            "security=%s  latency=%.1fms",
            rec_id, rec.final_action, rec.fused_confidence,
            security_constraints.security_tier, processing_ms,
        )

        return rec

    # ------------------------------------------------------------------
    # Convenience: build SystemState and call get_recommendation
    # ------------------------------------------------------------------

    def get_recommendation_from_state(
        self,
        state:    SystemState,
    ) -> SystemRecommendation:
        """
        Run the pipeline when the caller has already built a SystemState.
        Useful for testing and when module outputs are already parsed.
        """
        # Convert state back to the dict-based interface
        m7_stub = {
            "regime_probabilities": {
                "current_regime_label": state.current_regime,
                "current_probs":        state.regime_probs,
                "log_likelihood":       0.0,
            },
            "volatility_forecast": {
                "current_vol_ann":  state.garch_vol_current,
                "forecast_30d_vol": state.garch_vol_30d,
                "forecast_path":    {},
            },
            "forward_return_forecast": {
                "forward_distributions": {
                    "21d": {
                        "expected_return_ann": state.forward_return_21d,
                        "annualised_vol":      state.forward_vol_21d,
                        "cvar_95_percent":     state.forward_cvar_21d,
                        "var_95_percent":      state.var_95,
                    }
                }
            },
            "adaptive_parameter_shift": {
                "optimization_method": "multi_objective",
                "adapted_parameters":  {},
                "uncertainty_metrics": {
                    "regime_entropy": state.regime_entropy,
                    "shrinkage":      state.regime_entropy,
                },
            },
            "optimal_allocation": {
                "weights":             state.optimal_weights,
                "expected_return_ann": state.expected_return,
                "sharpe_ratio":        state.sharpe_ratio,
            },
            "transition_matrix": {
                "mixing_time_days": state.mixing_time_days,
            },
            "meta": {
                "tickers": state.tickers,
            },
        }
        m4_risk = {
            "volatility":    state.volatility_ann,
            "var_95":        state.var_95,
            "cvar_95":       state.cvar_95,
            "max_drawdown":  state.max_drawdown,
        }
        security = {
            "pqc": {
                "threat_level":    state.pqc_threat_level,
                "anomaly_score":   state.pqc_anomaly_score,
                "quarantine_status": "QUARANTINED" if state.transaction_quarantined else "CLEAR",
                "key_rotation":    state.key_rotation_active,
            },
            "bayesian": {
                "posterior_probability": state.bayesian_posterior,
                "threat_level":          state.bayesian_threat_level,
                "quarantine_status":     "QUARANTINED" if state.transaction_quarantined else "CLEAR",
                "key_rotation_signal":   state.key_rotation_active,
            },
        }
        return self.get_recommendation(
            m7_dict   = m7_stub,
            m4_risk   = m4_risk,
            security  = security,
            scenario  = state.scenario,
        )

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def get_decision_log(self, n: int = 20) -> List[Dict]:
        """Return the n most recent decision log entries."""
        return self._decision_log[-n:]

    def get_statistics(self) -> Dict:
        """Return aggregate statistics over all processed recommendations."""
        if not self._decision_log:
            return {"total_processed": 0}

        from collections import Counter
        action_counts  = Counter(e["final_action"]  for e in self._decision_log)
        tier_counts    = Counter(e["security_tier"] for e in self._decision_log)
        latencies      = [e["processing_ms"] for e in self._decision_log]

        return {
            "total_processed":   self._total_processed,
            "action_distribution": dict(action_counts),
            "security_tier_distribution": dict(tier_counts),
            "latency_ms": {
                "mean": round(float(np.mean(latencies)), 2),
                "p95":  round(float(np.percentile(latencies, 95)), 2),
                "max":  round(float(np.max(latencies)), 2),
            },
        }

    # ------------------------------------------------------------------
    # Internal logging
    # ------------------------------------------------------------------

    def _log_decision(
        self, rec_id: str, state: SystemState,
        rec: SystemRecommendation, processing_ms: float,
    ):
        """Write a structured audit record to the in-memory log."""
        # Hash the key inputs for audit fingerprint (no raw data stored)
        input_fingerprint = hashlib.sha256(
            json.dumps({
                "regime":   state.current_regime,
                "vol":      round(state.volatility_ann, 4),
                "var":      round(state.var_95, 4),
                "sharpe":   round(state.sharpe_ratio, 4),
                "pqc":      state.pqc_threat_level,
                "bay":      round(state.bayesian_posterior, 3),
                "scenario": state.scenario,
            }, sort_keys=True).encode()
        ).hexdigest()[:16]

        entry = {
            "recommendation_id": rec_id,
            "timestamp":         rec.timestamp,
            "final_action":      rec.final_action,
            "fused_confidence":  round(rec.fused_confidence, 4),
            "priority":          rec.priority,
            "security_tier":     rec.security_actions.get("security_tier", "CLEAR"),
            "regime":            state.current_regime,
            "vol_ann":           round(state.volatility_ann, 4),
            "sharpe":            round(state.sharpe_ratio, 4),
            "scenario":          state.scenario,
            "input_fingerprint": input_fingerprint,
            "processing_ms":     round(processing_ms, 2),
            "triggered_rules":   rec.decision_detail.get("triggered_rules", []),
        }

        self._decision_log.append(entry)

        # LRU eviction
        if len(self._decision_log) > self._max_log:
            self._decision_log = self._decision_log[-self._max_log:]
