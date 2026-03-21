"""
api_m8.py — Milestone 8: Decision Intelligence API
===================================================
Wraps the existing M8 logic into one clean function.
No print statements. Returns a single structured dictionary.

Usage:
    from portfolio.api_m8 import get_decision

    result = get_decision(
        m7_output    = get_market_regime(...),   # dict from api_m7
        m3_output    = get_portfolio_construction(...),  # dict from api_m3
        m6_output    = get_virtual_trade_and_security(...),  # dict from api_m6
        scenario     = None,   # optional: "market_crash" | "high_inflation" | etc.
    )
"""

import sys
import os
import types
import importlib.util

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_m8():
    """Load milestone8_complete.py from the portfolio folder."""
    portfolio_dir = os.path.dirname(__file__)
    project_dir   = os.path.join(portfolio_dir, "..")
    for folder in [portfolio_dir, project_dir]:
        path = os.path.normpath(os.path.join(folder, "milestone8_complete.py"))
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("milestone8_complete", path)
            mod  = types.ModuleType("milestone8_complete")
            mod.__spec__ = spec
            sys.modules["milestone8_complete"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "milestone8_complete.py not found in portfolio/ or project root."
    )


def get_decision(
    m7_output: dict,
    m3_output: dict = None,
    m6_output: dict = None,
    scenario: str = None,
) -> dict:
    """
    Runs the M8 decision intelligence pipeline.

    Accepts the structured dicts returned by api_m3, api_m6, and api_m7
    and fuses them into one recommendation.

    Parameters
    ----------
    m7_output : dict from api_m7.get_market_regime()
    m3_output : dict from api_m3.get_portfolio_construction()  (optional)
    m6_output : dict from api_m6.get_virtual_trade_and_security()  (optional)
    scenario  : optional scenario override — one of:
                "market_crash" | "high_inflation" | "liquidity_crisis" |
                "rate_shock" | "commodity_boom" | "geopolitical_risk"

    Returns
    -------
    dict with keys:
        recommendation_id    : unique run ID
        final_action         : "INCREASE_EXPOSURE" | "HOLD" |
                               "REDUCE_RISK" | "REBALANCE"
        financial_action     : action before security override (if any)
        security_override    : True if security changed the action
        fused_confidence     : confidence after security adjustment [0–1]
        priority             : "low" | "medium" | "high"

        portfolio_adjustments : {
            optimizer_params: {lam_return, lam_vol, lam_cvar,
                                lam_drawdown, max_weight, target_vol},
            leverage_scalar, min_assets, sector_cap,
            rebalance_urgency, confidence_blend,
            regime_applied, action_applied, notes
        }

        security_constraints  : {
            threat_composite, security_tier, security_action,
            restrictions, large_tx_blocked, multi_step_required,
            max_trades_per_session, account_flagged,
            key_rotation_required, overridden_action,
            action_was_overridden
        }

        explanation           : {
            summary, action_rationale, factors, narrative,
            risk_assessment, security_summary, confidence_statement
        }

        signals               : {regime, vol, var, cvar, dd, ret, sharpe}

        processing_ms         : float

        error                 : None or error string
    """

    result = {
        "recommendation_id":    None,
        "final_action":         None,
        "financial_action":     None,
        "security_override":    False,
        "fused_confidence":     None,
        "priority":             None,
        "portfolio_adjustments": None,
        "security_constraints": None,
        "explanation":          None,
        "signals":              None,
        "processing_ms":        None,
        "error":                None,
    }

    try:
        m8 = _load_m8()

        # ── Extract values from M7 output ─────────────────────────────────────
        probs = m7_output.get("regime_probabilities", {})
        trans_p = max(0.0,
                      1.0
                      - probs.get("Low-Vol Bull",  0.0)
                      - probs.get("High-Vol Bear", 0.0)
                      - probs.get("Crisis",        0.0))

        # ── Extract values from M3 output (or use M7 forward estimates) ───────
        if m3_output and not m3_output.get("error"):
            ms = m3_output.get("max_sharpe_portfolio") or {}
            vol      = float(ms.get("volatility",      m7_output.get("garch_current_vol", 0.20)))
            exp_ret  = float(ms.get("expected_return",
                             m7_output.get("forward_distributions", {})
                             .get("21d", {}).get("expected_return_ann", 0.08)))
            sharpe   = float(ms.get("sharpe_ratio",     0.50))
            rm       = m3_output.get("risk_metrics") or {}
            var_95   = float(rm.get("var_95_pct",       0.03))
            cvar_95  = float(rm.get("cvar_95_pct",      0.05))
            max_dd   = abs(float(rm.get("max_drawdown", 0.10)))
            weights  = ms.get("weights", {})
            tickers  = m3_output.get("tickers", [])
        else:
            fd21     = m7_output.get("forward_distributions", {}).get("21d", {})
            vol      = float(m7_output.get("garch_current_vol",  0.20))
            exp_ret  = float(fd21.get("expected_return_ann",     0.08))
            sharpe   = 0.50
            var_95   = float(fd21.get("var_95_pct",              0.03))
            cvar_95  = float(fd21.get("cvar_95_pct",             0.05))
            max_dd   = 0.10
            weights  = m7_output.get("optimal_weights", {})
            tickers  = m7_output.get("tickers", [])

        # ── Extract security signals from M6 output ───────────────────────────
        pqc_level     = "low"
        pqc_anomaly   = 0.0
        bay_posterior = 0.05
        bay_level     = "SAFE"
        quarantined   = False
        key_rot       = False

        if m6_output and not m6_output.get("error"):
            pqc           = m6_output.get("security_pqc") or {}
            bay           = m6_output.get("security_bayesian") or {}
            pqc_level     = pqc.get("threat_level",          "low")
            pqc_anomaly   = float(pqc.get("anomaly_score",    0.0))
            bay_posterior = float(bay.get("posterior_probability", 0.05))
            bay_level     = bay.get("threat_level",           "SAFE")
            quarantined   = bool(
                pqc.get("quarantine_status") == "QUARANTINED"
                or bay.get("quarantine_status") == "QUARANTINED"
            )
            key_rot       = bool(
                pqc.get("key_rotation", False)
                or bay.get("key_rotation_signal", False)
            )

        # ── Build SystemState ─────────────────────────────────────────────────
        engine = m8.RecommendationEngine()
        state  = m8.SystemState(
            volatility_ann          = vol,
            var_95                  = var_95,
            cvar_95                 = cvar_95,
            max_drawdown            = max_dd,
            expected_return         = exp_ret,
            sharpe_ratio            = sharpe,
            optimal_weights         = weights,
            tickers                 = tickers,
            current_regime          = m7_output.get("current_regime", "Transitional"),
            regime_probs            = {
                "Low-Vol Bull":  probs.get("Low-Vol Bull",  0.0),
                "High-Vol Bear": probs.get("High-Vol Bear", 0.0),
                "Crisis":        probs.get("Crisis",        0.0),
                "Transitional":  trans_p,
            },
            forward_return_21d      = float(m7_output.get("forward_distributions", {})
                                            .get("21d", {}).get("expected_return_ann", 0.0)),
            forward_vol_21d         = float(m7_output.get("forward_distributions", {})
                                            .get("21d", {}).get("annualised_vol", 0.20)),
            forward_cvar_21d        = float(m7_output.get("forward_distributions", {})
                                            .get("21d", {}).get("cvar_95_pct", 0.05)),
            garch_vol_current       = float(m7_output.get("garch_current_vol",  0.20)),
            garch_vol_30d           = float(m7_output.get("garch_30d_forecast", 0.20)),
            pqc_threat_level        = pqc_level,
            pqc_anomaly_score       = pqc_anomaly,
            bayesian_posterior      = bay_posterior,
            bayesian_threat_level   = bay_level,
            transaction_quarantined = quarantined,
            key_rotation_active     = key_rot,
            scenario                = scenario,
        )

        rec = engine.get_recommendation_from_state(state)
        d   = rec.to_dict()

        result.update({
            "recommendation_id":    d["recommendation_id"],
            "final_action":         d["final_action"],
            "financial_action":     d["decision_detail"]["action"],
            "security_override":    d["security_actions"]["action_was_overridden"],
            "fused_confidence":     round(float(d["fused_confidence"]), 4),
            "priority":             d["priority"],
            "portfolio_adjustments": d["portfolio_adjustments"],
            "security_constraints": d["security_actions"],
            "explanation":          d["explanation"],
            "signals":              d["decision_detail"]["signal_breakdown"],
            "processing_ms":        round(float(d["processing_ms"]), 3),
        })

    except Exception as e:
        result["error"] = str(e)

    return result
