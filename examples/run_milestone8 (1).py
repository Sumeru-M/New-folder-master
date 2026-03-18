"""
run_milestone8.py — Autonomous Decision Intelligence Demo
==========================================================

Tests the full M8 pipeline against four scenarios:
  1. Benign bull market   → HOLD / INCREASE_EXPOSURE
  2. Bear market stress   → REDUCE_RISK
  3. Crisis + security threat → REBALANCE overridden by security
  4. User-driven interactive mode: supply real M7 output + choose scenario

Does NOT modify any M1–M7 module.
"""

import sys, os, json, time, logging

logging.basicConfig(level=logging.WARNING)

# ── Path setup ────────────────────────────────────────────────────────────────
# Runner lives in examples/  →  portfolio/ is one level up (../portfolio)
import types, importlib.util

def _load_m8():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here,                    "milestone8_complete.py"),
        os.path.join(here, "portfolio",       "milestone8_complete.py"),
        os.path.join(here, "..",              "milestone8_complete.py"),
        os.path.join(here, "..", "portfolio", "milestone8_complete.py"),
    ]
    for path in candidates:
        path = os.path.normpath(path)
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("milestone8_complete", path)
            mod  = types.ModuleType("milestone8_complete")
            mod.__spec__ = spec
            sys.modules["milestone8_complete"] = mod
            spec.loader.exec_module(mod)
            return mod
    searched = "\n".join(f"  {os.path.normpath(p)}" for p in candidates)
    raise FileNotFoundError(
        f"milestone8_complete.py not found. Searched:\n{searched}\n"
        "Fix: place milestone8_complete.py inside your portfolio/ folder."
    )

_m8 = _load_m8()
RecommendationEngine = _m8.RecommendationEngine
SystemState          = _m8.SystemState


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def banner(title):
    print("\n" + "═" * 70)
    print(f"  {title}")
    print("═" * 70)

def print_rec(rec, label=""):
    d = rec.to_dict()
    action_icons = {
        "INCREASE_EXPOSURE": "📈",
        "HOLD":              "⏸️ ",
        "REDUCE_RISK":       "🛡️ ",
        "REBALANCE":         "⚖️ ",
    }
    icon = action_icons.get(d["final_action"], "•")
    tier_icons = {
        "CLEAR": "🟢", "MONITOR": "🟡",
        "ELEVATED": "🟠", "HIGH": "🟠", "CRITICAL": "🔴"
    }
    sec_tier = d["security_actions"]["security_tier"]
    sec_icon = tier_icons.get(sec_tier, "⚪")

    print(f"\n  {icon}  Final action   : {d['final_action']}")
    print(f"  📊  Confidence     : {d['fused_confidence']:.0%}  ({d['priority']} priority)")
    print(f"  {sec_icon}  Security tier  : {sec_tier}")
    if d["security_actions"]["action_was_overridden"]:
        print(f"  ⚠️   Security override: financial={d['decision_detail']['action']} "
              f"→ final={d['final_action']}")

    print(f"\n  Explanation summary:")
    print(f"    {d['explanation']['summary']}")
    print(f"\n  Primary drivers:")
    for f in d["explanation"]["factors"][:3]:
        print(f"    {f['tier']:20s}  {f['factor']:25s}  "
              f"contrib={f['contribution']:+.4f}  ({f['current_value']})")

    print(f"\n  Strategy parameters:")
    op = d["portfolio_adjustments"]["optimizer_params"]
    print(f"    λ_return={op['lam_return']:.3f}  λ_vol={op['lam_vol']:.3f}  "
          f"λ_cvar={op['lam_cvar']:.3f}  max_w={op['max_weight']:.0%}  "
          f"target_vol={op['target_vol']:.0%}")
    print(f"    leverage={d['portfolio_adjustments']['leverage_scalar']:.0%}  "
          f"urgency={d['portfolio_adjustments']['rebalance_urgency']}")

    if d["security_actions"]["restrictions"]:
        print(f"\n  Active restrictions:")
        for r in d["security_actions"]["restrictions"][:3]:
            print(f"    • {r}")

    print(f"\n  Processing time: {d['processing_ms']:.1f} ms")


# ═════════════════════════════════════════════════════════════════════════════
# Synthetic state builders for each test scenario
# ═════════════════════════════════════════════════════════════════════════════

def make_bull_state() -> SystemState:
    return SystemState(
        volatility_ann=0.13, var_95=0.018, cvar_95=0.028,
        max_drawdown=0.06, expected_return=0.18, sharpe_ratio=1.1,
        current_regime="Low-Vol Bull",
        regime_probs={"Low-Vol Bull":0.82,"High-Vol Bear":0.10,"Crisis":0.02,"Transitional":0.06},
        forward_return_21d=0.16, forward_vol_21d=0.13, forward_cvar_21d=0.025,
        garch_vol_current=0.13, garch_vol_30d=0.135,
        pqc_threat_level="low", pqc_anomaly_score=0.2,
        bayesian_posterior=0.04, bayesian_threat_level="SAFE",
    )

def make_bear_state() -> SystemState:
    return SystemState(
        volatility_ann=0.28, var_95=0.038, cvar_95=0.055,
        max_drawdown=0.22, expected_return=0.04, sharpe_ratio=0.14,
        current_regime="High-Vol Bear",
        regime_probs={"Low-Vol Bull":0.05,"High-Vol Bear":0.75,"Crisis":0.12,"Transitional":0.08},
        forward_return_21d=-0.08, forward_vol_21d=0.30, forward_cvar_21d=0.062,
        garch_vol_current=0.28, garch_vol_30d=0.31,
        pqc_threat_level="low", pqc_anomaly_score=0.5,
        bayesian_posterior=0.08, bayesian_threat_level="SAFE",
    )

def make_crisis_security_state() -> SystemState:
    return SystemState(
        volatility_ann=0.45, var_95=0.062, cvar_95=0.092,
        max_drawdown=0.38, expected_return=-0.15, sharpe_ratio=-0.85,
        current_regime="Crisis",
        regime_probs={"Low-Vol Bull":0.01,"High-Vol Bear":0.15,"Crisis":0.82,"Transitional":0.02},
        forward_return_21d=-0.22, forward_vol_21d=0.48, forward_cvar_21d=0.10,
        garch_vol_current=0.45, garch_vol_30d=0.50,
        pqc_threat_level="high", pqc_anomaly_score=12.5,
        bayesian_posterior=0.91, bayesian_threat_level="CRITICAL_THREAT",
        transaction_quarantined=True,
    )

def make_transitional_state() -> SystemState:
    return SystemState(
        volatility_ann=0.19, var_95=0.026, cvar_95=0.040,
        max_drawdown=0.11, expected_return=0.09, sharpe_ratio=0.52,
        current_regime="Transitional",
        regime_probs={"Low-Vol Bull":0.22,"High-Vol Bear":0.30,"Crisis":0.18,"Transitional":0.30},
        forward_return_21d=0.04, forward_vol_21d=0.21, forward_cvar_21d=0.043,
        garch_vol_current=0.20, garch_vol_30d=0.21,
        pqc_threat_level="medium", pqc_anomaly_score=2.8,
        bayesian_posterior=0.35, bayesian_threat_level="MONITOR",
        regime_entropy=0.65,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main():
    engine = RecommendationEngine()

    # ── TEST 1: Bull market ────────────────────────────────────────────────
    banner("TEST 1 — Bull Market (Low volatility, positive returns, clear regime)")
    rec1 = engine.get_recommendation_from_state(make_bull_state())
    print_rec(rec1)

    # ── TEST 2: Bear market ────────────────────────────────────────────────
    banner("TEST 2 — Bear Market (High volatility, falling returns)")
    rec2 = engine.get_recommendation_from_state(make_bear_state())
    print_rec(rec2)

    # ── TEST 3: Crisis + Critical security threat ──────────────────────────
    banner("TEST 3 — Crisis + Critical Security Threat (worst case)")
    rec3 = engine.get_recommendation_from_state(make_crisis_security_state())
    print_rec(rec3)

    # ── TEST 4: Transitional + Elevated security ───────────────────────────
    banner("TEST 4 — Transitional Market + Elevated Security (mixed signals)")
    rec4 = engine.get_recommendation_from_state(make_transitional_state())
    print_rec(rec4)

    # ── TEST 5: Scenario overlay ───────────────────────────────────────────
    banner("TEST 5 — Scenario Overlay: market_crash applied to Bull state")
    bull_crash       = make_bull_state()
    bull_crash.scenario = "market_crash"
    rec5 = engine.get_recommendation_from_state(bull_crash)
    print_rec(rec5)
    print(f"\n  Note: Bull state + market_crash scenario should override to REBALANCE")

    # ── Statistics ────────────────────────────────────────────────────────
    banner("ENGINE STATISTICS")
    stats = engine.get_statistics()
    print(f"\n  Total processed  : {stats['total_processed']}")
    print(f"  Action breakdown : {stats['action_distribution']}")
    print(f"  Security tiers   : {stats['security_tier_distribution']}")
    print(f"  Latency (mean)   : {stats['latency_ms']['mean']:.1f} ms")
    print(f"  Latency (p95)    : {stats['latency_ms']['p95']:.1f} ms")

    # ── Decision log sample ───────────────────────────────────────────────
    banner("DECISION LOG (last 5 entries)")
    for entry in engine.get_decision_log(5):
        print(f"  [{entry['recommendation_id']}]  "
              f"{entry['final_action']:20s}  "
              f"conf={entry['fused_confidence']:.2f}  "
              f"regime={entry['regime']:16s}  "
              f"security={entry['security_tier']:8s}  "
              f"{entry['processing_ms']:.1f}ms")

    # ── Full JSON output for reference ────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "artifacts", "milestone8_report.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {f"test_{i+1}": rec.to_dict()
             for i, rec in enumerate([rec1, rec2, rec3, rec4, rec5])},
            f, indent=2, default=str,
        )
    print(f"\n  Full JSON saved: {out_path}")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
