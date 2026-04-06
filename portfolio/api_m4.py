"""
api_m4.py — Milestone 4: Scenario Analysis & Risk API
======================================================
Wraps the existing M4 logic into one clean function.
No print statements. Returns a single structured dictionary.

Usage:
    from portfolio.api_m4 import get_scenario_analysis

    result = get_scenario_analysis(
        tickers          = ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        portfolio_value  = 1_000_000,
        risk_free_rate   = 0.07,
        confidence_level = 0.95,
        scenarios        = "ALL",   # or comma-separated codes e.g. "CRISIS_1,MOD_1"
    )
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_scenario_analysis(
    tickers: list,
    portfolio_value: float = 1_000_000,
    risk_free_rate: float = 0.07,
    confidence_level: float = 0.95,
    scenarios: str = "ALL",
) -> dict:
    """
    Runs M4 scenario analysis and risk metrics on live data.

    Parameters
    ----------
    tickers          : list of NSE ticker strings
    portfolio_value  : total portfolio value in INR
    risk_free_rate   : annual risk-free rate as decimal (0.07 = 7%)
    confidence_level : VaR/CVaR confidence level (0.95 = 95%)
    scenarios        : "ALL", "SEVERE", or comma-separated scenario codes
                       e.g. "CRISIS_1,MOD_1,MILD_1"

    Returns
    -------
    dict with keys:
        tickers           : tickers used
        portfolio_value   : INR value used
        risk_free_rate    : rate used

        optimal_portfolio : {weights, expected_return, volatility, sharpe_ratio}

        risk_metrics      : {
            var_parametric_inr, var_parametric_pct,
            var_historical_inr, var_historical_pct,
            cvar_inr, cvar_pct,
            risk_contribution: {ticker: pct}
        }

        scenarios_tested  : list of {
            code, name, category, likelihood, duration, recovery,
            base_return, stressed_return, return_change,
            base_vol, stressed_vol, vol_change_pct,
            base_sharpe, stressed_sharpe, sharpe_change,
            pnl_inr, pnl_pct
        }

        summary           : {
            total_tested,
            worst_loss:   {scenario, pnl_inr, pnl_pct},
            best_gain:    {scenario, pnl_inr, pnl_pct},
            risk_level:   "high" | "moderate" | "acceptable",
            risk_message: str
        }

        error             : None or error string
    """

    # ── Imports from existing M4 modules ─────────────────────────────────────
    from portfolio.portfolio_complete   import load_price_data, normalize_tickers_for_market_data
    from portfolio.portfolio_complete     import (
        PortfolioOptimizer, compute_daily_returns,
        compute_expected_returns, compute_covariance_matrix,
    )
    from portfolio.portfolio_complete import ScenarioEngine
    from portfolio.portfolio_complete    import (
        compute_parametric_var, compute_historical_var,
        compute_cvar, compute_component_var,
    )

    # Import the scenario library directly from the existing M4 run file
    # by adding examples/ to path and importing get_enhanced_scenarios
    _examples = os.path.join(os.path.dirname(__file__), "..", "examples")
    if _examples not in sys.path:
        sys.path.insert(0, _examples)
    from examples.run_milestone4_ENHANCED import get_enhanced_scenarios, analyze_impact

    result = {
        "tickers":           tickers,
        "portfolio_value":   portfolio_value,
        "risk_free_rate":    risk_free_rate,
        "optimal_portfolio": None,
        "risk_metrics":      None,
        "scenarios_tested":  [],
        "summary":           None,
        "error":             None,
    }

    try:
        tickers = normalize_tickers_for_market_data(tickers)

        # ── Load data and build optimal portfolio ─────────────────────────────
        prices  = load_price_data(tickers, period="2y")
        if prices is None or prices.empty:
            result["error"] = "Could not load price data."
            return result

        valid = [t for t in tickers if t in prices.columns]
        prices  = prices[valid].dropna()
        returns = compute_daily_returns(prices)
        mu      = compute_expected_returns(returns)
        sigma   = compute_covariance_matrix(returns)

        opt     = PortfolioOptimizer(mu, sigma, risk_free_rate)
        opt_res = opt.optimize_max_sharpe()
        weights = opt_res.weights

        result["optimal_portfolio"] = {
            "weights":         {t: round(float(w), 4) for t, w in weights.items()},
            "expected_return": round(float(opt_res.expected_return), 4),
            "volatility":      round(float(opt_res.volatility), 4),
            "sharpe_ratio":    round(float(opt_res.sharpe_ratio), 4),
        }

        # ── Risk metrics ──────────────────────────────────────────────────────
        port_ret   = returns[list(weights.index)].dot(weights)
        pvar       = compute_parametric_var(weights, mu, sigma, confidence_level, 1, portfolio_value)
        hvar       = compute_historical_var(port_ret, confidence_level, portfolio_value)
        cvar       = compute_cvar(port_ret, confidence_level, portfolio_value)
        cvar_comp  = compute_component_var(weights, sigma, confidence_level)

        result["risk_metrics"] = {
            "var_parametric_inr": round(float(pvar["var_amount"]), 2),
            "var_parametric_pct": round(float(pvar["var_percent"]), 4),
            "var_historical_inr": round(float(hvar["var_amount"]), 2),
            "var_historical_pct": round(float(hvar["var_percent"]), 4),
            "cvar_inr":           round(float(cvar["cvar_amount"]), 2),
            "cvar_pct":           round(float(cvar["cvar_percent"]), 4),
            "risk_contribution":  {
                t: round(float(cvar_comp.loc[t, "% Contribution"]), 2)
                for t in cvar_comp.index
            },
        }

        # ── Scenario analysis ─────────────────────────────────────────────────
        all_scenarios = get_enhanced_scenarios()
        engine        = ScenarioEngine(mu, sigma)

        # Resolve which scenarios to run
        sc_upper = scenarios.strip().upper()
        if sc_upper == "ALL":
            selected = list(all_scenarios.keys())
        elif sc_upper == "SEVERE":
            selected = [k for k, v in all_scenarios.items() if "SEVERE" in v["category"]]
        else:
            selected = [
                c.strip() for c in sc_upper.split(",")
                if c.strip() in all_scenarios
            ]

        rows = []
        for code in selected:
            info   = all_scenarios[code]
            shock  = info["shock"]
            st_mu, st_sigma = engine.apply_scenario(shock)
            impact = analyze_impact(
                weights, mu, sigma, st_mu, st_sigma,
                risk_free_rate, portfolio_value
            )
            rows.append({
                "code":            code,
                "name":            info["name"],
                "category":        info["category"],
                "likelihood":      info["likelihood"],
                "duration":        info["duration"],
                "recovery":        info["recovery"],
                "base_return":     round(float(impact["base_return"]),      4),
                "stressed_return": round(float(impact["stressed_return"]),  4),
                "return_change":   round(float(impact["return_change"]),    4),
                "base_vol":        round(float(impact["base_vol"]),         4),
                "stressed_vol":    round(float(impact["stressed_vol"]),     4),
                "vol_change_pct":  round(float(impact["vol_change_pct"]),   4),
                "base_sharpe":     round(float(impact["base_sharpe"]),      4),
                "stressed_sharpe": round(float(impact["stressed_sharpe"]),  4),
                "sharpe_change":   round(float(impact["sharpe_change"]),    4),
                "pnl_inr":         round(float(impact["portfolio_loss"]),   2),
                "pnl_pct":         round(float(impact["loss_pct"]),         4),
            })

        result["scenarios_tested"] = rows

        # ── Summary ───────────────────────────────────────────────────────────
        if rows:
            df         = pd.DataFrame(rows)
            worst_idx  = df["pnl_inr"].idxmin()
            best_idx   = df["pnl_inr"].idxmax()
            max_loss   = abs(df[df["pnl_pct"] < 0]["pnl_pct"].min() * 100) \
                         if (df["pnl_pct"] < 0).any() else 0.0

            risk_level   = "high" if max_loss > 35 else "moderate" if max_loss > 20 else "acceptable"
            risk_message = (
                "Portfolio is highly vulnerable — consider hedging."
                if risk_level == "high"
                else "Portfolio has moderate stress exposure — monitor closely."
                if risk_level == "moderate"
                else "Portfolio stress exposure is within acceptable limits."
            )

            result["summary"] = {
                "total_tested": len(rows),
                "worst_loss": {
                    "scenario": df.loc[worst_idx, "name"],
                    "pnl_inr":  round(float(df.loc[worst_idx, "pnl_inr"]), 2),
                    "pnl_pct":  round(float(df.loc[worst_idx, "pnl_pct"]), 4),
                },
                "best_gain": {
                    "scenario": df.loc[best_idx, "name"],
                    "pnl_inr":  round(float(df.loc[best_idx, "pnl_inr"]), 2),
                    "pnl_pct":  round(float(df.loc[best_idx, "pnl_pct"]), 4),
                },
                "risk_level":   risk_level,
                "risk_message": risk_message,
            }

    except Exception as e:
        result["error"] = str(e)

    return result
