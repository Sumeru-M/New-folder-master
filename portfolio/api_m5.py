"""
api_m5.py — Milestone 5: Institutional Optimisation API
========================================================
Wraps the existing M5 logic into one clean function.
No print statements. Returns a single structured dictionary.

Usage:
    from portfolio.api_m5 import get_institutional_optimisation

    result = get_institutional_optimisation(
        tickers          = ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        current_weights  = [0.33, 0.33, 0.34],   # optional
        portfolio_value  = 1_000_000,
        risk_free_rate   = 0.07,
        max_weight       = 0.40,
        sector_cap       = 0.60,
        confidence_level = 0.95,
        methods          = "all",   # or specific: "max_sharpe,risk_parity"
    )
"""

import sys
import os
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_institutional_optimisation(
    tickers: list,
    current_weights: list = None,
    portfolio_value: float = 1_000_000,
    risk_free_rate: float = 0.07,
    max_weight: float = 0.40,
    sector_cap: float = 0.60,
    confidence_level: float = 0.95,
    methods: str = "all",
) -> dict:
    """
    Runs the full M5 institutional optimisation pipeline on live data.

    Parameters
    ----------
    tickers          : list of NSE ticker strings
    current_weights  : optional list of floats (fractions, sum to 1)
    portfolio_value  : total INR value for rebalance calculations
    risk_free_rate   : annual risk-free rate as decimal
    max_weight       : maximum weight per single asset
    sector_cap       : maximum weight per sector
    confidence_level : CVaR confidence level
    methods          : "all" or comma-separated from:
                       mean_variance, min_variance, cvar,
                       risk_parity, max_diversification, multi_objective

    Returns
    -------
    dict with keys:
        tickers              : tickers used
        portfolio_value      : value used
        risk_free_rate       : rate used
        ledoit_wolf_delta    : shrinkage intensity (0=sample, 1=identity)

        current_portfolio    : {weights, expected_return, volatility, sharpe_ratio}

        methods              : {method_name: {
                                   weights, expected_return, volatility,
                                   sharpe_ratio, cvar_95, health_score,
                                   solve_status
                               }}

        best_method          : name of the method with the highest Sharpe ratio
        recommendation       : {expected_return, volatility, sharpe_ratio,
                                 cvar_95, health_score, weights}

        improvement_vs_current : {return_delta, vol_delta, sharpe_delta}

        risk_attribution     : {diversification_ratio, effective_n,
                                 risk_contribution: {ticker: pct}}

        rebalance_actions    : list of {ticker, action, current_pct,
                                         target_pct, rupees_delta}

        efficient_frontier   : list of {return, volatility, sharpe_ratio}

        robust               : {
            worst_case:        {weights, expected_return, volatility, sharpe_ratio},
            scenario_weighted: {weights, expected_return, volatility, sharpe_ratio}
        }

        method_comparison    : list of {method, expected_return, volatility,
                                         sharpe_ratio, cvar_95, health_score}

        error                : None or error string
    """

    # ── Imports from existing M5 modules ─────────────────────────────────────
    from portfolio.data_loader         import load_price_data
    from portfolio.optimizer           import compute_daily_returns
    from portfolio.optimization_engine import (
        optimize_mean_variance, optimize_minimum_variance, optimize_cvar,
        optimize_risk_parity, optimize_max_diversification,
        optimize_multi_objective, compute_efficient_frontier,
    )
    from portfolio.robust_optimizer    import (
        compute_ledoit_wolf_shrinkage_fixed, optimize_worst_case,
        build_stress_scenarios_from_engine, optimize_scenario_weighted,
    )
    from portfolio.constraints         import build_institutional_constraints
    from portfolio.allocation_scorer   import enrich_allocation_result
    from portfolio.risk_contribution   import build_risk_attribution_report

    # Import sector mapping from M5 run file
    _examples = os.path.join(os.path.dirname(__file__), "..", "examples")
    if _examples not in sys.path:
        sys.path.insert(0, _examples)
    from run_milestone5 import get_sector

    result = {
        "tickers":               tickers,
        "portfolio_value":       portfolio_value,
        "risk_free_rate":        risk_free_rate,
        "ledoit_wolf_delta":     None,
        "current_portfolio":     None,
        "methods":               {},
        "best_method":           None,
        "recommendation":        None,
        "improvement_vs_current": None,
        "risk_attribution":      None,
        "rebalance_actions":     [],
        "efficient_frontier":    [],
        "robust":                {},
        "method_comparison":     [],
        "error":                 None,
    }

    try:
        # ── Load data ─────────────────────────────────────────────────────────
        prices = load_price_data(tickers, period="2y")
        if prices is None or prices.empty:
            result["error"] = "Could not load price data."
            return result

        valid = [t for t in tickers if t in prices.columns]
        prices = prices[valid].dropna()
        N      = len(valid)

        log_ret        = compute_daily_returns(prices)
        mu_log_daily   = log_ret.mean()
        var_log_daily  = log_ret.var()
        mu_series      = np.exp(mu_log_daily * 252 + 0.5 * var_log_daily * 252) - 1
        simple_returns = np.expm1(log_ret.values)

        Sigma_shrink_df, lw_delta = compute_ledoit_wolf_shrinkage_fixed(log_ret)
        Sigma_shrink = Sigma_shrink_df.values
        result["ledoit_wolf_delta"] = round(float(lw_delta), 4)

        # ── Current portfolio stats ───────────────────────────────────────────
        if current_weights and len(current_weights) == N:
            w_arr = np.array(current_weights, dtype=float)
            w_arr = w_arr / w_arr.sum()
        else:
            w_arr = np.ones(N) / N
        current_w = pd.Series(w_arr, index=valid)

        curr_ret    = float(w_arr @ mu_series.values)
        curr_vol    = float(np.sqrt(w_arr @ Sigma_shrink @ w_arr))
        curr_sharpe = (curr_ret - risk_free_rate) / curr_vol if curr_vol > 0 else 0.0

        result["current_portfolio"] = {
            "weights":         {t: round(float(w), 4) for t, w in current_w.items()},
            "expected_return": round(curr_ret, 4),
            "volatility":      round(curr_vol, 4),
            "sharpe_ratio":    round(curr_sharpe, 4),
        }

        # ── Constraints ───────────────────────────────────────────────────────
        min_feasible = 1.0 / N
        eff_max_w    = max(max_weight, min_feasible + 0.01)
        sector_map   = {t: get_sector(t) for t in valid}
        cb = build_institutional_constraints(
            n_assets=N, tickers=valid,
            max_weight=eff_max_w,
            sector_map=sector_map,
            sector_cap=sector_cap,
        )
        rf = risk_free_rate

        # ── Choose which methods to run ───────────────────────────────────────
        method_list = {
            "mean_variance":       lambda: optimize_mean_variance(
                mu=mu_series.values, Sigma=Sigma_shrink, tickers=valid,
                constraint_builder=cb, lam=1.0,
                returns_history=simple_returns, rf=rf),
            "min_variance":        lambda: optimize_minimum_variance(
                mu=mu_series.values, Sigma=Sigma_shrink, tickers=valid,
                constraint_builder=cb,
                returns_history=simple_returns, rf=rf),
            "cvar":                lambda: optimize_cvar(
                mu=mu_series.values, Sigma=Sigma_shrink, tickers=valid,
                constraint_builder=cb, returns_history=simple_returns,
                confidence_level=confidence_level, lam_return=0.3, rf=rf),
            "risk_parity":         lambda: optimize_risk_parity(
                Sigma=Sigma_shrink, tickers=valid, mu=mu_series.values,
                returns_history=simple_returns, rf=rf),
            "max_diversification": lambda: optimize_max_diversification(
                Sigma=Sigma_shrink, tickers=valid, mu=mu_series.values,
                constraint_builder=cb, returns_history=simple_returns, rf=rf),
            "multi_objective":     lambda: optimize_multi_objective(
                mu=mu_series.values, Sigma=Sigma_shrink, tickers=valid,
                constraint_builder=cb, returns_history=simple_returns,
                lam_return=1.0, lam_vol=1.0, lam_cvar=0.5,
                lam_drawdown=0.3, lam_factor_conc=0.0,
                confidence_level=confidence_level, rf=rf),
        }

        if methods.strip().lower() == "all":
            run_methods = list(method_list.keys())
        else:
            run_methods = [m.strip() for m in methods.split(",") if m.strip() in method_list]

        # ── Run optimisations ─────────────────────────────────────────────────
        opt_results = {}
        for mname in run_methods:
            try:
                res = method_list[mname]()
                res = enrich_allocation_result(
                    result=res, covariance_matrix=Sigma_shrink_df,
                    mu=mu_series, current_weights=current_w,
                    portfolio_value=portfolio_value, rf=rf,
                )
                opt_results[mname] = res
            except Exception:
                continue

        if not opt_results:
            result["error"] = "All optimisation methods failed."
            return result

        # ── Pack method outputs ───────────────────────────────────────────────
        for mname, res in opt_results.items():
            result["methods"][mname] = {
                "weights":         {t: round(float(w), 4) for t, w in res.weights.items()},
                "expected_return": round(float(res.expected_return), 4),
                "volatility":      round(float(res.volatility), 4),
                "sharpe_ratio":    round(float(res.sharpe_ratio), 4),
                "cvar_95":         round(float(res.cvar_95), 4),
                "health_score":    int(getattr(res, "allocation_health_score", 0)),
                "solve_status":    str(res.solve_status),
            }

        # ── Best method and recommendation ────────────────────────────────────
        best_method = max(opt_results, key=lambda m: opt_results[m].sharpe_ratio)
        best        = opt_results[best_method]
        result["best_method"] = best_method

        result["recommendation"] = {
            "method":          best_method,
            "weights":         {t: round(float(w), 4) for t, w in best.weights.items()},
            "expected_return": round(float(best.expected_return), 4),
            "volatility":      round(float(best.volatility), 4),
            "sharpe_ratio":    round(float(best.sharpe_ratio), 4),
            "cvar_95":         round(float(best.cvar_95), 4),
            "health_score":    int(getattr(best, "allocation_health_score", 0)),
        }

        result["improvement_vs_current"] = {
            "return_delta": round(float(best.expected_return - curr_ret), 4),
            "vol_delta":    round(float(best.volatility - curr_vol), 4),
            "sharpe_delta": round(float(best.sharpe_ratio - curr_sharpe), 4),
        }

        # ── Risk attribution ──────────────────────────────────────────────────
        try:
            attr = build_risk_attribution_report(
                weights=best.weights, covariance_matrix=Sigma_shrink_df
            )
            result["risk_attribution"] = {
                "diversification_ratio": round(float(attr.get("diversification_ratio", 0)), 4),
                "effective_n":           round(float(attr.get("concentration", {}).get("effective_n", 0)), 2),
                "risk_contribution":     {
                    t: round(float(v), 4)
                    for t, v in attr.get("risk_contribution", {}).items()
                },
            }
        except Exception:
            pass

        # ── Rebalance actions ─────────────────────────────────────────────────
        if getattr(best, "rebalance_actions_rupees", None):
            for t, amt in best.rebalance_actions_rupees.items():
                curr_pct   = float(current_w.get(t, 0))
                target_pct = float(best.weights.get(t, 0))
                result["rebalance_actions"].append({
                    "ticker":      t,
                    "action":      "BUY" if amt > 0 else "SELL" if amt < 0 else "HOLD",
                    "current_pct": round(curr_pct, 4),
                    "target_pct":  round(target_pct, 4),
                    "rupees_delta":round(float(amt), 2),
                })

        # ── Efficient frontier ────────────────────────────────────────────────
        try:
            ef = compute_efficient_frontier(
                mu=mu_series.values, Sigma=Sigma_shrink, tickers=valid,
                constraint_builder=cb, n_points=20,
                returns_history=simple_returns, rf=rf,
            )
            ef_df = ef.to_dataframe()
            result["efficient_frontier"] = [
                {"return":       round(float(row["return"]), 4),
                 "volatility":   round(float(row["volatility"]), 4),
                 "sharpe_ratio": round(float(row["sharpe"]), 4)}
                for _, row in ef_df.iterrows()
            ]
        except Exception:
            pass

        # ── Robust optimisation ───────────────────────────────────────────────
        try:
            wc = optimize_worst_case(
                mu=mu_series.values, Sigma=Sigma_shrink, tickers=valid,
                constraint_builder=cb, uncertainty_level=0.10,
                optimization_method="min_variance",
                returns_history=simple_returns, rf=rf,
            )
            wc = enrich_allocation_result(wc, Sigma_shrink_df, mu_series, current_w, portfolio_value, rf=rf)
            result["robust"]["worst_case"] = {
                "weights":         {t: round(float(w), 4) for t, w in wc.weights.items()},
                "expected_return": round(float(wc.expected_return), 4),
                "volatility":      round(float(wc.volatility), 4),
                "sharpe_ratio":    round(float(wc.sharpe_ratio), 4),
            }
        except Exception:
            pass

        try:
            scenarios_sw = build_stress_scenarios_from_engine(
                base_mu=mu_series, base_Sigma=Sigma_shrink_df,
                base_returns=pd.DataFrame(simple_returns, columns=valid),
                scenario_definitions=[
                    {"label": "Base",            "probability": 0.50, "return_shock":  0.00, "volatility_shock": 1.0, "correlation_shock":  0.00},
                    {"label": "Moderate Stress", "probability": 0.25, "return_shock": -0.15, "volatility_shock": 1.5, "correlation_shock":  0.15},
                    {"label": "Severe Crash",    "probability": 0.15, "return_shock": -0.40, "volatility_shock": 2.5, "correlation_shock":  0.40},
                    {"label": "Recovery/Boom",   "probability": 0.10, "return_shock":  0.15, "volatility_shock": 0.8, "correlation_shock": -0.10},
                ],
            )
            sw = optimize_scenario_weighted(
                scenarios=scenarios_sw, tickers=valid,
                constraint_builder=cb, confidence_level=confidence_level, rf=rf,
            )
            sw = enrich_allocation_result(sw, Sigma_shrink_df, mu_series, current_w, portfolio_value, rf=rf)
            result["robust"]["scenario_weighted"] = {
                "weights":         {t: round(float(w), 4) for t, w in sw.weights.items()},
                "expected_return": round(float(sw.expected_return), 4),
                "volatility":      round(float(sw.volatility), 4),
                "sharpe_ratio":    round(float(sw.sharpe_ratio), 4),
            }
        except Exception:
            pass

        # ── Method comparison table ───────────────────────────────────────────
        all_r = {**opt_results, **{
            k: type("_R", (), v)() for k, v in [
                ("worst_case",        result["robust"].get("worst_case",        {})),
                ("scenario_weighted", result["robust"].get("scenario_weighted", {})),
            ] if v
        }}

        for mname, res in opt_results.items():
            result["method_comparison"].append({
                "method":          mname.replace("_", " ").title(),
                "expected_return": round(float(res.expected_return), 4),
                "volatility":      round(float(res.volatility), 4),
                "sharpe_ratio":    round(float(res.sharpe_ratio), 4),
                "cvar_95":         round(float(res.cvar_95), 4),
                "health_score":    int(getattr(res, "allocation_health_score", 0)),
            })

    except Exception as e:
        result["error"] = str(e)

    return result
