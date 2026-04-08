"""
api_m7.py — Milestone 7: Market Regime Intelligence API
========================================================
Wraps the existing M7 logic into one clean function.
No print statements. Returns a single structured dictionary.

Usage:
    from portfolio.api_m7 import get_market_regime

    result = get_market_regime(
        tickers      = ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        risk_free_rate = 0.07,
        horizons     = [21, 63],
        risk_appetite= "balanced",
    )
"""

import sys
import os
import types
import importlib.util
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _load_m7():
    """Load milestone7_complete.py from the portfolio folder."""
    portfolio_dir = os.path.dirname(__file__)
    project_dir   = os.path.join(portfolio_dir, "..")
    for folder in [portfolio_dir, project_dir]:
        path = os.path.normpath(os.path.join(folder, "milestone7_complete.py"))
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("milestone7_complete", path)
            mod  = types.ModuleType("milestone7_complete")
            mod.__spec__ = spec
            sys.modules["milestone7_complete"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "milestone7_complete.py not found in portfolio/ or project root."
    )


def get_market_regime(
    tickers: list,
    risk_free_rate: float = 0.07,
    horizons: list = None,
    risk_appetite: str = "balanced",
    hmm_restarts: int = 3,
    hmm_max_iter: int = 150,
    garch_n_sim: int = 300,
    uncertainty_n_boot: int = 200,
) -> dict:
    """
    Runs the full M7 regime intelligence pipeline on live data.

    Parameters
    ----------
    tickers             : list of NSE ticker strings
    risk_free_rate      : annual risk-free rate as decimal
    horizons            : two forecast horizons in trading days, e.g. [21, 63]
    risk_appetite       : "conservative" | "balanced" | "aggressive"
    hmm_restarts        : number of HMM random restarts (more = more reliable)
    hmm_max_iter        : maximum EM iterations per restart
    garch_n_sim         : bootstrap samples for GARCH CI
    uncertainty_n_boot  : bootstrap samples for regime probability CI

    Returns
    -------
    dict with keys:
        tickers                : tickers used
        risk_free_rate         : rate used
        elapsed_seconds        : pipeline runtime

        current_regime         : "Low-Vol Bull" | "High-Vol Bear" |
                                 "Crisis" | "Transitional"
        regime_probabilities   : {regime_name: probability}
        log_likelihood         : HMM model fit score

        expected_regime_duration_days : {regime_name: days}
        transition_matrix      : {from_regime: {to_regime: probability}}
        forward_regime_probs   : {
            "1d":  {regime_name: probability},
            "5d":  {...}, "21d": {...}, "63d": {...}
        }
        crisis_probability_by_horizon : {"1d": float, "5d": float, ...}
        mixing_time_days       : int

        garch_current_vol      : annualised current volatility
        garch_30d_forecast     : annualised 30-day vol forecast
        garch_forecast_path    : {day: {vol_ann, ci_lower_95, ci_upper_95}}

        forward_distributions  : {
            "21d": {
                expected_return_ann, annualised_vol, skewness,
                excess_kurtosis, var_95_pct, cvar_95_pct,
                prob_loss_over_10pct
            },
            "63d": {...}
        }

        adaptive_parameters    : {
            optimization_method, lam_return, lam_vol, lam_cvar,
            lam_drawdown, max_weight, target_vol_ann,
            cvar_confidence, position_scale,
            regime_entropy, shrinkage
        }
        parameter_notes        : list of plain-English notes

        optimal_weights        : {ticker: weight}
        portfolio_stats        : {expected_return, volatility, sharpe_ratio,
                                   cvar_95, optimization_type}

        uncertainty            : {
            effective_sample_size,
            regime_ci: {regime: {mean, lower_95, upper_95, half_width}},
            forecast_return_ci: {horizon: {mean_ann, lower_95, upper_95}},
            weight_sensitivity: {ticker: regime_sensitivity_index}
        }

        stationary_distribution: {regime_name: probability}

        error                  : None or error string
    """

    horizons = horizons or [21, 63]

    result = {
        "tickers":                      tickers,
        "risk_free_rate":               risk_free_rate,
        "elapsed_seconds":              None,
        "current_regime":               None,
        "regime_probabilities":         {},
        "log_likelihood":               None,
        "expected_regime_duration_days":{},
        "transition_matrix":            {},
        "forward_regime_probs":         {},
        "crisis_probability_by_horizon":{},
        "mixing_time_days":             None,
        "garch_current_vol":            None,
        "garch_30d_forecast":           None,
        "garch_forecast_path":          {},
        "forward_distributions":        {},
        "adaptive_parameters":          None,
        "parameter_notes":              [],
        "optimal_weights":              {},
        "portfolio_stats":              None,
        "uncertainty":                  None,
        "stationary_distribution":      {},
        "error":                        None,
    }

    try:
        m7 = _load_m7()

        from portfolio.portfolio_complete import load_price_data, normalize_tickers_for_market_data
        from portfolio.portfolio_complete   import compute_daily_returns

        tickers = normalize_tickers_for_market_data(tickers)

        # ── Load live data ────────────────────────────────────────────────────
        # load_price_data/get_stock_data does not support period="3y",
        # so request an explicit date range for a true ~3-year window.
        # If network/date-range fetch fails, fall back to period-based cached data.
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=365 * 3)
        try:
            prices = load_price_data(
                tickers,
                start_date=start_date.isoformat(),
                end_date=end_date.isoformat(),
            )
        except Exception:
            prices = load_price_data(tickers, period="2y")
            if prices is None or prices.empty:
                prices = load_price_data(tickers, period="5y")
        if prices is None or prices.empty:
            result["error"] = "Could not load price data."
            return result

        valid     = [t for t in tickers if t in prices.columns]
        prices    = prices[valid].dropna()
        daily_ret = compute_daily_returns(prices)

        if len(daily_ret) < 100:
            result["error"] = "Insufficient historical data (need at least 100 days)."
            return result

        result["tickers"] = valid

        # ── Run M7 pipeline ───────────────────────────────────────────────────
        report = m7.run_adaptive_intelligence(
            prices_or_returns  = daily_ret,
            tickers            = valid,
            is_returns         = True,
            horizons           = horizons,
            rf_base            = risk_free_rate,
            risk_appetite      = risk_appetite,
            hmm_restarts       = hmm_restarts,
            hmm_max_iter       = hmm_max_iter,
            garch_n_sim        = garch_n_sim,
            uncertainty_n_boot = uncertainty_n_boot,
            quiet              = True,
        )

        d = report.to_dict()
        result["elapsed_seconds"] = round(float(report.elapsed_seconds), 2)

        # ── Regime probabilities ──────────────────────────────────────────────
        rp = d["regime_probabilities"]
        result["current_regime"]       = rp.get("current_regime_label")
        result["regime_probabilities"] = {
            k: round(float(v), 4) for k, v in rp.get("current_probs", {}).items()
        }
        result["log_likelihood"]       = round(float(rp.get("log_likelihood", 0)), 2)

        # ── Transition matrix ─────────────────────────────────────────────────
        tm = d["transition_matrix"]
        result["transition_matrix"] = {
            row: {col: round(float(v), 6) for col, v in cols.items()}
            for row, cols in tm.get("transition_matrix", {}).items()
        }
        result["expected_regime_duration_days"] = {
            k: round(float(v), 1)
            for k, v in tm.get("expected_duration_days", {}).items()
            if v < 1e7
        }
        result["forward_regime_probs"] = {
            h: {r: round(float(p), 4) for r, p in probs.items()}
            for h, probs in tm.get("forward_probabilities", {}).items()
        }
        result["crisis_probability_by_horizon"] = {
            h: round(float(p), 4)
            for h, p in tm.get("crisis_probability_by_horizon", {}).items()
        }
        result["mixing_time_days"]       = int(tm.get("mixing_time_days", 0))
        result["stationary_distribution"]= {
            k: round(float(v), 4)
            for k, v in tm.get("stationary_distribution", {}).items()
        }

        # ── GARCH volatility forecast ─────────────────────────────────────────
        vf = d["volatility_forecast"]
        result["garch_current_vol"]  = round(float(vf.get("current_vol_ann",  0)), 4)
        result["garch_30d_forecast"] = round(float(vf.get("forecast_30d_vol", 0)), 4)
        result["garch_forecast_path"]= {
            h: {
                "vol_ann":      round(float(vals.get("vol_ann",      0)), 4),
                "ci_lower_95":  round(float(vals.get("ci_lower_95",  0)), 4),
                "ci_upper_95":  round(float(vals.get("ci_upper_95",  0)), 4),
            }
            for h, vals in vf.get("forecast_path", {}).items()
        }

        # ── Forward return distributions ──────────────────────────────────────
        frf = d["forward_return_forecast"].get("forward_distributions", {})
        result["forward_distributions"] = {
            h_str: {
                "expected_return_ann":   round(float(fd.get("expected_return_ann",    0)), 4),
                "annualised_vol":         round(float(fd.get("annualised_vol",         0)), 4),
                "skewness":               round(float(fd.get("skewness",               0)), 4),
                "excess_kurtosis":        round(float(fd.get("excess_kurtosis",        0)), 4),
                "var_95_pct":             round(float(fd.get("var_95_percent",         0)), 4),
                "cvar_95_pct":            round(float(fd.get("cvar_95_percent",        0)), 4),
                "prob_loss_over_10pct":   round(float(fd.get("tail_prob_minus10pct",   0)), 4),
            }
            for h_str, fd in frf.items()
        }

        # ── Adaptive parameters ───────────────────────────────────────────────
        ap     = d["adaptive_parameter_shift"]
        params = ap.get("adapted_parameters", {})
        ue     = ap.get("uncertainty_metrics", {})

        result["adaptive_parameters"] = {
            "optimization_method": ap.get("optimization_method", ""),
            "lam_return":          round(float(params.get("lam_return",          0)), 4),
            "lam_vol":             round(float(params.get("lam_vol",             0)), 4),
            "lam_cvar":            round(float(params.get("lam_cvar",            0)), 4),
            "lam_drawdown":        round(float(params.get("lam_drawdown",        0)), 4),
            "max_weight":          round(float(params.get("max_weight",          0)), 4),
            "target_vol_ann":      round(float(params.get("target_vol_ann",      0)), 4),
            "cvar_confidence":     round(float(params.get("cvar_confidence",     0)), 4),
            "position_scale":      round(float(params.get("position_scale",      0)), 4),
            "regime_entropy":      round(float(ue.get("regime_entropy",          0)), 4),
            "shrinkage":           round(float(ue.get("shrinkage",               0)), 4),
        }
        result["parameter_notes"] = list(ap.get("parameter_shift_notes", []))

        # ── Optimal allocation ────────────────────────────────────────────────
        oa = d["optimal_allocation"]
        result["optimal_weights"] = {
            t: round(float(w), 4) for t, w in oa.get("weights", {}).items()
        }
        result["portfolio_stats"] = {
            "expected_return":  round(float(oa.get("expected_return_ann", 0)), 4),
            "volatility":       round(float(oa.get("volatility_ann",      0)), 4),
            "sharpe_ratio":     round(float(oa.get("sharpe_ratio",        0)), 4),
            "cvar_95":          round(float(oa.get("cvar_95",             0)), 4),
            "optimization_type":str(oa.get("optimization_type", "")),
        }

        # ── Uncertainty metrics ───────────────────────────────────────────────
        um = d["uncertainty_metrics"]
        result["uncertainty"] = {
            "effective_sample_size": int(um.get("effective_sample_size", 0)),
            "regime_ci": {
                regime: {
                    "mean":       round(float(ci.get("mean",       0)), 4),
                    "lower_95":   round(float(ci.get("lower_95",   0)), 4),
                    "upper_95":   round(float(ci.get("upper_95",   0)), 4),
                    "half_width": round(float(ci.get("half_width", 0)), 4),
                }
                for regime, ci in um.get("regime_probability_ci_95", {}).items()
            },
            "forecast_return_ci": {
                h: {
                    "mean_ann":  round(float(fc.get("mean_ann",  0)), 4),
                    "lower_95":  round(float(fc.get("lower_95",  0)), 4),
                    "upper_95":  round(float(fc.get("upper_95",  0)), 4),
                }
                for h, fc in um.get("forecast_return_ci_95", {}).items()
            },
            "weight_sensitivity": {
                t: round(float(v.get("regime_sensitivity_index", 0)), 6)
                for t, v in um.get("weight_regime_sensitivity", {}).items()
            },
        }

    except Exception as e:
        result["error"] = str(e)

    return result
