"""
api_m3.py — Milestone 3: Portfolio Construction API
=====================================================
Wraps the existing M3 logic into one clean function.
No print statements. Returns a single structured dictionary.

Usage:
    from portfolio.api_m3 import get_portfolio_construction

    result = get_portfolio_construction(
        tickers         = ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        current_weights = [0.33, 0.33, 0.34],   # optional
        period          = "2y",
        risk_free_rate  = 0.07,
    )
"""

import sys
import os
import numpy as np
import pandas as pd

# Make sure portfolio package is importable when called from examples/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def get_portfolio_construction(
    tickers: list,
    current_weights: list = None,
    period: str = "2y",
    risk_free_rate: float = 0.07,
) -> dict:
    """
    Runs the full M3 portfolio construction pipeline on live data.

    Parameters
    ----------
    tickers         : list of NSE ticker strings, e.g. ["RELIANCE.NS", "TCS.NS"]
    current_weights : optional list of floats (fractions summing to 1).
                      If None, equal weights are used.
    period          : yfinance period string — "1y", "2y", "5y", etc.
    risk_free_rate  : annual risk-free rate as a decimal (0.07 = 7%)

    Returns
    -------
    dict with keys:
        tickers                 : list of tickers actually used (some may be dropped)
        period                  : data period used
        risk_free_rate          : rate used

        current_portfolio       : {expected_return, volatility, sharpe_ratio}
        min_variance_portfolio  : {weights, expected_return, volatility, sharpe_ratio}
        max_sharpe_portfolio    : {weights, expected_return, volatility, sharpe_ratio}

        robust_comparison       : {min_variance, max_sharpe}  — standard vs shrinkage
        efficient_frontier      : list of {return, volatility, sharpe_ratio}

        risk_metrics            : {var_95_pct, cvar_95_pct, max_drawdown,
                                   recovery_days, ulcer_index}
        factor_analysis         : {systematic_volatility, idiosyncratic_volatility,
                                   r_squared}  or None if Nifty data unavailable
        market_regime           : {regime, current_vol_annualized}  or None
        correlation_matrix      : {ticker: {ticker: float}}

        error                   : None or error message string
    """

    # ── Imports from existing M3 modules ─────────────────────────────────────
    from portfolio.portfolio_complete  import load_price_data
    from portfolio.portfolio_complete    import (
        PortfolioOptimizer,
        compute_daily_returns,
        compute_expected_returns,
        compute_covariance_matrix,
        compare_covariance_methods,
        compute_weight_dispersion,
    )

    result = {
        "tickers": tickers,
        "period": period,
        "risk_free_rate": risk_free_rate,
        "current_portfolio": None,
        "min_variance_portfolio": None,
        "max_sharpe_portfolio": None,
        "robust_comparison": None,
        "efficient_frontier": [],
        "risk_metrics": None,
        "factor_analysis": None,
        "market_regime": None,
        "correlation_matrix": None,
        "error": None,
    }

    try:
        # ── Load prices ───────────────────────────────────────────────────────
        prices = load_price_data(tickers, period=period)
        if prices is None or prices.empty:
            result["error"] = "Could not load price data. Check tickers and connection."
            return result

        # Drop tickers with too many NaN values
        valid_tickers = [
            t for t in tickers
            if t in prices.columns and prices[t].isna().mean() < 0.30
        ]
        if len(valid_tickers) < 2:
            result["error"] = (
                f"Only {len(valid_tickers)} valid ticker(s) found. "
                "Need at least 2."
            )
            return result

        result["tickers"] = valid_tickers
        prices       = prices[valid_tickers].dropna()
        daily_returns = compute_daily_returns(prices)
        expected_returns = compute_expected_returns(daily_returns, annualized=True)
        covariance_matrix = compute_covariance_matrix(daily_returns, annualized=True)

        # ── Parse current weights ─────────────────────────────────────────────
        N = len(valid_tickers)
        if current_weights and len(current_weights) == N:
            w = np.array(current_weights, dtype=float)
            w = w / w.sum()
        else:
            w = np.ones(N) / N
        w_series = pd.Series(w, index=valid_tickers)

        # ── Current portfolio stats ───────────────────────────────────────────
        curr_ret = float(np.dot(w, expected_returns.values))
        curr_vol = float(np.sqrt(w @ covariance_matrix.values @ w))
        curr_sharpe = (curr_ret - risk_free_rate) / curr_vol if curr_vol > 0 else 0.0

        result["current_portfolio"] = {
            "weights":         {t: round(float(wv), 4) for t, wv in w_series.items()},
            "expected_return": round(curr_ret, 4),
            "volatility":      round(curr_vol, 4),
            "sharpe_ratio":    round(curr_sharpe, 4),
        }

        # ── Optimised portfolios ──────────────────────────────────────────────
        optimizer = PortfolioOptimizer(expected_returns, covariance_matrix, risk_free_rate)

        min_var = optimizer.optimize_min_variance()
        result["min_variance_portfolio"] = {
            "weights":         {t: round(float(wv), 4) for t, wv in min_var.weights.items()},
            "expected_return": round(float(min_var.expected_return), 4),
            "volatility":      round(float(min_var.volatility), 4),
            "sharpe_ratio":    round(float(min_var.sharpe_ratio), 4),
        }

        max_sharpe = optimizer.optimize_max_sharpe()
        result["max_sharpe_portfolio"] = {
            "weights":         {t: round(float(wv), 4) for t, wv in max_sharpe.weights.items()},
            "expected_return": round(float(max_sharpe.expected_return), 4),
            "volatility":      round(float(max_sharpe.volatility), 4),
            "sharpe_ratio":    round(float(max_sharpe.sharpe_ratio), 4),
        }

        # ── Robust comparison (standard vs Ledoit-Wolf shrinkage) ─────────────
        try:
            comp_mv  = compare_covariance_methods(daily_returns, expected_returns, risk_free_rate, "min_variance")
            comp_ms  = compare_covariance_methods(daily_returns, expected_returns, risk_free_rate, "max_sharpe")
            result["robust_comparison"] = {
                "min_variance": {
                    "standard": {
                        "weights": {t: round(float(wv), 4) for t, wv in comp_mv.sample_result.weights.items()},
                        "expected_return": round(float(comp_mv.sample_result.expected_return), 4),
                        "volatility":      round(float(comp_mv.sample_result.volatility), 4),
                        "sharpe_ratio":    round(float(comp_mv.sample_result.sharpe_ratio), 4),
                    },
                    "robust": {
                        "weights": {t: round(float(wv), 4) for t, wv in comp_mv.shrinkage_result.weights.items()},
                        "expected_return": round(float(comp_mv.shrinkage_result.expected_return), 4),
                        "volatility":      round(float(comp_mv.shrinkage_result.volatility), 4),
                        "sharpe_ratio":    round(float(comp_mv.shrinkage_result.sharpe_ratio), 4),
                    },
                },
                "max_sharpe": {
                    "standard": {
                        "weights": {t: round(float(wv), 4) for t, wv in comp_ms.sample_result.weights.items()},
                        "expected_return": round(float(comp_ms.sample_result.expected_return), 4),
                        "volatility":      round(float(comp_ms.sample_result.volatility), 4),
                        "sharpe_ratio":    round(float(comp_ms.sample_result.sharpe_ratio), 4),
                    },
                    "robust": {
                        "weights": {t: round(float(wv), 4) for t, wv in comp_ms.shrinkage_result.weights.items()},
                        "expected_return": round(float(comp_ms.shrinkage_result.expected_return), 4),
                        "volatility":      round(float(comp_ms.shrinkage_result.volatility), 4),
                        "sharpe_ratio":    round(float(comp_ms.shrinkage_result.sharpe_ratio), 4),
                    },
                },
            }
        except Exception:
            pass  # robust comparison is supplementary

        # ── Efficient frontier ────────────────────────────────────────────────
        try:
            ef_r, ef_v, ef_s = optimizer.compute_efficient_frontier(n_points=50)
            result["efficient_frontier"] = [
                {"return": round(float(r), 4),
                 "volatility": round(float(v), 4),
                 "sharpe_ratio": round(float(s), 4)}
                for r, v, s in zip(ef_r, ef_v, ef_s)
            ]
        except Exception:
            pass

        # ── Risk metrics (VaR, CVaR, drawdown) on max-Sharpe portfolio ────────
        try:
            from portfolio.portfolio_complete import (
                compute_portfolio_risk_metrics,
                compute_max_drawdown,
                compute_ulcer_index,
            )
            port_daily = (daily_returns * max_sharpe.weights).sum(axis=1)
            risk_stats = compute_portfolio_risk_metrics(
                max_sharpe.weights, port_daily, expected_returns, covariance_matrix
            )
            dd_stats   = compute_max_drawdown(port_daily)
            ulcer      = compute_ulcer_index(port_daily)

            result["risk_metrics"] = {
                "var_95_pct":      round(float(risk_stats["parametric_var"]["var_percent"]), 4),
                "var_95_hist_pct": round(float(risk_stats["historical_var"]["var_percent"]), 4),
                "cvar_95_pct":     round(float(risk_stats["cvar"]["cvar_percent"]), 4),
                "max_drawdown":    round(float(dd_stats["max_drawdown"]), 4),
                "recovery_days":   int(dd_stats.get("max_drawdown_duration_days", 0)),
                "ulcer_index":     round(float(ulcer), 6),
            }
        except Exception:
            pass

        # ── Factor analysis vs Nifty ──────────────────────────────────────────
        try:
            from portfolio.portfolio_complete import FactorModel
            from portfolio.portfolio_complete import detect_market_regime

            idx_data   = load_price_data(["^NSEI"], period=period)
            idx_ret    = compute_daily_returns(idx_data)
            common     = daily_returns.index.intersection(idx_ret.index)
            fm         = FactorModel(idx_ret.loc[common])
            decomp     = fm.decompose_portfolio_risk(max_sharpe.weights, daily_returns.loc[common])

            result["factor_analysis"] = {
                "systematic_volatility":    round(float(decomp["systematic_volatility"]), 4),
                "idiosyncratic_volatility": round(float(decomp["idiosyncratic_volatility"]), 4),
                "r_squared":                round(float(decomp["r_squared"]), 4),
            }

            reg = detect_market_regime(idx_ret.iloc[:, 0])
            result["market_regime"] = {
                "regime":                reg.get("regime", "UNKNOWN"),
                "current_vol_annualized": round(float(reg.get("current_vol_annualized", 0)), 4),
            }
        except Exception:
            pass  # factor analysis is supplementary

        # ── Correlation matrix ────────────────────────────────────────────────
        corr = daily_returns.corr()
        result["correlation_matrix"] = {
            t: {t2: round(float(corr.loc[t, t2]), 4) for t2 in valid_tickers}
            for t in valid_tickers
        }

    except Exception as e:
        result["error"] = str(e)

    return result
