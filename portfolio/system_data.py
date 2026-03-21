"""
system_data.py — System Aggregator
====================================
Combines outputs from all milestone APIs into one unified response.
Each milestone can also be called independently.

Usage:
    # Full system view
    from portfolio.system_data import get_system_data

    data = get_system_data(
        tickers         = ["RELIANCE.NS", "TCS.NS", "INFY.NS"],
        risk_free_rate  = 0.07,
        portfolio_value = 1_000_000,
        current_weights = None,    # optional
        trade_ticker    = None,    # set to run M6, e.g. "INFY.NS"
        trade_quantity  = 10,      # shares to buy in virtual trade
        trade_price     = None,    # will fetch live price if None
        scenario        = None,    # optional M8 scenario override
    )

    # Individual milestone calls
    from portfolio.system_data import (
        get_portfolio_data,    # M3
        get_risk_data,         # M4
        get_optimisation_data, # M5
        get_security_data,     # M6
        get_regime_data,       # M7
        get_decision_data,     # M8
    )
"""

import os
import sys

# Ensure portfolio package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from portfolio.api_m3 import get_portfolio_construction
from portfolio.api_m4 import get_scenario_analysis
from portfolio.api_m5 import get_institutional_optimisation
from portfolio.api_m6 import get_virtual_trade_and_security
from portfolio.api_m7 import get_market_regime
from portfolio.api_m8 import get_decision


# ── Thin public aliases so the frontend imports use simple names ──────────────

def get_portfolio_data(tickers, current_weights=None, period="2y", risk_free_rate=0.07):
    """
    Milestone 3 — Portfolio construction and optimisation.
    Returns efficient frontier, weights, risk metrics, and factor analysis.
    """
    return get_portfolio_construction(
        tickers         = tickers,
        current_weights = current_weights,
        period          = period,
        risk_free_rate  = risk_free_rate,
    )


def get_risk_data(tickers, portfolio_value=1_000_000, risk_free_rate=0.07,
                  scenarios="ALL"):
    """
    Milestone 4 — Scenario analysis and downside risk metrics.
    Returns VaR, CVaR, drawdown, and scenario P&L under market shocks.
    """
    return get_scenario_analysis(
        tickers         = tickers,
        portfolio_value = portfolio_value,
        risk_free_rate  = risk_free_rate,
        scenarios       = scenarios,
    )


def get_optimisation_data(tickers, current_weights=None, portfolio_value=1_000_000,
                          risk_free_rate=0.07, max_weight=0.40, methods="all"):
    """
    Milestone 5 — Institutional multi-method optimisation.
    Returns six optimisation methods, the best one, rebalance plan,
    risk attribution, and the efficient frontier.
    """
    return get_institutional_optimisation(
        tickers         = tickers,
        current_weights = current_weights,
        portfolio_value = portfolio_value,
        risk_free_rate  = risk_free_rate,
        max_weight      = max_weight,
        methods         = methods,
    )


def get_security_data(ticker, quantity, price, holdings, current_prices,
                      total_value, risk_free_rate=0.07):
    """
    Milestone 6 — Secure virtual trade simulation.
    Returns a post-quantum signed transaction, portfolio impact,
    Monte Carlo projections, and dual security threat assessment.
    """
    return get_virtual_trade_and_security(
        ticker         = ticker,
        quantity       = quantity,
        price          = price,
        holdings       = holdings,
        current_prices = current_prices,
        total_value    = total_value,
        risk_free_rate = risk_free_rate,
    )


def get_regime_data(tickers, risk_free_rate=0.07, horizons=None):
    """
    Milestone 7 — Market regime intelligence.
    Returns HMM regime detection, GARCH volatility forecast,
    forward return distributions, and uncertainty quantification.
    """
    return get_market_regime(
        tickers        = tickers,
        risk_free_rate = risk_free_rate,
        horizons       = horizons or [21, 63],
    )


def get_decision_data(m7_output, m3_output=None, m6_output=None, scenario=None):
    """
    Milestone 8 — Autonomous decision intelligence.
    Fuses regime, risk, and security signals into one recommendation
    with a full plain-English explanation.
    """
    return get_decision(
        m7_output = m7_output,
        m3_output = m3_output,
        m6_output = m6_output,
        scenario  = scenario,
    )


# ── Full system aggregator ────────────────────────────────────────────────────

def get_system_data(
    tickers: list,
    risk_free_rate: float = 0.07,
    portfolio_value: float = 1_000_000,
    current_weights: list = None,
    trade_ticker: str = None,
    trade_quantity: float = 10,
    trade_price: float = None,
    scenario: str = None,
) -> dict:
    """
    Runs all milestones in sequence and returns the combined output.

    Each milestone's result is stored under its own key. If a milestone
    fails, its key will contain {"error": "..."} and the remaining
    milestones continue running.

    Parameters
    ----------
    tickers         : list of NSE ticker strings
    risk_free_rate  : annual risk-free rate as decimal (0.07 = 7%)
    portfolio_value : total portfolio value in INR
    current_weights : optional list of current allocation weights
    trade_ticker    : ticker to run M6 virtual trade on (first ticker if None)
    trade_quantity  : number of shares for the virtual trade
    trade_price     : price per share (fetches live price if None)
    scenario        : optional M8 scenario — "market_crash" | "high_inflation" |
                      "liquidity_crisis" | "rate_shock" | "commodity_boom" |
                      "geopolitical_risk"

    Returns
    -------
    dict with keys:
        portfolio        : M3 output  (construction and optimisation)
        risk             : M4 output  (scenario analysis)
        optimisation     : M5 output  (institutional methods)
        security         : M6 output  (virtual trade + threat check)
        regime           : M7 output  (market intelligence)
        decision         : M8 output  (recommendation)
        meta             : {tickers, portfolio_value, risk_free_rate, scenario}
    """

    result = {
        "portfolio":    None,
        "risk":         None,
        "optimisation": None,
        "security":     None,
        "regime":       None,
        "decision":     None,
        "meta": {
            "tickers":         tickers,
            "portfolio_value": portfolio_value,
            "risk_free_rate":  risk_free_rate,
            "scenario":        scenario,
        },
    }

    # ── M3: portfolio construction ────────────────────────────────────────────
    result["portfolio"] = get_portfolio_data(
        tickers         = tickers,
        current_weights = current_weights,
        risk_free_rate  = risk_free_rate,
    )

    # ── M4: scenario risk analysis ────────────────────────────────────────────
    result["risk"] = get_risk_data(
        tickers         = tickers,
        portfolio_value = portfolio_value,
        risk_free_rate  = risk_free_rate,
    )

    # ── M5: institutional optimisation ────────────────────────────────────────
    result["optimisation"] = get_optimisation_data(
        tickers         = tickers,
        current_weights = current_weights,
        portfolio_value = portfolio_value,
        risk_free_rate  = risk_free_rate,
    )

    # ── M6: virtual trade + security ─────────────────────────────────────────
    # Determine trade ticker and price
    active_ticker = trade_ticker or tickers[0]

    if trade_price is None:
        # Try to get the current price from M3's data
        try:
            m3_weights = (result["portfolio"] or {}).get("max_sharpe_portfolio", {}).get("weights", {})
            # Use first ticker with a weight as the trade target
            if m3_weights:
                active_ticker = max(m3_weights, key=m3_weights.get)
        except Exception:
            pass
        trade_price_to_use = 1000.0  # fallback; M6 fetches live price internally
    else:
        trade_price_to_use = trade_price

    # Build holdings from current_weights or equal weight
    N = len(tickers)
    if current_weights and len(current_weights) == N:
        w = [cw / sum(current_weights) for cw in current_weights]
    else:
        w = [1.0 / N] * N

    holdings       = {t: round(wi * portfolio_value / max(trade_price_to_use, 1), 0)
                      for t, wi in zip(tickers, w)}
    current_prices = {t: trade_price_to_use for t in tickers}

    result["security"] = get_security_data(
        ticker         = active_ticker,
        quantity       = trade_quantity,
        price          = trade_price_to_use,
        holdings       = holdings,
        current_prices = current_prices,
        total_value    = portfolio_value,
        risk_free_rate = risk_free_rate,
    )

    # ── M7: market regime ─────────────────────────────────────────────────────
    result["regime"] = get_regime_data(
        tickers        = tickers,
        risk_free_rate = risk_free_rate,
    )

    # ── M8: decision ──────────────────────────────────────────────────────────
    result["decision"] = get_decision_data(
        m7_output = result["regime"],
        m3_output = result["portfolio"],
        m6_output = result["security"],
        scenario  = scenario,
    )

    return result
