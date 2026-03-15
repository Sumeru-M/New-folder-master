"""
pipeline.py
===========
Milestone 6 End-to-End Pipeline Orchestrator.

This module wires together all four M6 sub-engines into a single
call that matches the specified output schema exactly:

    {
      "transaction_security": {
          "tx_id":              str,
          "tx_hash":            str,
          "signature_verified": bool,
          "public_key_fingerprint": str,
          "signed_at":          float
      },
      "portfolio_impact": {
          "expected_return_change": float,
          "volatility_change":      float,
          "sharpe_change":          float,
          "cvar_change":            float,
          "diversification_change": float,
          "hhi_change":             float,
          "effective_n_change":     float
      },
      "factor_shift": {
          "market_beta_delta":             float,
          "systematic_variance_delta":     float,
          "idiosyncratic_variance_delta":  float,
          "r_squared_delta":               float,
          "tracking_error_delta":          float
      },
      "monte_carlo_projection": {
          "n_paths":  int,
          "1Y": { real: ..., virtual: ..., deltas: ... },
          "3Y": { ... },
          "5Y": { ... },
          "best_horizon":    str,
          "overall_verdict": str
      },
      "portfolio_comparison": {
          "real":    { return, vol, sharpe, cvar, dr, hhi, effective_n },
          "virtual": { ... }
      },
      "risk_summary": str,
      "transaction_record": { full tx object }
    }

Public API
----------
    run_virtual_trade_simulation(
        ticker, quantity, price,
        real_holdings, real_prices, daily_returns,
        total_value, risk_free_rate,
        n_mc_paths, keypair
    ) -> dict
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from milestone6.crypto_layer import PrivateKey, PublicKey, generate_keypair
from milestone6.virtual_trade_engine import (
    RealPortfolioSnapshot,
    VirtualTrade,
    VirtualTradeEngine,
)
from milestone6.impact_analyzer import ImpactAnalyzer
from milestone6.projection_engine import ProjectionEngine


# ---------------------------------------------------------------------------
# Helper: build expected_returns and covariance from daily_returns
# ---------------------------------------------------------------------------

def _build_params(
    daily_returns: pd.DataFrame,
    tickers: List[str],
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute annualised arithmetic expected returns and Ledoit-Wolf-lite
    shrinkage covariance from a historical daily log-return DataFrame.

    Mirrors the approach in run_milestone5.py (prepare_inputs) without
    importing it.
    """
    cols = [t for t in tickers if t in daily_returns.columns]
    ret  = daily_returns[cols]

    # Arithmetic expected return with Itô / lognormal correction
    mu_log  = ret.mean()
    var_log = ret.var()
    mu_arith = np.exp(mu_log * 252 + 0.5 * var_log * 252) - 1
    expected_returns = pd.Series(mu_arith, index=cols)

    # Ledoit-Wolf shrinkage (analytical form)
    N = len(cols)
    T = len(ret)
    S = ret.cov() * 252
    mu_tr = float(np.trace(S.values)) / N
    F     = pd.DataFrame(np.eye(N) * mu_tr, index=cols, columns=cols)
    delta = 1.0 / (1.0 + T / N)
    cov   = (1 - delta) * S + delta * F
    cov   = pd.DataFrame((cov.values + cov.values.T) / 2, index=cols, columns=cols)

    # Ensure PSD
    eigv = np.linalg.eigvalsh(cov.values)
    if eigv.min() < 1e-8:
        cov = pd.DataFrame(
            cov.values + (abs(eigv.min()) + 1e-8) * np.eye(N),
            index=cols, columns=cols
        )

    return expected_returns, cov


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_virtual_trade_simulation(
    ticker:          str,
    quantity:        float,
    price:           float,
    real_holdings:   Dict[str, float],
    real_prices:     Dict[str, float],
    daily_returns:   pd.DataFrame,
    total_value:     float,
    risk_free_rate:  float                         = 0.07,
    n_mc_paths:      int                           = 1_000,
    keypair:         Optional[Tuple[PublicKey, PrivateKey]] = None,
    trade_timestamp: Optional[float]               = None,
    trade_note:      Optional[str]                 = None,
) -> Dict[str, Any]:
    """
    End-to-end virtual trade simulation pipeline.

    Parameters
    ----------
    ticker          : NSE ticker to virtually buy, e.g. "INFY.NS".
    quantity        : Number of shares to simulate purchasing.
    price           : Price per share at simulation time (INR).
    real_holdings   : {ticker: shares_held} — current real portfolio.
    real_prices     : {ticker: price} — current prices for all held tickers.
    daily_returns   : pd.DataFrame of historical daily log returns.
                      Columns must cover all real_holdings tickers.
                      If the traded ticker is new, it will be synthesised.
    total_value     : Total INR value of the real portfolio.
    risk_free_rate  : Annual risk-free rate (default 0.07 = 7%).
    n_mc_paths      : Monte Carlo paths per horizon (minimum 1,000).
    keypair         : (PublicKey, PrivateKey) — supply for persistent keys.
                      If None, a fresh ML-DSA key pair is auto-generated.
    trade_timestamp : Unix timestamp of the trade (defaults to now).
    trade_note      : Optional annotation attached to the transaction record.

    Returns
    -------
    dict matching the Milestone 6 output schema (fully JSON-serialisable).
    """
    ts = trade_timestamp or time.time()

    # ── 1. Validate inputs ────────────────────────────────────────────────
    if quantity <= 0:
        raise ValueError(f"quantity must be positive for a virtual BUY, got {quantity}.")
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}.")
    if total_value <= 0:
        raise ValueError("total_value must be positive.")
    if not real_holdings:
        raise ValueError("real_holdings cannot be empty.")

    # ── 2. Build trade instruction ────────────────────────────────────────
    trade = VirtualTrade(
        ticker    = ticker,
        quantity  = quantity,
        price     = price,
        timestamp = ts,
        note      = trade_note or f"Virtual buy: {quantity} × {ticker} @ ₹{price:,.2f}",
    )

    # ── 3. Build real portfolio params ────────────────────────────────────
    real_tickers = list(real_holdings.keys())
    real_mu, real_cov = _build_params(daily_returns, real_tickers)

    snapshot = RealPortfolioSnapshot(
        holdings       = real_holdings,
        prices         = real_prices,
        daily_returns  = daily_returns,
        total_value    = total_value,
        risk_free_rate = risk_free_rate,
        name           = "Real Portfolio",
    )

    # ── 4. Execute virtual trade (crypto-sealed) ──────────────────────────
    engine = VirtualTradeEngine(keypair=keypair, risk_free_rate=risk_free_rate)
    vp, tx_record = engine.execute(trade, snapshot)

    # ── 5. Impact analysis ────────────────────────────────────────────────
    analyzer = ImpactAnalyzer(confidence=0.95)
    real_weights = snapshot.weights
    impact_report = analyzer.analyze(
        vp                    = vp,
        real_weights          = real_weights,
        real_daily_returns    = daily_returns,
        real_expected_returns = real_mu,
        real_covariance       = real_cov,
        real_total_value      = total_value,
        risk_free_rate        = risk_free_rate,
    )

    # ── 6. Monte Carlo projections ────────────────────────────────────────
    projector = ProjectionEngine(n_paths=n_mc_paths)
    mc_report = projector.run(
        vp                    = vp,
        real_weights          = real_weights,
        real_expected_returns = real_mu,
        real_covariance       = real_cov,
        real_total_value      = total_value,
        risk_free_rate        = risk_free_rate,
    )

    # ── 7. Assemble output JSON ───────────────────────────────────────────
    ir  = impact_report
    mcr = mc_report

    output = {
        # ── Transaction security ─────────────────────────────────────────
        "transaction_security": {
            "tx_id":                   tx_record["tx_id"],
            "tx_hash":                 tx_record["sha3_hash"],
            "signature_verified":      tx_record["verification_status"],
            "public_key_fingerprint":  tx_record["public_key"]["fingerprint"],
            "signed_at":               tx_record["signed_at"],
            "signature_scheme":        tx_record["signature"]["scheme"],
            "z_infinity_norm":         tx_record["signature"]["z_infinity_norm"],
            "acceptance_bound":        tx_record["signature"]["acceptance_bound"],
        },

        # ── Portfolio impact deltas ───────────────────────────────────────
        "portfolio_impact": {
            "expected_return_change": round(ir.portfolio_impact["expected_return_change"], 6),
            "volatility_change":      round(ir.portfolio_impact["volatility_change"],      6),
            "sharpe_change":          round(ir.portfolio_impact["sharpe_change"],          4),
            "cvar_change":            round(ir.portfolio_impact["cvar_change"],            6),
            "diversification_change": round(ir.portfolio_impact["diversification_change"], 4),
            "hhi_change":             round(ir.portfolio_impact["hhi_change"],             4),
            "effective_n_change":     round(ir.portfolio_impact["effective_n_change"],     2),
        },

        # ── Factor shift ──────────────────────────────────────────────────
        "factor_shift": {
            "market_beta_delta":            ir.factor_shift.get("market_beta",            0),
            "systematic_variance_delta":    ir.factor_shift.get("systematic_variance",    0),
            "idiosyncratic_variance_delta": ir.factor_shift.get("idiosyncratic_variance", 0),
            "r_squared_delta":              ir.factor_shift.get("r_squared",              0),
            "tracking_error_delta":         ir.factor_shift.get("tracking_error_vs_ew",   0),
        },

        # ── Monte Carlo projections ───────────────────────────────────────
        "monte_carlo_projection": {
            "n_paths":         mcr.n_paths,
            "trade_ticker":    mcr.trade_ticker,
            "best_horizon":    f"{mcr.best_horizon}Y",
            "overall_verdict": mcr.overall_verdict,
            **{
                f"{h.horizon_years}Y": {
                    "real":    h.real.to_dict(),
                    "virtual": h.virtual.to_dict(),
                    "deltas": {
                        "expected_value_delta": round(h.expected_value_delta, 2),
                        "downside_prob_delta":  round(h.downside_prob_delta,  4),
                        "p5_delta":             round(h.p5_delta,             2),
                        "median_cagr_delta":    round(h.median_cagr_delta,    4),
                    },
                }
                for h in mcr.horizons
            },
        },

        # ── Portfolio comparison (before / after) ─────────────────────────
        "portfolio_comparison": {
            "real":    ir.real_metrics.to_dict(),
            "virtual": ir.virtual_metrics.to_dict(),
        },

        # ── Human-readable risk summary ───────────────────────────────────
        "risk_summary": ir.risk_summary,

        # ── Full transaction record (for audit / P2P broadcast) ───────────
        "transaction_record": tx_record,
    }

    return output
