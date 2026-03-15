"""
milestone6 — Encrypted P2P Virtual Trade Simulation
====================================================

Modules
-------
crypto_layer        : ML-DSA-style cryptographic security layer.
virtual_trade_engine: Immutable virtual portfolio construction.
impact_analyzer     : Portfolio impact analytics (deltas vs real).
projection_engine   : Monte Carlo future projection engine.

Quick-start
-----------
    from milestone6 import run_virtual_trade_simulation

    result = run_virtual_trade_simulation(
        ticker          = "INFY.NS",
        quantity        = 100,
        price           = 1_800.0,
        real_holdings   = {"TCS.NS": 50, "HDFCBANK.NS": 30},
        real_prices     = {"TCS.NS": 3_500.0, "HDFCBANK.NS": 1_600.0},
        daily_returns   = returns_df,        # pd.DataFrame of log returns
        total_value     = 2_55_000.0,
        risk_free_rate  = 0.07,
    )
    # result is a JSON-serialisable dict matching the M6 output schema.

See milestone6/run_milestone6.py for a full runnable example with
synthetic NSE data.
"""

from milestone6.crypto_layer import (
    PublicKey,
    PrivateKey,
    TransactionSignature,
    generate_keypair,
    sign_transaction,
    verify_transaction,
    hash_payload,
    build_transaction_object,
)

from milestone6.virtual_trade_engine import (
    VirtualTrade,
    RealPortfolioSnapshot,
    VirtualPortfolio,
    VirtualTradeEngine,
)

from milestone6.impact_analyzer import (
    ImpactAnalyzer,
    ImpactReport,
    PortfolioMetrics,
)

from milestone6.projection_engine import (
    ProjectionEngine,
    MonteCarloReport,
    ProjectionResult,
    HorizonComparison,
)

from milestone6.pipeline import run_virtual_trade_simulation

__all__ = [
    # crypto
    "PublicKey", "PrivateKey", "TransactionSignature",
    "generate_keypair", "sign_transaction", "verify_transaction",
    "hash_payload", "build_transaction_object",
    # trade engine
    "VirtualTrade", "RealPortfolioSnapshot", "VirtualPortfolio",
    "VirtualTradeEngine",
    # analytics
    "ImpactAnalyzer", "ImpactReport", "PortfolioMetrics",
    # projection
    "ProjectionEngine", "MonteCarloReport", "ProjectionResult",
    "HorizonComparison",
    # pipeline
    "run_virtual_trade_simulation",
]
