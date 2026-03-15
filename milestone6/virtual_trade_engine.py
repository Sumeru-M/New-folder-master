"""
virtual_trade_engine.py
=======================
Virtual Trade Engine for Milestone 6 — Encrypted P2P Virtual Trade Simulation.

Responsibilities
----------------
1. Accept a virtual trade instruction (ticker, quantity, price, timestamp).
2. Validate the trade without touching the caller's real portfolio object.
3. Construct an immutable VirtualPortfolio that blends real holdings with
   the simulated position — expressed as weight vectors suitable for all
   downstream M3–M5 analytics.
4. Cryptographically seal the trade record via crypto_layer.

No existing module (M3–M5) is imported here.  The engine is fully isolated;
it receives real portfolio data as plain Python dicts / numpy arrays and
returns standardised data structures that impact_analyzer.py and
projection_engine.py can consume without knowing about the source.

Public API
----------
    VirtualTrade               dataclass — trade instruction
    VirtualPortfolio           dataclass — blended portfolio state
    VirtualTradeEngine         class     — orchestrates everything

    VirtualTradeEngine.execute(trade, real_portfolio, prices_at_trade)
        -> (VirtualPortfolio, dict)      # portfolio + signed tx record
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from milestone6.crypto_layer import (
    PrivateKey,
    PublicKey,
    build_transaction_object,
    generate_keypair,
    hash_payload,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VirtualTrade:
    """
    A single virtual (simulated) trade instruction.

    Attributes
    ----------
    ticker      : NSE ticker, e.g. "INFY.NS".
    quantity    : Number of shares to virtually purchase (positive = buy).
    price       : Price per share at time of simulation (INR).
    timestamp   : Unix timestamp of the trade (defaults to now).
    trade_id    : Auto-generated UUID — uniquely identifies this instruction.
    note        : Optional free-text annotation.
    """
    ticker:    str
    quantity:  float
    price:     float
    timestamp: float          = field(default_factory=time.time)
    trade_id:  str            = field(default_factory=lambda: str(uuid.uuid4()))
    note:      Optional[str]  = None

    def __post_init__(self):
        if self.quantity == 0:
            raise ValueError("quantity cannot be zero — use a positive number for BUY.")
        if self.price <= 0:
            raise ValueError(f"price must be positive, got {self.price}.")
        self.ticker = self.ticker.upper().strip()

    @property
    def trade_value(self) -> float:
        """Total rupee value of the virtual position."""
        return abs(self.quantity) * self.price

    def to_payload(self) -> Dict[str, Any]:
        """Serialisable dict used as the cryptographic payload."""
        return {
            "trade_id":   self.trade_id,
            "ticker":     self.ticker,
            "quantity":   self.quantity,
            "price":      self.price,
            "trade_value": self.trade_value,
            "timestamp":  self.timestamp,
            "note":       self.note or "",
            "type":       "VIRTUAL_BUY" if self.quantity > 0 else "VIRTUAL_SELL",
        }


@dataclass
class RealPortfolioSnapshot:
    """
    Caller-supplied snapshot of the real portfolio at the time of simulation.
    This is the only representation of real holdings the engine ever sees;
    the caller's live PortfolioState is never modified.

    Attributes
    ----------
    holdings        : {ticker: shares_held}      — can be empty for a fresh portfolio.
    prices          : {ticker: current_price}    — latest prices for all held tickers.
    daily_returns   : pd.DataFrame               — historical log returns, cols = tickers.
    total_value     : float                      — total portfolio value in INR.
    risk_free_rate  : float                      — annual risk-free rate (e.g. 0.07).
    name            : str                        — label for display purposes.
    """
    holdings:       Dict[str, float]
    prices:         Dict[str, float]
    daily_returns:  pd.DataFrame
    total_value:    float
    risk_free_rate: float  = 0.07
    name:           str    = "Real Portfolio"

    def __post_init__(self):
        if self.total_value <= 0:
            raise ValueError("total_value must be positive.")
        # Validate every holding has a price
        missing = [t for t in self.holdings if t not in self.prices]
        if missing:
            raise ValueError(f"Missing prices for held tickers: {missing}")

    @property
    def tickers(self) -> List[str]:
        return list(self.holdings.keys())

    @property
    def weights(self) -> pd.Series:
        """Current portfolio weights derived from holdings × prices."""
        values = {t: self.holdings[t] * self.prices[t]
                  for t in self.holdings}
        total  = sum(values.values()) or self.total_value
        return pd.Series({t: v / total for t, v in values.items()})


@dataclass
class VirtualPortfolio:
    """
    Immutable blended portfolio: real holdings + simulated trade.

    This is the sole output of VirtualTradeEngine.execute().  It is consumed
    by impact_analyzer.py and projection_engine.py as if it were a real
    portfolio; they never know the difference.

    Attributes
    ----------
    weights         : Normalised weight vector over the combined asset universe.
    expected_returns: Annualised arithmetic expected returns (per asset).
    covariance      : Annualised covariance matrix of returns.
    daily_returns   : Historical log returns (combined universe).
    total_value     : Combined portfolio value (real + virtual trade value).
    risk_free_rate  : Annual risk-free rate.
    tickers         : List of all tickers in the combined universe.
    trade           : The VirtualTrade that produced this portfolio.
    real_weights    : Weights of the real portfolio (pre-trade, same universe).
    weight_delta    : weights − real_weights (the trade's marginal contribution).
    new_ticker      : True if the traded ticker is new to the real portfolio.
    snapshot_id     : UUID for this virtual state.
    """
    weights:          pd.Series
    expected_returns: pd.Series
    covariance:       pd.DataFrame
    daily_returns:    pd.DataFrame
    total_value:      float
    risk_free_rate:   float
    tickers:          List[str]
    trade:            VirtualTrade
    real_weights:     pd.Series
    weight_delta:     pd.Series
    new_ticker:       bool
    snapshot_id:      str = field(default_factory=lambda: str(uuid.uuid4()))

    # Computed on first access
    _port_return:  Optional[float] = field(default=None, repr=False)
    _port_vol:     Optional[float] = field(default=None, repr=False)
    _sharpe:       Optional[float] = field(default=None, repr=False)

    @property
    def portfolio_return(self) -> float:
        if self._port_return is None:
            self._port_return = float(
                np.dot(self.weights.values, self.expected_returns.values)
            )
        return self._port_return

    @property
    def portfolio_volatility(self) -> float:
        if self._port_vol is None:
            w = self.weights.values
            S = self.covariance.values
            self._port_vol = float(np.sqrt(w @ S @ w))
        return self._port_vol

    @property
    def sharpe_ratio(self) -> float:
        if self._sharpe is None:
            vol = self.portfolio_volatility
            self._sharpe = (
                (self.portfolio_return - self.risk_free_rate) / vol
                if vol > 0 else 0.0
            )
        return self._sharpe

    def to_dict(self) -> Dict[str, Any]:
        """Summary dict for JSON serialisation."""
        return {
            "snapshot_id":       self.snapshot_id,
            "tickers":           self.tickers,
            "total_value":       round(self.total_value, 2),
            "portfolio_return":  round(self.portfolio_return, 6),
            "portfolio_vol":     round(self.portfolio_volatility, 6),
            "sharpe_ratio":      round(self.sharpe_ratio, 4),
            "weights":           {t: round(float(w), 6)
                                  for t, w in self.weights.items()},
            "real_weights":      {t: round(float(w), 6)
                                  for t, w in self.real_weights.items()},
            "weight_delta":      {t: round(float(d), 6)
                                  for t, d in self.weight_delta.items()},
            "trade": {
                "ticker":     self.trade.ticker,
                "quantity":   self.trade.quantity,
                "price":      self.trade.price,
                "trade_value": self.trade.trade_value,
                "new_ticker": self.new_ticker,
            },
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VirtualTradeEngine:
    """
    Orchestrates the full virtual trade pipeline:

        execute(trade, snapshot)
            ├── validate inputs
            ├── compute virtual portfolio (blend real + trade)
            ├── build cryptographic transaction record
            └── return (VirtualPortfolio, tx_record)

    The engine generates a fresh ML-DSA key pair per session.  In a
    production P2P system the caller would supply their own persistent key
    pair; here we auto-generate for hermetic, dependency-free operation.
    """

    def __init__(
        self,
        keypair: Optional[Tuple[PublicKey, PrivateKey]] = None,
        risk_free_rate: float = 0.07,
    ):
        """
        Parameters
        ----------
        keypair        : (PublicKey, PrivateKey) — supply for persistent keys.
                         If None, a fresh ML-DSA key pair is generated.
        risk_free_rate : Annual risk-free rate fallback.
        """
        if keypair is None:
            self.pk, self.sk = generate_keypair()
        else:
            self.pk, self.sk = keypair

        self.risk_free_rate = risk_free_rate

    # ------------------------------------------------------------------
    # Public method
    # ------------------------------------------------------------------

    def execute(
        self,
        trade:    VirtualTrade,
        snapshot: RealPortfolioSnapshot,
    ) -> Tuple[VirtualPortfolio, Dict[str, Any]]:
        """
        Execute a virtual trade against a real portfolio snapshot.

        Parameters
        ----------
        trade    : VirtualTrade   — the simulated instruction.
        snapshot : RealPortfolioSnapshot — caller's current real state.

        Returns
        -------
        (VirtualPortfolio, tx_record)
            VirtualPortfolio — blended state ready for impact analysis.
            tx_record        — cryptographically signed transaction dict.
        """
        self._validate(trade, snapshot)
        vp        = self._build_virtual_portfolio(trade, snapshot)
        tx_record = self._sign_trade(trade, vp)
        return vp, tx_record

    # ------------------------------------------------------------------
    # Internal: validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate(trade: VirtualTrade, snapshot: RealPortfolioSnapshot) -> None:
        """Raise ValueError with a clear message if inputs are inconsistent."""
        if trade.price <= 0:
            raise ValueError(f"Trade price must be positive, got {trade.price}.")
        if trade.quantity == 0:
            raise ValueError("Trade quantity cannot be zero.")
        if snapshot.total_value <= 0:
            raise ValueError("Snapshot total_value must be positive.")

        # Warn if ticker is not in the returns history (new ticker is allowed)
        # but returns must exist for at least the real holdings
        held = set(snapshot.holdings.keys())
        historical = set(snapshot.daily_returns.columns.tolist())
        missing_history = held - historical
        if missing_history:
            raise ValueError(
                f"daily_returns missing columns for held tickers: {missing_history}. "
                "Ensure snapshot.daily_returns covers all real holdings."
            )

    # ------------------------------------------------------------------
    # Internal: portfolio construction
    # ------------------------------------------------------------------

    def _build_virtual_portfolio(
        self,
        trade:    VirtualTrade,
        snapshot: RealPortfolioSnapshot,
    ) -> VirtualPortfolio:
        """
        Blend real holdings + virtual trade into a single weight vector.

        Steps
        -----
        1. Determine combined asset universe (real ∪ {trade.ticker}).
        2. For the traded ticker: if it has no historical returns, synthesise
           a proxy return series from the cross-sectional mean of existing
           assets plus Gaussian noise calibrated to the average volatility.
        3. Compute combined daily_returns DataFrame over the full universe.
        4. Derive expected_returns and covariance from combined history.
        5. Compute virtual holdings = real_holdings + {trade.ticker: trade.quantity}.
        6. Derive normalised weights from virtual holdings × prices.
        7. Map real weights onto the combined universe (zeros for new ticker).
        8. Return immutable VirtualPortfolio.
        """
        rf = snapshot.risk_free_rate or self.risk_free_rate

        # ── 1. Universe ──────────────────────────────────────────────────
        real_tickers  = list(snapshot.holdings.keys())
        trade_ticker  = trade.ticker
        new_ticker    = trade_ticker not in real_tickers
        all_tickers   = real_tickers.copy()
        if new_ticker:
            all_tickers.append(trade_ticker)
        N = len(all_tickers)

        # ── 2. Return history for combined universe ───────────────────────
        real_returns = snapshot.daily_returns[real_tickers].copy()

        if new_ticker:
            synthetic = self._synthesise_returns(trade_ticker, real_returns)
            combined_returns = pd.concat([real_returns, synthetic], axis=1)
        else:
            combined_returns = real_returns.copy()

        combined_returns = combined_returns[all_tickers].dropna()

        # ── 3. Expected returns (annualised arithmetic) ───────────────────
        mu_log   = combined_returns.mean()
        var_log  = combined_returns.var()
        # Lognormal correction: arithmetic mu = exp(mu_log*252 + 0.5*var_log*252) - 1
        mu_arith = np.exp(mu_log * 252 + 0.5 * var_log * 252) - 1
        expected_returns = pd.Series(mu_arith, index=all_tickers)

        # ── 4. Covariance (annualised, Ledoit-Wolf-lite shrinkage) ─────────
        T = len(combined_returns)
        S = combined_returns.cov() * 252           # sample covariance

        # Simple analytical shrinkage target: scaled identity
        mu_trace = float(np.trace(S.values)) / N
        F        = pd.DataFrame(
            np.eye(N) * mu_trace, index=all_tickers, columns=all_tickers
        )
        # Shrinkage intensity: 1/(1 + T/N) — pulls toward identity when T is small
        delta = 1.0 / (1.0 + T / N)
        covariance = (1 - delta) * S + delta * F
        # Enforce symmetry and positive-definiteness
        cov_vals = (covariance.values + covariance.values.T) / 2
        eigvals  = np.linalg.eigvalsh(cov_vals)
        if eigvals.min() < 1e-8:
            cov_vals += (abs(eigvals.min()) + 1e-8) * np.eye(N)
        covariance = pd.DataFrame(cov_vals, index=all_tickers, columns=all_tickers)

        # ── 5. Virtual holdings ───────────────────────────────────────────
        virtual_holdings = dict(snapshot.holdings)  # shallow copy — real untouched
        virtual_holdings[trade_ticker] = (
            virtual_holdings.get(trade_ticker, 0.0) + trade.quantity
        )

        # ── 6. Virtual prices ─────────────────────────────────────────────
        virtual_prices = dict(snapshot.prices)
        if trade_ticker not in virtual_prices:
            virtual_prices[trade_ticker] = trade.price

        # ── 7. Weights from holdings × prices ────────────────────────────
        virtual_values = {
            t: virtual_holdings[t] * virtual_prices[t]
            for t in all_tickers
        }
        virtual_total  = sum(virtual_values.values())
        # If all real holdings have value 0 (fresh portfolio), add trade value
        if virtual_total <= 0:
            virtual_total = trade.trade_value

        virtual_weights = pd.Series(
            {t: virtual_values[t] / virtual_total for t in all_tickers}
        )
        virtual_weights = virtual_weights.clip(lower=0)
        s = virtual_weights.sum()
        if s > 0:
            virtual_weights /= s

        # ── 8. Real weights on combined universe (zeros for new ticker) ───
        real_weights_raw = snapshot.weights
        real_weights_combined = pd.Series(
            {t: float(real_weights_raw.get(t, 0.0)) for t in all_tickers}
        )
        s = real_weights_combined.sum()
        if s > 0:
            real_weights_combined /= s

        weight_delta = virtual_weights - real_weights_combined

        return VirtualPortfolio(
            weights          = virtual_weights,
            expected_returns = expected_returns,
            covariance       = covariance,
            daily_returns    = combined_returns,
            total_value      = snapshot.total_value + trade.trade_value,
            risk_free_rate   = rf,
            tickers          = all_tickers,
            trade            = trade,
            real_weights     = real_weights_combined,
            weight_delta     = weight_delta,
            new_ticker       = new_ticker,
        )

    @staticmethod
    def _synthesise_returns(
        ticker: str,
        existing_returns: pd.DataFrame,
        seed: int = 42,
    ) -> pd.Series:
        """
        Synthesise a return series for a ticker with no history.

        Method: cross-sectional mean return + zero-mean Gaussian noise
        calibrated to the average pairwise volatility of existing assets.
        This preserves the time index and produces plausible but conservative
        return characteristics.

        The synthesis is clearly labelled; in production the caller would
        supply real historical data for any new ticker.
        """
        rng         = np.random.default_rng(seed)
        mean_return = existing_returns.mean(axis=1)      # cross-sectional daily mean
        avg_vol     = existing_returns.std().mean()      # average daily vol

        noise = rng.normal(0.0, avg_vol * 0.8, size=len(existing_returns))
        synthetic = mean_return + noise
        return pd.Series(synthetic.values, index=existing_returns.index, name=ticker)

    # ------------------------------------------------------------------
    # Internal: cryptographic sealing
    # ------------------------------------------------------------------

    def _sign_trade(
        self,
        trade: VirtualTrade,
        vp:    VirtualPortfolio,
    ) -> Dict[str, Any]:
        """
        Build and cryptographically sign the transaction record.

        The payload includes:
        - The full VirtualTrade instruction.
        - A summary of the resulting VirtualPortfolio.
        - A SHA3-256 fingerprint of the portfolio weight vector (tamper-evident).
        """
        weights_bytes  = np.array(list(vp.weights.values)).tobytes()
        weights_hash   = hash_payload({"w": list(vp.weights.values),
                                       "t": vp.tickers})

        payload = {
            **trade.to_payload(),
            "virtual_portfolio_snapshot_id": vp.snapshot_id,
            "virtual_total_value":           round(vp.total_value, 2),
            "virtual_return":                round(vp.portfolio_return, 6),
            "virtual_volatility":            round(vp.portfolio_volatility, 6),
            "virtual_sharpe":                round(vp.sharpe_ratio, 4),
            "weight_fingerprint":            weights_hash,
            "n_assets":                      len(vp.tickers),
            "new_ticker_added":              vp.new_ticker,
        }

        return build_transaction_object(payload, self.sk, self.pk)
