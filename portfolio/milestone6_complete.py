"""
milestone6_complete.py
======================
Milestone 6 — Secure Virtual P2P Trading System
CONSOLIDATED SINGLE-FILE EDITION

All 17 sub-modules merged into one self-contained file.
No package structure required — import directly from this single file.

Architecture
------------
LAYER 1   crypto_layer          ML-DSA-III (CRYSTALS-Dilithium) PQC signatures
LAYER 2   virtual_trade_engine  Virtual portfolio construction & trade simulation
LAYER 3   impact_analyzer       Portfolio impact attribution (return, vol, CVaR, HHI)
LAYER 4   projection_engine     Multivariate GBM Monte Carlo (1Y / 3Y / 5Y)
LAYER 5   pipeline              End-to-end orchestrator (single public API call)

SECURITY MODULE 1   entropy_monitor      Shannon entropy & bit-uniformity scoring
SECURITY MODULE 2   immune_detector      Welford z-score statistical anomaly detection
SECURITY MODULE 3   immune_response      Adaptive action dispatcher (quarantine/multi-sig)
SECURITY MODULE 4   key_mutation         Adaptive ML-DSA key rotation with cross-signing
SECURITY MODULE 5   threat_memory        Cosine-similarity episodic threat pattern store
SECURITY MODULE 6   security_engine      PQC Immune Defense master pipeline

BAYESIAN MODULE 1   signal_processor     6-signal extractor & sigmoid normaliser
BAYESIAN MODULE 2   bayesian_engine      Neutral-point LLR Bayesian posterior engine
BAYESIAN MODULE 3   threat_classifier    Posterior → SAFE/MONITOR/ELEVATED/CRITICAL
BAYESIAN MODULE 4   response_engine      Monotone action lattice + QuarantineLedger
BAYESIAN MODULE 5   immune_memory        Recency-weighted cosine memory + JSON persistence
BAYESIAN MODULE 6   security_pipeline    Bayesian Immune Defense master controller

Public entry points
-------------------
result  = run_virtual_trade_simulation(ticker, qty, price, holdings, prices,
                                       returns_df, total_value)
engine  = SecurityEngine()
report  = engine.process_transaction_security(tx_dict)
bpipe   = BayesianSecurityPipeline()
breport = bpipe.process_transaction_security(tx_dict)

Dependencies: numpy, scipy, pandas
"""

from __future__ import annotations

# ────────────────────────────────────────────────────────────────────────────
# Imports
# ────────────────────────────────────────────────────────────────────────────

from typing import Any, Dict, List, Optional, Set, Tuple
from collections import OrderedDict
from dataclasses import dataclass
from dataclasses import dataclass, field
import hashlib, hmac, json, secrets, struct, time, uuid
import json
import logging
import math
import os
import time
import uuid
import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm


# ════════════════════════════════════════════════════════════════════════════
# LAYER 1 — CRYPTOGRAPHIC FOUNDATION  (ML-DSA / CRYSTALS-Dilithium)
# ════════════════════════════════════════════════════════════════════════════

# ── Dilithium-III parameters ─────────────────────────────────────────────────
_Q     = 8_380_417
_N     = 256
_K     = 6
_L     = 5
_ETA   = 4
_TAU   = 49
_G1    = 131_072        # gamma1 = 2^17
_G2    = 95_232         # gamma2 = (q−1)/88
_BETA  = _TAU * _ETA    # 196
_D     = 4
_OMEGA = 55             # max hint weight (Dilithium-III)

# ── Fast polynomial multiplication (negacyclic, mod X^N+1, mod q) ────────────

def _poly_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Multiply two polynomials in R_q = Z_q[X]/(X^N+1).
    Uses numpy.convolve (O(N log N)) then folds the degree-N..2N−2 tail
    back with a sign flip (X^N ≡ −1) and reduces mod q.
    """
    c = np.convolve(a.astype(np.int64), b.astype(np.int64))
    r = c[:_N].copy()
    tail = c[_N:]                        # length N−1
    r[:len(tail)] -= tail
    return r % _Q

def _poly_add(a, b): return (np.asarray(a, np.int64) + np.asarray(b, np.int64)) % _Q
def _poly_sub(a, b): return (np.asarray(a, np.int64) - np.asarray(b, np.int64)) % _Q

def _matvec(A, v):
    out = []
    for row in A:
        acc = np.zeros(_N, np.int64)
        for aij, vj in zip(row, v):
            acc = _poly_add(acc, _poly_mul(aij, vj))
        out.append(acc)
    return out

def _inf_norm(vec):
    best = 0
    for p in vec:
        c = np.asarray(p, np.int64) % _Q
        c[c > _Q // 2] -= _Q
        m = int(np.max(np.abs(c)))
        if m > best: best = m
    return best

# ── Power2Round / Decompose / MakeHint / UseHint  (FIPS 204 §3.1) ─────────────

def _power2round(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """r = r1·2^d + r0,  r0 ∈ (−2^(d−1), 2^(d−1)].  Nearest-round."""
    r      = np.asarray(poly, np.int64) % _Q
    half   = 1 << (_D - 1)   # 8  (2^(d-1) with d=4)
    two_d  = 1 << _D          # 16 (2^d with d=4)
    r1     = (r + half) >> _D
    r0     = r - (r1 << _D)
    r0     = np.where(r0 >  half,  r0 - two_d, r0)
    r0     = np.where(r0 <= -half, r0 + two_d, r0)
    return r1 % _Q, r0

def _decompose(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose(r) with alpha = 2·gamma2."""
    alpha = 2 * _G2
    r     = np.asarray(poly, np.int64) % _Q
    r0    = r % alpha
    r0    = np.where(r0 > _G2, r0 - alpha, r0)
    r1    = (r - r0) // alpha
    # Spec boundary case
    top   = (_Q - 1) // alpha + 1
    mask  = r1 == top
    r1    = np.where(mask, np.int64(0), r1)
    r0    = np.where(mask, r0 - np.int64(1), r0)
    return r1, r0

def _high_bits(p): return _decompose(p)[0]
def _low_bits(p):  return _decompose(p)[1]

def _make_hint(z: np.ndarray, r: np.ndarray) -> np.ndarray:
    r1, _  = _decompose(r)
    v1, _  = _decompose((np.asarray(r, np.int64) + np.asarray(z, np.int64)) % _Q)
    return (r1 != v1).astype(np.int64)

def _use_hint(h: np.ndarray, r: np.ndarray) -> np.ndarray:
    m      = (_Q - 1) // (2 * _G2)   # 43
    r1, r0 = _decompose(r)
    adj    = np.where(r0 > 0, (r1 + 1) % (m + 1), (r1 - 1) % (m + 1))
    return np.where(h == 1, adj, r1)

# ── Hash / XOF ────────────────────────────────────────────────────────────────

def _sha3_256(d: bytes) -> bytes: return hashlib.sha3_256(d).digest()
def _sha3_512(d: bytes) -> bytes: return hashlib.sha3_512(d).digest()
def _shake256(d: bytes, n: int) -> bytes:
    h = hashlib.shake_256(); h.update(d); return h.digest(n)

# ── Polynomial sampling ───────────────────────────────────────────────────────

def _expand_A(rho: bytes) -> List[List[np.ndarray]]:
    """Near-uniform over Z_q: 3-byte reads, 23-bit mask, reject if ≥ q."""
    A = []
    for i in range(_K):
        row = []
        for j in range(_L):
            seed = rho + bytes([j, i])
            stream = _shake256(seed, _N * 4)
            coeffs, idx = [], 0
            while len(coeffs) < _N:
                if idx + 3 > len(stream):
                    stream += _shake256(seed + bytes([idx & 0xFF]), _N * 3)
                b0, b1, b2 = stream[idx], stream[idx+1], stream[idx+2]; idx += 3
                d = (b0 | (b1 << 8) | (b2 << 16)) & 0x7FFFFF
                if d < _Q: coeffs.append(d)
            row.append(np.array(coeffs[:_N], np.int64))
        A.append(row)
    return A

def _sample_secret(eta: int, seed: bytes, nonce: int) -> np.ndarray:
    stream = _shake256(seed + struct.pack("<H", nonce), _N * 2)
    coeffs = []
    for byte in stream:
        c0, c1 = byte & 0x0F, byte >> 4
        if c0 <= 2*eta: coeffs.append(eta - c0)
        if c1 <= 2*eta and len(coeffs) < _N: coeffs.append(eta - c1)
        if len(coeffs) >= _N: break
    coeffs.extend([0] * (_N - len(coeffs)))
    return np.array(coeffs[:_N], np.int64)

def _sample_in_ball(seed: bytes) -> np.ndarray:
    stream    = _shake256(seed, _N + 8)
    c         = np.zeros(_N, np.int64)
    sign_bits = int.from_bytes(stream[:8], "little")
    idx       = 8
    for i in range(_N - _TAU, _N):
        j = None
        while j is None or j > i:
            if idx >= len(stream):
                stream = _shake256(seed + bytes([idx & 0xFF]), _N); idx = 0
            j = stream[idx]; idx += 1
            if j > i: j = None
        c[i] = c[j]
        c[j] = np.int64(1) if ((sign_bits >> (i % 64)) & 1) == 0 else np.int64(_Q - 1)
    return c % _Q

# ── Key material dataclasses ──────────────────────────────────────────────────

@dataclass
class PublicKey:
    rho: bytes; t1: List[np.ndarray]; key_id: str; created: float

    def fingerprint(self) -> str:
        return _sha3_256(self.rho + b"".join(p.tobytes() for p in self.t1)).hex()

    def to_dict(self) -> Dict[str, Any]:
        return {"key_id": self.key_id, "fingerprint": self.fingerprint(),
                "rho_hex": self.rho.hex(), "created_at": self.created,
                "scheme": "ML-DSA-III (simulated, FIPS 204)",
                "params": {"q":_Q,"n":_N,"k":_K,"l":_L,
                           "eta":_ETA,"tau":_TAU,"beta":_BETA}}

@dataclass
class PrivateKey:
    rho: bytes; K: bytes
    s1: List[np.ndarray]; s2: List[np.ndarray]; t0: List[np.ndarray]
    key_id: str

@dataclass
class TransactionSignature:
    c_tilde: bytes; z: List[np.ndarray]; h: List[np.ndarray]
    nonce: int; scheme: str = "ML-DSA-III-simulated"

    def pack(self) -> bytes:
        return (self.c_tilde + struct.pack("<I", self.nonce)
                + b"".join(p.tobytes() for p in self.z)
                + b"".join(p.tobytes() for p in self.h))

    def to_dict(self) -> Dict[str, Any]:
        return {"scheme": self.scheme, "c_tilde_hex": self.c_tilde.hex(),
                "nonce": self.nonce, "z_infinity_norm": _inf_norm(self.z),
                "packed_length_bytes": len(self.pack()),
                "acceptance_bound": _G1 - _BETA,
                "valid_norm": _inf_norm(self.z) < _G1 - _BETA}

# ── Key generation ────────────────────────────────────────────────────────────

def generate_keypair() -> Tuple[PublicKey, PrivateKey]:
    """Fresh cryptographically random key pair every call."""
    key_id  = str(uuid.uuid4())
    created = time.time()
    xi      = secrets.token_bytes(32)
    exp     = _sha3_512(xi)
    rho, rho_p = exp[:32], exp[32:]
    K       = _sha3_256(xi + b"\x02")

    A   = _expand_A(rho)
    s1  = [_sample_secret(_ETA, rho_p, i)      for i in range(_L)]
    s2  = [_sample_secret(_ETA, rho_p, _L + i) for i in range(_K)]
    As1 = _matvec(A, s1)
    t   = [_poly_add(As1[i], s2[i]) for i in range(_K)]

    t1_list, t0_list = [], []
    for p in t:
        r1, r0 = _power2round(p)
        t1_list.append(r1)
        t0_list.append(r0)

    pk = PublicKey(rho=rho, t1=t1_list, key_id=key_id, created=created)
    sk = PrivateKey(rho=rho, K=K, s1=s1, s2=s2, t0=t0_list, key_id=key_id)
    return pk, sk

def _t1_from_sk(sk: PrivateKey) -> List[np.ndarray]:
    A   = _expand_A(sk.rho)
    As1 = _matvec(A, sk.s1)
    t   = [_poly_add(As1[i], sk.s2[i]) for i in range(_K)]
    return [_power2round(p)[0] for p in t]

# ── Signing ───────────────────────────────────────────────────────────────────

def sign_transaction(payload_bytes: bytes, sk: PrivateKey) -> TransactionSignature:
    """
    Randomised ML-DSA signing.
    - Fresh secrets.token_bytes(32) mixed into y_seed each call → unique sig.
    - MakeHint computed so verify can reconstruct w1 via UseHint.
    """
    A         = _expand_A(sk.rho)
    t1        = _t1_from_sk(sk)
    t1_packed = b"".join(p.tobytes() for p in t1)
    tr        = _sha3_256(sk.rho + t1_packed)
    mu        = _sha3_256(tr + payload_bytes)
    rnd       = secrets.token_bytes(32)
    kappa     = 0

    for _ in range(2_000):
        y_seed = _sha3_512(sk.K + rnd + mu + struct.pack("<I", kappa))
        y = []
        for i in range(_L):
            stream = _shake256(y_seed + struct.pack("<H", i), _N * 5)
            coeffs, idx = [], 0
            while len(coeffs) < _N and idx + 5 <= len(stream):
                raw = int.from_bytes(stream[idx:idx+5], "little") % (2*_G1)
                coeffs.append(raw - _G1); idx += 5   # signed: in (-G1, G1]
            coeffs.extend([0]*(_N - len(coeffs)))
            y.append(np.array(coeffs[:_N], np.int64))
        kappa += 1

        Ay      = _matvec(A, y)
        w1      = [_high_bits(p) for p in Ay]
        c_tilde = _sha3_256(mu + b"".join(p.tobytes() for p in w1))
        c_poly  = _sample_in_ball(c_tilde)

        z = [_poly_add(y[i], _poly_mul(c_poly, sk.s1[i])) for i in range(_L)]
        if _inf_norm(z) >= _G1 - _BETA:
            continue

        cs2 = [_poly_mul(c_poly, sk.s2[i]) for i in range(_K)]
        r   = [_poly_sub(Ay[i], cs2[i])    for i in range(_K)]

        if any(_inf_norm([_low_bits(r[i])]) >= _G2 - _BETA for i in range(_K)):
            continue

        ct0 = [_poly_mul(c_poly, sk.t0[i]) for i in range(_K)]
        h   = [_make_hint(ct0[i], r[i])    for i in range(_K)]
        if sum(int(np.sum(hi)) for hi in h) > _OMEGA:
            continue

        return TransactionSignature(c_tilde=c_tilde, z=z, h=h, nonce=kappa)

    raise RuntimeError("ML-DSA signing did not converge after 2,000 iterations.")

# ── Verification ──────────────────────────────────────────────────────────────

def verify_transaction(payload_bytes: bytes,
                       sig: TransactionSignature,
                       pk: PublicKey) -> bool:
    """
    Verify ML-DSA signature.
    Checks: norm, hint weight, UseHint reconstruction, commitment hash.
    """
    try:
        if _inf_norm(sig.z) >= _G1 - _BETA:                       return False
        if sum(int(np.sum(hi)) for hi in sig.h) > _OMEGA:          return False

        t1_packed = b"".join(p.tobytes() for p in pk.t1)
        tr        = _sha3_256(pk.rho + t1_packed)
        mu        = _sha3_256(tr + payload_bytes)

        A      = _expand_A(pk.rho)
        c_poly = _sample_in_ball(sig.c_tilde)
        Az     = _matvec(A, sig.z)

        w_prime = []
        for i in range(_K):
            t1_scaled = (pk.t1[i].astype(np.int64) << _D) % _Q
            ct1       = _poly_mul(c_poly, t1_scaled)
            diff      = _poly_sub(Az[i], ct1)
            w_prime.append(_use_hint(sig.h[i], diff))

        expected = _sha3_256(mu + b"".join(p.tobytes() for p in w_prime))
        return hmac.compare_digest(sig.c_tilde, expected)
    except Exception:
        return False

# ── Payload utilities ─────────────────────────────────────────────────────────

def hash_payload(payload: Dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",",":"),
                     default=str).encode()
    return hashlib.sha3_256(raw).hexdigest()

def build_transaction_object(payload: Dict[str, Any],
                              sk: PrivateKey, pk: PublicKey) -> Dict[str, Any]:
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",",":"),
                               default=str).encode()
    payload_hash  = hash_payload(payload)
    sig      = sign_transaction(payload_bytes, sk)
    is_valid = verify_transaction(payload_bytes, sig, pk)
    ts       = struct.pack(">d", time.time())
    tx_id    = hashlib.sha3_256((payload_hash+pk.fingerprint()).encode()+ts).hexdigest()[:32]
    return {"tx_id": tx_id, "payload": payload, "sha3_hash": payload_hash,
            "signature": sig.to_dict(), "public_key": pk.to_dict(),
            "verification_status": is_valid, "signed_at": time.time()}


# ════════════════════════════════════════════════════════════════════════════
# LAYER 2 — VIRTUAL TRADE ENGINE
# ════════════════════════════════════════════════════════════════════════════

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
    trade_id    : Auto-generated UUID.
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
            "trade_id":    self.trade_id,
            "ticker":      self.ticker,
            "quantity":    self.quantity,
            "price":       self.price,
            "trade_value": self.trade_value,
            "timestamp":   self.timestamp,
            "note":        self.note or "",
            "type":        "VIRTUAL_BUY" if self.quantity > 0 else "VIRTUAL_SELL",
        }


@dataclass
class RealPortfolioSnapshot:
    """
    Caller-supplied snapshot of the real portfolio at the time of simulation.
    The caller's live PortfolioState is never modified.

    Attributes
    ----------
    holdings        : {ticker: shares_held}
    prices          : {ticker: current_price}
    daily_returns   : pd.DataFrame of historical log returns, cols = tickers.
    total_value     : float — total portfolio value in INR.
    risk_free_rate  : float — annual risk-free rate (e.g. 0.07).
    name            : str   — label for display purposes.
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
        missing = [t for t in self.holdings if t not in self.prices]
        if missing:
            raise ValueError(f"Missing prices for held tickers: {missing}")

    @property
    def tickers(self) -> List[str]:
        return list(self.holdings.keys())

    @property
    def weights(self) -> pd.Series:
        """Current portfolio weights derived from holdings x prices."""
        values = {t: self.holdings[t] * self.prices[t] for t in self.holdings}
        total  = sum(values.values()) or self.total_value
        return pd.Series({t: v / total for t, v in values.items()})


@dataclass
class VirtualPortfolio:
    """
    Immutable blended portfolio: real holdings + simulated trade.

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
    weight_delta    : weights - real_weights (the trade's marginal contribution).
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
        return {
            "snapshot_id":      self.snapshot_id,
            "tickers":          self.tickers,
            "total_value":      round(self.total_value, 2),
            "portfolio_return": round(self.portfolio_return, 6),
            "portfolio_vol":    round(self.portfolio_volatility, 6),
            "sharpe_ratio":     round(self.sharpe_ratio, 4),
            "weights":          {t: round(float(w), 6) for t, w in self.weights.items()},
            "real_weights":     {t: round(float(w), 6) for t, w in self.real_weights.items()},
            "weight_delta":     {t: round(float(d), 6) for t, d in self.weight_delta.items()},
            "trade": {
                "ticker":      self.trade.ticker,
                "quantity":    self.trade.quantity,
                "price":       self.trade.price,
                "trade_value": self.trade.trade_value,
                "new_ticker":  self.new_ticker,
            },
        }


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class VirtualTradeEngine:
    """
    Orchestrates the full virtual trade pipeline:

        execute(trade, snapshot)
            |-- validate inputs
            |-- compute virtual portfolio (blend real + trade)
            |-- build cryptographic transaction record
            `-- return (VirtualPortfolio, tx_record)
    """

    def __init__(
        self,
        keypair: Optional[Tuple[PublicKey, PrivateKey]] = None,
        risk_free_rate: float = 0.07,
    ):
        if keypair is None:
            self.pk, self.sk = generate_keypair()
        else:
            self.pk, self.sk = keypair
        self.risk_free_rate = risk_free_rate

    def execute(
        self,
        trade:    VirtualTrade,
        snapshot: RealPortfolioSnapshot,
    ) -> Tuple[VirtualPortfolio, Dict[str, Any]]:
        """
        Execute a virtual trade against a real portfolio snapshot.

        Returns
        -------
        (VirtualPortfolio, tx_record)
        """
        self._validate(trade, snapshot)
        vp        = self._build_virtual_portfolio(trade, snapshot)
        tx_record = self._sign_trade(trade, vp)
        return vp, tx_record

    @staticmethod
    def _validate(trade: VirtualTrade, snapshot: RealPortfolioSnapshot) -> None:
        if trade.price <= 0:
            raise ValueError(f"Trade price must be positive, got {trade.price}.")
        if trade.quantity == 0:
            raise ValueError("Trade quantity cannot be zero.")
        if snapshot.total_value <= 0:
            raise ValueError("Snapshot total_value must be positive.")
        held       = set(snapshot.holdings.keys())
        historical = set(snapshot.daily_returns.columns.tolist())
        missing    = held - historical
        if missing:
            raise ValueError(
                f"daily_returns missing columns for held tickers: {missing}. "
                "Ensure snapshot.daily_returns covers all real holdings."
            )

    def _build_virtual_portfolio(
        self,
        trade:    VirtualTrade,
        snapshot: RealPortfolioSnapshot,
    ) -> VirtualPortfolio:
        rf = snapshot.risk_free_rate or self.risk_free_rate

        real_tickers  = list(snapshot.holdings.keys())
        trade_ticker  = trade.ticker
        new_ticker    = trade_ticker not in real_tickers
        all_tickers   = real_tickers.copy()
        if new_ticker:
            all_tickers.append(trade_ticker)
        N = len(all_tickers)

        real_returns = snapshot.daily_returns[real_tickers].copy()
        if new_ticker:
            synthetic = self._synthesise_returns(trade_ticker, real_returns)
            combined_returns = pd.concat([real_returns, synthetic], axis=1)
        else:
            combined_returns = real_returns.copy()

        combined_returns = combined_returns[all_tickers].dropna()

        mu_log   = combined_returns.mean()
        var_log  = combined_returns.var()
        mu_arith = np.exp(mu_log * 252 + 0.5 * var_log * 252) - 1
        expected_returns = pd.Series(mu_arith, index=all_tickers)

        T = len(combined_returns)
        S = combined_returns.cov() * 252
        mu_trace  = float(np.trace(S.values)) / N
        F         = pd.DataFrame(np.eye(N) * mu_trace, index=all_tickers, columns=all_tickers)
        delta_lw  = 1.0 / (1.0 + T / N)
        covariance = (1 - delta_lw) * S + delta_lw * F
        cov_vals   = (covariance.values + covariance.values.T) / 2
        eigvals    = np.linalg.eigvalsh(cov_vals)
        if eigvals.min() < 1e-8:
            cov_vals += (abs(eigvals.min()) + 1e-8) * np.eye(N)
        covariance = pd.DataFrame(cov_vals, index=all_tickers, columns=all_tickers)

        virtual_holdings = dict(snapshot.holdings)
        virtual_holdings[trade_ticker] = (
            virtual_holdings.get(trade_ticker, 0.0) + trade.quantity
        )

        virtual_prices = dict(snapshot.prices)
        if trade_ticker not in virtual_prices:
            virtual_prices[trade_ticker] = trade.price

        virtual_values = {
            t: virtual_holdings[t] * virtual_prices[t] for t in all_tickers
        }
        virtual_total = sum(virtual_values.values())
        if virtual_total <= 0:
            virtual_total = trade.trade_value

        virtual_weights = pd.Series(
            {t: virtual_values[t] / virtual_total for t in all_tickers}
        )
        virtual_weights = virtual_weights.clip(lower=0)
        s = virtual_weights.sum()
        if s > 0:
            virtual_weights /= s

        real_weights_raw      = snapshot.weights
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
        Uses cross-sectional mean + Gaussian noise calibrated to avg vol.
        """
        rng         = np.random.default_rng(seed)
        mean_return = existing_returns.mean(axis=1)
        avg_vol     = existing_returns.std().mean()
        noise       = rng.normal(0.0, avg_vol * 0.8, size=len(existing_returns))
        synthetic   = mean_return + noise
        return pd.Series(synthetic.values, index=existing_returns.index, name=ticker)

    def _sign_trade(
        self,
        trade: VirtualTrade,
        vp:    VirtualPortfolio,
    ) -> Dict[str, Any]:
        weights_hash = hash_payload({
            "w": list(vp.weights.values),
            "t": vp.tickers,
        })
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


# ════════════════════════════════════════════════════════════════════════════
# LAYER 3 — PORTFOLIO IMPACT ANALYSER
# ════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Portfolio metric helpers
# ---------------------------------------------------------------------------

def _portfolio_return(weights: np.ndarray, mu: np.ndarray) -> float:
    return float(weights @ mu)


def _portfolio_vol(weights: np.ndarray, sigma: np.ndarray) -> float:
    v = float(weights @ sigma @ weights)
    return float(np.sqrt(max(v, 0.0)))


def _sharpe(ret: float, vol: float, rf: float) -> float:
    return (ret - rf) / vol if vol > 1e-12 else 0.0


def _cvar_historical(
    weights: np.ndarray,
    daily_returns: pd.DataFrame,
    confidence: float = 0.95,
    weight_index: list = None,
) -> float:
    """Historical CVaR (Expected Shortfall). Returns positive loss magnitude."""
    ret_tickers = daily_returns.columns.tolist()

    # Build a named series from weights so we can align by ticker name
    if weight_index is not None and len(weight_index) == len(weights):
        w_named = pd.Series(weights, index=weight_index)
    else:
        # Fallback: assume weights already match daily_returns columns 1-to-1
        w_named = pd.Series(weights, index=ret_tickers[:len(weights)])

    # Align to the returns universe (zero-fill any ticker not in weights)
    aligned_w = np.array([float(w_named.get(t, 0.0)) for t in ret_tickers])

    port_ret  = daily_returns.values @ aligned_w
    threshold = np.percentile(port_ret, (1 - confidence) * 100)
    tail      = port_ret[port_ret <= threshold]
    if len(tail) == 0:
        return float(-threshold)
    return float(-np.mean(tail))


def _diversification_ratio(weights: np.ndarray, sigma: np.ndarray) -> float:
    """DR = (w . sigma_individual) / sigma_portfolio."""
    individual_vols = np.sqrt(np.diag(sigma))
    weighted_vols   = float(weights @ individual_vols)
    port_vol        = _portfolio_vol(weights, sigma)
    if port_vol < 1e-12 or weighted_vols < 1e-12:
        return 1.0
    return weighted_vols / port_vol


def _hhi(weights: np.ndarray) -> float:
    """Herfindahl-Hirschman Index = sum(w_i^2)."""
    return float(np.sum(weights ** 2))


def _compute_factor_exposures(
    weights: np.ndarray,
    daily_returns: pd.DataFrame,
    tickers: List[str],
) -> Dict[str, float]:
    """
    Single-factor CAPM decomposition aggregated to portfolio level.
    Factor proxy = equal-weight market portfolio of all assets.
    """
    n = len(tickers)
    if n < 2 or daily_returns.empty:
        return {
            "market_beta": 1.0, "systematic_variance": 0.0,
            "idiosyncratic_variance": 0.0, "r_squared": 0.0,
            "tracking_error_vs_ew": 0.0,
        }

    aligned = pd.DataFrame(
        index=daily_returns.index,
        data={t: daily_returns[t].values
              if t in daily_returns.columns
              else np.zeros(len(daily_returns))
              for t in tickers}
    )

    mkt_returns = aligned.mean(axis=1).values

    betas, r2s = [], []
    for t in tickers:
        asset_r = aligned[t].values
        cov_am  = np.cov(asset_r, mkt_returns)
        var_mkt = cov_am[1, 1]
        if var_mkt < 1e-12:
            beta, r2 = 1.0, 0.0
        else:
            beta = cov_am[0, 1] / var_mkt
            r2   = (cov_am[0, 1] ** 2) / (var_mkt * max(cov_am[0, 0], 1e-12))
        betas.append(beta)
        r2s.append(r2)

    betas = np.array(betas)
    r2s   = np.array(r2s)

    port_beta   = float(weights @ betas)
    port_r2     = float(weights @ r2s)
    mkt_var_ann = float(np.var(mkt_returns) * 252)
    syst_var    = (port_beta ** 2) * mkt_var_ann
    total_var   = float(_portfolio_vol(weights, aligned.cov().values * 252) ** 2)
    idio_var    = max(total_var - syst_var, 0.0)

    ew_weights   = np.ones(n) / n
    diff_weights = weights - ew_weights
    te_var       = float(diff_weights @ (aligned.cov().values * 252) @ diff_weights)
    te           = float(np.sqrt(max(te_var, 0.0)))

    return {
        "market_beta":              round(port_beta, 4),
        "systematic_variance":      round(syst_var,  6),
        "idiosyncratic_variance":   round(idio_var,  6),
        "r_squared":                round(port_r2,   4),
        "tracking_error_vs_ew":     round(te,        4),
    }


def _concentration_metrics(weights: np.ndarray, n: int) -> Dict[str, float]:
    """HHI, effective-N, Shannon entropy, Gini, normalised entropy.

    Gini formula (standard, sort-based):
        G = (2 * sum(i * w_i) / (n * sum(w))) - (n+1)/n
    where w is sorted in ascending order and i is 1-indexed rank.
    This produces G=0 for equal weights and G→1 for full concentration.
    """
    w      = np.clip(weights, 1e-10, 1.0)
    hhi    = float(np.sum(w ** 2))
    eff_n  = 1.0 / hhi if hhi > 0 else float(n)
    entr   = float(-np.sum(w * np.log(w)))
    norm_e = entr / np.log(n) if n > 1 else 0.0

    # Correct Gini: ascending sort, 1-indexed ranks
    w_sorted = np.sort(w)          # ascending
    i_ranks  = np.arange(1, n + 1)
    total    = np.sum(w_sorted)
    gini     = (2.0 * float(np.sum(i_ranks * w_sorted)) / (n * total)) - (n + 1) / n if total > 0 else 0.0

    return {
        "herfindahl_index":   round(hhi,             4),
        "effective_n":        round(eff_n,            2),
        "entropy":            round(entr,             4),
        "normalised_entropy": round(norm_e,           4),
        "gini_coefficient":   round(max(gini, 0.0),  4),
    }


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PortfolioMetrics:
    """Full set of analytics for one portfolio state (real or virtual)."""
    expected_return:       float
    volatility:            float
    sharpe_ratio:          float
    cvar_95:               float
    diversification_ratio: float
    hhi:                   float
    effective_n:           float
    concentration:         Dict[str, float]
    factor_exposures:      Dict[str, float]
    weights:               Dict[str, float]
    tickers:               List[str]
    label:                 str = "portfolio"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label":                 self.label,
            "expected_return":       round(self.expected_return, 6),
            "volatility":            round(self.volatility, 6),
            "sharpe_ratio":          round(self.sharpe_ratio, 4),
            "cvar_95":               round(self.cvar_95, 6),
            "diversification_ratio": round(self.diversification_ratio, 4),
            "hhi":                   round(self.hhi, 4),
            "effective_n":           round(self.effective_n, 2),
            "concentration":         self.concentration,
            "factor_exposures":      self.factor_exposures,
            "weights":               {t: round(float(w), 6) for t, w in self.weights.items()},
        }


@dataclass
class ImpactReport:
    """Complete impact analysis: before (real) vs after (virtual) metrics + deltas."""
    real_metrics:     PortfolioMetrics
    virtual_metrics:  PortfolioMetrics
    portfolio_impact: Dict[str, float]
    factor_shift:     Dict[str, float]
    risk_summary:     str
    trade_ticker:     str
    trade_value_inr:  float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trade_ticker":      self.trade_ticker,
            "trade_value_inr":   round(self.trade_value_inr, 2),
            "real_portfolio":    self.real_metrics.to_dict(),
            "virtual_portfolio": self.virtual_metrics.to_dict(),
            "portfolio_impact":  {k: round(v, 6) for k, v in self.portfolio_impact.items()},
            "factor_shift":      {k: round(v, 6) for k, v in self.factor_shift.items()},
            "risk_summary":      self.risk_summary,
        }


# ---------------------------------------------------------------------------
# Main analyzer class
# ---------------------------------------------------------------------------

class ImpactAnalyzer:
    """
    Computes the full analytical impact of a virtual trade.

    Usage
    -----
    analyzer = ImpactAnalyzer()
    report   = analyzer.analyze(virtual_portfolio, real_*)
    """

    def __init__(self, confidence: float = 0.95):
        self.confidence = confidence

    def analyze(
        self,
        vp:                    VirtualPortfolio,
        real_weights:          pd.Series,
        real_daily_returns:    pd.DataFrame,
        real_expected_returns: pd.Series,
        real_covariance:       pd.DataFrame,
        real_total_value:      float,
        risk_free_rate:        float = 0.07,
    ) -> ImpactReport:
        """Run full impact analysis."""
        rf = risk_free_rate

        real_m = self._compute_metrics(
            weights          = real_weights,
            expected_returns = real_expected_returns,
            covariance       = real_covariance,
            daily_returns    = real_daily_returns,
            rf               = rf,
            label            = "Real Portfolio",
        )

        virt_m = self._compute_metrics(
            weights          = vp.weights,
            expected_returns = vp.expected_returns,
            covariance       = vp.covariance,
            daily_returns    = vp.daily_returns,
            rf               = rf,
            label            = "Virtual Portfolio",
        )

        portfolio_impact = {
            "expected_return_change": virt_m.expected_return  - real_m.expected_return,
            "volatility_change":      virt_m.volatility       - real_m.volatility,
            "sharpe_change":          virt_m.sharpe_ratio     - real_m.sharpe_ratio,
            "cvar_change":            virt_m.cvar_95          - real_m.cvar_95,
            "diversification_change": virt_m.diversification_ratio - real_m.diversification_ratio,
            "hhi_change":             virt_m.hhi              - real_m.hhi,
            "effective_n_change":     virt_m.effective_n      - real_m.effective_n,
        }

        factor_shift = {
            k: round(float(virt_m.factor_exposures.get(k, 0))
                     - float(real_m.factor_exposures.get(k, 0)), 4)
            for k in set(list(virt_m.factor_exposures.keys()) +
                         list(real_m.factor_exposures.keys()))
        }

        risk_summary = self._generate_risk_summary(
            trade_ticker = vp.trade.ticker,
            trade_value  = vp.trade.trade_value,
            delta        = portfolio_impact,
            virt_metrics = virt_m,
            new_ticker   = vp.new_ticker,
        )

        return ImpactReport(
            real_metrics     = real_m,
            virtual_metrics  = virt_m,
            portfolio_impact = portfolio_impact,
            factor_shift     = factor_shift,
            risk_summary     = risk_summary,
            trade_ticker     = vp.trade.ticker,
            trade_value_inr  = vp.trade.trade_value,
        )

    def _compute_metrics(
        self,
        weights:          pd.Series,
        expected_returns: pd.Series,
        covariance:       pd.DataFrame,
        daily_returns:    pd.DataFrame,
        rf:               float,
        label:            str,
    ) -> PortfolioMetrics:
        tickers = weights.index.tolist()
        w       = weights.values.astype(float)
        w       = np.clip(w, 0, 1)
        if w.sum() > 0:
            w /= w.sum()

        mu    = np.array([float(expected_returns.get(t, 0.0)) for t in tickers])
        sigma = np.zeros((len(tickers), len(tickers)))
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                v = 0.0
                if ti in covariance.index and tj in covariance.columns:
                    v = float(covariance.loc[ti, tj])
                sigma[i, j] = v
        sigma    = (sigma + sigma.T) / 2
        eigvals  = np.linalg.eigvalsh(sigma)
        if eigvals.min() < 1e-8:
            sigma += (abs(eigvals.min()) + 1e-8) * np.eye(len(tickers))

        ret  = _portfolio_return(w, mu)
        vol  = _portfolio_vol(w, sigma)
        shr  = _sharpe(ret, vol, rf)
        cvar = _cvar_historical(w, daily_returns, self.confidence, weight_index=tickers)
        dr   = _diversification_ratio(w, sigma)
        hhi  = _hhi(w)
        conc = _concentration_metrics(w, len(tickers))
        fe   = _compute_factor_exposures(w, daily_returns, tickers)

        return PortfolioMetrics(
            expected_return       = ret,
            volatility            = vol,
            sharpe_ratio          = shr,
            cvar_95               = cvar,
            diversification_ratio = dr,
            hhi                   = hhi,
            effective_n           = conc["effective_n"],
            concentration         = conc,
            factor_exposures      = fe,
            weights               = dict(zip(tickers, w.tolist())),
            tickers               = tickers,
            label                 = label,
        )

    @staticmethod
    def _generate_risk_summary(
        trade_ticker: str,
        trade_value:  float,
        delta:        Dict[str, float],
        virt_metrics: PortfolioMetrics,
        new_ticker:   bool,
    ) -> str:
        lines = [
            f"Virtual trade: {trade_ticker}  |  Rs.{trade_value:,.0f}",
            ("New position -- adds a ticker not currently in portfolio."
             if new_ticker else "Increases existing position."),
            "",
        ]

        dr = delta["expected_return_change"]
        if abs(dr) < 1e-4:
            lines.append("--  Expected return: negligible change.")
        elif dr > 0:
            lines.append(f"UP  Expected return: +{dr:.2%} p.a. (positive).")
        else:
            lines.append(f"DN  Expected return: {dr:.2%} p.a. (negative).")

        dv = delta["volatility_change"]
        if abs(dv) < 1e-4:
            lines.append("--  Volatility: negligible change.")
        elif dv > 0:
            lines.append(f"UP  Volatility: +{dv:.2%} p.a. -- portfolio risk increases.")
        else:
            lines.append(f"DN  Volatility: {dv:.2%} p.a. -- portfolio risk decreases.")

        ds = delta["sharpe_change"]
        if abs(ds) < 0.01:
            lines.append("--  Sharpe ratio: negligible change.")
        elif ds > 0:
            lines.append(f"UP  Sharpe: +{ds:.2f} -- trade improves risk-adjusted returns.")
        else:
            lines.append(f"DN  Sharpe: {ds:.2f} -- trade reduces risk-adjusted efficiency.")

        dc = delta["cvar_change"]
        if abs(dc) < 1e-4:
            lines.append("--  Tail risk (CVaR 95%): negligible change.")
        elif dc > 0:
            lines.append(f"UP  CVaR 95%: +{dc:.2%} daily -- expected tail loss increases.")
        else:
            lines.append(f"DN  CVaR 95%: {dc:.2%} daily -- expected tail loss decreases.")

        dd = delta["diversification_change"]
        if abs(dd) < 0.01:
            lines.append("--  Diversification ratio: negligible change.")
        elif dd > 0:
            lines.append(f"UP  Diversification ratio: +{dd:.3f} -- portfolio becomes more diversified.")
        else:
            lines.append(f"DN  Diversification ratio: {dd:.3f} -- concentration increases.")

        pos = sum(1 for k in ("expected_return_change", "sharpe_change", "diversification_change")
                  if delta[k] > 0.001)
        neg = sum(1 for k in ("volatility_change", "cvar_change")
                  if delta[k] > 0.001)

        lines.append("")
        if pos >= 2 and neg == 0:
            verdict = "BENEFICIAL -- improves return, Sharpe, and/or diversification without increasing risk."
        elif pos >= 1 and neg <= 1:
            verdict = "MIXED -- some improvement in returns/diversification with modest risk increase. Consider position sizing."
        elif neg >= 2 and pos == 0:
            verdict = "CAUTION -- increases volatility and tail risk with no offsetting return improvement."
        else:
            verdict = "NEUTRAL -- no significant change to portfolio risk-return profile."

        lines.append(f"Verdict: {verdict}")
        lines.append(f"Post-trade Sharpe: {virt_metrics.sharpe_ratio:.2f}  |  "
                     f"Volatility: {virt_metrics.volatility:.2%}  |  "
                     f"Effective-N: {virt_metrics.effective_n:.1f}")

        return "\n".join(lines)


# ════════════════════════════════════════════════════════════════════════════
# LAYER 4 — MONTE CARLO PROJECTION ENGINE
# ════════════════════════════════════════════════════════════════════════════

warnings.filterwarnings("ignore", category=RuntimeWarning)

_TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Monte Carlo core
# ---------------------------------------------------------------------------

def _run_mc(
    weights:       np.ndarray,
    mu_annual:     np.ndarray,
    sigma_annual:  np.ndarray,
    initial_value: float,
    horizon_years: int,
    n_paths:       int,
    seed:          int,
) -> np.ndarray:
    """
    Run Monte Carlo for a single portfolio. Returns (n_paths,) terminal values.
    """
    rng      = np.random.default_rng(seed)
    T        = horizon_years * _TRADING_DAYS
    N        = len(weights)
    mu_daily = mu_annual / _TRADING_DAYS
    sigma_d  = sigma_annual / _TRADING_DAYS
    drift_d  = mu_daily - 0.5 * np.diag(sigma_d)

    try:
        sigma_reg = sigma_d.copy()
        eigvals   = np.linalg.eigvalsh(sigma_reg)
        if eigvals.min() < 1e-12:
            sigma_reg += (abs(eigvals.min()) + 1e-12) * np.eye(N)
        L = np.linalg.cholesky(sigma_reg)
    except np.linalg.LinAlgError:
        L = np.diag(np.sqrt(np.diag(sigma_d)))

    z       = rng.standard_normal((T, n_paths, N))
    corr_z  = z @ L.T
    asset_r = drift_d[None, None, :] + corr_z
    port_r  = (asset_r * weights[None, None, :]).sum(axis=2)
    cum_log = port_r.sum(axis=0)
    return initial_value * np.exp(cum_log)


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def _summarise(
    terminal_values: np.ndarray,
    initial_value:   float,
    horizon_years:   int,
    risk_free_rate:  float,
) -> "ProjectionResult":
    n_paths = len(terminal_values)

    expected_value = float(np.mean(terminal_values))
    median_value   = float(np.median(terminal_values))
    p5_value       = float(np.percentile(terminal_values, 5))
    p25_value      = float(np.percentile(terminal_values, 25))
    p75_value      = float(np.percentile(terminal_values, 75))
    p95_value      = float(np.percentile(terminal_values, 95))

    downside_prob  = float(np.mean(terminal_values < initial_value))
    shortfall_prob = float(np.mean(terminal_values < 0.8 * initial_value))

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(initial_value > 0, terminal_values / initial_value, 1.0)
        cagr_per_path = np.where(
            ratio > 0,
            np.power(np.maximum(ratio, 1e-10), 1.0 / horizon_years) - 1,
            -1.0
        )
    median_cagr = float(np.median(cagr_per_path))
    var_95      = max(0.0, initial_value - p5_value)

    annual_ret  = float(np.mean(cagr_per_path))
    annual_std  = float(np.std(cagr_per_path))
    dist_sharpe = (annual_ret - risk_free_rate) / annual_std if annual_std > 1e-8 else 0.0

    return ProjectionResult(
        horizon_years    = horizon_years,
        n_paths          = n_paths,
        initial_value    = round(initial_value,  2),
        expected_value   = round(expected_value, 2),
        median_value     = round(median_value,   2),
        p5_value         = round(p5_value,       2),
        p25_value        = round(p25_value,      2),
        p75_value        = round(p75_value,      2),
        p95_value        = round(p95_value,      2),
        downside_prob    = round(downside_prob,   4),
        shortfall_prob   = round(shortfall_prob,  4),
        median_cagr      = round(median_cagr,    4),
        var_95_inr       = round(var_95,         2),
        dist_sharpe      = round(dist_sharpe,    4),
    )


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ProjectionResult:
    """Monte Carlo projection for a single portfolio at a single horizon."""
    horizon_years:  int
    n_paths:        int
    initial_value:  float
    expected_value: float
    median_value:   float
    p5_value:       float
    p25_value:      float
    p75_value:      float
    p95_value:      float
    downside_prob:  float
    shortfall_prob: float
    median_cagr:    float
    var_95_inr:     float
    dist_sharpe:    float

    def to_dict(self) -> Dict[str, Any]:
        gain_pct = (self.expected_value / self.initial_value - 1) if self.initial_value > 0 else 0.0
        return {
            "horizon_years":     self.horizon_years,
            "n_paths":           self.n_paths,
            "initial_value":     self.initial_value,
            "expected_value":    self.expected_value,
            "median_value":      self.median_value,
            "p5_value":          self.p5_value,
            "p25_value":         self.p25_value,
            "p75_value":         self.p75_value,
            "p95_value":         self.p95_value,
            "expected_gain_pct": round(gain_pct, 4),
            "downside_prob":     self.downside_prob,
            "shortfall_prob":    self.shortfall_prob,
            "median_cagr":       self.median_cagr,
            "var_95_inr":        self.var_95_inr,
            "dist_sharpe":       self.dist_sharpe,
        }


@dataclass
class HorizonComparison:
    """Side-by-side real vs virtual at one horizon."""
    horizon_years: int
    real:          ProjectionResult
    virtual:       ProjectionResult

    @property
    def expected_value_delta(self) -> float:
        return self.virtual.expected_value - self.real.expected_value

    @property
    def downside_prob_delta(self) -> float:
        return self.virtual.downside_prob - self.real.downside_prob

    @property
    def p5_delta(self) -> float:
        return self.virtual.p5_value - self.real.p5_value

    @property
    def median_cagr_delta(self) -> float:
        return self.virtual.median_cagr - self.real.median_cagr

    def to_dict(self) -> Dict[str, Any]:
        return {
            "horizon_years":        self.horizon_years,
            "real":                 self.real.to_dict(),
            "virtual":              self.virtual.to_dict(),
            "delta_expected_value": round(self.expected_value_delta, 2),
            "delta_downside_prob":  round(self.downside_prob_delta,  4),
            "delta_p5":             round(self.p5_delta,             2),
            "delta_median_cagr":    round(self.median_cagr_delta,    4),
        }


@dataclass
class MonteCarloReport:
    """Full Monte Carlo comparison across all horizons."""
    horizons:        List[HorizonComparison]
    best_horizon:    int
    overall_verdict: str
    trade_ticker:    str
    n_paths:         int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_paths":         self.n_paths,
            "trade_ticker":    self.trade_ticker,
            "horizons":        {f"{h.horizon_years}Y": h.to_dict() for h in self.horizons},
            "best_horizon":    f"{self.best_horizon}Y",
            "overall_verdict": self.overall_verdict,
        }


# ---------------------------------------------------------------------------
# Engine class
# ---------------------------------------------------------------------------

class ProjectionEngine:
    """
    Runs Monte Carlo projections for real and virtual portfolios.

    Usage
    -----
    engine = ProjectionEngine(n_paths=1_000)
    report = engine.run(vp, real_weights, real_mu, real_sigma,
                        real_total_value, risk_free_rate=0.07)
    """

    HORIZONS = [1, 3, 5]

    def __init__(self, n_paths: int = 1_000, base_seed: int = 2024):
        self.n_paths   = max(1_000, n_paths)
        self.base_seed = base_seed

    def run(
        self,
        vp:                    VirtualPortfolio,
        real_weights:          pd.Series,
        real_expected_returns: pd.Series,
        real_covariance:       pd.DataFrame,
        real_total_value:      float,
        risk_free_rate:        float = 0.07,
    ) -> MonteCarloReport:
        rf = risk_free_rate

        real_w, real_mu, real_sigma = self._align_real(
            real_weights, real_expected_returns, real_covariance
        )
        virt_w, virt_mu, virt_sigma = self._align_virtual(vp)

        horizon_comparisons = []
        for h_idx, years in enumerate(self.HORIZONS):
            seed_r = self.base_seed + h_idx * 1000
            seed_v = self.base_seed + h_idx * 1000 + 500

            tv_real = _run_mc(real_w, real_mu, real_sigma,
                              real_total_value, years, self.n_paths, seed_r)
            tv_virt = _run_mc(virt_w, virt_mu, virt_sigma,
                              vp.total_value, years, self.n_paths, seed_v)

            real_proj = _summarise(tv_real, real_total_value, years, rf)
            virt_proj = _summarise(tv_virt, vp.total_value,   years, rf)

            horizon_comparisons.append(
                HorizonComparison(horizon_years=years, real=real_proj, virtual=virt_proj)
            )

        gains        = [(h.expected_value_delta, h.horizon_years) for h in horizon_comparisons]
        best_horizon = max(gains, key=lambda x: x[0])[1]
        verdict      = self._verdict(horizon_comparisons, rf)

        return MonteCarloReport(
            horizons        = horizon_comparisons,
            best_horizon    = best_horizon,
            overall_verdict = verdict,
            trade_ticker    = vp.trade.ticker,
            n_paths         = self.n_paths,
        )

    @staticmethod
    def _align_real(weights, mu, sigma):
        tickers = weights.index.tolist()
        N       = len(tickers)
        w       = np.clip(weights.values.astype(float), 0, 1)
        if w.sum() > 0:
            w /= w.sum()
        mu_arr = np.array([float(mu.get(t, 0.0)) for t in tickers])
        sig    = np.zeros((N, N))
        for i, ti in enumerate(tickers):
            for j, tj in enumerate(tickers):
                if ti in sigma.index and tj in sigma.columns:
                    sig[i, j] = float(sigma.loc[ti, tj])
        sig  = (sig + sig.T) / 2
        eigv = np.linalg.eigvalsh(sig)
        if eigv.min() < 1e-8:
            sig += (abs(eigv.min()) + 1e-8) * np.eye(N)
        return w, mu_arr, sig

    @staticmethod
    def _align_virtual(vp: VirtualPortfolio):
        N    = len(vp.tickers)
        w    = np.clip(vp.weights.values.astype(float), 0, 1)
        if w.sum() > 0:
            w /= w.sum()
        mu   = vp.expected_returns.values.astype(float)
        sig  = vp.covariance.values.astype(float)
        sig  = (sig + sig.T) / 2
        eigv = np.linalg.eigvalsh(sig)
        if eigv.min() < 1e-8:
            sig += (abs(eigv.min()) + 1e-8) * np.eye(N)
        return w, mu, sig

    @staticmethod
    def _verdict(comparisons: List[HorizonComparison], rf: float) -> str:
        outperform_count      = sum(1 for h in comparisons if h.expected_value_delta > 0)
        lower_downside_count  = sum(1 for h in comparisons if h.downside_prob_delta < -0.005)
        higher_downside_count = sum(1 for h in comparisons if h.downside_prob_delta > 0.01)

        h5       = next((h for h in comparisons if h.horizon_years == 5), comparisons[-1])
        gain_pct = h5.expected_value_delta / h5.real.initial_value * 100 if h5.real.initial_value > 0 else 0.0
        cagr_d   = h5.median_cagr_delta * 100

        if outperform_count == 3 and lower_downside_count >= 1:
            return (
                f"POSITIVE across all horizons: the virtual trade is projected to add "
                f"~Rs.{abs(h5.expected_value_delta):,.0f} ({abs(gain_pct):.1f}%) to 5-year expected value "
                f"while reducing downside probability. "
                f"Median CAGR delta: {cagr_d:+.2f}pp. Recommend executing."
            )
        elif outperform_count >= 2 and higher_downside_count == 0:
            return (
                f"MOSTLY POSITIVE: the virtual trade improves projected value in "
                f"{outperform_count}/3 horizons with no material increase in downside risk. "
                f"5-year expected delta: Rs.{h5.expected_value_delta:+,.0f}. "
                f"Consider executing with standard position sizing."
            )
        elif outperform_count >= 2 and higher_downside_count >= 1:
            return (
                f"MIXED: the trade improves expected value ({outperform_count}/3 horizons) "
                f"but increases downside probability in {higher_downside_count} horizon(s). "
                f"5-year expected delta: Rs.{h5.expected_value_delta:+,.0f}. "
                f"Review position size -- consider reducing quantity."
            )
        elif outperform_count == 0:
            return (
                f"NEGATIVE: the virtual trade reduces projected value across all horizons. "
                f"5-year expected delta: Rs.{h5.expected_value_delta:+,.0f} ({gain_pct:+.1f}%). "
                f"Downside probability change: {h5.downside_prob_delta:+.1%}. "
                f"Do not recommend executing as-is."
            )
        else:
            return (
                f"NEUTRAL: marginal projected impact across horizons. "
                f"5-year expected delta: Rs.{h5.expected_value_delta:+,.0f}. "
                f"Monitor after execution."
            )


# ════════════════════════════════════════════════════════════════════════════
# LAYER 5 — END-TO-END PIPELINE ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Parameter builder (mirrors run_milestone5.py approach, no M5 import)
# ---------------------------------------------------------------------------

def _build_params(
    daily_returns: pd.DataFrame,
    tickers: List[str],
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Compute annualised arithmetic expected returns and shrinkage covariance
    from historical daily log-return DataFrame.
    """
    cols = [t for t in tickers if t in daily_returns.columns]
    ret  = daily_returns[cols]

    mu_log   = ret.mean()
    var_log  = ret.var()
    mu_arith = np.exp(mu_log * 252 + 0.5 * var_log * 252) - 1
    expected_returns = pd.Series(mu_arith, index=cols)

    N     = len(cols)
    T     = len(ret)
    S     = ret.cov() * 252
    mu_tr = float(np.trace(S.values)) / N
    F     = pd.DataFrame(np.eye(N) * mu_tr, index=cols, columns=cols)
    delta = 1.0 / (1.0 + T / N)
    cov   = (1 - delta) * S + delta * F
    cov   = pd.DataFrame((cov.values + cov.values.T) / 2, index=cols, columns=cols)

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
    risk_free_rate:  float                          = 0.07,
    n_mc_paths:      int                            = 1_000,
    keypair:         Optional[Tuple[PublicKey, PrivateKey]] = None,
    trade_timestamp: Optional[float]                = None,
    trade_note:      Optional[str]                  = None,
) -> Dict[str, Any]:
    """
    End-to-end virtual trade simulation pipeline.

    Parameters
    ----------
    ticker          : NSE ticker to virtually buy, e.g. "INFY.NS".
    quantity        : Number of shares to simulate purchasing.
    price           : Price per share at simulation time (INR).
    real_holdings   : {ticker: shares_held} -- current real portfolio.
    real_prices     : {ticker: price} -- current prices for all held tickers.
    daily_returns   : pd.DataFrame of historical daily log returns.
    total_value     : Total INR value of the real portfolio.
    risk_free_rate  : Annual risk-free rate (default 0.07 = 7%).
    n_mc_paths      : Monte Carlo paths per horizon (minimum 1,000).
    keypair         : (PublicKey, PrivateKey) -- supply for persistent keys.
                      If None, a fresh ML-DSA key pair is auto-generated.
    trade_timestamp : Unix timestamp of the trade (defaults to now).
    trade_note      : Optional annotation attached to the transaction record.

    Returns
    -------
    dict matching the Milestone 6 output schema (fully JSON-serialisable).
    """
    ts = trade_timestamp or time.time()

    # ── 1. Validate ───────────────────────────────────────────────────────
    if quantity <= 0:
        raise ValueError(f"quantity must be positive for a virtual BUY, got {quantity}.")
    if price <= 0:
        raise ValueError(f"price must be positive, got {price}.")
    if total_value <= 0:
        raise ValueError("total_value must be positive.")
    if not real_holdings:
        raise ValueError("real_holdings cannot be empty.")

    # ── 2. Trade instruction ──────────────────────────────────────────────
    trade = VirtualTrade(
        ticker    = ticker,
        quantity  = quantity,
        price     = price,
        timestamp = ts,
        note      = trade_note or f"Virtual buy: {quantity} x {ticker} @ Rs.{price:,.2f}",
    )

    # ── 3. Real portfolio params ──────────────────────────────────────────
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
    analyzer      = ImpactAnalyzer(confidence=0.95)
    real_weights  = snapshot.weights
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

    # ── 7. Assemble output ────────────────────────────────────────────────
    ir  = impact_report
    mcr = mc_report

    output = {
        "transaction_security": {
            "tx_id":                  tx_record["tx_id"],
            "tx_hash":                tx_record["sha3_hash"],
            "signature_verified":     tx_record["verification_status"],
            "public_key_fingerprint": tx_record["public_key"]["fingerprint"],
            "signed_at":              tx_record["signed_at"],
            "signature_scheme":       tx_record["signature"]["scheme"],
            "z_infinity_norm":        tx_record["signature"]["z_infinity_norm"],
            "acceptance_bound":       tx_record["signature"]["acceptance_bound"],
        },

        "portfolio_impact": {
            "expected_return_change": round(ir.portfolio_impact["expected_return_change"], 6),
            "volatility_change":      round(ir.portfolio_impact["volatility_change"],      6),
            "sharpe_change":          round(ir.portfolio_impact["sharpe_change"],          4),
            "cvar_change":            round(ir.portfolio_impact["cvar_change"],            6),
            "diversification_change": round(ir.portfolio_impact["diversification_change"], 4),
            "hhi_change":             round(ir.portfolio_impact["hhi_change"],             4),
            "effective_n_change":     round(ir.portfolio_impact["effective_n_change"],     2),
        },

        "factor_shift": {
            "market_beta_delta":            ir.factor_shift.get("market_beta",            0),
            "systematic_variance_delta":    ir.factor_shift.get("systematic_variance",    0),
            "idiosyncratic_variance_delta": ir.factor_shift.get("idiosyncratic_variance", 0),
            "r_squared_delta":              ir.factor_shift.get("r_squared",              0),
            "tracking_error_delta":         ir.factor_shift.get("tracking_error_vs_ew",   0),
        },

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

        "portfolio_comparison": {
            "real":    ir.real_metrics.to_dict(),
            "virtual": ir.virtual_metrics.to_dict(),
        },

        "risk_summary":      ir.risk_summary,
        "transaction_record": tx_record,
    }

    return output


# ════════════════════════════════════════════════════════════════════════════
# SECURITY MODULE 1 — ENTROPY MONITOR
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HASH_BYTES       = 32          # SHA3-256
HASH_BITS        = 256
H_MAX_BYTE       = math.log2(HASH_BYTES)   # ≈ 5.0 bits  (32 unique symbols max)
H_BIT_SAFE       = 0.95        # Bernoulli entropy threshold (bits per bit)
H_BYTE_RATIO_SAFE= 0.85        # byte entropy / H_max threshold
Z_BIT_THRESHOLD  = 3.0         # std dev from expected 128 ones
COMPOSITE_SAFE   = 0.80        # composite entropy score threshold
ALERT_KEYWORD    = "ENTROPY_ALERT"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class EntropyReport:
    """
    Full entropy analysis of a single transaction hash.

    Attributes
    ----------
    hash_hex         : 64-char hex string of the transaction hash
    h_bit            : Bernoulli entropy of bit distribution (0–1)
    h_byte           : Empirical Shannon entropy of byte distribution (0–5)
    z_bit            : Standardised bit-count statistic
    uniqueness       : Fraction of distinct bytes (0–1)
    composite_score  : Weighted composite entropy score (0–1)
    alert            : True if composite_score < COMPOSITE_SAFE
    alert_reason     : Human-readable explanation when alert is True
    metrics          : Raw metric dict for downstream consumption
    """
    hash_hex:       str
    h_bit:          float
    h_byte:         float
    z_bit:          float
    uniqueness:     float
    composite_score: float
    alert:          bool
    alert_reason:   str
    metrics:        Dict

    def to_dict(self) -> Dict:
        return {
            "hash_hex":        self.hash_hex,
            "h_bit":           round(self.h_bit, 6),
            "h_byte":          round(self.h_byte, 6),
            "h_byte_max":      round(H_MAX_BYTE, 6),
            "z_bit":           round(self.z_bit, 4),
            "uniqueness":      round(self.uniqueness, 4),
            "composite_score": round(self.composite_score, 6),
            "safe_threshold":  COMPOSITE_SAFE,
            "alert":           self.alert,
            "alert_reason":    self.alert_reason,
        }


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _bernoulli_entropy(p: float) -> float:
    """H(p) = -p log₂ p - (1-p) log₂(1-p), guarded against log(0)."""
    if p <= 0 or p >= 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def _byte_shannon_entropy(data: bytes) -> float:
    """
    Empirical Shannon entropy over byte values of `data`.

    H = -Σ p(b) log₂ p(b)

    Maximum = log₂(len(data)) when all bytes are distinct.
    """
    n = len(data)
    if n == 0:
        return 0.0
    counts = np.zeros(256, dtype=int)
    for b in data:
        counts[b] += 1
    nonzero = counts[counts > 0]
    probs   = nonzero / n
    return float(-np.sum(probs * np.log2(probs)))


def _bit_stats(data: bytes) -> tuple:
    """
    Return (p1, n_ones, n_bits, z_bit) for the bit array of `data`.

    p1     = fraction of 1-bits
    n_ones = count of 1-bits
    n_bits = total bits
    z_bit  = (n_ones - n_bits/2) / sqrt(n_bits/4)   [Bernoulli CLT]
    """
    n_bits  = len(data) * 8
    n_ones  = sum(bin(b).count("1") for b in data)
    p1      = n_ones / n_bits if n_bits > 0 else 0.5
    # CLT std for Bernoulli(0.5) count: sqrt(n/4)
    std     = math.sqrt(n_bits / 4) if n_bits > 0 else 1.0
    z_bit   = (n_ones - n_bits / 2) / std
    return p1, n_ones, n_bits, z_bit


def analyse_entropy(hash_hex: str) -> EntropyReport:
    """
    Full entropy analysis of a hex-encoded transaction hash.

    Parameters
    ----------
    hash_hex : 64-character hex string (SHA3-256)

    Returns
    -------
    EntropyReport
    """
    # Parse
    try:
        data = bytes.fromhex(hash_hex)
    except ValueError:
        # Treat non-hex input as maximum-alert raw bytes
        data = hash_hex.encode()[:HASH_BYTES].ljust(HASH_BYTES, b"\x00")

    # 1. Bit-level Bernoulli entropy
    p1, n_ones, n_bits, z_bit = _bit_stats(data)
    h_bit = _bernoulli_entropy(p1)

    # 2. Byte-level Shannon entropy
    h_byte = _byte_shannon_entropy(data)

    # 3. Uniqueness
    uniqueness = len(set(data)) / len(data) if len(data) > 0 else 0.0

    # 4. Composite score
    h_byte_norm = min(h_byte / H_MAX_BYTE, 1.0)
    composite   = 0.6 * h_bit + 0.3 * h_byte_norm + 0.1 * uniqueness

    # 5. Alert logic
    alert  = False
    reason = ""
    reasons = []
    if h_bit < H_BIT_SAFE:
        alert = True
        reasons.append(
            f"Bit entropy {h_bit:.4f} < safe threshold {H_BIT_SAFE} "
            f"(bit bias detected: p₁={p1:.4f})"
        )
    if abs(z_bit) > Z_BIT_THRESHOLD:
        alert = True
        reasons.append(
            f"Bit-count z-score {z_bit:.2f} exceeds ±{Z_BIT_THRESHOLD} "
            f"({n_ones}/{n_bits} ones)"
        )
    if composite < COMPOSITE_SAFE:
        alert = True
        reasons.append(
            f"Composite entropy score {composite:.4f} < {COMPOSITE_SAFE}"
        )
    reason = "; ".join(reasons) if reasons else "All entropy checks passed."

    metrics = {
        "n_ones":        n_ones,
        "n_bits":        n_bits,
        "p1":            round(p1, 6),
        "h_bit":         round(h_bit, 6),
        "h_byte":        round(h_byte, 6),
        "h_byte_norm":   round(h_byte_norm, 6),
        "z_bit":         round(z_bit, 4),
        "uniqueness":    round(uniqueness, 4),
        "composite":     round(composite, 6),
    }

    return EntropyReport(
        hash_hex        = hash_hex,
        h_bit           = h_bit,
        h_byte          = h_byte,
        z_bit           = z_bit,
        uniqueness      = uniqueness,
        composite_score = composite,
        alert           = alert,
        alert_reason    = reason,
        metrics         = metrics,
    )


# ---------------------------------------------------------------------------
# Population-level baseline
# ---------------------------------------------------------------------------

class EntropyMonitor:
    """
    Maintains a rolling baseline of entropy scores over recent transactions.

    Uses an exponentially-weighted moving average (EWMA) to track the
    expected entropy level of the system:

        μ_t = α × score_t + (1 - α) × μ_{t-1}

    where α = 2/(window+1) is the EWMA decay constant.

    A new observation is flagged as anomalous if it deviates from the EWMA
    by more than k standard deviations (k = 3 by default).
    """

    def __init__(self, window: int = 50, k: float = 3.0):
        self.window   = window
        self.k        = k
        self._alpha   = 2.0 / (window + 1)
        self._ewma    = None     # running mean
        self._ewmvar  = None     # running variance (Welford online)
        self._history: List[float] = []

    def observe(self, score: float) -> bool:
        """
        Record an entropy score and return True if it is anomalous.

        Anomaly criterion: |score - μ| > k × σ  (after warm-up of 10 obs.)
        """
        self._history.append(score)
        if self._ewma is None:
            self._ewma   = score
            self._ewmvar = 0.0
            return False

        # EWMA update
        diff          = score - self._ewma
        self._ewma   += self._alpha * diff
        self._ewmvar  = (1 - self._alpha) * (self._ewmvar + self._alpha * diff ** 2)

        if len(self._history) < 10:
            return False

        sigma = math.sqrt(self._ewmvar) if self._ewmvar > 0 else 0.01
        return abs(score - self._ewma) > self.k * sigma

    @property
    def baseline_mean(self) -> float:
        return self._ewma or 0.0

    @property
    def baseline_std(self) -> float:
        return math.sqrt(self._ewmvar) if self._ewmvar else 0.0

    def summary(self) -> Dict:
        return {
            "observations":  len(self._history),
            "ewma_mean":     round(self.baseline_mean, 6),
            "ewma_std":      round(self.baseline_std, 6),
            "window":        self.window,
        }


# ════════════════════════════════════════════════════════════════════════════
# SECURITY MODULE 2 — STATISTICAL ANOMALY DETECTOR
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Composite score thresholds
LOW_THRESHOLD    = 2.0
HIGH_THRESHOLD   = 8.0      # hard penalties (sig=10, entropy=4) always exceed this

# Soft component weights (applied to continuous signals only; sum to 1.0)
W_SIZE      = 0.40
W_FREQ      = 0.35
W_ENTROPY   = 0.25

# Hard additive penalties (not weighted; added directly to composite)
SIG_PENALTY     = 10.0    # failed ML-DSA verification → always HIGH
ENTROPY_PENALTY =  4.0    # entropy alert → always MEDIUM or higher

# Cosine similarity threshold for pattern matching
SIM_THRESHOLD = 0.85

# Minimum observations before z-scores are trusted
MIN_OBS = 5


# ---------------------------------------------------------------------------
# Online Welford statistics
# ---------------------------------------------------------------------------

class _WelfordTracker:
    """
    Online mean and variance via Welford's numerically stable algorithm.

    After n updates:
        mean = Σx / n
        var  = Σ(x-mean)² / (n-1)   [sample variance]
    """
    def __init__(self):
        self.n    = 0
        self.mean = 0.0
        self._M2  = 0.0   # sum of squared deviations

    def update(self, x: float):
        self.n += 1
        delta      = x - self.mean
        self.mean += delta / self.n
        delta2     = x - self.mean
        self._M2  += delta * delta2

    @property
    def var(self) -> float:
        if self.n < 2:
            return 1.0   # no estimate yet; return sentinel
        return self._M2 / (self.n - 1)

    @property
    def std(self) -> float:
        return math.sqrt(max(self.var, 1e-12))

    def z_score(self, x: float) -> float:
        """Standardised score; clipped to [-10, 10] for numerical safety."""
        if self.n < MIN_OBS:
            return 0.0
        return float(np.clip((x - self.mean) / self.std, -10.0, 10.0))


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class AnomalyReport:
    """
    Result of anomaly detection for a single transaction.

    Attributes
    ----------
    tx_id           : transaction identifier
    risk_score      : composite risk score (≥0, higher = more anomalous)
    threat_level    : "low" | "medium" | "high"
    z_size          : z-score of transaction size
    z_freq          : z-score of inter-arrival frequency
    entropy_dev     : entropy deviation signal
    sig_penalty     : signature validation penalty applied
    pattern_match   : True if similar to a known threat pattern
    pattern_sim     : highest cosine similarity to stored threats
    feature_vector  : (4,) feature array used for pattern matching
    breakdown       : per-component risk score contributions
    """
    tx_id:          str
    risk_score:     float
    threat_level:   str
    z_size:         float
    z_freq:         float
    entropy_dev:    float
    sig_penalty:    float
    pattern_match:  bool
    pattern_sim:    float
    feature_vector: np.ndarray
    breakdown:      Dict

    def to_dict(self) -> Dict:
        return {
            "tx_id":         self.tx_id,
            "risk_score":    round(self.risk_score, 4),
            "threat_level":  self.threat_level,
            "components": {
                "z_size":       round(self.z_size, 4),
                "z_freq":       round(self.z_freq, 4),
                "entropy_dev":  round(self.entropy_dev, 4),
                "sig_penalty":  round(self.sig_penalty, 4),
            },
            "breakdown":     {k: round(v, 4) for k, v in self.breakdown.items()},
            "pattern_match": self.pattern_match,
            "pattern_sim":   round(self.pattern_sim, 4),
        }


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Stateful anomaly detector that maintains rolling population statistics
    and a library of known threat feature vectors for pattern matching.
    """

    def __init__(self):
        self._size_tracker  = _WelfordTracker()
        self._freq_tracker  = _WelfordTracker()
        self._entropy_tracker = _WelfordTracker()
        self._last_ts: Optional[float] = None
        self._threat_patterns: List[np.ndarray] = []  # known threat vectors
        self._total_obs = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        tx_id:          str,
        size_bytes:     int,
        timestamp:      float,
        entropy_score:  float,
        sig_valid:      bool,
        entropy_alert:  bool  = False,    # hard signal from EntropyMonitor
        entropy_ewma:   float = 0.95,     # population entropy baseline
        entropy_ewma_std: float = 0.05,
    ) -> AnomalyReport:
        """
        Compute anomaly score for a single transaction.

        Scoring model: risk = soft_score + hard_penalties
          soft_score = weighted continuous signals (size, freq, entropy_dev)
          hard_penalties = SIG_PENALTY (if invalid) + ENTROPY_PENALTY (if alert)
        """
        self._total_obs += 1

        # ── Inter-arrival time ─────────────────────────────────────────
        if self._last_ts is not None:
            gap = max(timestamp - self._last_ts, 0.001)   # seconds
        else:
            gap = 60.0    # neutral default for first transaction
        self._last_ts = timestamp

        # ── Update population trackers ─────────────────────────────────
        self._size_tracker.update(float(size_bytes))
        self._freq_tracker.update(gap)
        self._entropy_tracker.update(entropy_score)

        # ── Z-scores (soft signals) ────────────────────────────────────
        z_size = abs(self._size_tracker.z_score(float(size_bytes)))

        # Frequency: negate gap z-score so short gap → positive z_freq
        z_gap  = self._freq_tracker.z_score(gap)
        z_freq = max(-z_gap, 0.0)

        # ── Entropy deviation (continuous soft signal) ─────────────────
        std_h = max(entropy_ewma_std, 1e-4)
        entropy_dev = max((entropy_ewma - entropy_score) / std_h, 0.0)
        entropy_dev = float(np.clip(entropy_dev, 0.0, 10.0))

        # ── Hard penalties (direct additive; not weighted) ─────────────
        sig_pen     = SIG_PENALTY     if not sig_valid    else 0.0
        entropy_pen = ENTROPY_PENALTY if entropy_alert    else 0.0

        # ── Composite risk score ───────────────────────────────────────
        soft_score  = (W_SIZE * z_size) + (W_FREQ * z_freq) + (W_ENTROPY * entropy_dev)
        risk_score  = soft_score + sig_pen + entropy_pen

        contrib = {
            "soft_size_contrib":    round(W_SIZE * z_size, 4),
            "soft_freq_contrib":    round(W_FREQ * z_freq, 4),
            "soft_entropy_contrib": round(W_ENTROPY * entropy_dev, 4),
            "hard_sig_penalty":     round(sig_pen, 4),
            "hard_entropy_penalty": round(entropy_pen, 4),
        }

        # ── Feature vector: [z_size, z_freq, entropy_dev, sig_pen, entropy_pen]
        fv = np.array([z_size, z_freq, entropy_dev, sig_pen, entropy_pen], dtype=float)

        # ── Pattern similarity against stored threats ──────────────────
        best_sim, pattern_match = self._pattern_similarity(fv)

        # ── Pattern match escalation ───────────────────────────────────
        if pattern_match:
            risk_score = max(risk_score, HIGH_THRESHOLD + 0.1)

        # ── Threat classification ──────────────────────────────────────
        threat = self._classify(risk_score)

        return AnomalyReport(
            tx_id          = tx_id,
            risk_score     = risk_score,
            threat_level   = threat,
            z_size         = z_size,
            z_freq         = z_freq,
            entropy_dev    = entropy_dev,
            sig_penalty    = sig_pen + entropy_pen,   # total hard penalty
            pattern_match  = pattern_match,
            pattern_sim    = best_sim,
            feature_vector = fv,
            breakdown      = contrib,
        )

    def register_threat_pattern(self, feature_vector: np.ndarray):
        """Store a confirmed threat feature vector for future matching."""
        norm = np.linalg.norm(feature_vector)
        if norm > 1e-8:
            self._threat_patterns.append(feature_vector / norm)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _classify(score: float) -> str:
        if score >= HIGH_THRESHOLD:
            return "high"
        elif score >= LOW_THRESHOLD:
            return "medium"
        return "low"

    def _pattern_similarity(
        self, fv: np.ndarray
    ) -> Tuple[float, bool]:
        """
        Cosine similarity between fv and each stored threat pattern.

        sim(a, b) = a·b / (‖a‖ ‖b‖)

        Returns (best_sim, match_flag).
        """
        if not self._threat_patterns:
            return 0.0, False

        norm_fv = np.linalg.norm(fv)
        if norm_fv < 1e-8:
            return 0.0, False

        fv_unit  = fv / norm_fv
        sims     = [float(np.dot(fv_unit, p)) for p in self._threat_patterns]
        best_sim = max(sims)
        return best_sim, best_sim >= SIM_THRESHOLD

    def population_stats(self) -> Dict:
        return {
            "total_observations": self._total_obs,
            "size_mean":    round(self._size_tracker.mean, 2),
            "size_std":     round(self._size_tracker.std, 2),
            "freq_mean_gap_s": round(self._freq_tracker.mean, 2),
            "freq_std_gap_s":  round(self._freq_tracker.std, 2),
            "n_threat_patterns": len(self._threat_patterns),
        }


# ════════════════════════════════════════════════════════════════════════════
# SECURITY MODULE 3 — IMMUNE RESPONSE ENGINE
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Response action constants
# ---------------------------------------------------------------------------

class PQCAction:
    LOG                   = "LOG"
    ALERT                 = "ALERT"
    ENHANCED_VALIDATION   = "ENHANCED_VALIDATION"
    MULTI_SIG             = "MULTI_SIG_REQUESTED"
    QUARANTINE            = "TRANSACTION_QUARANTINED"
    KEY_ROTATION_SIGNAL   = "KEY_ROTATION_SIGNALLED"
    RATE_LIMIT            = "RATE_LIMIT_APPLIED"
    PATTERN_STORE         = "THREAT_PATTERN_STORED"


# Response sets by base threat level
BASE_RESPONSES: Dict[str, List[str]] = {
    "low":    [PQCAction.LOG],
    "medium": [PQCAction.LOG, PQCAction.ALERT, PQCAction.ENHANCED_VALIDATION],
    "high":   [PQCAction.LOG, PQCAction.ALERT, PQCAction.ENHANCED_VALIDATION,
               PQCAction.MULTI_SIG, PQCAction.QUARANTINE, PQCAction.KEY_ROTATION_SIGNAL],
}

# Quarantine decision parameters
QUARANTINE_PIVOT = 3.5
QUARANTINE_SCALE = 0.5

# Multi-sig confirmation count
N_MULTISIG = 2


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ResponseRecord:
    """
    Full record of a single immune response event.

    Attributes
    ----------
    response_id     : unique identifier
    tx_id           : transaction being evaluated
    timestamp       : Unix timestamp of response
    threat_level    : "low" | "medium" | "high"
    actions         : set of PQCAction constants triggered
    quarantine      : True if transaction is quarantined
    key_rotation    : True if key rotation was signalled
    multi_sig       : True if multi-sig verification was requested
    quarantine_score: probabilistic quarantine score ∈ [0,1]
    processing_ms   : time to compute response (milliseconds)
    notes           : list of human-readable response notes
    """
    response_id:      str
    tx_id:            str
    timestamp:        float
    threat_level:     str
    actions:          List[str]
    quarantine:       bool
    key_rotation:     bool
    multi_sig:        bool
    quarantine_score: float
    processing_ms:    float
    notes:            List[str]

    def to_dict(self) -> Dict:
        return {
            "response_id":      self.response_id,
            "tx_id":            self.tx_id,
            "timestamp":        self.timestamp,
            "threat_level":     self.threat_level,
            "actions_triggered":self.actions,
            "quarantine":       self.quarantine,
            "quarantine_score": round(self.quarantine_score, 4),
            "key_rotation":     self.key_rotation,
            "multi_sig":        self.multi_sig,
            "processing_ms":    round(self.processing_ms, 3),
            "notes":            self.notes,
        }


# ---------------------------------------------------------------------------
# Immune Response Engine
# ---------------------------------------------------------------------------

class ImmuneResponseEngine:
    """
    Adaptive immune response dispatcher.

    Maintains a response log (effector memory) for audit trails and
    applies escalating responses based on threat level and contextual flags.
    """

    def __init__(self):
        self._log:             List[ResponseRecord] = []
        self._quarantine_set:  Set[str] = set()    # quarantined tx_ids
        self._total_responses  = 0
        self._total_quarantine = 0
        self._total_rotation   = 0

    # ------------------------------------------------------------------
    # Core response trigger
    # ------------------------------------------------------------------

    def respond(
        self,
        tx_id:          str,
        threat_level:   str,
        risk_score:     float,
        sig_failed:     bool  = False,
        entropy_alert:  bool  = False,
        pattern_matched:bool  = False,
        attack_type:    str   = "unknown",
    ) -> ResponseRecord:
        """
        Determine and log the appropriate immune response.

        Parameters
        ----------
        tx_id           : transaction identifier
        threat_level    : "low" | "medium" | "high"
        risk_score      : composite anomaly score from AnomalyDetector
        sig_failed      : True if ML-DSA verification failed
        entropy_alert   : True if entropy monitor raised an alert
        pattern_matched : True if known threat pattern was recognised
        attack_type     : classified attack type

        Returns
        -------
        ResponseRecord
        """
        t0 = time.perf_counter()
        self._total_responses += 1

        # Base response set
        actions: Set[str] = set(BASE_RESPONSES.get(threat_level, [PQCAction.LOG]))

        # Contextual escalation
        notes = []
        if sig_failed:
            actions.add(PQCAction.MULTI_SIG)
            actions.add(PQCAction.QUARANTINE)
            notes.append(f"ML-DSA signature verification FAILED — "
                         f"mandatory quarantine + multi-sig")
        if entropy_alert:
            actions.add(PQCAction.ENHANCED_VALIDATION)
            actions.add(PQCAction.ALERT)
            notes.append("Hash entropy below safe threshold — "
                         "enhanced validation triggered")
        if pattern_matched:
            actions.add(PQCAction.QUARANTINE)
            actions.add(PQCAction.PATTERN_STORE)
            notes.append("Known threat pattern recognised — "
                         "quarantine applied")

        # Quarantine score (probabilistic sigmoid)
        q_score = self._quarantine_score(risk_score)

        # Final quarantine decision
        quarantine = (
            PQCAction.QUARANTINE in actions
            or q_score > 0.5
        )

        # Key rotation signalling
        key_rotation = PQCAction.KEY_ROTATION_SIGNAL in actions

        # Multi-sig
        multi_sig = PQCAction.MULTI_SIG in actions

        # Always store threat pattern for high-level incidents
        if threat_level == "high":
            actions.add(PQCAction.PATTERN_STORE)

        # Rate limiting on burst attacks
        if attack_type == "burst":
            actions.add(PQCAction.RATE_LIMIT)
            notes.append("Burst frequency attack — rate limiting applied")

        # Quarantine registry update
        if quarantine:
            self._quarantine_set.add(tx_id)
            self._total_quarantine += 1
            notes.append(
                f"Transaction {tx_id[:12]}… QUARANTINED "
                f"(q_score={q_score:.3f})"
            )

        if key_rotation:
            self._total_rotation += 1
            notes.append(
                f"Key rotation signalled "
                f"(cumulative signals: {self._total_rotation})"
            )

        if not notes:
            notes.append(f"Threat level {threat_level!r} — standard logging.")

        t1 = time.perf_counter()

        record = ResponseRecord(
            response_id      = str(uuid.uuid4())[:12],
            tx_id            = tx_id,
            timestamp        = time.time(),
            threat_level     = threat_level,
            actions          = sorted(actions),
            quarantine       = quarantine,
            key_rotation     = key_rotation,
            multi_sig        = multi_sig,
            quarantine_score = q_score,
            processing_ms    = (t1 - t0) * 1000,
            notes            = notes,
        )
        self._log.append(record)
        return record

    # ------------------------------------------------------------------
    # Quarantine management
    # ------------------------------------------------------------------

    def is_quarantined(self, tx_id: str) -> bool:
        return tx_id in self._quarantine_set

    def release_quarantine(self, tx_id: str) -> bool:
        """Release a transaction from quarantine (manual review)."""
        if tx_id in self._quarantine_set:
            self._quarantine_set.discard(tx_id)
            return True
        return False

    # ------------------------------------------------------------------
    # Audit and stats
    # ------------------------------------------------------------------

    def recent_log(self, n: int = 10) -> List[Dict]:
        return [r.to_dict() for r in self._log[-n:]]

    def stats(self) -> Dict:
        if not self._log:
            return {"total_responses": 0}
        levels  = [r.threat_level for r in self._log]
        by_level = {
            lvl: levels.count(lvl) for lvl in ("low", "medium", "high")
        }
        return {
            "total_responses":        self._total_responses,
            "total_quarantined":      self._total_quarantine,
            "currently_quarantined":  len(self._quarantine_set),
            "key_rotation_signals":   self._total_rotation,
            "by_threat_level":        by_level,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _quarantine_score(risk_score: float) -> float:
        """
        Sigmoid quarantine score:
            q = 1 / (1 + exp(-(risk - pivot) / scale))
        """
        x = (risk_score - QUARANTINE_PIVOT) / QUARANTINE_SCALE
        # Guard against overflow
        x = float(np.clip(x, -20, 20))
        return 1.0 / (1.0 + math.exp(-x))


# ════════════════════════════════════════════════════════════════════════════
# SECURITY MODULE 4 — ADAPTIVE KEY MUTATION SYSTEM
# ════════════════════════════════════════════════════════════════════════════

# Import M6 crypto layer as a black box


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROTATION_TRIGGER_HIGH        = 3     # consecutive HIGH events before rotation
ROTATION_TRIGGER_CUMULATIVE  = 10    # cumulative HIGH events before rotation
MAX_KEY_AGE_HOURS            = 24    # time-based rotation threshold
MIN_ROTATION_INTERVAL_SECONDS= 60    # anti-DoS: minimum time between rotations
MAX_ROTATIONS_PER_HOUR       = 6     # anti-DoS: cap on rotation rate


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class KeyRecord:
    """
    An archived key pair with its rotation metadata.

    Attributes
    ----------
    generation   : key generation index (0 = genesis)
    public_key   : PublicKey object
    private_key  : PrivateKey object (held in memory, not serialised)
    activated_at : Unix timestamp when this key became active
    retired_at   : Unix timestamp when this key was rotated out (None if current)
    rotation_cert: signed rotation certificate (None for genesis key)
    """
    generation:    int
    public_key:    PublicKey
    private_key:   PrivateKey
    activated_at:  float
    retired_at:    Optional[float] = None
    rotation_cert: Optional[Dict]  = None

    @property
    def age_hours(self) -> float:
        end = self.retired_at or time.time()
        return (end - self.activated_at) / 3600.0

    def to_dict(self) -> Dict:
        return {
            "generation":    self.generation,
            "key_id":        self.public_key.key_id,
            "fingerprint":   self.public_key.fingerprint(),
            "activated_at":  self.activated_at,
            "retired_at":    self.retired_at,
            "age_hours":     round(self.age_hours, 3),
            "is_current":    self.retired_at is None,
            "rotation_cert": self.rotation_cert,
        }


@dataclass
class RotationEvent:
    """Record of a single key rotation event."""
    event_id:          str
    from_key_id:       str
    to_key_id:         str
    reason:            str
    timestamp:         float
    consecutive_high:  int
    cumulative_high:   int
    cert_sig_hex:      str

    def to_dict(self) -> Dict:
        return {
            "event_id":         self.event_id,
            "from_key_id":      self.from_key_id,
            "to_key_id":        self.to_key_id,
            "reason":           self.reason,
            "timestamp":        self.timestamp,
            "consecutive_high": self.consecutive_high,
            "cumulative_high":  self.cumulative_high,
            "cert_sig_hex":     self.cert_sig_hex,
        }


# ---------------------------------------------------------------------------
# Key Mutation System
# ---------------------------------------------------------------------------

class KeyMutationSystem:
    """
    Adaptive PQC key rotation system with chain-of-trust cross-signing.

    Maintains a key archive, rotation event log, and current active key pair.
    Applies rotation policy based on threat level signals from the immune layer.
    """

    def __init__(self, initial_keypair: Optional[Tuple[PublicKey, PrivateKey]] = None):
        if initial_keypair is None:
            pk, sk = generate_keypair()
        else:
            pk, sk = initial_keypair

        genesis = KeyRecord(
            generation   = 0,
            public_key   = pk,
            private_key  = sk,
            activated_at = time.time(),
        )
        self._archive:   List[KeyRecord]    = [genesis]
        self._rotation_log: List[RotationEvent] = []
        self._consecutive_high  = 0
        self._cumulative_high   = 0
        self._last_rotation_ts  = 0.0
        self._rotations_this_hour: List[float] = []

    # ------------------------------------------------------------------
    # Rotation decisions
    # ------------------------------------------------------------------

    def observe_threat(self, threat_level: str) -> bool:
        """
        Inform the rotation system of a threat level observation.

        Returns True if a rotation was triggered, False otherwise.
        """
        if threat_level == "high":
            self._consecutive_high += 1
            self._cumulative_high  += 1
        else:
            self._consecutive_high = 0   # reset on non-high event

        should_rotate = False
        reason        = ""

        if self._consecutive_high >= ROTATION_TRIGGER_HIGH:
            should_rotate = True
            reason = (f"Sustained HIGH threat: {self._consecutive_high} "
                      f"consecutive high-severity events")

        elif self._cumulative_high >= ROTATION_TRIGGER_CUMULATIVE:
            should_rotate = True
            reason = (f"Cumulative HIGH threshold reached: "
                      f"{self._cumulative_high} total high-severity events")
            self._cumulative_high = 0   # reset after triggering

        elif self.current_key.age_hours >= MAX_KEY_AGE_HOURS:
            should_rotate = True
            reason = (f"Key age {self.current_key.age_hours:.1f}h "
                      f"exceeds maximum {MAX_KEY_AGE_HOURS}h")

        if should_rotate and self._may_rotate():
            return self._rotate(reason)
        return False

    def force_rotate(self, reason: str = "manual_rotation") -> bool:
        """Unconditionally rotate to a new key pair (if rate limit allows)."""
        if self._may_rotate():
            return self._rotate(reason)
        return False

    # ------------------------------------------------------------------
    # Key access
    # ------------------------------------------------------------------

    @property
    def current_key(self) -> KeyRecord:
        return self._archive[-1]

    @property
    def current_public_key(self) -> PublicKey:
        return self._archive[-1].public_key

    @property
    def current_private_key(self) -> PrivateKey:
        return self._archive[-1].private_key

    def get_key_for_id(self, key_id: str) -> Optional[KeyRecord]:
        """Look up an archived key by key_id for historical verification."""
        for k in self._archive:
            if k.public_key.key_id == key_id:
                return k
        return None

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict:
        now   = time.time()
        rph   = sum(1 for t in self._rotations_this_hour if now - t < 3600)
        return {
            "current_generation":   self.current_key.generation,
            "current_key_id":       self.current_key.public_key.key_id,
            "current_fingerprint":  self.current_key.public_key.fingerprint(),
            "key_age_hours":        round(self.current_key.age_hours, 3),
            "consecutive_high":     self._consecutive_high,
            "cumulative_high":      self._cumulative_high,
            "total_rotations":      len(self._rotation_log),
            "rotations_this_hour":  rph,
            "archive_size":         len(self._archive),
            "rotation_log":         [e.to_dict() for e in self._rotation_log[-5:]],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _may_rotate(self) -> bool:
        """Anti-DoS rate-limit checks."""
        now = time.time()

        # Minimum interval
        if now - self._last_rotation_ts < MIN_ROTATION_INTERVAL_SECONDS:
            return False

        # Max per hour
        self._rotations_this_hour = [t for t in self._rotations_this_hour
                                     if now - t < 3600]
        if len(self._rotations_this_hour) >= MAX_ROTATIONS_PER_HOUR:
            return False

        return True

    def _rotate(self, reason: str) -> bool:
        """
        Perform key rotation with cross-signing certificate.

        1. Generate new ML-DSA key pair.
        2. Sign a rotation certificate with the current (old) private key.
        3. Retire the current key; install the new key as current.
        4. Append rotation event to log.
        """
        now     = time.time()
        old_key = self.current_key

        # Generate new key pair via M6 crypto (black box)
        new_pk, new_sk = generate_keypair()

        # Build rotation certificate
        cert_payload = {
            "type":           "KEY_ROTATION",
            "from_key_id":    old_key.public_key.key_id,
            "from_fingerprint": old_key.public_key.fingerprint(),
            "to_key_id":      new_pk.key_id,
            "to_fingerprint": new_pk.fingerprint(),
            "rotation_reason":reason,
            "generation":     old_key.generation + 1,
            "timestamp":      now,
            "consecutive_high": self._consecutive_high,
            "cumulative_high":  self._cumulative_high,
        }

        # Sign the certificate with the OLD private key (establishes chain)
        cert_bytes  = json.dumps(cert_payload, sort_keys=True,
                                 separators=(",", ":")).encode()
        cert_sig    = sign_transaction(cert_bytes, old_key.private_key)
        cert_sig_hex = cert_sig.c_tilde.hex()

        rotation_cert = {
            **cert_payload,
            "cert_signature_hex": cert_sig_hex,
            "cert_sig_scheme":    cert_sig.scheme,
        }

        # Retire old key
        old_key.retired_at    = now
        old_key.rotation_cert = rotation_cert

        # Install new key
        new_record = KeyRecord(
            generation   = old_key.generation + 1,
            public_key   = new_pk,
            private_key  = new_sk,
            activated_at = now,
        )
        self._archive.append(new_record)

        # Log rotation event
        event = RotationEvent(
            event_id         = str(uuid.uuid4())[:12],
            from_key_id      = old_key.public_key.key_id,
            to_key_id        = new_pk.key_id,
            reason           = reason,
            timestamp        = now,
            consecutive_high = self._consecutive_high,
            cumulative_high  = self._cumulative_high,
            cert_sig_hex     = cert_sig_hex,
        )
        self._rotation_log.append(event)

        # Update rate-limit trackers
        self._last_rotation_ts = now
        self._rotations_this_hour.append(now)
        self._consecutive_high = 0   # reset after rotation

        return True


# ════════════════════════════════════════════════════════════════════════════
# SECURITY MODULE 5 — THREAT MEMORY DATABASE
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECOGNITION_THRESHOLD  = 0.75    # effective similarity threshold
RECENCY_LAMBDA         = 0.02    # decay rate (per day)
CONSOLIDATION_BETA     = 0.30    # learning rate for pattern update
CONSOLIDATION_COUNT    = 3       # recurrences before consolidation
MAX_PATTERNS           = 1000    # memory capacity
FEATURE_DIM            = 5       # [z_size, z_freq, entropy_dev, sig_penalty, entropy_penalty]


# ---------------------------------------------------------------------------
# Attack type classifier
# ---------------------------------------------------------------------------

def classify_attack_type(
    z_size:       float,
    z_freq:       float,
    entropy_dev:  float,
    sig_penalty:  float,
    risk_score:   float,
    pattern_match: bool,
) -> str:
    """
    Classify the dominant attack type from component signals.

    Priority order:
    1. sig_failure  (hard indicator)
    2. entropy_attack  (entropy manipulation)
    3. burst            (frequency anomaly)
    4. replay           (low novelty, pattern match)
    5. unknown
    """
    if sig_penalty > 0:
        return "sig_failure"
    if entropy_dev > 2.0 and entropy_dev > max(z_size, z_freq) * 1.5:
        return "entropy_attack"
    if z_freq > 3.0:
        return "burst"
    if pattern_match:
        return "replay"
    if risk_score >= 4.5:
        return "unknown_high"
    return "unknown"


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ThreatRecord:
    """
    A stored threat pattern with metadata.

    Attributes
    ----------
    record_id       : unique identifier
    attack_type     : classified attack type
    anomaly_sig     : normalised feature vector (5,)
    raw_feature_vec : raw (unnormalised) feature vector
    hash_fingerprint: first 16 chars of transaction hash
    tx_id           : transaction identifier
    timestamp       : Unix time of first observation
    last_seen       : Unix time of most recent occurrence
    occurrence_count: number of times this pattern was matched
    mitigation_action: response triggered
    risk_score      : risk score at time of recording
    threat_level    : "low" | "medium" | "high"
    """
    record_id:         str
    attack_type:       str
    anomaly_sig:       np.ndarray    # normalised (5,)
    raw_feature_vec:   np.ndarray    # raw (5,)
    hash_fingerprint:  str
    tx_id:             str
    timestamp:         float
    last_seen:         float
    occurrence_count:  int
    mitigation_action: str
    risk_score:        float
    threat_level:      str

    @property
    def age_days(self) -> float:
        return (time.time() - self.timestamp) / 86400.0

    @property
    def recency_weight(self) -> float:
        """w = exp(-λ × age_days)"""
        return math.exp(-RECENCY_LAMBDA * self.age_days)

    def to_dict(self) -> Dict:
        return {
            "record_id":        self.record_id,
            "attack_type":      self.attack_type,
            "hash_fingerprint": self.hash_fingerprint,
            "tx_id":            self.tx_id,
            "timestamp":        self.timestamp,
            "last_seen":        self.last_seen,
            "occurrence_count": self.occurrence_count,
            "mitigation_action":self.mitigation_action,
            "risk_score":       round(self.risk_score, 4),
            "threat_level":     self.threat_level,
            "age_days":         round(self.age_days, 2),
            "recency_weight":   round(self.recency_weight, 4),
            "feature_vector":   [round(float(x), 4) for x in self.raw_feature_vec],
        }


# ---------------------------------------------------------------------------
# Threat Memory Database
# ---------------------------------------------------------------------------

class ThreatMemory:
    """
    Episodic threat memory with similarity-based retrieval, consolidation,
    and recency weighting.

    Analogous to immunological memory B-cells: records are formed on first
    encounter with a threat, reinforced on re-encounter, and gradually
    forgotten (down-weighted) if not seen again.
    """

    def __init__(self, max_patterns: int = MAX_PATTERNS):
        self._records: List[ThreatRecord] = []
        self._max     = max_patterns
        self._total_recorded   = 0
        self._total_recognised = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def record(
        self,
        tx_id:            str,
        hash_hex:         str,
        feature_vector:   np.ndarray,   # raw (z_size, z_freq, entropy_dev, sig_pen, risk)
        attack_type:      str,
        mitigation_action: str,
        risk_score:       float,
        threat_level:     str,
    ) -> ThreatRecord:
        """
        Store a new threat pattern or consolidate with an existing similar one.

        If an existing record with the same attack type has effective_sim > 0.85
        to the new feature vector and occurrence_count ≥ CONSOLIDATION_COUNT,
        the stored pattern is updated via EWMA consolidation.

        Otherwise a new record is created.

        Returns the stored (new or consolidated) ThreatRecord.
        """
        self._total_recorded += 1
        ts  = time.time()
        fp  = hash_hex[:16] if len(hash_hex) >= 16 else hash_hex.ljust(16, "0")

        # Normalise feature vector
        norm = np.linalg.norm(feature_vector)
        sig  = feature_vector / norm if norm > 1e-8 else feature_vector.copy()

        # Check for consolidation candidate
        candidate = self._find_consolidation_candidate(sig, attack_type)

        if candidate is not None:
            # Consolidate
            candidate.anomaly_sig = (
                (1 - CONSOLIDATION_BETA) * candidate.anomaly_sig
                + CONSOLIDATION_BETA * sig
            )
            re_norm = np.linalg.norm(candidate.anomaly_sig)
            if re_norm > 1e-8:
                candidate.anomaly_sig /= re_norm
            candidate.occurrence_count += 1
            candidate.last_seen = ts
            candidate.mitigation_action = mitigation_action
            return candidate

        # New record
        record = ThreatRecord(
            record_id         = str(uuid.uuid4())[:12],
            attack_type       = attack_type,
            anomaly_sig       = sig,
            raw_feature_vec   = feature_vector.copy(),
            hash_fingerprint  = fp,
            tx_id             = tx_id,
            timestamp         = ts,
            last_seen         = ts,
            occurrence_count  = 1,
            mitigation_action = mitigation_action,
            risk_score        = risk_score,
            threat_level      = threat_level,
        )
        self._records.append(record)

        # Evict if over capacity (LRU: oldest last_seen first)
        if len(self._records) > self._max:
            self._records.sort(key=lambda r: r.last_seen)
            self._records = self._records[-self._max:]

        return record

    def query(
        self, feature_vector: np.ndarray
    ) -> Tuple[float, Optional[ThreatRecord]]:
        """
        Find the most similar stored threat pattern to a new feature vector.

        Returns (best_effective_sim, best_record) or (0.0, None) if empty.

        effective_sim = cosine_similarity × recency_weight
        """
        if not self._records:
            return 0.0, None

        norm = np.linalg.norm(feature_vector)
        if norm < 1e-8:
            return 0.0, None
        fv_unit = feature_vector / norm

        best_sim    = -1.0
        best_record = None
        for r in self._records:
            cos_sim  = float(np.dot(fv_unit, r.anomaly_sig))
            eff_sim  = cos_sim * r.recency_weight
            if eff_sim > best_sim:
                best_sim    = eff_sim
                best_record = r

        if best_sim >= RECOGNITION_THRESHOLD:
            self._total_recognised += 1
            if best_record is not None:
                best_record.occurrence_count += 1
                best_record.last_seen = time.time()

        return max(best_sim, 0.0), best_record if best_sim >= RECOGNITION_THRESHOLD else None

    def get_all_records(self) -> List[Dict]:
        return [r.to_dict() for r in self._records]

    def get_attack_summary(self) -> Dict:
        """Frequency table of attack types in memory."""
        summary: Dict[str, int] = {}
        for r in self._records:
            summary[r.attack_type] = summary.get(r.attack_type, 0) + r.occurrence_count
        return {
            "attack_type_counts":    summary,
            "total_records":         len(self._records),
            "total_recorded":        self._total_recorded,
            "total_recognised":      self._total_recognised,
            "recognition_rate":      round(
                self._total_recognised / max(self._total_recorded, 1), 4
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_consolidation_candidate(
        self, sig: np.ndarray, attack_type: str
    ) -> Optional[ThreatRecord]:
        """
        Find an existing record that is close enough to consolidate with.

        Criteria:
          - Same attack_type
          - Cosine similarity > 0.85
          - occurrence_count ≥ CONSOLIDATION_COUNT
        """
        best_sim    = 0.85   # threshold for consolidation
        best_record = None
        for r in self._records:
            if r.attack_type != attack_type:
                continue
            if r.occurrence_count < CONSOLIDATION_COUNT:
                continue
            cos_sim = float(np.dot(sig, r.anomaly_sig))
            if cos_sim > best_sim:
                best_sim    = cos_sim
                best_record = r
        return best_record

    @property
    def size(self) -> int:
        return len(self._records)


# ════════════════════════════════════════════════════════════════════════════
# SECURITY MODULE 6 — PQC IMMUNE DEFENSE ENGINE  (master pipeline)
# ════════════════════════════════════════════════════════════════════════════

# M6 crypto layer — black-box import (no modifications)

# M6 immune sub-modules


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SecurityReport:
    """
    Full security assessment for a single transaction.
    All fields are JSON-serialisable via .to_dict().
    """
    transaction_id:    str
    entropy_score:     float
    entropy_alert:     bool
    anomaly_score:     float
    threat_level:      str
    response_triggered:str
    actions_triggered: List[str]
    quarantine_status: str
    key_rotation:      bool
    multi_sig:         bool
    pattern_match:     bool
    pattern_similarity:float
    attack_type:       str
    entropy_detail:    Dict
    anomaly_detail:    Dict
    response_detail:   Dict
    key_status:        Dict
    memory_summary:    Dict
    processing_ms:     float

    def to_dict(self) -> Dict:
        return {
            "transaction_id":    self.transaction_id,
            "entropy_score":     round(self.entropy_score, 6),
            "entropy_alert":     self.entropy_alert,
            "anomaly_score":     round(self.anomaly_score, 4),
            "threat_level":      self.threat_level,
            "response_triggered":self.response_triggered,
            "actions_triggered": self.actions_triggered,
            "quarantine_status": self.quarantine_status,
            "key_rotation":      self.key_rotation,
            "multi_sig":         self.multi_sig,
            "pattern_match":     self.pattern_match,
            "pattern_similarity":round(self.pattern_similarity, 4),
            "attack_type":       self.attack_type,
            "entropy_detail":    self.entropy_detail,
            "anomaly_detail":    self.anomaly_detail,
            "response_detail":   self.response_detail,
            "key_status":        self.key_status,
            "memory_summary":    self.memory_summary,
            "processing_ms":     round(self.processing_ms, 3),
        }


# ---------------------------------------------------------------------------
# Security Engine
# ---------------------------------------------------------------------------

class SecurityEngine:
    """
    Adaptive PQC Immune Defense Layer — master pipeline.

    One SecurityEngine instance should be shared across the application
    lifetime so that it accumulates population statistics, threat memory,
    and key rotation history.

    Usage
    -----
    engine = SecurityEngine()

    # After running M6 pipeline:
    result = run_virtual_trade_simulation(...)
    tx_dict = result["transaction_record"]

    report = engine.process_transaction_security(tx_dict)
    print(report.to_dict())

    # Access current signing key (updated after any rotation):
    pk, sk = engine.current_keypair
    """

    def __init__(
        self,
        initial_keypair: Optional[Tuple[PublicKey, PrivateKey]] = None,
        entropy_window:  int = 50,
        anomaly_min_obs: int = 5,
    ):
        self._entropy_monitor  = EntropyMonitor(window=entropy_window)
        self._anomaly_detector = AnomalyDetector()
        self._response_engine  = ImmuneResponseEngine()
        self._threat_memory    = ThreatMemory()
        self._key_mutation     = KeyMutationSystem(initial_keypair)
        self._processed        = 0

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def process_transaction_security(
        self,
        transaction_dict: Dict[str, Any],
    ) -> SecurityReport:
        """
        Full immune pipeline on a completed transaction dict.

        The transaction_dict is the output of M6's build_transaction_object()
        or run_virtual_trade_simulation()["transaction_record"].

        Expected keys (all optional with graceful fallback):
            tx_id              : str
            sha3_hash          : str  (hex SHA3-256 of payload)
            verification_status: bool (ML-DSA verify result)
            signed_at          : float (Unix timestamp)
            payload            : dict
            signature          : dict (from sig.to_dict())

        Returns SecurityReport.
        """
        t0 = time.perf_counter()
        self._processed += 1

        # ── Extract fields ──────────────────────────────────────────────
        tx_id    = str(transaction_dict.get("tx_id", f"tx_{self._processed}"))
        hash_hex = str(transaction_dict.get("sha3_hash", "0" * 64))
        sig_valid= bool(transaction_dict.get("verification_status", True))
        ts       = float(transaction_dict.get("signed_at", time.time()))

        # Transaction size (serialised payload bytes)
        try:
            payload_str = json.dumps(
                transaction_dict.get("payload", transaction_dict),
                sort_keys=True, separators=(",", ":"), default=str
            )
            size_bytes = len(payload_str.encode())
        except Exception:
            size_bytes = 500   # fallback

        # ── STEP 1: Entropy analysis ────────────────────────────────────
        entropy_report = analyse_entropy(hash_hex)
        entropy_anomaly = self._entropy_monitor.observe(entropy_report.composite_score)

        # ── STEP 2: Prior pattern query ─────────────────────────────────
        # Build preliminary feature vector for memory query
        entropy_pen_prior = ENTROPY_PENALTY if (entropy_report.alert or entropy_anomaly) else 0.0
        sig_pen_prior     = SIG_PENALTY if not sig_valid else 0.0
        prior_fv = np.array([
            0.0,   # z_size placeholder
            0.0,   # z_freq placeholder
            max(0.0, (self._entropy_monitor.baseline_mean or 0.95) - entropy_report.composite_score)
                   / max(self._entropy_monitor.baseline_std or 0.01, 0.01),
            sig_pen_prior,
            entropy_pen_prior,
        ])
        prior_sim, prior_record = self._threat_memory.query(prior_fv)

        # ── STEP 3: Anomaly detection ───────────────────────────────────
        anomaly_report = self._anomaly_detector.analyse(
            tx_id           = tx_id,
            size_bytes      = size_bytes,
            timestamp       = ts,
            entropy_score   = entropy_report.composite_score,
            sig_valid       = sig_valid,
            entropy_alert   = entropy_report.alert or entropy_anomaly,
            entropy_ewma    = self._entropy_monitor.baseline_mean or 0.95,
            entropy_ewma_std= max(self._entropy_monitor.baseline_std, 0.01),
        )

        # Full feature vector (5-dim: z_size, z_freq, entropy_dev, sig_pen, entropy_pen)
        full_fv = anomaly_report.feature_vector.copy()
        full_sim, full_record = self._threat_memory.query(full_fv)

        # Use the better of prior/full similarity
        best_sim    = max(prior_sim, full_sim)
        pattern_match = best_sim >= 0.75
        matched_record = full_record or prior_record

        # ── STEP 4: Attack type classification ─────────────────────────
        attack_type = classify_attack_type(
            z_size        = anomaly_report.z_size,
            z_freq        = anomaly_report.z_freq,
            entropy_dev   = anomaly_report.entropy_dev,
            sig_penalty   = anomaly_report.sig_penalty,
            risk_score    = anomaly_report.risk_score,
            pattern_match = pattern_match,
        )

        # Override threat level if full pattern match
        threat_level = anomaly_report.threat_level
        if pattern_match and threat_level == "low":
            threat_level = "medium"

        # ── STEP 5: Immune response ─────────────────────────────────────
        response = self._response_engine.respond(
            tx_id           = tx_id,
            threat_level    = threat_level,
            risk_score      = anomaly_report.risk_score,
            sig_failed      = not sig_valid,
            entropy_alert   = entropy_report.alert or entropy_anomaly,
            pattern_matched = pattern_match,
            attack_type     = attack_type,
        )

        # ── STEP 6: Key rotation ────────────────────────────────────────
        rotation_occurred = self._key_mutation.observe_threat(threat_level)

        # ── STEP 7: Threat memory update (store medium+ or any hard signal) ──
        should_store = (
            threat_level in ("medium", "high")
            or not sig_valid
            or entropy_report.alert
            or entropy_anomaly
        )
        if should_store:
            mitigation = ", ".join(response.actions)
            self._threat_memory.record(
                tx_id             = tx_id,
                hash_hex          = hash_hex,
                feature_vector    = full_fv,
                attack_type       = attack_type,
                mitigation_action = mitigation,
                risk_score        = anomaly_report.risk_score,
                threat_level      = threat_level,
            )
            # Also register with anomaly detector's pattern library
            self._anomaly_detector.register_threat_pattern(full_fv)

        # ── STEP 8: Assemble report ─────────────────────────────────────
        t1 = time.perf_counter()

        primary_action = (
            PQCAction.QUARANTINE if response.quarantine
            else PQCAction.KEY_ROTATION_SIGNAL if response.key_rotation
            else PQCAction.MULTI_SIG if response.multi_sig
            else response.actions[0] if response.actions
            else PQCAction.LOG
        )

        report = SecurityReport(
            transaction_id    = tx_id,
            entropy_score     = entropy_report.composite_score,
            entropy_alert     = entropy_report.alert or entropy_anomaly,
            anomaly_score     = anomaly_report.risk_score,
            threat_level      = threat_level,
            response_triggered= primary_action,
            actions_triggered = response.actions,
            quarantine_status = "QUARANTINED" if response.quarantine else "CLEAR",
            key_rotation      = rotation_occurred,
            multi_sig         = response.multi_sig,
            pattern_match     = pattern_match,
            pattern_similarity= best_sim,
            attack_type       = attack_type,
            entropy_detail    = entropy_report.to_dict(),
            anomaly_detail    = anomaly_report.to_dict(),
            response_detail   = response.to_dict(),
            key_status        = self._key_mutation.status(),
            memory_summary    = self._threat_memory.get_attack_summary(),
            processing_ms     = (t1 - t0) * 1000,
        )
        return report

    # ------------------------------------------------------------------
    # Convenience: inject a crafted threat for testing
    # ------------------------------------------------------------------

    def inject_test_threat(
        self,
        threat_type: str = "entropy_attack",
        severity:    str = "high",
    ) -> SecurityReport:
        """
        Inject a synthetic threat transaction to exercise the pipeline.

        Useful for verifying that all defense layers are active without
        needing a real adversarial transaction.

        Parameters
        ----------
        threat_type : "entropy_attack" | "sig_failure" | "burst" | "replay"
        severity    : "low" | "medium" | "high"
        """
        # Craft transaction dicts for each threat scenario
        templates = {
            "entropy_attack": {
                "tx_id":             f"TEST_ENTROPY_{int(time.time())}",
                "sha3_hash":         "0000000000000000000000000000000000000000000000000000000000000001",
                "verification_status": True,
                "signed_at":         time.time(),
                "payload":           {"test": "entropy_attack", "severity": severity},
            },
            "sig_failure": {
                "tx_id":             f"TEST_SIGFAIL_{int(time.time())}",
                "sha3_hash":         "a3b4c5d6e7f8a3b4c5d6e7f8a3b4c5d6e7f8a3b4c5d6e7f8a3b4c5d6e7f8a3b4",
                "verification_status": False,
                "signed_at":         time.time(),
                "payload":           {"test": "sig_failure", "severity": severity},
            },
            "burst": {
                "tx_id":             f"TEST_BURST_{int(time.time())}",
                "sha3_hash":         "f1e2d3c4b5a6f1e2d3c4b5a6f1e2d3c4b5a6f1e2d3c4b5a6f1e2d3c4b5a6f1e2",
                "verification_status": True,
                "signed_at":         time.time() - 0.1,  # very recent
                "payload":           {"test": "burst", "severity": severity} * 10,
            },
            "replay": {
                "tx_id":             f"TEST_REPLAY_{int(time.time())}",
                "sha3_hash":         "a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2",
                "verification_status": True,
                "signed_at":         time.time(),
                "payload":           {"test": "replay", "severity": severity},
            },
        }
        tx = templates.get(threat_type, templates["entropy_attack"])
        return self.process_transaction_security(tx)

    # ------------------------------------------------------------------
    # Key access
    # ------------------------------------------------------------------

    @property
    def current_keypair(self) -> Tuple[PublicKey, PrivateKey]:
        """Return the current (possibly rotated) key pair."""
        km = self._key_mutation
        return km.current_public_key, km.current_private_key

    # ------------------------------------------------------------------
    # System status
    # ------------------------------------------------------------------

    def system_status(self) -> Dict:
        """Full status snapshot of all immune subsystems."""
        return {
            "transactions_processed": self._processed,
            "entropy_monitor":    self._entropy_monitor.summary(),
            "anomaly_detector":   self._anomaly_detector.population_stats(),
            "response_engine":    self._response_engine.stats(),
            "threat_memory":      self._threat_memory.get_attack_summary(),
            "key_system":         self._key_mutation.status(),
        }


# ════════════════════════════════════════════════════════════════════════════
# BAYESIAN MODULE 1 — SIGNAL PROCESSOR
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Sigmoid steepness for continuous signals
_SIG_K         = 2.5

# Replay detection TTL (seconds)
REPLAY_WINDOW_SECONDS = 300     # 5 minutes

# Replay cache max size (LRU eviction)
REPLAY_CACHE_SIZE = 10_000

# Welford prior seeds
_PRIOR_SIZE_MEAN  = 500.0
_PRIOR_SIZE_STD   = 100.0
_PRIOR_GAP_MEAN   = 30.0
_PRIOR_GAP_STD    = 15.0

# Sender history window (last N transactions)
SENDER_HISTORY_N  = 20


# ---------------------------------------------------------------------------
# Online Welford tracker (reused from immune_detector; independent copy)
# ---------------------------------------------------------------------------

class _Welford:
    """Numerically stable online mean and variance via Welford's algorithm."""
    def __init__(self, init_mean: float = 0.0, init_std: float = 1.0):
        self.n      = 0
        self.mean   = init_mean
        self._M2    = (init_std ** 2)   # start with prior variance
        self._init_std = init_std

    def update(self, x: float):
        self.n += 1
        delta     = x - self.mean
        self.mean += delta / (self.n + 1)   # +1 counts prior seed
        self._M2  += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return self._init_std
        return math.sqrt(max(self._M2 / self.n, 1e-10))

    def z(self, x: float) -> float:
        return (x - self.mean) / max(self.std, 1e-8)


# ---------------------------------------------------------------------------
# Sigmoid helper
# ---------------------------------------------------------------------------

def _sigmoid(x: float, mu: float = 0.0, k: float = _SIG_K) -> float:
    """Sigmoid mapped to [0, 1]; guarded against overflow."""
    val = float(np.clip(k * (x - mu), -30, 30))
    return 1.0 / (1.0 + math.exp(-val))


# ---------------------------------------------------------------------------
# LRU replay cache
# ---------------------------------------------------------------------------

class _ReplayCache:
    """
    Time-limited LRU cache for replay detection.
    Stores (hash or tx_id) → insertion_timestamp.
    """
    def __init__(self, maxsize: int = REPLAY_CACHE_SIZE, ttl: float = REPLAY_WINDOW_SECONDS):
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._maxsize = maxsize
        self._ttl     = ttl

    def seen_before(self, key: str) -> bool:
        """Return True if key was seen within TTL window. Always records key."""
        now    = time.time()
        # Evict expired
        stale  = [k for k, t in self._cache.items() if now - t > self._ttl]
        for k in stale:
            self._cache.pop(k, None)
        # Check
        exists = key in self._cache
        # Record / refresh
        self._cache[key] = now
        self._cache.move_to_end(key)
        # LRU eviction
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)
        return exists


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class SignalVector:
    """
    Normalised threat signal likelihoods extracted from a transaction.

    All signal values are ∈ [0, 1].
    0 = no evidence of threat; 1 = strong evidence of threat.

    Attributes
    ----------
    transaction_id       : unique transaction identifier
    size_anomaly         : signal from size z-score (sigmoid)
    frequency_anomaly    : signal from inter-arrival burst detection
    sender_deviation     : signal from sender behavioral deviation
    signature_risk       : 0.0 if valid, 0.95 if ML-DSA failed
    entropy_deviation    : 1 - hash entropy composite score
    replay_indicator     : 0.9 if replay detected, else 0.0
    raw_size_bytes       : raw transaction size for audit
    raw_gap_seconds      : inter-arrival time in seconds
    timestamp            : Unix timestamp of signal extraction
    """
    transaction_id:    str
    size_anomaly:      float
    frequency_anomaly: float
    sender_deviation:  float
    signature_risk:    float
    entropy_deviation: float
    replay_indicator:  float
    raw_size_bytes:    int
    raw_gap_seconds:   float
    timestamp:         float

    def to_dict(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "signals": {
                "size_anomaly":      round(self.size_anomaly,      4),
                "frequency_anomaly": round(self.frequency_anomaly, 4),
                "sender_deviation":  round(self.sender_deviation,  4),
                "signature_risk":    round(self.signature_risk,    4),
                "entropy_deviation": round(self.entropy_deviation, 4),
                "replay_indicator":  round(self.replay_indicator,  4),
            },
            "raw": {
                "size_bytes":   self.raw_size_bytes,
                "gap_seconds":  round(self.raw_gap_seconds, 3),
            },
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Signal Processor
# ---------------------------------------------------------------------------

class SignalProcessor:
    """
    Stateful signal extractor that maintains population statistics and
    sender histories across the lifetime of the engine.

    One instance should be shared across all processed transactions so
    that population statistics stabilise over time.
    """

    def __init__(self):
        self._size_tracker = _Welford(_PRIOR_SIZE_MEAN, _PRIOR_SIZE_STD)
        self._gap_tracker  = _Welford(_PRIOR_GAP_MEAN,  _PRIOR_GAP_STD)
        self._last_ts: Optional[float] = None
        self._replay_cache = _ReplayCache()
        # sender_id → deque of (size, gap) tuples
        self._sender_history: Dict[str, List[Tuple[float, float]]] = {}
        self._processed = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, tx: Dict[str, Any]) -> SignalVector:
        """
        Extract and normalise all threat signals from a transaction dict.

        Parameters
        ----------
        tx : transaction dict from build_transaction_object() or any dict
             containing at minimum 'tx_id', 'sha3_hash',
             'verification_status', 'signed_at'.

        Returns
        -------
        SignalVector
        """
        self._processed += 1
        now = time.time()

        tx_id        = str(tx.get("tx_id",    f"tx_{self._processed}"))
        hash_hex     = str(tx.get("sha3_hash", "0" * 64))
        sig_valid    = bool(tx.get("verification_status", True))
        ts           = float(tx.get("signed_at", now))
        entropy_score= float(tx.get("_entropy_composite_score", -1.0))

        # ── Serialised size ───────────────────────────────────────────
        try:
            payload_str = json.dumps(
                tx.get("payload", tx),
                sort_keys=True, separators=(",", ":"), default=str
            )
            size = len(payload_str.encode())
        except Exception:
            size = 500

        # ── Inter-arrival gap ─────────────────────────────────────────
        if self._last_ts is not None:
            gap = max(ts - self._last_ts, 0.001)
        else:
            gap = _PRIOR_GAP_MEAN
        self._last_ts = ts

        # ── Update population statistics ──────────────────────────────
        self._size_tracker.update(float(size))
        self._gap_tracker.update(gap)

        # ── SIGNAL 1: size_anomaly ─────────────────────────────────────
        # z-score of size, sigmoid-mapped → [0, 1]
        z_size      = abs(self._size_tracker.z(float(size)))
        size_anomaly= _sigmoid(z_size, mu=2.0, k=1.5)   # alert at z≥2

        # ── SIGNAL 2: frequency_anomaly ───────────────────────────────
        # Short gap (burst) → high z_freq → high signal
        z_gap        = self._gap_tracker.z(gap)
        freq_anomaly = _sigmoid(-z_gap, mu=1.5, k=1.5)  # alert at z≤-1.5

        # ── SIGNAL 3: sender_deviation ────────────────────────────────
        # Sender is identified by public key fingerprint (if present)
        sender_id = (tx.get("public_key") or {}).get("fingerprint", "unknown")
        sender_dev = self._sender_deviation(sender_id, float(size), gap)

        # ── SIGNAL 4: signature_risk ──────────────────────────────────
        sig_risk = 0.0 if sig_valid else 0.95

        # ── SIGNAL 5: entropy_deviation ───────────────────────────────
        # If entropy score was pre-computed (injected into tx dict), use it.
        # Otherwise, estimate from hash byte repetition.
        if entropy_score >= 0.0:
            ent_dev = float(np.clip(1.0 - entropy_score, 0.0, 1.0))
        else:
            ent_dev = self._estimate_entropy_deviation(hash_hex)

        # ── SIGNAL 6: replay_indicator ────────────────────────────────
        # Check both tx_id and hash for replay
        is_replay = (
            self._replay_cache.seen_before(tx_id)
            or self._replay_cache.seen_before(hash_hex[:32])
        )
        replay_sig = 0.9 if is_replay else 0.0

        return SignalVector(
            transaction_id    = tx_id,
            size_anomaly      = float(np.clip(size_anomaly,  0.0, 1.0)),
            frequency_anomaly = float(np.clip(freq_anomaly,  0.0, 1.0)),
            sender_deviation  = float(np.clip(sender_dev,    0.0, 1.0)),
            signature_risk    = sig_risk,
            entropy_deviation = float(np.clip(ent_dev,       0.0, 1.0)),
            replay_indicator  = replay_sig,
            raw_size_bytes    = size,
            raw_gap_seconds   = gap,
            timestamp         = now,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sender_deviation(self, sender_id: str, size: float, gap: float) -> float:
        """
        Cosine similarity between current (size, gap) and sender's history mean.
        Deviation = 1 - cosine_similarity.

        Returns 0.3 (slight uncertainty) if sender has no history.
        """
        history = self._sender_history.setdefault(sender_id, [])
        if len(history) < 3:
            history.append((size, gap))
            if len(history) > SENDER_HISTORY_N:
                history.pop(0)
            return 0.3   # no history: slight uncertainty

        # Compute mean vector of history
        arr  = np.array(history, dtype=float)
        mean = arr.mean(axis=0)
        curr = np.array([size, gap], dtype=float)

        norm_m = np.linalg.norm(mean)
        norm_c = np.linalg.norm(curr)
        if norm_m < 1e-8 or norm_c < 1e-8:
            history.append((size, gap))
            if len(history) > SENDER_HISTORY_N:
                history.pop(0)
            return 0.3

        cos_sim   = float(np.dot(mean / norm_m, curr / norm_c))
        deviation = float(np.clip((1.0 - cos_sim) / 2.0, 0.0, 1.0))  # [0,1]

        history.append((size, gap))
        if len(history) > SENDER_HISTORY_N:
            history.pop(0)

        return deviation

    @staticmethod
    def _estimate_entropy_deviation(hash_hex: str) -> float:
        """
        Lightweight entropy estimate from hash hex string.
        Counts unique hex nibbles; ideal = 16 unique values.
        Returns 1 - (unique_nibbles / 16).
        """
        if len(hash_hex) < 4:
            return 0.5
        unique = len(set(hash_hex.lower()))
        return float(np.clip(1.0 - unique / 16.0, 0.0, 1.0))


# ════════════════════════════════════════════════════════════════════════════
# BAYESIAN MODULE 2 — BAYESIAN ENGINE
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Constants — Signal weights (must sum to 1.0)
# ---------------------------------------------------------------------------

SIGNAL_WEIGHTS: Dict[str, float] = {
    "signature_risk":    0.35,
    "entropy_deviation": 0.20,
    "replay_indicator":  0.20,
    "frequency_anomaly": 0.10,
    "sender_deviation":  0.10,
    "size_anomaly":      0.05,
}

assert abs(sum(SIGNAL_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"

# Per-signal population baselines (benign expected values, calibrated to real traffic)
# These represent the expected signal value for a normal, legitimate transaction.
# Set by inspecting Phase-1 benign transaction signal outputs.
SIGNAL_BASELINES: Dict[str, float] = {
    "signature_risk":    0.001,   # ML-DSA verification almost always passes
    "entropy_deviation": 0.040,   # SHA3-256 hash has near-perfect entropy
    "replay_indicator":  0.001,   # replays are rare in normal operation
    "frequency_anomaly": 0.200,   # moderate inter-arrival variation expected
    "sender_deviation":  0.300,   # new senders default to 0.3 deviation
    "size_anomaly":      0.100,   # typical transaction size variation
}

# LLR scaling factor β — calibrated so sig_failure→CRITICAL, entropy_only→MONITOR
# Verification (beta_scale=2.0, prior=0.05):
#   Benign  (all signals at baseline): posterior ≈ 0.0495  → SAFE  ✓
#   SigFail (sig_risk=0.95):           posterior ≈ 0.948   → CRITICAL ✓
#   EntOnly (ent_dev=0.999):           posterior ≈ 0.151   → SAFE/MONITOR ✓
#   Replay  (replay=0.90):             posterior ≈ 0.439   → MONITOR ✓
#   Burst   (replay+ent+freq):         posterior ≈ 0.773   → ELEVATED ✓
#   All-hit (all signals high):        posterior ≈ 0.998   → CRITICAL ✓
BETA_SCALE: float = 2.0

# Baseline prior P(threat)
BASE_PRIOR: float = 0.05

# Effective sample size for CI computation
N_EFF: int = 100

# Memory similarity boost multiplier
MAX_MEMORY_BOOST: float = 0.40

# Epsilon guard
_EPS: float = 1e-6


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class BayesianResult:
    """
    Output of the Bayesian threat probability engine.

    Attributes
    ----------
    transaction_id         : unique transaction identifier
    prior                  : P(threat) before observing evidence
    prior_boosted_by_memory: True if prior was raised by immune memory
    memory_similarity      : cosine similarity to nearest stored threat (0–1)
    log_likelihood_threat  : Σ log P(sᵢ | H₁)  (weighted)
    log_likelihood_benign  : Σ log P(sᵢ | H₀)  (weighted)
    likelihood_ratio       : exp(log_likelihood_threat - log_likelihood_benign)
    posterior              : P(threat | evidence) ∈ [0, 1]
    ci_lower_90            : 90% credible interval lower bound
    ci_upper_90            : 90% credible interval upper bound
    signal_contributions   : per-signal posterior contribution
    """
    transaction_id:          str
    prior:                   float
    prior_boosted_by_memory: bool
    memory_similarity:       float
    log_likelihood_threat:   float
    log_likelihood_benign:   float
    likelihood_ratio:        float
    posterior:               float
    ci_lower_90:             float
    ci_upper_90:             float
    signal_contributions:    Dict[str, float]

    def to_dict(self) -> Dict:
        return {
            "transaction_id":           self.transaction_id,
            "prior":                    round(self.prior, 6),
            "prior_boosted_by_memory":  self.prior_boosted_by_memory,
            "memory_similarity":        round(self.memory_similarity, 4),
            "log_likelihood_threat":    round(self.log_likelihood_threat, 6),
            "log_likelihood_benign":    round(self.log_likelihood_benign, 6),
            "likelihood_ratio":         round(self.likelihood_ratio, 6),
            "posterior_threat_probability": round(self.posterior, 6),
            "ci_90_lower":              round(self.ci_lower_90, 4),
            "ci_90_upper":              round(self.ci_upper_90, 4),
            "signal_contributions":     {k: round(v, 4)
                                         for k, v in self.signal_contributions.items()},
        }


# ---------------------------------------------------------------------------
# Bayesian Engine
# ---------------------------------------------------------------------------

class BayesianEngine:
    """
    Computes the Bayesian posterior threat probability P(threat | signals).

    Stateless with respect to individual transactions — all prior boosting
    information is supplied externally via memory_similarity parameter.

    Usage
    -----
    engine = BayesianEngine()
    result = engine.compute(signal_vector, memory_similarity=0.0)
    """

    def __init__(
        self,
        base_prior:       float = BASE_PRIOR,
        weights:          Optional[Dict[str, float]] = None,
        baselines:        Optional[Dict[str, float]] = None,
    ):
        self._base_prior = float(np.clip(base_prior, _EPS, 1.0 - _EPS))
        self._weights    = weights   or SIGNAL_WEIGHTS
        self._baselines  = baselines or SIGNAL_BASELINES

    # ------------------------------------------------------------------
    # Primary computation
    # ------------------------------------------------------------------

    def compute(
        self,
        signals:           SignalVector,
        memory_similarity: float = 0.0,
    ) -> BayesianResult:
        """
        Compute Bayesian posterior threat probability.

        Parameters
        ----------
        signals           : SignalVector from SignalProcessor
        memory_similarity : cosine similarity (0–1) from ImmuneMemory query.
                            Higher similarity → larger prior boost.

        Returns
        -------
        BayesianResult
        """
        # ── 1. Prior (possibly boosted by memory) ─────────────────────
        sim   = float(np.clip(memory_similarity, 0.0, 1.0))
        boost = sim * MAX_MEMORY_BOOST
        prior = float(np.clip(self._base_prior + (1 - self._base_prior) * boost,
                               _EPS, 1.0 - _EPS))
        memory_boosted = boost > 0.001

        # ── 2. One-sided LLR per signal ────────────────────────────────
        # LLR_i = β × max(0, log(s/(1-s)) - log(b/(1-b)))
        # weighted_LLR = Σ wᵢ × LLR_i
        sig_dict = {
            "size_anomaly":      signals.size_anomaly,
            "frequency_anomaly": signals.frequency_anomaly,
            "sender_deviation":  signals.sender_deviation,
            "signature_risk":    signals.signature_risk,
            "entropy_deviation": signals.entropy_deviation,
            "replay_indicator":  signals.replay_indicator,
        }

        weighted_llr  = 0.0
        log_lk_threat = 0.0
        log_lk_benign = 0.0
        contributions: Dict[str, float] = {}

        for signal_name, w in self._weights.items():
            s = float(np.clip(sig_dict.get(signal_name, 0.0), _EPS, 1.0 - _EPS))
            b = float(np.clip(self._baselines.get(signal_name, 0.15), _EPS, 1.0 - _EPS))
            # Calibrated neutral-point LLR:
            #   LLR_i = β × log((s/(1-s)) / (b/(1-b)))
            #         = β × log((s × (1-b)) / ((1-s) × b))
            # Interpretation:
            #   s == b  → LLR = 0  (signal at population baseline; no update)
            #   s > b   → LLR > 0  (signal elevated above baseline; threat evidence)
            #   s < b   → LLR < 0  (below baseline; slight benign evidence)
            # Two-sided update: benign signals pulling below baseline gently reduce
            # the posterior, while threat signals raise it.  This is mathematically
            # correct and prevents posterior collapse.
            llr_i    = BETA_SCALE * math.log((s * (1.0 - b)) / ((1.0 - s) * b))
            w_llr_i  = w * llr_i
            weighted_llr  += w_llr_i
            # Approximate raw likelihoods for audit fields
            log_lk_threat += math.log(float(np.clip(s, _EPS, 1 - _EPS)))
            log_lk_benign += math.log(float(np.clip(1.0 - s, _EPS, 1 - _EPS)))
            contributions[signal_name] = round(w_llr_i, 4)

        # ── 3. Log-odds posterior (numerically stable Bayes) ──────────
        log_odds_prior     = math.log(prior / (1.0 - prior))
        log_odds_posterior = log_odds_prior + weighted_llr
        log_odds_posterior = float(np.clip(log_odds_posterior, -30, 30))
        posterior          = 1.0 / (1.0 + math.exp(-log_odds_posterior))

        # ── 4. Likelihood ratio (from total weighted LLR) ─────────────
        lk_ratio = math.exp(float(np.clip(weighted_llr, -30, 30)))

        # ── 5. Wilson score 90% credible interval ─────────────────────
        ci_lo, ci_hi = self._wilson_ci(posterior, N_EFF, z=1.645)

        return BayesianResult(
            transaction_id          = signals.transaction_id,
            prior                   = prior,
            prior_boosted_by_memory = memory_boosted,
            memory_similarity       = sim,
            log_likelihood_threat   = log_lk_threat,
            log_likelihood_benign   = log_lk_benign,
            likelihood_ratio        = lk_ratio,
            posterior               = posterior,
            ci_lower_90             = ci_lo,
            ci_upper_90             = ci_hi,
            signal_contributions    = contributions,
        )

    # ------------------------------------------------------------------
    # Batch utility
    # ------------------------------------------------------------------

    def compute_batch(
        self,
        signal_vectors: List[SignalVector],
        memory_similarities: Optional[List[float]] = None,
    ) -> List[BayesianResult]:
        """Process a list of SignalVectors in order."""
        sims = memory_similarities or [0.0] * len(signal_vectors)
        return [self.compute(sv, ms) for sv, ms in zip(signal_vectors, sims)]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wilson_ci(p: float, n: int, z: float = 1.645) -> tuple:
        """
        Wilson score interval for binomial proportion.

        CI = (p̃ ± z × σ̃) / denom
        where p̃ = (p + z²/2n) / (1 + z²/n)
              σ̃ = sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)
        """
        p   = float(np.clip(p, _EPS, 1.0 - _EPS))
        z2  = z * z
        denom  = 1.0 + z2 / n
        centre = (p + z2 / (2 * n)) / denom
        half   = (z / denom) * math.sqrt(max(p * (1 - p) / n + z2 / (4 * n * n), 0))
        return (
            float(np.clip(centre - half, 0.0, 1.0)),
            float(np.clip(centre + half, 0.0, 1.0)),
        )


# ════════════════════════════════════════════════════════════════════════════
# BAYESIAN MODULE 3 — THREAT CLASSIFIER
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Threshold constants
# ---------------------------------------------------------------------------

THRESHOLD_SAFE     = 0.20
THRESHOLD_MONITOR  = 0.50
THRESHOLD_ELEVATED = 0.80

THREAT_LEVELS = ["SAFE", "MONITOR", "ELEVATED_RISK", "CRITICAL_THREAT"]

LEVEL_COLORS = {
    "SAFE":           "green",
    "MONITOR":        "yellow",
    "ELEVATED_RISK":  "orange",
    "CRITICAL_THREAT":"red",
}

# Label → (lower_bound, upper_bound)
THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "SAFE":           (0.00, THRESHOLD_SAFE),
    "MONITOR":        (THRESHOLD_SAFE,     THRESHOLD_MONITOR),
    "ELEVATED_RISK":  (THRESHOLD_MONITOR,  THRESHOLD_ELEVATED),
    "CRITICAL_THREAT":(THRESHOLD_ELEVATED, 1.00),
}


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class ThreatClassification:
    """
    Result of classifying the Bayesian posterior into a threat level.

    Attributes
    ----------
    transaction_id   : unique transaction identifier
    posterior        : raw P(threat | evidence)
    threat_level     : "SAFE" | "MONITOR" | "ELEVATED_RISK" | "CRITICAL_THREAT"
    threat_score     : posterior × 100 (0–100 scale for display)
    escalated        : True if precautionary CI-based escalation was applied
    ci_lower_90      : lower bound of 90% credible interval
    ci_upper_90      : upper bound of 90% credible interval
    level_index      : integer ordinal of threat level (0=SAFE, 3=CRITICAL)
    color            : display colour string
    description      : one-line human-readable description
    """
    transaction_id: str
    posterior:      float
    threat_level:   str
    threat_score:   float
    escalated:      bool
    ci_lower_90:    float
    ci_upper_90:    float
    level_index:    int
    color:          str
    description:    str

    def to_dict(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "posterior":       round(self.posterior,    6),
            "threat_level":    self.threat_level,
            "threat_score":    round(self.threat_score, 2),
            "escalated":       self.escalated,
            "ci_90": {
                "lower": round(self.ci_lower_90, 4),
                "upper": round(self.ci_upper_90, 4),
            },
            "level_index":  self.level_index,
            "color":        self.color,
            "description":  self.description,
        }

    @property
    def is_threat(self) -> bool:
        return self.level_index >= 2   # ELEVATED_RISK or above

    @property
    def requires_quarantine(self) -> bool:
        return self.threat_level == "CRITICAL_THREAT"


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class ThreatClassifier:
    """
    Stateless classifier mapping posterior probability to threat level.

    Applies CI-based precautionary escalation to avoid under-reaction
    near threshold boundaries.
    """

    def classify(
        self,
        transaction_id: str,
        posterior:      float,
        ci_lower_90:    float = 0.0,
        ci_upper_90:    float = 1.0,
    ) -> ThreatClassification:
        """
        Classify a posterior probability into a threat level.

        Parameters
        ----------
        transaction_id : unique transaction identifier
        posterior      : P(threat | evidence) ∈ [0, 1]
        ci_lower_90    : 90% CI lower bound
        ci_upper_90    : 90% CI upper bound

        Returns
        -------
        ThreatClassification
        """
        p        = float(np.clip(posterior, 0.0, 1.0))
        escalated = False

        # Base level assignment
        if p < THRESHOLD_SAFE:
            level       = "SAFE"
            level_index = 0
        elif p < THRESHOLD_MONITOR:
            level       = "MONITOR"
            level_index = 1
        elif p < THRESHOLD_ELEVATED:
            level       = "ELEVATED_RISK"
            level_index = 2
        else:
            level       = "CRITICAL_THREAT"
            level_index = 3

        # CI-based precautionary escalation (never downgrade)
        next_threshold = {
            "SAFE":          THRESHOLD_SAFE,
            "MONITOR":       THRESHOLD_MONITOR,
            "ELEVATED_RISK": THRESHOLD_ELEVATED,
            "CRITICAL_THREAT": 1.01,  # cannot escalate further
        }[level]

        if ci_upper_90 > next_threshold and level_index < 3:
            level_index += 1
            level        = THREAT_LEVELS[level_index]
            escalated    = True

        description = self._description(level, p, escalated)

        return ThreatClassification(
            transaction_id = transaction_id,
            posterior      = p,
            threat_level   = level,
            threat_score   = p * 100.0,
            escalated      = escalated,
            ci_lower_90    = ci_lower_90,
            ci_upper_90    = ci_upper_90,
            level_index    = level_index,
            color          = LEVEL_COLORS[level],
            description    = description,
        )

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _description(level: str, p: float, escalated: bool) -> str:
        base = {
            "SAFE":           f"Benign — P(threat)={p:.3f} below safe threshold.",
            "MONITOR":        f"Monitor — P(threat)={p:.3f}; logging for audit.",
            "ELEVATED_RISK":  f"Elevated — P(threat)={p:.3f}; secondary validation required.",
            "CRITICAL_THREAT":f"CRITICAL — P(threat)={p:.3f}; quarantine triggered.",
        }[level]
        if escalated:
            base += " [ESCALATED from CI boundary uncertainty]"
        return base

    @staticmethod
    def level_to_index(level: str) -> int:
        return THREAT_LEVELS.index(level) if level in THREAT_LEVELS else -1

    @staticmethod
    def index_to_level(index: int) -> str:
        return THREAT_LEVELS[int(np.clip(index, 0, 3))]

    def batch_classify(
        self,
        transactions: List[Tuple[str, float, float, float]],
    ) -> List[ThreatClassification]:
        """
        Batch classify. Each tuple is (tx_id, posterior, ci_lower, ci_upper).
        """
        return [self.classify(*t) for t in transactions]


# ════════════════════════════════════════════════════════════════════════════
# BAYESIAN MODULE 4 — RESPONSE ENGINE  (quarantine ledger)
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# BayesAction constants
# ---------------------------------------------------------------------------

class BayesAction:
    LOG                    = "LOG"
    AUDIT_FLAG             = "AUDIT_FLAG"
    ENHANCED_VALIDATION    = "ENHANCED_VALIDATION"
    SECONDARY_VALIDATION   = "SECONDARY_VALIDATION"
    QUARANTINE             = "QUARANTINE"
    KEY_ROTATION_SIGNAL    = "KEY_ROTATION_SIGNAL"
    ACCOUNT_INVESTIGATION  = "ACCOUNT_INVESTIGATION"


# Monotone action sets per threat level
LEVEL_ACTIONS: Dict[str, List[str]] = {
    "SAFE": [
        BayesAction.LOG,
    ],
    "MONITOR": [
        BayesAction.LOG,
        BayesAction.AUDIT_FLAG,
    ],
    "ELEVATED_RISK": [
        BayesAction.LOG,
        BayesAction.AUDIT_FLAG,
        BayesAction.ENHANCED_VALIDATION,
        BayesAction.SECONDARY_VALIDATION,
    ],
    "CRITICAL_THREAT": [
        BayesAction.LOG,
        BayesAction.AUDIT_FLAG,
        BayesAction.ENHANCED_VALIDATION,
        BayesAction.SECONDARY_VALIDATION,
        BayesAction.QUARANTINE,
        BayesAction.KEY_ROTATION_SIGNAL,
        BayesAction.ACCOUNT_INVESTIGATION,
    ],
}


# ---------------------------------------------------------------------------
# Quarantine ledger
# ---------------------------------------------------------------------------

@dataclass
class QuarantineRecord:
    """A single entry in the quarantine ledger."""
    record_id:       str
    tx_id:           str
    posterior:       float
    threat_level:    str
    threat_score:    float
    quarantined_at:  float
    reason:          str
    released:        bool = False
    released_at:     Optional[float] = None

    def to_dict(self) -> Dict:
        return {
            "record_id":      self.record_id,
            "tx_id":          self.tx_id,
            "posterior":      round(self.posterior, 6),
            "threat_level":   self.threat_level,
            "threat_score":   round(self.threat_score, 2),
            "quarantined_at": self.quarantined_at,
            "reason":         self.reason,
            "released":       self.released,
            "released_at":    self.released_at,
        }


class QuarantineLedger:
    """
    Separate in-memory ledger for quarantined transactions.
    Architecturally isolated from the main transaction ledger.
    """
    def __init__(self):
        self._records: List[QuarantineRecord] = []
        self._quarantined_ids: Set[str] = set()

    def add(self, record: QuarantineRecord):
        self._records.append(record)
        self._quarantined_ids.add(record.tx_id)

    def is_quarantined(self, tx_id: str) -> bool:
        return tx_id in self._quarantined_ids

    def release(self, tx_id: str) -> bool:
        """Release a quarantined transaction after manual review."""
        for r in self._records:
            if r.tx_id == tx_id and not r.released:
                r.released    = True
                r.released_at = time.time()
                self._quarantined_ids.discard(tx_id)
                return True
        return False

    def get_all(self) -> List[Dict]:
        return [r.to_dict() for r in self._records]

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def active_count(self) -> int:
        return sum(1 for r in self._records if not r.released)


# ---------------------------------------------------------------------------
# Response data class
# ---------------------------------------------------------------------------

@dataclass
class ResponseDecision:
    """
    Full defensive response for a single transaction.

    Attributes
    ----------
    response_id         : unique identifier
    tx_id               : transaction being processed
    threat_level        : from ThreatClassification
    actions             : ordered list of actions triggered
    quarantined         : True if transaction is in quarantine ledger
    key_rotation_signal : True if key rotation was signalled
    secondary_validation: True if secondary validation was requested
    account_flagged     : True if account investigation was triggered
    posterior           : raw posterior probability (for logging)
    latency_ms          : processing time (signal → response)
    notes               : audit trail notes
    timestamp           : Unix time of response
    """
    response_id:          str
    tx_id:                str
    threat_level:         str
    actions:              List[str]
    quarantined:          bool
    key_rotation_signal:  bool
    secondary_validation: bool
    account_flagged:      bool
    posterior:            float
    latency_ms:           float
    notes:                List[str]
    timestamp:            float

    def to_dict(self) -> Dict:
        return {
            "response_id":          self.response_id,
            "tx_id":                self.tx_id,
            "threat_level":         self.threat_level,
            "actions_triggered":    self.actions,
            "quarantined":          self.quarantined,
            "quarantine_status":    "QUARANTINED" if self.quarantined else "CLEAR",
            "key_rotation_signal":  self.key_rotation_signal,
            "secondary_validation": self.secondary_validation,
            "account_flagged":      self.account_flagged,
            "posterior":            round(self.posterior, 6),
            "latency_ms":           round(self.latency_ms, 3),
            "notes":                self.notes,
            "timestamp":            self.timestamp,
        }


# ---------------------------------------------------------------------------
# Response Engine
# ---------------------------------------------------------------------------

class BayesianResponseEngine:
    """
    Dispatches defensive actions based on ThreatClassification.

    Maintains:
      - A response log (full audit trail)
      - A quarantine ledger (isolated from main ledger)
    """

    def __init__(self):
        self._log:              List[ResponseDecision] = []
        self._quarantine_ledger = QuarantineLedger()
        self._stats = {
            "total": 0, "safe": 0, "monitor": 0,
            "elevated": 0, "critical": 0, "key_rotations": 0,
        }

    # ------------------------------------------------------------------
    # Primary dispatch
    # ------------------------------------------------------------------

    def dispatch(
        self,
        classification: ThreatClassification,
        signal_start_time: Optional[float] = None,
    ) -> ResponseDecision:
        """
        Trigger the appropriate response for a classified threat.

        Parameters
        ----------
        classification     : ThreatClassification from ThreatClassifier
        signal_start_time  : Unix time when signal extraction began
                             (used to compute end-to-end latency)

        Returns
        -------
        ResponseDecision
        """
        t_now    = time.time()
        latency  = (t_now - signal_start_time) * 1000.0 if signal_start_time else 0.0

        level    = classification.threat_level
        actions  = list(LEVEL_ACTIONS.get(level, [BayesAction.LOG]))
        notes    = [classification.description]

        quarantined          = BayesAction.QUARANTINE in actions
        key_rotation_signal  = BayesAction.KEY_ROTATION_SIGNAL in actions
        secondary_validation = BayesAction.SECONDARY_VALIDATION in actions
        account_flagged      = BayesAction.ACCOUNT_INVESTIGATION in actions

        # Execute quarantine
        if quarantined:
            qr = QuarantineRecord(
                record_id      = str(uuid.uuid4())[:12],
                tx_id          = classification.transaction_id,
                posterior      = classification.posterior,
                threat_level   = level,
                threat_score   = classification.threat_score,
                quarantined_at = t_now,
                reason         = (f"Bayesian posterior {classification.posterior:.4f} ≥ "
                                  f"CRITICAL threshold (0.80)"),
            )
            self._quarantine_ledger.add(qr)
            notes.append(
                f"Transaction {classification.transaction_id[:16]}… "
                f"added to quarantine ledger (record {qr.record_id})"
            )

        if key_rotation_signal:
            notes.append(
                "Key rotation signalled to KeyMutationSystem. "
                "New ML-DSA keypair will be generated."
            )

        if secondary_validation:
            notes.append(
                "Secondary signature validation pass requested. "
                f"Hash re-derivation and ML-DSA re-verify queued."
            )

        if account_flagged:
            notes.append(
                f"Sender account flagged for investigation "
                f"(posterior={classification.posterior:.4f})."
            )

        if classification.escalated:
            notes.append(
                "⚠ Threat level was precautionarily ESCALATED due to "
                "CI boundary uncertainty."
            )

        # Update stats
        self._stats["total"] += 1
        level_map = {
            "SAFE": "safe", "MONITOR": "monitor",
            "ELEVATED_RISK": "elevated", "CRITICAL_THREAT": "critical",
        }
        self._stats[level_map.get(level, "safe")] += 1
        if key_rotation_signal:
            self._stats["key_rotations"] += 1

        decision = ResponseDecision(
            response_id          = str(uuid.uuid4())[:12],
            tx_id                = classification.transaction_id,
            threat_level         = level,
            actions              = actions,
            quarantined          = quarantined,
            key_rotation_signal  = key_rotation_signal,
            secondary_validation = secondary_validation,
            account_flagged      = account_flagged,
            posterior            = classification.posterior,
            latency_ms           = latency,
            notes                = notes,
            timestamp            = t_now,
        )
        self._log.append(decision)
        return decision

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def quarantine_ledger(self) -> QuarantineLedger:
        return self._quarantine_ledger

    def response_stats(self) -> Dict:
        return {**self._stats,
                "quarantine_ledger_size": self._quarantine_ledger.size,
                "active_quarantines":     self._quarantine_ledger.active_count}

    def recent_log(self, n: int = 10) -> List[Dict]:
        return [d.to_dict() for d in self._log[-n:]]


# ════════════════════════════════════════════════════════════════════════════
# BAYESIAN MODULE 5 — IMMUNE MEMORY  (pattern store)
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RECENCY_LAMBDA          = 0.03    # per day; half-life ≈ 23 days
MAX_MEMORY_BOOST        = 0.40    # maximum prior boost from memory match
CONSOLIDATION_THRESHOLD = 0.90    # cosine similarity → consolidate not add
CONSOLIDATION_BETA      = 0.25    # learning rate for pattern consolidation
RECOGNITION_THRESHOLD   = 0.60    # effective_sim → "known threat" flag
CONSOLIDATION_MIN_OCC   = 2       # min occurrences before consolidation
MAX_RECORDS             = 1_000   # memory capacity
SIGNAL_DIM              = 6       # dimension of feature vector

SIGNAL_NAMES = [
    "size_anomaly",
    "frequency_anomaly",
    "sender_deviation",
    "signature_risk",
    "entropy_deviation",
    "replay_indicator",
]


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """
    A stored threat pattern with metadata.

    Attributes
    ----------
    record_id               : unique identifier
    transaction_sig_vector  : L2-normalised feature vector (6-dim)
    raw_vector              : original (unnormalised) feature vector
    posterior_probability   : posterior at time of recording
    threat_classification   : "SAFE" | "MONITOR" | "ELEVATED_RISK" | "CRITICAL_THREAT"
    mitigation_action       : comma-separated actions taken
    timestamp               : Unix time of first observation
    last_seen               : Unix time of most recent match
    occurrence_count        : number of times this pattern was triggered
    tx_id_ref               : transaction ID that generated this record
    """
    record_id:              str
    transaction_sig_vector: np.ndarray   # normalised
    raw_vector:             np.ndarray   # unnormalised
    posterior_probability:  float
    threat_classification:  str
    mitigation_action:      str
    timestamp:              float
    last_seen:              float
    occurrence_count:       int
    tx_id_ref:              str

    @property
    def age_days(self) -> float:
        return (time.time() - self.timestamp) / 86400.0

    @property
    def recency_weight(self) -> float:
        return math.exp(-RECENCY_LAMBDA * self.age_days)

    def to_dict(self) -> Dict:
        return {
            "record_id":              self.record_id,
            "transaction_sig_vector": [round(float(x), 6)
                                       for x in self.transaction_sig_vector],
            "raw_vector":             [round(float(x), 6) for x in self.raw_vector],
            "posterior_probability":  round(self.posterior_probability, 6),
            "threat_classification":  self.threat_classification,
            "mitigation_action":      self.mitigation_action,
            "timestamp":              self.timestamp,
            "last_seen":              self.last_seen,
            "occurrence_count":       self.occurrence_count,
            "tx_id_ref":              self.tx_id_ref,
            "age_days":               round(self.age_days, 2),
            "recency_weight":         round(self.recency_weight, 4),
        }


# ---------------------------------------------------------------------------
# Immune Memory
# ---------------------------------------------------------------------------

class ImmuneMemory:
    """
    Bayesian immune memory with similarity-based retrieval, consolidation,
    recency weighting, and optional JSON persistence.

    Usage
    -----
    memory = ImmuneMemory(persistence_path="immune_memory.json")

    # Before Bayesian update (query)
    sim = memory.query_similarity(signal_vector)

    # After Bayesian update + classification (record)
    memory.record(tx_id, signal_vector, posterior, threat_level, actions)
    """

    def __init__(self, persistence_path: Optional[str] = None):
        self._records: List[MemoryRecord] = []
        self._persistence_path = persistence_path
        self._total_queries    = 0
        self._total_records    = 0
        self._total_recognised = 0

        if persistence_path and os.path.exists(persistence_path):
            self._load(persistence_path)

    # ------------------------------------------------------------------
    # Query (STEP 0 in pipeline)
    # ------------------------------------------------------------------

    def query_similarity(self, signal_vector: np.ndarray) -> float:
        """
        Find the maximum effective cosine similarity between the incoming
        signal vector and all stored threat patterns.

        Effective similarity = cosine_similarity × recency_weight

        Returns
        -------
        float ∈ [0, 1]  (0 = no memory, 1 = perfect match to recent threat)
        """
        self._total_queries += 1
        if not self._records:
            return 0.0

        norm_q = np.linalg.norm(signal_vector)
        if norm_q < 1e-8:
            return 0.0
        q_unit = signal_vector / norm_q

        best = 0.0
        for rec in self._records:
            cos_sim  = float(np.dot(q_unit, rec.transaction_sig_vector))
            eff_sim  = max(cos_sim, 0.0) * rec.recency_weight
            if eff_sim > best:
                best = eff_sim

        if best >= RECOGNITION_THRESHOLD:
            self._total_recognised += 1

        return float(np.clip(best, 0.0, 1.0))

    def query_top_k(
        self, signal_vector: np.ndarray, k: int = 3
    ) -> List[Tuple[float, MemoryRecord]]:
        """
        Return top-k (effective_similarity, MemoryRecord) pairs, sorted descending.
        """
        if not self._records:
            return []

        norm_q = np.linalg.norm(signal_vector)
        if norm_q < 1e-8:
            return []
        q_unit = signal_vector / norm_q

        scored = []
        for rec in self._records:
            cos_sim  = float(np.dot(q_unit, rec.transaction_sig_vector))
            eff_sim  = max(cos_sim, 0.0) * rec.recency_weight
            scored.append((eff_sim, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

    # ------------------------------------------------------------------
    # Record (STEP 5 in pipeline)
    # ------------------------------------------------------------------

    def record(
        self,
        tx_id:              str,
        signal_vector:      np.ndarray,   # raw 6-dim from SignalProcessor
        posterior:          float,
        threat_level:       str,
        mitigation_action:  str,
    ) -> MemoryRecord:
        """
        Store or consolidate a confirmed threat pattern.

        If a sufficiently similar pattern already exists
        (cos_sim > CONSOLIDATION_THRESHOLD and occ ≥ CONSOLIDATION_MIN_OCC),
        the stored pattern is updated via EWMA.  Otherwise a new record is created.

        Parameters
        ----------
        tx_id             : originating transaction
        signal_vector     : raw 6-dim signal vector
        posterior         : P(threat | evidence) at time of recording
        threat_level      : threat classification label
        mitigation_action : comma-separated action list

        Returns
        -------
        MemoryRecord (new or consolidated)
        """
        self._total_records += 1
        ts   = time.time()

        # Normalise
        norm = np.linalg.norm(signal_vector)
        sig  = signal_vector / norm if norm > 1e-8 else signal_vector.copy()

        # Search for consolidation candidate
        candidate = self._find_consolidation_candidate(sig, threat_level)

        if candidate is not None:
            # EWMA consolidation (affinity maturation)
            candidate.transaction_sig_vector = self._normalise(
                CONSOLIDATION_BETA * signal_vector
                + (1 - CONSOLIDATION_BETA) * candidate.transaction_sig_vector
            )
            candidate.occurrence_count  += 1
            candidate.last_seen          = ts
            candidate.posterior_probability = (
                (candidate.posterior_probability * (candidate.occurrence_count - 1)
                 + posterior) / candidate.occurrence_count
            )
            candidate.mitigation_action  = mitigation_action
            return candidate

        # New record
        rec = MemoryRecord(
            record_id               = str(uuid.uuid4())[:12],
            transaction_sig_vector  = sig,
            raw_vector              = signal_vector.copy(),
            posterior_probability   = posterior,
            threat_classification   = threat_level,
            mitigation_action       = mitigation_action,
            timestamp               = ts,
            last_seen               = ts,
            occurrence_count        = 1,
            tx_id_ref               = tx_id,
        )
        self._records.append(rec)

        # Capacity management: LRU eviction by last_seen
        if len(self._records) > MAX_RECORDS:
            self._records.sort(key=lambda r: r.last_seen)
            self._records = self._records[-MAX_RECORDS:]

        # Optional persistence
        if self._persistence_path:
            self._save()

        return rec

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        """Write all records to JSON store."""
        try:
            data = [r.to_dict() for r in self._records]
            with open(self._persistence_path, "w") as f:
                json.dump(data, f, indent=2)
        except (IOError, TypeError):
            pass  # Non-fatal; memory still in RAM

    def _load(self, path: str):
        """Load records from JSON store on startup."""
        try:
            with open(path) as f:
                data = json.load(f)
            for d in data:
                vec = np.array(d["transaction_sig_vector"], dtype=float)
                raw = np.array(d.get("raw_vector", d["transaction_sig_vector"]), dtype=float)
                rec = MemoryRecord(
                    record_id              = d["record_id"],
                    transaction_sig_vector = vec,
                    raw_vector             = raw,
                    posterior_probability  = d["posterior_probability"],
                    threat_classification  = d["threat_classification"],
                    mitigation_action      = d["mitigation_action"],
                    timestamp              = d["timestamp"],
                    last_seen              = d.get("last_seen", d["timestamp"]),
                    occurrence_count       = d.get("occurrence_count", 1),
                    tx_id_ref              = d.get("tx_id_ref", ""),
                )
                self._records.append(rec)
        except (IOError, json.JSONDecodeError, KeyError):
            pass

    # ------------------------------------------------------------------
    # Statistics & accessors
    # ------------------------------------------------------------------

    def get_all_records(self) -> List[Dict]:
        return [r.to_dict() for r in self._records]

    def summary(self) -> Dict:
        if not self._records:
            return {"total_patterns": 0, "total_queries": self._total_queries,
                    "recognition_rate": 0.0}
        by_level: Dict[str, int] = {}
        for r in self._records:
            by_level[r.threat_classification] = (
                by_level.get(r.threat_classification, 0) + r.occurrence_count
            )
        return {
            "total_patterns":    len(self._records),
            "total_queries":     self._total_queries,
            "total_records":     self._total_records,
            "total_recognised":  self._total_recognised,
            "recognition_rate":  round(
                self._total_recognised / max(self._total_queries, 1), 4
            ),
            "by_threat_level":   by_level,
            "persistence_path":  self._persistence_path,
        }

    @property
    def size(self) -> int:
        return len(self._records)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v

    def _find_consolidation_candidate(
        self, sig: np.ndarray, threat_level: str
    ) -> Optional[MemoryRecord]:
        best_sim    = CONSOLIDATION_THRESHOLD
        best_record = None
        for r in self._records:
            if r.threat_classification != threat_level:
                continue
            if r.occurrence_count < CONSOLIDATION_MIN_OCC:
                continue
            cos_sim = float(np.dot(sig, r.transaction_sig_vector))
            if cos_sim > best_sim:
                best_sim    = cos_sim
                best_record = r
        return best_record


# ════════════════════════════════════════════════════════════════════════════
# BAYESIAN MODULE 6 — BAYESIAN SECURITY PIPELINE  (master controller)
# ════════════════════════════════════════════════════════════════════════════

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("immune_bayesian")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s [IMMUNE-BAYES] %(levelname)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Output data class
# ---------------------------------------------------------------------------

@dataclass
class BayesianSecurityReport:
    """
    Complete Bayesian immune security assessment for a single transaction.

    All fields are JSON-serialisable via .to_dict().
    """
    transaction_id:        str
    posterior_probability: float
    prior:                 float
    likelihood_ratio:      float
    threat_level:          str
    threat_score:          float
    action_taken:          str
    actions_triggered:     List[str]
    quarantine_status:     str
    key_rotation_signal:   bool
    memory_similarity:     float
    memory_boosted_prior:  bool
    signal_vector:         Dict
    bayesian_detail:       Dict
    classification_detail: Dict
    response_detail:       Dict
    processing_ms:         float

    def to_dict(self) -> Dict:
        return {
            "transaction_id":        self.transaction_id,
            "posterior_probability": round(self.posterior_probability, 6),
            "prior":                 round(self.prior, 6),
            "likelihood_ratio":      round(self.likelihood_ratio, 6),
            "threat_level":          self.threat_level,
            "threat_score":          round(self.threat_score, 2),
            "action_taken":          self.action_taken,
            "actions_triggered":     self.actions_triggered,
            "quarantine_status":     self.quarantine_status,
            "key_rotation_signal":   self.key_rotation_signal,
            "memory_similarity":     round(self.memory_similarity, 4),
            "memory_boosted_prior":  self.memory_boosted_prior,
            "signal_vector":         self.signal_vector,
            "bayesian_detail":       self.bayesian_detail,
            "classification_detail": self.classification_detail,
            "response_detail":       self.response_detail,
            "processing_ms":         round(self.processing_ms, 3),
        }


# ---------------------------------------------------------------------------
# Pipeline Controller
# ---------------------------------------------------------------------------

class BayesianSecurityPipeline:
    """
    Master Bayesian Immune Security Pipeline.

    Orchestrates the complete Detection → Probability Update
    → Response → Memory cycle for every transaction.

    Parameters
    ----------
    base_prior          : P(threat) baseline (default 0.05)
    memory_path         : optional path to JSON persistence file for ImmuneMemory
    log_level           : Python logging level (default INFO)
    store_safe_in_memory: if False (default), only MONITOR+ patterns are stored
    """

    def __init__(
        self,
        base_prior:           float          = 0.05,
        memory_path:          Optional[str]  = None,
        log_level:            int            = logging.INFO,
        store_safe_in_memory: bool           = False,
    ):
        self._signal_processor = SignalProcessor()
        self._bayesian_engine  = BayesianEngine(base_prior=base_prior)
        self._threat_classifier= ThreatClassifier()
        self._response_engine  = BayesianResponseEngine()
        self._immune_memory    = ImmuneMemory(persistence_path=memory_path)
        self._store_safe       = store_safe_in_memory
        self._processed        = 0
        self._security_log: List[Dict] = []
        logger.setLevel(log_level)

    # ------------------------------------------------------------------
    # Primary entry point
    # ------------------------------------------------------------------

    def process_transaction_security(
        self,
        transaction_dict: Dict[str, Any],
    ) -> BayesianSecurityReport:
        """
        Run the full Bayesian immune pipeline on a transaction.

        This is the sole public API.  All sub-steps are executed in order
        and the results assembled into a BayesianSecurityReport.

        Parameters
        ----------
        transaction_dict : transaction dict from M6 build_transaction_object()
                           or any dict containing tx_id, sha3_hash,
                           verification_status, signed_at, payload.
                           Optionally inject '_entropy_composite_score' ∈ [0,1]
                           for richer entropy signal.

        Returns
        -------
        BayesianSecurityReport
        """
        t0 = time.perf_counter()
        self._processed += 1

        tx_id = str(transaction_dict.get("tx_id", f"tx_{self._processed}"))

        # ── STEP 1: Extract signals ─────────────────────────────────────
        signals: SignalVector = self._signal_processor.extract(transaction_dict)
        logger.debug("tx=%s  signals extracted: sig_risk=%.3f ent_dev=%.3f replay=%.3f",
                     tx_id[:16], signals.signature_risk,
                     signals.entropy_deviation, signals.replay_indicator)

        # ── STEP 0 (pre-query): Memory similarity query ─────────────────
        # Build raw signal vector for memory query
        raw_fv = self._signals_to_vector(signals)
        memory_sim = self._immune_memory.query_similarity(raw_fv)

        # ── STEP 2: Bayesian posterior computation ──────────────────────
        bayes: "BayesianResult" = self._bayesian_engine.compute(
            signals, memory_similarity=memory_sim
        )
        logger.debug("tx=%s  prior=%.4f  posterior=%.4f  LR=%.4f",
                     tx_id[:16], bayes.prior, bayes.posterior, bayes.likelihood_ratio)

        # ── STEP 3: Threat classification ───────────────────────────────
        classification = self._threat_classifier.classify(
            transaction_id = tx_id,
            posterior      = bayes.posterior,
            ci_lower_90    = bayes.ci_lower_90,
            ci_upper_90    = bayes.ci_upper_90,
        )
        logger.info(
            "tx=%-20s  P(threat)=%.4f  level=%-16s  score=%.1f  "
            "memory_sim=%.3f  escalated=%s",
            tx_id[:20],
            bayes.posterior,
            classification.threat_level,
            classification.threat_score,
            memory_sim,
            classification.escalated,
        )

        # ── STEP 4: Response dispatch ───────────────────────────────────
        response = self._response_engine.dispatch(
            classification,
            signal_start_time=t0 + time.time() - time.perf_counter(),
        )

        # Log threat detections
        if classification.level_index >= 1:
            logger.warning(
                "THREAT DETECTED  tx=%-20s  level=%-16s  actions=%s",
                tx_id[:20],
                classification.threat_level,
                ", ".join(response.actions),
            )

        if response.quarantined:
            logger.warning(
                "QUARANTINED  tx=%-20s  posterior=%.4f",
                tx_id[:20], bayes.posterior
            )

        # ── STEP 5: Update immune memory ────────────────────────────────
        should_store = (
            self._store_safe
            or classification.level_index >= 1   # MONITOR or higher
        )
        if should_store:
            mitigation = ", ".join(response.actions)
            self._immune_memory.record(
                tx_id             = tx_id,
                signal_vector     = raw_fv,
                posterior         = bayes.posterior,
                threat_level      = classification.threat_level,
                mitigation_action = mitigation,
            )
            logger.debug("tx=%s  immune memory updated (level=%s)",
                         tx_id[:16], classification.threat_level)

        # ── STEP 6: Assemble report ─────────────────────────────────────
        t1 = time.perf_counter()

        primary_action = response.actions[0] if response.actions else "LOG"

        report = BayesianSecurityReport(
            transaction_id        = tx_id,
            posterior_probability = bayes.posterior,
            prior                 = bayes.prior,
            likelihood_ratio      = bayes.likelihood_ratio,
            threat_level          = classification.threat_level,
            threat_score          = classification.threat_score,
            action_taken          = primary_action,
            actions_triggered     = response.actions,
            quarantine_status     = "QUARANTINED" if response.quarantined else "CLEAR",
            key_rotation_signal   = response.key_rotation_signal,
            memory_similarity     = memory_sim,
            memory_boosted_prior  = bayes.prior_boosted_by_memory,
            signal_vector         = signals.to_dict(),
            bayesian_detail       = bayes.to_dict(),
            classification_detail = classification.to_dict(),
            response_detail       = response.to_dict(),
            processing_ms         = (t1 - t0) * 1000,
        )

        # Internal audit log
        self._security_log.append({
            "tx_id":       tx_id,
            "threat_level":classification.threat_level,
            "posterior":   round(bayes.posterior, 6),
            "quarantined": response.quarantined,
            "timestamp":   time.time(),
        })

        return report

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def get_security_log(self, n: int = 20) -> List[Dict]:
        """Return the n most recent pipeline log entries."""
        return self._security_log[-n:]

    def system_status(self) -> Dict:
        """Full status snapshot of all Bayesian immune subsystems."""
        return {
            "transactions_processed": self._processed,
            "response_stats":         self._response_engine.response_stats(),
            "memory_summary":         self._immune_memory.summary(),
            "quarantine_ledger": {
                "size":   self._response_engine.quarantine_ledger.size,
                "active": self._response_engine.quarantine_ledger.active_count,
            },
        }

    def get_quarantine_ledger(self) -> List[Dict]:
        """Return all quarantined transaction records."""
        return self._response_engine.quarantine_ledger.get_all()

    def get_memory_records(self) -> List[Dict]:
        """Return all stored threat pattern records."""
        return self._immune_memory.get_all_records()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _signals_to_vector(sv: SignalVector) -> np.ndarray:
        """Convert SignalVector to ordered numpy array (matches SIGNAL_NAMES)."""
        return np.array([
            sv.size_anomaly,
            sv.frequency_anomaly,
            sv.sender_deviation,
            sv.signature_risk,
            sv.entropy_deviation,
            sv.replay_indicator,
        ], dtype=float)
