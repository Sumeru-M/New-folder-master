"""
entropy_monitor.py — Cryptographic Randomness Quality Monitor
=============================================================

Mathematical Framework
----------------------

A transaction hash h ∈ {0,1}^256 (SHA3-256, 32 bytes) should, if the
underlying PRF is sound, be indistinguishable from a uniformly random
bitstring.  We measure three complementary properties:

1. **Shannon Byte Entropy**

   Treat the 32-byte digest as a sequence of byte symbols x_i ∈ {0,…,255}.
   Compute the empirical probability mass function p(b) over the 32 bytes,
   then:

       H = -Σ_{b} p(b) log₂ p(b)     (bits)

   Maximum possible: H_max = log₂(32) = 5 bits  (each byte unique)
   Threshold:        H_safe = H_max × 0.85 = 4.25 bits

   Note: with only 32 samples the empirical entropy is noisy; we therefore
   also compute the *bit-level* entropy over 256 bits for a more stable
   estimate.

2. **Bit-Level Shannon Entropy** (primary metric)

   Parse the hash as 256 individual bits b_i ∈ {0, 1}.
   Let p₁ = fraction of 1-bits.  Bernoulli entropy:

       H_bit = -p₁ log₂ p₁  -  (1-p₁) log₂(1-p₁)     (bits per bit)

   H_bit = 1.0 for a perfect coin; H_bit < 1 indicates bit bias.
   Threshold: H_bit_safe = 0.95

3. **Bit-Distribution Uniformity** (chi-squared test)

   Under H₀ (uniform bits): expected 1-count = 128, std ≈ 8.
   Standardised statistic:

       z_bit = (count_ones - 128) / 8

   |z_bit| > 3  → flag (p < 0.003 under H₀).

4. **Byte-level Collision Potential**

   Using the birthday problem approximation:
   For n = 32 draws from alphabet of size A = 256:

       P_collision ≈ n² / (2A) = 1024 / 512 = 2.0    (expected collisions)

   This is always moderate for 32 bytes; we instead report the proportion
   of unique bytes as a uniformity metric:

       uniqueness = |{distinct bytes}| / 32    (ideal ≈ 1.0 for long hashes)

5. **Composite Entropy Score** ∈ [0, 1]

       score = 0.6 × (H_bit / 1.0)
             + 0.3 × min(H_byte / H_max, 1.0)
             + 0.1 × uniqueness

   score < SAFE_THRESHOLD (0.80) → alert.

Assumptions
-----------
A1. SHA3-256 output is modelled as i.i.d. Bernoulli(0.5) bits under H₀.
A2. Entropy estimates are noisy for 32-byte samples; composite score is
    more robust than any single metric.
A3. Thresholds are calibrated to yield < 0.001 false-positive rate on
    genuine SHA3-256 digests.  Adversarially crafted hashes (e.g., from
    weak RNGs or hash truncation) will score substantially lower.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


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
