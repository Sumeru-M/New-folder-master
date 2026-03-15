"""
immune_detector.py — Statistical Transaction Anomaly Detector
=============================================================

Mathematical Framework
----------------------

Each transaction is characterised by a feature vector:

    x = [size_bytes, time_gap_seconds, entropy_score, sig_valid]

We detect anomalies by computing a **composite risk score** from four
independent signals, then classifying into threat levels.

1. **Z-score of Transaction Size**

   Z-scores measure deviation from population mean in standard deviations:

       z_size = (size - μ_size) / σ_size

   Online mean and variance are maintained via Welford's algorithm
   (numerically stable, O(1) per update):

       δ  = x - mean
       mean += δ / n
       M2  += δ × (x - mean)     [Welford update]
       var  = M2 / (n-1)

   z_size > threshold → unusually large or small transaction.

2. **Z-score of Transaction Frequency (inter-arrival time)**

   Let Δt_i = t_i - t_{i-1} (seconds between consecutive transactions).
   Low Δt (burst) → high frequency → anomaly:

       z_freq = -(Δt - μ_Δt) / σ_Δt

   Negated so that high frequency → positive z_freq.

3. **Entropy Deviation**

   Given composite entropy score s ∈ [0,1] with expected μ_H under normal
   operation (estimated from history):

       entropy_dev = max(0, μ_H - s) / σ_H    (one-sided; low entropy is bad)

4. **Signature Validation Penalty**

   Binary indicator: 0 if valid, P_sig if invalid (default P_sig = 5).
   A failed verification is a hard signal requiring immediate escalation.

5. **Composite Risk Score**

   The score has two additive components: a **soft score** from continuous
   signals and a **hard penalty** from binary security failures.

   Soft score (continuous signals, weighted sum):

       soft = w_size × |z_size| + w_freq × max(z_freq, 0) + w_entropy × entropy_dev

   Hard penalties (direct additive; not down-weighted):

       hard = sig_penalty    if sig_invalid   (= 10.0)
            + entropy_penalty if entropy_alert (=  4.0)

       risk_score = soft + hard

   This ensures:
     - sig_failure alone  → risk ≥ 10.0 → HIGH unconditionally
     - entropy attack alone → risk ≥ 4.0 → MEDIUM
     - both               → risk ≥ 14.0 → HIGH

6. **Threat Classification**

       LOW    :  risk_score < 2.0
       MEDIUM :  2.0 ≤ risk_score < 8.0
       HIGH   :  risk_score ≥ 8.0

   The HIGH threshold of 8.0 is chosen so a failed signature (hard = 10)
   always crosses it, while normal statistical noise (soft ≤ 2) stays LOW.

7. **Historical Comparison**

   Each transaction's feature vector is compared to stored threat patterns
   using **cosine similarity**:

       sim(a, b) = a·b / (‖a‖ ‖b‖)

   A similarity score > 0.85 to any stored HIGH-threat pattern triggers
   an automatic escalation to HIGH regardless of raw score.

Assumptions
-----------
A1. Transaction sizes and inter-arrival times are approximately stationary
    within a rolling window.  Welford online statistics adapt to structural
    changes over time.
A2. The four signals are approximately independent conditional on being
    benign.  This is a simplifying assumption; in practice there may be
    correlations (e.g., large transactions may arrive less frequently).
A3. Signature validation is a hard indicator.  We do not apply shrinkage
    to the signature penalty.
A4. The threat classification thresholds are set conservatively (prefer
    false positives over false negatives in a security context).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


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
