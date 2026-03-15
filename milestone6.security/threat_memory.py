"""
threat_memory.py — Adaptive Threat Pattern Memory
===================================================

Mathematical Framework
----------------------

The threat memory implements an **episodic immunological memory** system
analogous to adaptive immune memory cells.  Once a threat is identified,
its "signature" is encoded as a normalised feature vector and stored for
rapid future recognition.

**Threat Encoding**

Each threat record encodes:
    1. Feature vector f ∈ ℝ⁵  (size, frequency, entropy, sig_valid, risk)
    2. Hash fingerprint — first 16 chars of tx hash (compact identifier)
    3. Attack type label (entropy_attack, replay, burst, sig_failure, unknown)
    4. Mitigation action taken
    5. Timestamp and occurrence count

**Similarity Scoring (Cosine Distance)**

For a new transaction feature vector q and stored threat t:

    sim(q, t) = q·t / (‖q‖ · ‖t‖)   ∈ [-1, 1]

We use cosine similarity because it is scale-invariant — a transaction
that is 10× larger but otherwise identical to a threat should match.

**Exponential Recency Weighting**

Older threats are down-weighted in matching to allow the system to
"forget" patterns that are no longer active:

    w(t) = exp(-λ × age_days)

where λ = 0.02 (≈ half-life of 35 days).

**Effective Similarity**

    effective_sim(q, t_i) = sim(q, t_i) × w(t_i)

A new observation is considered a **known threat** if:

    max_i  effective_sim(q, t_i) > RECOGNITION_THRESHOLD (0.75)

**Attack Type Classification**

Based on which component of the feature vector dominates:

    entropy_attack  : entropy_dev > 2.0 × other components
    replay          : z_size ≈ 0, z_freq ≈ 0, pattern_match = True
    burst           : z_freq > 3.0
    sig_failure     : sig_penalty > 0
    unknown         : no dominant signal

**Memory Consolidation**

If the same attack type recurs ≥ 3 times, the stored pattern is updated
via an exponentially-weighted mean:

    f_consolidated = (1 - β) × f_stored + β × f_new   (β = 0.3)

This ensures the memory adapts to evolving attack patterns.

Assumptions
-----------
A1. Feature vectors are normalised before storage and lookup.
A2. Cosine similarity is appropriate because we care about the direction
    (attack type profile) not the magnitude (severity).
A3. Recency weighting uses a half-life of 35 days; this should be tuned
    to the expected threat persistence in production.
A4. Memory capacity is bounded at MAX_PATTERNS (default 1000) with
    LRU eviction of the oldest, least-similar patterns.
"""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


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
