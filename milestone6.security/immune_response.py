"""
immune_response.py — Adaptive Immune Response Engine
=====================================================

Mathematical Framework
----------------------

The immune response engine maps a (threat_level, anomaly_report) pair to
a set of **defensive actions**, modelled as a partially-ordered lattice of
escalating responses:

    LEVEL 0  (low)    : LOG only
    LEVEL 1  (medium) : enhanced validation + alert
    LEVEL 2  (high)   : quarantine + key rotation trigger + multi-sig

This mirrors the **clonal selection** principle in biological immunity:
the strength of the response is proportional to the degree of threat
recognition, and multiple defence arms are activated in parallel.

**Response Selection Algorithm**

Given threat_level ∈ {low, medium, high} and boolean flags:
    - sig_failed      : signature validation failure
    - entropy_alert   : hash entropy below safe threshold
    - pattern_matched : known threat pattern recognised

The response set R is constructed as:

    R = base_responses[threat_level]              (always included)
    if sig_failed:     R ∪= {MULTI_SIG, QUARANTINE}
    if entropy_alert:  R ∪= {ENHANCED_VALIDATION, ALERT}
    if pattern_matched: R ∪= {QUARANTINE}

where QUARANTINE takes precedence over all other actions (hard stop).

**Quarantine Score** ∈ [0, 1]

Probabilistic quarantine criterion for borderline cases (risk_score
between LOW and HIGH threshold):

    q_score = sigmoid((risk_score - QUARANTINE_PIVOT) / QUARANTINE_SCALE)

    sigmoid(x) = 1 / (1 + exp(-x))

    QUARANTINE_PIVOT = 3.5   (midpoint between medium/high thresholds)
    QUARANTINE_SCALE = 0.5   (softness of decision boundary)

q_score > 0.5 → quarantine (deterministic above HIGH threshold by design,
probabilistic near boundary, never below LOW threshold).

**Multi-signature Verification**

When multi-sig is triggered, the engine requests N_MULTISIG additional
signature validations.  In this simulation, this is represented as a
flag in the security report; in production it would involve contacting
N additional signing authorities.

N_MULTISIG = 2 by default (total 3-of-3 confirmation).

**Response Logging**

Every response is logged with:
    - timestamp
    - threat_level
    - actions_triggered
    - quarantine decision
    - key_rotation_requested
    - processing time

This log serves as the immune system's "effector memory".

Assumptions
-----------
A1. Quarantine is a soft block: the transaction is flagged and held,
    not permanently rejected.  Release requires manual review.
A2. Key rotation is "requested" (signalled) by the response engine;
    the actual rotation is executed by KeyMutationSystem.observe_threat().
A3. Multi-signature verification is simulated (flag-only) since this
    is a single-node implementation.  In production, it would call
    N additional verification endpoints.
A4. All responses are idempotent: applying the same response twice has
    the same effect as applying it once.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np


# ---------------------------------------------------------------------------
# Response action constants
# ---------------------------------------------------------------------------

class Action:
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
    "low":    [Action.LOG],
    "medium": [Action.LOG, Action.ALERT, Action.ENHANCED_VALIDATION],
    "high":   [Action.LOG, Action.ALERT, Action.ENHANCED_VALIDATION,
               Action.MULTI_SIG, Action.QUARANTINE, Action.KEY_ROTATION_SIGNAL],
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
    actions         : set of Action constants triggered
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
        actions: Set[str] = set(BASE_RESPONSES.get(threat_level, [Action.LOG]))

        # Contextual escalation
        notes = []
        if sig_failed:
            actions.add(Action.MULTI_SIG)
            actions.add(Action.QUARANTINE)
            notes.append(f"ML-DSA signature verification FAILED — "
                         f"mandatory quarantine + multi-sig")
        if entropy_alert:
            actions.add(Action.ENHANCED_VALIDATION)
            actions.add(Action.ALERT)
            notes.append("Hash entropy below safe threshold — "
                         "enhanced validation triggered")
        if pattern_matched:
            actions.add(Action.QUARANTINE)
            actions.add(Action.PATTERN_STORE)
            notes.append("Known threat pattern recognised — "
                         "quarantine applied")

        # Quarantine score (probabilistic sigmoid)
        q_score = self._quarantine_score(risk_score)

        # Final quarantine decision
        quarantine = (
            Action.QUARANTINE in actions
            or q_score > 0.5
        )

        # Key rotation signalling
        key_rotation = Action.KEY_ROTATION_SIGNAL in actions

        # Multi-sig
        multi_sig = Action.MULTI_SIG in actions

        # Always store threat pattern for high-level incidents
        if threat_level == "high":
            actions.add(Action.PATTERN_STORE)

        # Rate limiting on burst attacks
        if attack_type == "burst":
            actions.add(Action.RATE_LIMIT)
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
