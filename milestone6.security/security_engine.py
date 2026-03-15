"""
security_engine.py — Adaptive PQC Immune Defense Master Pipeline
================================================================

This module is the single public entry point for the immune defense layer.

Orchestration: Detection → Response → Memory → Reinforcement
-------------------------------------------------------------

    transaction_dict
        │
        ├─► [1] EntropyMonitor.analyse(hash_hex)
        │       → EntropyReport  (Shannon entropy, bit uniformity, z_bit)
        │
        ├─► [2] ThreatMemory.query(feature_vector)
        │       → (similarity, matched_record)
        │           prior knowledge: is this a known threat pattern?
        │
        ├─► [3] AnomalyDetector.analyse(tx metadata)
        │       → AnomalyReport  (risk_score, z-scores, threat_level)
        │
        ├─► [4] ImmuneResponseEngine.respond(threat_level, flags)
        │       → ResponseRecord  (actions, quarantine, key_rotation)
        │
        ├─► [5] KeyMutationSystem.observe_threat(threat_level)
        │       → bool  (rotation_occurred)
        │
        ├─► [6] ThreatMemory.record(…)  [if medium/high]
        │       → ThreatRecord  (stored for future matching)
        │
        └─► [7] Assemble SecurityReport (full structured output)

Isolation guarantee
-------------------
This module imports from milestone6.crypto_layer (read-only, never modifies
existing functions) and from the new milestone6.security.* sub-modules only.
No M1–M5 code is touched.

The existing run_virtual_trade_simulation() and all M6 pipeline code
are completely unaffected.  The SecurityEngine is an additive layer:
callers pass in an already-built transaction dict and receive a security
report back.

Output schema
-------------
{
  "transaction_id":    str,
  "entropy_score":     float,          # composite entropy ∈ [0,1]
  "entropy_alert":     bool,
  "anomaly_score":     float,          # composite risk score ≥ 0
  "threat_level":      "low|medium|high",
  "response_triggered":str,            # primary action label
  "actions_triggered": [str, ...],
  "quarantine_status": "QUARANTINED" | "CLEAR",
  "key_rotation":      bool,
  "multi_sig":         bool,
  "pattern_match":     bool,
  "pattern_similarity":float,
  "attack_type":       str,
  "entropy_detail":    {...},
  "anomaly_detail":    {...},
  "response_detail":   {...},
  "key_status":        {...},
  "memory_summary":    {...},
  "processing_ms":     float,
}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# M6 crypto layer — black-box import (no modifications)
from milestone6.crypto_layer import (
    PublicKey, PrivateKey,
    generate_keypair,
    verify_transaction,
    TransactionSignature,
)

# M6 immune sub-modules
from milestone6.security.entropy_monitor  import EntropyMonitor, analyse_entropy
from milestone6.security.immune_detector  import AnomalyDetector, SIG_PENALTY, ENTROPY_PENALTY
from milestone6.security.immune_response  import ImmuneResponseEngine, Action
from milestone6.security.threat_memory    import (
    ThreatMemory, classify_attack_type,
)
from milestone6.security.key_mutation     import KeyMutationSystem


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
            Action.QUARANTINE if response.quarantine
            else Action.KEY_ROTATION_SIGNAL if response.key_rotation
            else Action.MULTI_SIG if response.multi_sig
            else response.actions[0] if response.actions
            else Action.LOG
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
