"""
key_mutation.py — Adaptive PQC Key Rotation System
===================================================

Mathematical Framework
----------------------

Key rotation is the operational response to sustained cryptographic threats.
The challenge is to rotate to a new key pair while preserving the ability
to verify historical signatures created under the old key.

**Rotation Policy**

The system maintains a key chain:

    K_0  →  K_1  →  …  →  K_t  (current)

Each key K_i consists of (pk_i, sk_i) generated via `generate_keypair()`
from the existing ML-DSA implementation.

Rotation is triggered when:
    (a) HIGH threat level is sustained for ≥ ROTATION_TRIGGER_HIGH consecutive
        transactions, OR
    (b) Total HIGH-threat count crosses ROTATION_TRIGGER_CUMULATIVE, OR
    (c) A key's age exceeds MAX_KEY_AGE_HOURS (time-based rotation), OR
    (d) Explicit external call to `force_rotate()`.

**Cross-signing (chain of trust)**

At rotation time t:

    1. New key pair (pk_{t+1}, sk_{t+1}) is generated.
    2. The transition is authenticated by signing a rotation certificate
       with sk_t:

           cert = {
               "type":        "KEY_ROTATION",
               "from_key_id": pk_t.key_id,
               "to_key_id":   pk_{t+1}.key_id,
               "rotation_reason": ...,
               "timestamp": ...,
           }
           sig_cert = sign_transaction(hash(cert), sk_t)

    3. The certificate + signature is stored in the rotation log.
    4. All future transactions use (pk_{t+1}, sk_{t+1}).
    5. Verification of historical transactions uses the key archived at
       their signing time, found via key_id lookup in the key archive.

This ensures historical signatures remain verifiable even after key rotation.

**Rotation Frequency Constraints**

To prevent rotation DoS (an attacker forcing continuous rotation):

    - Minimum interval between rotations: MIN_ROTATION_INTERVAL_SECONDS (60s)
    - Maximum rotations per hour: MAX_ROTATIONS_PER_HOUR (6)

**Entropy Seed for New Keys**

New key pairs use `secrets.token_bytes(32)` from the OS CSPRNG.
No deterministic derivation from the old key (compromise of old key
does not compromise new key).

Assumptions
-----------
A1. `generate_keypair()` from M6 crypto_layer is cryptographically secure.
A2. The rotation certificate binds the old key to the new key via an
    ML-DSA signature — the chain of trust is only as strong as the
    most recent uncompromised key.
A3. Key archive is in-memory; in production this should be persisted to
    a tamper-evident log (e.g., WORM storage or blockchain).
A4. Time-based rotation (A3 criterion) uses wall-clock time; in a real
    system this should use a trusted time source.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import M6 crypto layer as a black box
from milestone6.crypto_layer import (
    PublicKey, PrivateKey,
    generate_keypair,
    sign_transaction,
    hash_payload,
)


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
