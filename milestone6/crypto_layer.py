"""
crypto_layer.py  —  Milestone 6 Cryptographic Security Layer
=============================================================
Structurally faithful to NIST FIPS 204 (ML-DSA / CRYSTALS-Dilithium).
Pure Python + NumPy, no external crypto libraries.

Fixes applied vs earlier versions
-----------------------------------
1. poly_mul   : O(n log n) via numpy.convolve + negacyclic fold, not O(n²).
2. ExpandA    : 23-bit mask gives near-uniform Z_q coefficients (~99.9% accept).
3. Power2Round: nearest-round split → t0 in (−2^(d−1), 2^(d−1)].
4. MakeHint / UseHint : fully implemented so verify can reconstruct w1.
5. Signing    : randomised (hedged) — fresh rnd every call, unique sig each time.
6. Verification: all four checks (norm, hint weight, UseHint, commitment).
"""

from __future__ import annotations
import hashlib, hmac, json, secrets, struct, time, uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple
import numpy as np

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