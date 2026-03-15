"""
run_milestone6_security.py — Adaptive PQC Immune Defense Demo
==============================================================
Demonstrates the full immune pipeline against:
  1. Benign M6 transactions (should score LOW)
  2. Entropy-attack (weak RNG hash — scores MEDIUM/HIGH)
  3. Signature failure (hard anomaly — scores HIGH)
  4. Burst transactions (frequency anomaly)
  5. Key rotation under sustained threat
  6. Threat memory and pattern recognition

Does NOT modify any existing M6 modules.
"""

import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from milestone6.security import SecurityEngine
from milestone6.crypto_layer import generate_keypair, build_transaction_object, hash_payload


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_real_tx(ticker: str, quantity: int, price: float) -> dict:
    """Build a genuine M6 transaction object."""
    pk, sk = generate_keypair()
    payload = {
        "type":     "VIRTUAL_BUY",
        "ticker":   ticker,
        "quantity": quantity,
        "price":    price,
        "timestamp": time.time(),
    }
    return build_transaction_object(payload, sk, pk)


def section(title: str):
    print("\n" + "═" * 68)
    print(f"  {title}")
    print("═" * 68)


def print_report(report, idx: int):
    d = report.to_dict()
    threat_icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
    icon = threat_icons.get(d["threat_level"], "⚪")
    print(f"\n  [{idx:02d}] {icon} tx={d['transaction_id'][:14]}…"
          f"  threat={d['threat_level'].upper():6s}"
          f"  risk={d['anomaly_score']:5.2f}"
          f"  entropy={d['entropy_score']:.4f}")
    print(f"       quarantine={d['quarantine_status']:12s}"
          f"  key_rotation={d['key_rotation']}"
          f"  pattern_match={d['pattern_match']}")
    print(f"       attack_type={d['attack_type']:16s}"
          f"  actions: {', '.join(d['actions_triggered'])}")
    if d.get("entropy_alert"):
        print(f"       ⚠ ENTROPY ALERT: {d['entropy_detail']['alert_reason']}")
    if d["response_detail"]["notes"]:
        for note in d["response_detail"]["notes"]:
            print(f"       ↳ {note}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    engine = SecurityEngine(entropy_window=20)
    results = []

    # ── PHASE 1: Warm-up with benign transactions ──────────────────────────
    section("PHASE 1 — Benign Transactions (warm-up population statistics)")
    benign_txs = [
        make_real_tx("RELIANCE.NS", 10, 2850.0),
        make_real_tx("TCS.NS",       5, 3900.0),
        make_real_tx("HDFCBANK.NS",  8, 1650.0),
        make_real_tx("INFY.NS",     12, 1580.0),
        make_real_tx("ITC.NS",      20,  480.0),
        make_real_tx("WIPRO.NS",    15,  525.0),
        make_real_tx("RELIANCE.NS",  6, 2860.0),
        make_real_tx("TCS.NS",       3, 3905.0),
    ]
    for i, tx in enumerate(benign_txs):
        r = engine.process_transaction_security(tx)
        results.append(r)
        print_report(r, i + 1)

    # ── PHASE 2: Entropy attack (crafted weak-RNG hash) ───────────────────
    section("PHASE 2 — Entropy Attacks (crafted low-entropy hashes)")
    entropy_attacks = [
        {"tx_id": "ATK_ENT_001", "sha3_hash": "00" * 32,
         "verification_status": True, "signed_at": time.time(),
         "payload": {"type": "VIRTUAL_BUY", "ticker": "FAKE.NS", "qty": 100}},
        {"tx_id": "ATK_ENT_002", "sha3_hash": "ff" * 32,
         "verification_status": True, "signed_at": time.time() + 1,
         "payload": {"type": "VIRTUAL_BUY", "ticker": "FAKE.NS", "qty": 100}},
        {"tx_id": "ATK_ENT_003", "sha3_hash": "01" * 32,
         "verification_status": True, "signed_at": time.time() + 2,
         "payload": {"type": "VIRTUAL_BUY", "ticker": "FAKE.NS", "qty": 100}},
    ]
    for i, tx in enumerate(entropy_attacks):
        r = engine.process_transaction_security(tx)
        results.append(r)
        print_report(r, len(results))

    # ── PHASE 3: Signature failures ───────────────────────────────────────
    section("PHASE 3 — Signature Failures (hard anomaly)")
    sig_fail_txs = [
        {"tx_id": "ATK_SIG_001", "sha3_hash": "a3b4" * 16,
         "verification_status": False, "signed_at": time.time(),
         "payload": {"type": "VIRTUAL_BUY", "ticker": "FRAUD.NS", "qty": 9999}},
        {"tx_id": "ATK_SIG_002", "sha3_hash": "dead" * 16,
         "verification_status": False, "signed_at": time.time() + 0.5,
         "payload": {"type": "VIRTUAL_BUY", "ticker": "FRAUD.NS", "qty": 9999}},
    ]
    for tx in sig_fail_txs:
        r = engine.process_transaction_security(tx)
        results.append(r)
        print_report(r, len(results))

    # ── PHASE 4: Burst / frequency attack ────────────────────────────────
    section("PHASE 4 — Burst Attack (high-frequency transactions)")
    t_burst = time.time()
    burst_txs = []
    for i in range(10):
        burst_txs.append({
            "tx_id":              f"ATK_BURST_{i:03d}",
            "sha3_hash":          f"{i:064x}",
            "verification_status": True,
            "signed_at":          t_burst + i * 0.05,   # 50ms apart
            "payload":            {"type": "VIRTUAL_BUY", "ticker": "SPAM.NS",
                                   "qty": 1},
        })
    for tx in burst_txs:
        r = engine.process_transaction_security(tx)
        results.append(r)
        if r.threat_level != "low" or r == burst_txs[-1]:
            print_report(r, len(results))
        else:
            print(f"  [{len(results):02d}] 🟢 burst tx ok  risk={r.anomaly_score:.2f}")

    # ── PHASE 5: Pattern replay (previously seen threat) ─────────────────
    section("PHASE 5 — Pattern Replay (known threat recognised from memory)")
    # Re-send an entropy attack after the memory has stored it
    replay = {"tx_id": "ATK_REPLAY_001", "sha3_hash": "00" * 32,
              "verification_status": True, "signed_at": time.time(),
              "payload": {"type": "VIRTUAL_BUY", "ticker": "FAKE.NS", "qty": 100}}
    r = engine.process_transaction_security(replay)
    results.append(r)
    print_report(r, len(results))

    # ── PHASE 6: Key rotation status ──────────────────────────────────────
    section("PHASE 6 — Key Mutation System Status")
    ks = engine._key_mutation.status()
    print(f"  Current generation : {ks['current_generation']}")
    print(f"  Key ID             : {ks['current_key_id']}")
    print(f"  Fingerprint        : {ks['current_fingerprint'][:32]}…")
    print(f"  Key age            : {ks['key_age_hours']:.3f} hours")
    print(f"  Total rotations    : {ks['total_rotations']}")
    print(f"  Consecutive HIGH   : {ks['consecutive_high']}")
    print(f"  Archive size       : {ks['archive_size']}")
    if ks["rotation_log"]:
        print("  Recent rotations:")
        for ev in ks["rotation_log"]:
            print(f"    [{ev['event_id']}] {ev['reason'][:50]}…")

    # ── PHASE 7: Force key rotation ───────────────────────────────────────
    section("PHASE 7 — Force Key Rotation (manual trigger)")
    rotated = engine._key_mutation.force_rotate("DEMO_FORCE_ROTATE")
    print(f"  Rotation executed: {rotated}")
    ks2 = engine._key_mutation.status()
    print(f"  New generation   : {ks2['current_generation']}")
    print(f"  New key ID       : {ks2['current_key_id']}")
    if ks2["rotation_log"]:
        last_ev = ks2["rotation_log"][-1]
        print(f"  Cert sig (hex)   : {last_ev['cert_sig_hex'][:32]}…")

    # ── FINAL SYSTEM STATUS ────────────────────────────────────────────────
    section("SYSTEM STATUS SUMMARY")
    status = engine.system_status()
    print(f"  Transactions processed : {status['transactions_processed']}")
    print(f"  Entropy monitor        : "
          f"mean={status['entropy_monitor']['ewma_mean']:.4f}  "
          f"std={status['entropy_monitor']['ewma_std']:.4f}")
    print(f"  Response engine        : "
          f"total={status['response_engine']['total_responses']}  "
          f"quarantined={status['response_engine']['total_quarantined']}  "
          f"rotation_signals={status['response_engine']['key_rotation_signals']}")
    print(f"  Threat memory          : "
          f"{status['threat_memory']['total_records']} patterns  "
          f"recognition_rate={status['threat_memory']['recognition_rate']:.2%}")
    print(f"  Attack type counts:")
    for at, cnt in status["threat_memory"].get("attack_type_counts", {}).items():
        print(f"    {at:20s} : {cnt}")

    # ── Threat level breakdown ─────────────────────────────────────────────
    section("RESULTS — Threat Level Distribution")
    by_level = status["response_engine"]["by_threat_level"]
    total    = sum(by_level.values())
    for lvl, cnt in by_level.items():
        icons = {"low": "🟢", "medium": "🟡", "high": "🔴"}
        pct   = cnt / total * 100 if total else 0
        bar   = "█" * int(pct / 3)
        print(f"  {icons.get(lvl,'⚪')} {lvl:8s}  {cnt:3d} / {total}  "
              f"({pct:5.1f}%)  {bar}")

    # Save full JSON report
    out = {
        "system_status":  status,
        "all_reports":    [r.to_dict() for r in results],
        "key_history":    [k.to_dict() for k in engine._key_mutation._archive],
        "threat_memory":  engine._threat_memory.get_all_records(),
        "rotation_log":   [e.to_dict() for e in engine._key_mutation._rotation_log],
    }
    out_path = os.path.join(os.path.dirname(__file__), "milestone6_security_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Full JSON saved: {out_path}")
    print("═" * 68 + "\n")


if __name__ == "__main__":
    main()
