"""
run_milestone6_complete.py
==========================
Interactive Virtual P2P Trade Simulator with Advanced Security Testing.

Flow:
  PART 1 — You describe your portfolio and the trade you want to simulate.
           The system runs the virtual trade and shows you the impact.

  PART 2 — Eight real-world DeFi/trading attack patterns are simulated
           against your transaction. The PQC Immune Defense and Bayesian
           Immune Defense both analyse each one. The Bayesian engine stores
           threat memory persistently in MongoDB so it learns across sessions.

Run:
    python run_milestone6_complete.py

MongoDB (optional):
    Set MONGO_URI in your environment or .env file:
        MONGO_URI=mongodb://localhost:27017
    If unset the Bayesian memory falls back to the JSON file store.
"""

import sys, os, time, logging, json, math

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("immune_bayesian").setLevel(logging.CRITICAL)

# ── Load milestone6_complete.py ───────────────────────────────────────────────
import types, importlib.util, uuid as _uuid_mod

def _find_and_load():
    here = os.path.dirname(os.path.abspath(__file__))
    for folder in [
        here,
        os.path.join(here, "portfolio"),
        os.path.join(here, ".."),
        os.path.join(here, "..", "portfolio"),
        os.path.join(here, "..", ".."),
        os.path.join(here, "..", "src"),
    ]:
        candidate = os.path.normpath(os.path.join(folder, "milestone6_complete.py"))
        if os.path.isfile(candidate):
            spec = importlib.util.spec_from_file_location("milestone6_complete", candidate)
            mod  = types.ModuleType("milestone6_complete")
            mod.__spec__ = spec
            sys.modules["milestone6_complete"] = mod
            sys.path.insert(0, os.path.dirname(candidate))
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Cannot find milestone6_complete.py. "
        "Place it in the same folder as this runner and try again."
    )

_mod = _find_and_load()

run_virtual_trade_simulation = _mod.run_virtual_trade_simulation
generate_keypair             = _mod.generate_keypair
build_transaction_object     = _mod.build_transaction_object
SecurityEngine               = _mod.SecurityEngine
BayesianSecurityPipeline     = _mod.BayesianSecurityPipeline

import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_PROJECT_ROOT   = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
)
_ARTIFACTS_DIR  = os.path.join(_PROJECT_ROOT, "artifacts", "milestone6")
_SHARED_TX_PATH = os.path.join(_ARTIFACTS_DIR, "latest_cli_transaction.json")
_MEMORY_PATH    = os.path.join(_ARTIFACTS_DIR, "bayesian_memory.json")

# Unique ID for this run session (used to group MongoDB records)
_SESSION_ID = str(_uuid_mod.uuid4())[:8]


# =============================================================================
# MONGODB PERSISTENT MEMORY
# =============================================================================

class MongoMemoryStore:
    """
    Persistent memory backend for the Bayesian Immune System using MongoDB.

    Each threat pattern is stored as one document:
    {
        record_id:             str,     unique pattern ID
        signal_vector:         [float], 6-element feature vector
        posterior_probability: float,   P(threat) at time of detection
        threat_level:          str,     SAFE/MONITOR/ELEVATED_RISK/CRITICAL_THREAT
        mitigation_action:     str,     comma-separated response actions
        tx_id:                 str,     originating transaction ID
        attack_type:           str,     name of the attack pattern
        attack_key:            str,     menu key (1-8)
        first_seen:            float,   Unix timestamp (first occurrence)
        last_seen:             float,   Unix timestamp (most recent occurrence)
        occurrence_count:      int,     how many times this pattern matched
        session_id:            str,     run session that created this record
    }
    """

    COLLECTION = "bayesian_threat_memory"

    def __init__(self, uri: str, db_name: str = "portfolioiq_security"):
        try:
            from pymongo import MongoClient
            self._client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            self._client.admin.command("ping")
            self._db  = self._client[db_name]
            self._col = self._db[self.COLLECTION]
            # Ensure fast lookups
            self._col.create_index("record_id",   unique=True)
            self._col.create_index("threat_level")
            self._col.create_index("last_seen")
            self._col.create_index("attack_type")
            self._connected = True
            print(f"  ✓ MongoDB connected  uri={uri}  db={db_name}  "
                  f"collection={self.COLLECTION}")
        except Exception as e:
            self._connected = False
            print(f"  ✗ MongoDB unavailable ({e}).  Falling back to JSON memory.")

    @property
    def connected(self) -> bool:
        return self._connected

    # ------------------------------------------------------------------
    # Upsert a threat pattern
    # ------------------------------------------------------------------

    def upsert_pattern(self, record: dict) -> str:
        """
        Store a new threat pattern.  If a closely matching pattern already
        exists (cosine similarity >= 0.90), increment its occurrence_count
        instead of inserting a duplicate.  Returns the record_id used.
        """
        if not self._connected:
            return record.get("record_id", "")
        try:
            existing = self._find_similar(record.get("signal_vector", []))
            if existing:
                self._col.update_one(
                    {"record_id": existing["record_id"]},
                    {"$inc": {"occurrence_count": 1},
                     "$set": {
                         "last_seen":            time.time(),
                         "posterior_probability": record["posterior_probability"],
                         "mitigation_action":     record["mitigation_action"],
                     }}
                )
                return existing["record_id"]
            else:
                record.setdefault("occurrence_count", 1)
                record.setdefault("first_seen",       time.time())
                record.setdefault("last_seen",        time.time())
                self._col.insert_one(record)
                return record["record_id"]
        except Exception as e:
            logging.debug("MongoMemoryStore.upsert_pattern: %s", e)
            return record.get("record_id", "")

    def _find_similar(self, signal_vector: list, threshold: float = 0.90) -> dict:
        """Return the most similar stored pattern or None."""
        if not signal_vector or not self._connected:
            return None
        try:
            v     = np.array(signal_vector, dtype=float)
            nv    = np.linalg.norm(v)
            if nv < 1e-8:
                return None
            best_sim, best_doc = 0.0, None
            cursor = self._col.find(
                {"threat_level": {"$in": ["MONITOR", "ELEVATED_RISK", "CRITICAL_THREAT"]}},
                {"record_id": 1, "signal_vector": 1}
            ).limit(500)
            for doc in cursor:
                sv = doc.get("signal_vector", [])
                if len(sv) != len(signal_vector):
                    continue
                u  = np.array(sv, dtype=float)
                nu = np.linalg.norm(u)
                if nu < 1e-8:
                    continue
                sim = float(np.dot(v / nv, u / nu))
                if sim > best_sim:
                    best_sim, best_doc = sim, doc
            return best_doc if best_sim >= threshold else None
        except Exception as e:
            logging.debug("MongoMemoryStore._find_similar: %s", e)
            return None

    def get_all_patterns(self, limit: int = 1000) -> list:
        """Return all stored patterns, most recent first."""
        if not self._connected:
            return []
        try:
            return list(
                self._col.find({}, {"_id": 0})
                .sort("last_seen", -1)
                .limit(limit)
            )
        except Exception:
            return []

    def summary(self) -> dict:
        if not self._connected:
            return {"connected": False}
        try:
            total    = self._col.count_documents({})
            by_level = {
                lvl: self._col.count_documents({"threat_level": lvl})
                for lvl in ["SAFE", "MONITOR", "ELEVATED_RISK", "CRITICAL_THREAT"]
            }
            by_attack = {
                atk: self._col.count_documents({"attack_type": atk})
                for atk in self._col.distinct("attack_type")
            }
            return {
                "connected":       True,
                "total_patterns":  total,
                "by_threat_level": by_level,
                "by_attack_type":  by_attack,
            }
        except Exception:
            return {"connected": True, "total_patterns": 0}

    def close(self):
        if self._connected:
            try:
                self._client.close()
            except Exception:
                pass


def _get_mongo_uri() -> str:
    """Read MongoDB URI from environment or .env file. Returns None if absent."""
    uri = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
    if uri:
        return uri
    env_path = os.path.join(_PROJECT_ROOT, ".env")
    if os.path.isfile(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("MONGO_URI=") or line.startswith("MONGODB_URI="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


# =============================================================================
# UI helpers
# =============================================================================

def banner(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def sub(title):
    print(f"\n  -- {title} --")

def ask(prompt, default=None):
    suffix = f" [{default}]" if default is not None else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val if val else str(default) if default is not None else val

def ask_float(prompt, default=None, min_val=None):
    while True:
        raw = ask(prompt, default)
        try:
            val = float(raw)
            if min_val is not None and val < min_val:
                print(f"    X  Please enter a value of at least {min_val}.")
                continue
            return val
        except ValueError:
            print(f"    X  '{raw}' is not a number. Try again.")

def ask_yes_no(prompt, default="y"):
    raw = ask(prompt + " (y/n)", default).lower()
    return raw.startswith("y")

def fmt_inr(x):    return f"Rs.{x:,.2f}"
def fmt_pct(x):    return f"{x*100:.2f}%"

def pct_change_arrow(x):
    if x > 0.0001:  return f"+{fmt_pct(x)} up"
    if x < -0.0001: return f"{fmt_pct(x)} down"
    return "no change"


# =============================================================================
# Synthetic returns helper
# =============================================================================

def make_synthetic_returns(tickers, n_days=504, seed=42):
    rng  = np.random.default_rng(seed)
    data = rng.normal(0.0003, 0.014, size=(n_days, len(tickers)))
    return pd.DataFrame(data, columns=tickers)


# =============================================================================
# PART 1 -- Virtual Trade Setup
# =============================================================================

def collect_portfolio():
    banner("PART 1 -- YOUR CURRENT PORTFOLIO")
    print("""
  Tell us what stocks you hold so we can show how your proposed trade
  would change your portfolio.  Press Enter on the ticker to finish.
    """)
    holdings, prices = {}, {}
    print("  Example:  Ticker = RELIANCE.NS  |  Shares = 10  |  Price = 2850")
    print()
    while True:
        raw = input("  Stock ticker (or press Enter to finish): ").strip().upper()
        if not raw:
            if not holdings:
                print("  You need at least one stock.")
                continue
            break
        ticker = raw if "." in raw else raw + ".NS"
        shares = ask_float(f"  How many shares of {ticker} do you hold?", min_val=0.01)
        price  = ask_float(f"  Current price of {ticker} (Rs.)?",         min_val=0.01)
        holdings[ticker] = shares
        prices[ticker]   = price
        print(f"  Added: {ticker}  --  {shares} shares x {fmt_inr(price)} = {fmt_inr(shares*price)}")
        print()
    total = sum(holdings[t] * prices[t] for t in holdings)
    print(f"\n  Portfolio: {len(holdings)} stock(s)  |  Total value: {fmt_inr(total)}")
    return holdings, prices, total


def collect_trade(holdings, prices):
    banner("THE TRADE YOU WANT TO TEST")
    print("""
  This is a VIRTUAL trade -- no real money moves.
  We will show you how it would affect your portfolio.
    """)
    raw      = ask("Which stock do you want to buy? (e.g. INFY.NS)").strip().upper()
    ticker   = raw if "." in raw else raw + ".NS"
    quantity = ask_float(f"How many shares of {ticker}?", min_val=1)
    price    = ask_float(f"At what price per share (Rs.)?", min_val=0.01)
    print(f"\n  Trade cost: {quantity} x {fmt_inr(price)} = {fmt_inr(quantity * price)}")
    return ticker, quantity, price


def run_trade_and_show_results(holdings, prices, total_value, ticker, quantity, price):
    banner("RUNNING YOUR VIRTUAL TRADE")
    print("  Please wait -- running simulation and Monte Carlo projections...\n")
    all_tickers = list(set(list(holdings.keys()) + [ticker]))
    returns_df  = make_synthetic_returns(all_tickers)
    try:
        result = run_virtual_trade_simulation(
            ticker        = ticker,
            quantity      = quantity,
            price         = price,
            real_holdings = holdings,
            real_prices   = prices,
            daily_returns = returns_df,
            total_value   = total_value,
            n_mc_paths    = 1_000,
        )
    except Exception as e:
        print(f"  X Simulation failed: {e}")
        return None

    sub("Transaction Receipt")
    tx = result.get("transaction_record", {})
    print(f"  Transaction ID  : {tx.get('tx_id', 'N/A')}")
    print(f"  Signature       : {tx.get('signature',{}).get('scheme','N/A')}")
    verified = tx.get("verification_status", False)
    print(f"  Verified        : {'OK Authentic' if verified else 'FAIL Signature failed'}")

    sub("How This Trade Changes Your Portfolio")
    imp = result.get("portfolio_impact", {})
    print(f"  Expected annual return : {pct_change_arrow(imp.get('expected_return_change', 0))}")
    print(f"  Volatility             : {pct_change_arrow(imp.get('volatility_change', 0))}  (lower is safer)")
    print(f"  Sharpe ratio           : {imp.get('sharpe_change', 0):+.3f}")
    print(f"  Tail risk (CVaR 95%)   : {pct_change_arrow(imp.get('cvar_change', 0))}  (lower is safer)")
    print(f"  Diversification        : {pct_change_arrow(imp.get('diversification_change', 0))}")

    sub("Long-Term Projection (Monte Carlo -- 1,000 paths)")
    mc = result.get("monte_carlo_projection", {})
    print(f"  Best horizon : {mc.get('best_horizon', '?')}")
    print(f"\n  Verdict: {mc.get('overall_verdict', 'N/A')}")
    for h in ["1Y", "3Y", "5Y"]:
        hd = mc.get(h)
        if not hd:
            continue
        real_v = hd.get("real",    {}).get("expected_value", 0)
        virt_v = hd.get("virtual", {}).get("expected_value", 0)
        delta  = hd.get("deltas",  {}).get("expected_value_delta", 0)
        sign   = "+" if delta >= 0 else ""
        print(f"  {h}  |  Without: {fmt_inr(real_v)}  -->  With: {fmt_inr(virt_v)}  ({sign}{fmt_inr(delta)})")

    sub("Risk Summary")
    print(f"  {result.get('risk_summary', 'N/A')}")
    return result


# =============================================================================
# PART 2 -- Eight Advanced Attack Scenarios
# =============================================================================

ATTACK_MENU = {
    "1": {
        "name":    "Adversarial Evasion",
        "desc":    "A carefully crafted transaction designed to slip past "
                   "signature checks with minimal modification.",
        "mechanism": (
            "The attacker makes the payload look legitimate but corrupts one "
            "bit of the signature so verification barely fails.  The hash "
            "still looks random (high entropy) so entropy checks pass.  "
            "Only the signature check catches it."
        ),
        "what_changes": [
            "verification_status -> False  (signature corrupted by 1 bit)",
            "_entropy_composite_score -> 0.91  (hash still looks valid)",
            "Payload and hash left unchanged to avoid entropy flags",
        ],
        "signals_hit": {
            "signature_risk":    0.95,
            "entropy_deviation": 0.09,
            "replay_indicator":  0.0,
            "frequency_anomaly": 0.2,
        },
        "tx_override": {"verification_status": False},
        "entropy":     0.91,
    },
    "2": {
        "name":    "Sybil Trust Poisoning",
        "desc":    "Multiple fake identities flood the system to corrupt "
                   "the sender trust baseline.",
        "mechanism": (
            "The attacker creates many new sender identities each submitting "
            "one or two transactions.  Individually each looks legitimate. "
            "Over time this poisons the sender-deviation baseline making "
            "future attacks appear normal.  Modelled as 6 burst submissions "
            "each with a different spoofed sender fingerprint."
        ),
        "what_changes": [
            "tx_id changes each attempt: SYBIL_001, SYBIL_002...",
            "Sender fingerprint randomised each attempt",
            "signed_at offset by 0.2s per attempt",
            "All signatures valid -- attack relies on volume not forgery",
        ],
        "signals_hit": {
            "signature_risk":    0.0,
            "frequency_anomaly": 0.85,
            "sender_deviation":  0.9,
            "size_anomaly":      0.3,
        },
        "burst":        True,
        "sender_spoof": True,
        "entropy":      0.94,
    },
    "3": {
        "name":    "Model Extraction",
        "desc":    "Systematic probing to discover where the detection "
                   "thresholds are set.",
        "mechanism": (
            "The attacker sweeps transactions with incrementally varying "
            "entropy scores observing which ones get flagged.  This reveals "
            "the detection threshold.  Modelled as 6 submissions with "
            "entropy stepping from 0.95 down to 0.60 and payload size "
            "growing 200 bytes per attempt."
        ),
        "what_changes": [
            "6 attempts with entropy: 0.95 -> 0.82 -> 0.72 -> 0.65 -> 0.61 -> 0.60",
            "Payload size grows by 200 bytes per attempt",
            "All signatures remain valid",
        ],
        "signals_hit": {
            "entropy_deviation": 0.40,
            "size_anomaly":      0.55,
            "frequency_anomaly": 0.75,
            "signature_risk":    0.0,
        },
        "burst":         True,
        "model_extract": True,
        "entropy":       0.80,
    },
    "4": {
        "name":    "Side-Channel Timing Leak",
        "desc":    "Transactions submitted at precisely measured intervals "
                   "to infer threat scores from response latency.",
        "mechanism": (
            "The attacker discovers that high-risk transactions take slightly "
            "longer to process.  By measuring latency they infer threat scores "
            "without being flagged.  Modelled as 6 transactions with sub-"
            "millisecond uniform timing (gap = 0.001s) -- a statistical anomaly "
            "in the frequency tracker."
        ),
        "what_changes": [
            "signed_at gaps set to 0.001s (highly uniform -- not normal traffic)",
            "Payload unchanged -- attack is in the timing not the content",
            "Signatures remain valid",
        ],
        "signals_hit": {
            "frequency_anomaly": 0.95,
            "sender_deviation":  0.45,
            "signature_risk":    0.0,
            "entropy_deviation": 0.05,
        },
        "burst":   True,
        "timing":  True,
        "entropy": 0.96,
    },
    "5": {
        "name":    "Fault Injection",
        "desc":    "Hardware-level fault corrupts both the hash and "
                   "signature simultaneously.",
        "mechanism": (
            "The attacker injects a fault during signing (e.g. via malware "
            "corrupting the RNG).  This produces a hash with near-zero entropy "
            "AND an invalid signature.  Both engines receive maximum-severity "
            "signals simultaneously."
        ),
        "what_changes": [
            "verification_status -> False  (signature invalid)",
            "sha3_hash -> 0000...0000  (all-zero: zero entropy)",
            "_entropy_composite_score -> 0.003  (near-zero randomness)",
            "Both hard penalties fire: SIG_PENALTY + ENTROPY_PENALTY",
        ],
        "signals_hit": {
            "signature_risk":    0.95,
            "entropy_deviation": 0.997,
            "replay_indicator":  0.0,
            "frequency_anomaly": 0.2,
        },
        "tx_override":   {"verification_status": False},
        "hash_override": "00" * 32,
        "entropy":       0.003,
    },
    "6": {
        "name":    "Triangular Fraud",
        "desc":    "Three coordinated transactions form a circular trade "
                   "to manipulate recorded prices.",
        "mechanism": (
            "Attacker A sells to B, B sells to C, C sells back to A -- all "
            "at manipulated prices.  Each individual transaction looks "
            "legitimate but the circular flow is fraudulent.  Modelled as "
            "3 burst submissions with payload prices in a circular pattern."
        ),
        "what_changes": [
            "3 transactions fired in sequence (A->B, B->C, C->A)",
            "Payload price multiplied by 2.0 / 0.5 / 1.0 per leg",
            "Each has a unique tx_id and valid signature",
            "Size slightly larger than normal (richer payload)",
        ],
        "signals_hit": {
            "size_anomaly":      0.65,
            "frequency_anomaly": 0.70,
            "sender_deviation":  0.60,
            "signature_risk":    0.0,
        },
        "burst":      True,
        "n_burst":    3,
        "triangular": True,
        "entropy":    0.92,
    },
    "7": {
        "name":    "Price Oracle Manipulation",
        "desc":    "A transaction with a 10x inflated price to corrupt "
                   "the on-chain price feed.",
        "mechanism": (
            "The attacker submits a transaction whose payload price is 10x "
            "market value.  If accepted this corrupts any system reading the "
            "transaction log as a price oracle.  The payload is also much "
            "larger than normal."
        ),
        "what_changes": [
            "Payload price set to 10x normal (oracle poisoning)",
            "Transaction size inflated 4x (large data payload)",
            "Signature valid -- attack is in the data not the crypto",
            "_entropy_composite_score -> 0.85 (hash still random-looking)",
        ],
        "signals_hit": {
            "size_anomaly":      0.92,
            "sender_deviation":  0.55,
            "signature_risk":    0.0,
            "entropy_deviation": 0.15,
        },
        "size_inflate": 4.0,
        "oracle":       True,
        "entropy":      0.85,
    },
    "8": {
        "name":    "Front-Running (Sandwich Attack)",
        "desc":    "Two transactions bracket a victim trade to extract "
                   "value from price movement.",
        "mechanism": (
            "The attacker buys just before the victim's large trade "
            "(transaction 1) then sells immediately after it pushes the "
            "price up (transaction 2).  Modelled as two submissions with "
            "50ms gap and payload prices stepping up then down."
        ),
        "what_changes": [
            "Transaction 1: BUY at 0.99x market price",
            "Transaction 2 (50ms later): SELL at 1.01x market price",
            "Gap between them: 0.05s -- statistically anomalous timing",
            "Both signatures valid -- attack is in sequencing",
        ],
        "signals_hit": {
            "frequency_anomaly": 0.88,
            "sender_deviation":  0.50,
            "size_anomaly":      0.30,
            "signature_risk":    0.0,
        },
        "burst":    True,
        "n_burst":  2,
        "frontrun": True,
        "entropy":  0.93,
    },
}


# ---------------------------------------------------------------------------
# Build modified transaction for each attack
# ---------------------------------------------------------------------------

def build_attack_tx(original_tx: dict, attack_def: dict, burst_index: int = 0) -> dict:
    """Corrupt or modify the transaction according to the attack definition."""
    import secrets as _secrets
    tx = dict(original_tx)

    # Direct field overrides
    for k, v in attack_def.get("tx_override", {}).items():
        tx[k] = v

    # Hash override
    if "hash_override" in attack_def:
        tx["sha3_hash"] = attack_def["hash_override"]

    # Entropy injection
    tx["_entropy_composite_score"] = attack_def.get("entropy", 0.92)

    # Burst / timing overrides
    is_multi = (
        attack_def.get("burst") or attack_def.get("timing") or
        attack_def.get("frontrun") or attack_def.get("triangular")
    )
    if is_multi:
        if attack_def.get("timing"):
            gap = 0.001 * (burst_index + 1)
        elif attack_def.get("frontrun"):
            gap = 0.05 * (burst_index + 1)
        else:
            gap = 0.05 * (burst_index + 1)

        prefix = attack_def["name"].split()[0].upper()[:6]
        tx["tx_id"]     = f"{prefix}_{burst_index:03d}_{original_tx.get('tx_id','')[:8]}"
        tx["signed_at"] = time.time() + gap

    # Sybil: randomise sender fingerprint
    if attack_def.get("sender_spoof"):
        tx["public_key"] = {"fingerprint": _secrets.token_hex(8)}

    # Model extraction: step entropy and payload size down
    if attack_def.get("model_extract"):
        entropy_steps = [0.95, 0.82, 0.72, 0.65, 0.61, 0.60]
        tx["_entropy_composite_score"] = entropy_steps[min(burst_index, len(entropy_steps)-1)]
        payload = dict(tx.get("payload", {}))
        payload["_probe_padding"] = "X" * (200 * burst_index)
        tx["payload"] = payload

    # Oracle: inflate price and payload
    if attack_def.get("oracle"):
        payload = dict(tx.get("payload", {}))
        payload["price"] = payload.get("price", 1000) * 10
        payload["_bulk"] = "0" * 1000
        tx["payload"] = payload

    # Triangular: rotate price multiplier across 3 legs
    if attack_def.get("triangular"):
        legs = [2.0, 0.5, 1.0]
        payload = dict(tx.get("payload", {}))
        payload["price"] = payload.get("price", 1000) * legs[burst_index % 3]
        tx["payload"] = payload

    # Front-run: buy then sell
    if attack_def.get("frontrun"):
        payload = dict(tx.get("payload", {}))
        base    = payload.get("price", 1000)
        if burst_index == 0:
            payload["price"] = base * 0.99
            payload["type"]  = "VIRTUAL_BUY"
        else:
            payload["price"] = base * 1.01
            payload["type"]  = "VIRTUAL_SELL"
        tx["payload"] = payload

    return tx


# ---------------------------------------------------------------------------
# Print the attack menu
# ---------------------------------------------------------------------------

def print_attack_menu():
    print("""
  Choose an attack scenario to test.
  Both the PQC and Bayesian engines will analyse your real transaction.
    """)
    print("  " + "-" * 66)
    for key, atk in ATTACK_MENU.items():
        print(f"  [{key}]  {atk['name']}")
        print(f"       {atk['desc']}")
        print()
    print("  [A]  Run all 8 attacks in sequence")
    print("  [0]  Exit security testing")
    print("  " + "-" * 66)


# ---------------------------------------------------------------------------
# Show attack setup before running
# ---------------------------------------------------------------------------

def show_attack_setup(attack_def: dict):
    name = attack_def["name"]
    print()
    print(f"  ATTACK: {name}")
    print("  " + "=" * 60)
    print("  How it works:")
    words, line = attack_def["mechanism"].split(), ""
    for word in words:
        if len(line) + len(word) + 1 > 63:
            print(f"    {line}")
            line = word
        else:
            line = (line + " " + word).strip()
    if line:
        print(f"    {line}")
    print()
    print("  What this attack changes:")
    for item in attack_def["what_changes"]:
        print(f"    * {item}")
    print()
    print("  Expected signals (0=none, 1=maximum):")
    for sig, val in attack_def["signals_hit"].items():
        bar = "#" * int(val * 20) + "." * (20 - int(val * 20))
        print(f"    {sig:<26} [{bar}] {val:.2f}")
    print()


# ---------------------------------------------------------------------------
# Run one attack
# ---------------------------------------------------------------------------

def run_security_test(
    original_tx:  dict,
    attack_key:   str,
    attack_def:   dict,
    sec_engine,
    bay_engine,
    mongo_store=None,
):
    """Execute one attack through both engines and print full explanations."""
    banner(f"ATTACK {attack_key} -- {attack_def['name'].upper()}")
    show_attack_setup(attack_def)
    input("  Press Enter to run this attack...")
    print()

    n_burst  = attack_def.get("n_burst", 6 if (
        attack_def.get("burst") or attack_def.get("timing") or
        attack_def.get("frontrun") or attack_def.get("triangular")
    ) else 1)
    is_multi = n_burst > 1

    pqc_results = []
    bay_results = []

    for i in range(n_burst):
        atk_tx = build_attack_tx(original_tx, attack_def, burst_index=i)

        if attack_def.get("replay") and i == 0:
            sec_engine.process_transaction_security(dict(original_tx))
            bay_engine.process_transaction_security(dict(original_tx))

        pqc_r = sec_engine.process_transaction_security(atk_tx)
        bay_r = bay_engine.process_transaction_security(atk_tx)
        pqc_results.append(pqc_r)
        bay_results.append(bay_r)

        # Persist to MongoDB
        if mongo_store and mongo_store.connected and bay_r.posterior_probability > 0.15:
            sv = []
            if isinstance(bay_r.signal_vector, dict):
                sv = [
                    float(bay_r.signal_vector.get("size_anomaly",      0)),
                    float(bay_r.signal_vector.get("frequency_anomaly", 0)),
                    float(bay_r.signal_vector.get("sender_deviation",  0)),
                    float(bay_r.signal_vector.get("signature_risk",    0)),
                    float(bay_r.signal_vector.get("entropy_deviation", 0)),
                    float(bay_r.signal_vector.get("replay_indicator",  0)),
                ]
            record = {
                "record_id":             str(_uuid_mod.uuid4()),
                "signal_vector":         sv,
                "posterior_probability": round(bay_r.posterior_probability, 6),
                "threat_level":          bay_r.threat_level,
                "mitigation_action":     ", ".join(bay_r.actions_triggered),
                "tx_id":                 atk_tx.get("tx_id", ""),
                "attack_type":           attack_def["name"],
                "attack_key":            attack_key,
                "session_id":            _SESSION_ID,
            }
            saved_id = mongo_store.upsert_pattern(record)
            if i == 0:
                print(f"  [MongoDB] pattern saved  id={saved_id[:16]}...")

    # PQC results
    print("  PQC IMMUNE DEFENSE ENGINE")
    print("  Rule-based: entropy + signature + anomaly scoring")
    print()
    for i, r in enumerate(pqc_results):
        label  = f"Attempt {i+1}/{n_burst}" if is_multi else "Result"
        status = "BLOCKED" if r.quarantine_status == "QUARANTINED" else "ALLOWED"
        print(f"  {label}: threat={r.threat_level.upper():<8}  "
              f"entropy={r.entropy_score:.4f}  anomaly={r.anomaly_score:.4f}  "
              f"outcome={status}")
        if r.quarantine_status == "QUARANTINED":
            print(f"           Actions: {', '.join(r.actions_triggered)}")
        if r.key_rotation:
            print(f"           KEY ROTATION triggered")
    print()

    # Bayesian results
    print("  BAYESIAN IMMUNE DEFENSE ENGINE")
    print("  Probabilistic: prior -> LLR update -> posterior")
    print()
    for i, r in enumerate(bay_results):
        label  = f"Attempt {i+1}/{n_burst}" if is_multi else "Result"
        prob   = r.posterior_probability
        status = "QUARANTINED" if r.quarantine_status == "QUARANTINED" else "CLEAR"
        print(f"  {label}: level={r.threat_level:<18}  "
              f"P(threat)={prob:.4f} ({prob*100:.1f}%)")
        print(f"           prior={r.prior:.4f}  LR={r.likelihood_ratio:.4f}  "
              f"mem_sim={r.memory_similarity:.4f}  "
              f"mem_boost={'YES' if r.memory_boosted_prior else 'no'}  "
              f"status={status}")
        sv = r.signal_vector if isinstance(r.signal_vector, dict) else {}
        if sv:
            sig_parts = [
                f"{k[:4]}={float(v):.2f}"
                for k, v in sv.items()
                if isinstance(v, (int, float))
            ]
            print(f"           signals: {' '.join(sig_parts)}")
        print()

    # Outcome
    final_pqc = pqc_results[-1]
    final_bay = bay_results[-1]
    pqc_blocked = final_pqc.quarantine_status == "QUARANTINED"
    bay_blocked = final_bay.quarantine_status == "QUARANTINED"

    print("  OUTCOME")
    print("  " + "-" * 50)
    if pqc_blocked and bay_blocked:
        print("  Both engines blocked this attack independently.")
        print("  The transaction would never reach the trading system.")
    elif pqc_blocked:
        print("  PQC engine blocked this attack.")
        print("  Bayesian engine flagged but did not quarantine.")
    elif bay_blocked:
        print("  Bayesian engine blocked this attack.")
        print("  PQC engine flagged but did not quarantine.")
    else:
        prob = final_bay.posterior_probability
        if final_pqc.threat_level == "medium" or prob > 0.20:
            print(f"  Neither engine quarantined, but both raised flags.")
            print(f"  P(threat)={prob*100:.1f}% -- would trigger enhanced monitoring.")
        else:
            print(f"  Attack not strongly detected in isolation.")
            print(f"  Repeated attempts accumulate in memory over time.")
    print()


# ---------------------------------------------------------------------------
# Security section main loop
# ---------------------------------------------------------------------------

def run_security_section(original_tx: dict, mongo_store=None):
    banner("PART 2 -- ADVANCED SECURITY ATTACK TESTING")
    print("""
  Your virtual trade has been cryptographically signed.
  Test 8 real-world attack patterns against it.

  Each test shows:
    - What the attack changes in the transaction
    - Which signals it activates and how strongly
    - How both engines respond (PQC + Bayesian)
    - Whether the transaction is blocked or allowed
    """)

    if mongo_store and mongo_store.connected:
        mdb = mongo_store.summary()
        print(f"  [MongoDB] {mdb.get('total_patterns', 0)} patterns stored "
              f"from previous sessions.")
    else:
        print("  [Memory] JSON file store (MongoDB not configured)")
    print()

    sec = SecurityEngine()
    bay = BayesianSecurityPipeline(
        base_prior  = 0.05,
        memory_path = _MEMORY_PATH,
        log_level   = logging.CRITICAL,
    )

    sec.process_transaction_security(dict(original_tx))
    bay.process_transaction_security(dict(original_tx))
    print("  Both engines primed with the baseline transaction.")
    print(f"  Bayesian prior: 5%  |  Session ID: {_SESSION_ID}")
    print()

    while True:
        print_attack_menu()
        choice = input("  Your choice (1-8, A for all, 0 to exit): ").strip().upper()

        if choice == "0":
            print("\n  Security testing complete.\n")
            break
        if choice == "A":
            keys_to_run = list(ATTACK_MENU.keys())
        elif choice in ATTACK_MENU:
            keys_to_run = [choice]
        else:
            print("  X Invalid choice. Please enter 1-8, A, or 0.")
            continue

        for key in keys_to_run:
            run_security_test(
                original_tx = original_tx,
                attack_key  = key,
                attack_def  = ATTACK_MENU[key],
                sec_engine  = sec,
                bay_engine  = bay,
                mongo_store = mongo_store,
            )

        banner("SESSION SUMMARY")
        ps  = sec.system_status()
        bs  = bay.system_status()
        mem = bs.get("memory_summary", {})

        print(f"""
  Transactions evaluated   : {ps['transactions_processed']} (PQC)  /  {bs['transactions_processed']} (Bayesian)
  Transactions blocked     : {ps['response_engine']['total_quarantined']} (PQC)  /  {bs['quarantine_ledger']['size']} (Bayesian)
  Key rotations triggered  : {ps['key_system']['total_rotations']}
  Threat patterns (memory) : {mem.get('total_patterns', 0)}  (recognition: {mem.get('recognition_rate', 0)*100:.1f}%)
        """)

        if mongo_store and mongo_store.connected:
            mdb = mongo_store.summary()
            print(f"  MongoDB patterns stored  : {mdb.get('total_patterns', 0)}")
            for atk, cnt in mdb.get("by_attack_type", {}).items():
                print(f"    {atk:<44} {cnt} pattern(s)")
            print()

        again = ask_yes_no("  Would you like to test another attack?", default="y")
        if not again:
            print("\n  All security tests complete.\n")
            break


# =============================================================================
# MAIN
# =============================================================================

def main():
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)

    print()
    print("  =" * 35)
    print("  VIRTUAL P2P TRADE SIMULATOR  +  ADVANCED SECURITY TESTER")
    print("  Milestone 6  --  Post-Quantum Secure Trading")
    print("  =" * 35)
    print("""
  PART 1 -- Describe your portfolio and the trade you want to test.
            The system simulates the trade and shows the portfolio impact.

  PART 2 -- Choose from 8 real-world attack patterns.
            Two independent engines analyse each attack.
            Threat patterns are persisted to MongoDB across sessions.

  Press Ctrl+C at any time to exit.
    """)

    # Initialise MongoDB
    mongo_store = None
    uri = _get_mongo_uri()
    if uri:
        mongo_store = MongoMemoryStore(uri=uri)
    else:
        print("  MONGO_URI not set in environment or .env file.")
        print("  Bayesian memory will use the local JSON file store.")
        print(f"  To enable MongoDB add: MONGO_URI=mongodb://localhost:27017")
        print()

    try:
        holdings, prices, total_value = collect_portfolio()
        ticker, quantity, price       = collect_trade(holdings, prices)
        result = run_trade_and_show_results(
            holdings, prices, total_value, ticker, quantity, price
        )

        if result is None:
            print("  Trade simulation failed. Exiting.\n")
            return

        original_tx = result.get("transaction_record", {})
        if not original_tx:
            print("  No transaction record found. Cannot run security tests.\n")
            return

        # Save transaction for frontend if needed
        try:
            saved = {
                "tx_id":               original_tx.get("tx_id"),
                "sha3_hash":           original_tx.get("sha3_hash"),
                "verification_status": bool(original_tx.get("verification_status", False)),
                "signed_at":           float(original_tx.get("signed_at", time.time())),
                "payload":             original_tx.get("payload", {}),
                "signature":           original_tx.get("signature", {}),
                "saved_at":            time.time(),
            }
            with open(_SHARED_TX_PATH, "w") as f:
                json.dump(saved, f, indent=2)
            print(f"\n  Transaction saved: {_SHARED_TX_PATH}")
        except Exception as e:
            print(f"  Could not save transaction: {e}")

        do_security = ask_yes_no(
            "\n  Would you like to run the advanced security attack tests?",
            default="y"
        )
        if do_security:
            run_security_section(original_tx, mongo_store=mongo_store)
        else:
            print("\n  Security testing skipped. Goodbye!\n")

    except KeyboardInterrupt:
        print("\n\n  Exited by user.\n")
    finally:
        if mongo_store:
            mongo_store.close()


if __name__ == "__main__":
    main()
