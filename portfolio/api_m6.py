"""
api_m6.py — Milestone 6: Secure Virtual Trade API
==================================================
Wraps the existing M6 logic into one clean function.
No print statements. Returns a single structured dictionary.

Usage:
    from portfolio.api_m6 import get_virtual_trade_and_security

    result = get_virtual_trade_and_security(
        ticker          = "RELIANCE.NS",
        quantity        = 10,
        price           = 2850.0,
        holdings        = {"RELIANCE.NS": 10, "TCS.NS": 5},
        current_prices  = {"RELIANCE.NS": 2850.0, "TCS.NS": 3900.0},
        total_value     = 67250.0,
    )
"""

import sys
import os
import time
import types
import importlib.util
from datetime import datetime, timezone
from typing import Any

import numpy as np

try:
    from pymongo import MongoClient
except Exception:  # pragma: no cover
    MongoClient = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


DEFAULT_BASE_PRIOR = 0.05
MEMORY_LOOKBACK_DAYS = 30
MEMORY_REPLAY_LIMIT = 400

_MONGO_CLIENT_CACHE = None

class MongoBayesianMemory:
    """
    Optional Mongo-backed persistent memory for the Bayesian immune layer.
    Uses a singleton MongoClient to optimize connection pooling.
    """

    def __init__(self):
        global _MONGO_CLIENT_CACHE
        self._enabled = False
        self._reason = ""
        self._memory_col = None
        self._events_col = None

        uri = os.getenv("MONGODB_URI", "").strip()
        db_name = os.getenv("MONGODB_DB", "clearview_analytics").strip()
        memory_collection = os.getenv("MONGODB_MEMORY_COLLECTION", "bayesian_immune_memory").strip()
        events_collection = os.getenv("MONGODB_EVENTS_COLLECTION", "bayesian_security_events").strip()

        if not uri:
            self._reason = "MONGODB_URI is not set"
            return
        if MongoClient is None:
            self._reason = "pymongo is not installed"
            return

        try:
            if _MONGO_CLIENT_CACHE is None:
                _MONGO_CLIENT_CACHE = MongoClient(uri, serverSelectionTimeoutMS=1500)
            
            client = _MONGO_CLIENT_CACHE
            # Fast ping check
            client.admin.command('ping')
            
            db = client[db_name]
            self._memory_col = db[memory_collection]
            self._events_col = db[events_collection]
            self._memory_col.create_index("record_id", unique=True)
            self._memory_col.create_index("last_seen")
            self._events_col.create_index("timestamp")
            self._events_col.create_index("threat_level")
            self._enabled = True
        except Exception as exc:  # pragma: no cover
            self._reason = f"Mongo unavailable: {exc}"
            _MONGO_CLIENT_CACHE = None # Reset cache on failure

    @property
    def enabled(self) -> bool:
        return self._enabled

    def estimate_prior(self, default_prior: float = DEFAULT_BASE_PRIOR) -> float:
        """Estimate prior risk from recent event history."""
        if not self._enabled:
            return default_prior
        try:
            cutoff = time.time() - (MEMORY_LOOKBACK_DAYS * 86400)
            total = self._events_col.count_documents({"timestamp": {"$gte": cutoff}})
            if total <= 0:
                return default_prior
            risky = self._events_col.count_documents({
                "timestamp": {"$gte": cutoff},
                "threat_level": {"$in": ["MONITOR", "ELEVATED_RISK", "CRITICAL_THREAT", "HIGH", "MEDIUM"]},
            })
            prior = risky / total
            return float(np.clip(prior, 0.02, 0.35))
        except Exception:  # pragma: no cover
            return default_prior

    def seed_pipeline_memory(self, bayesian_pipeline: Any) -> int:
        """Replay stored patterns into the in-memory Bayesian immune system."""
        if not self._enabled:
            return 0
        try:
            records = list(
                self._memory_col.find({})
                .sort("last_seen", -1)
                .limit(MEMORY_REPLAY_LIMIT)
            )
            loaded = 0
            for rec in reversed(records):
                raw = rec.get("raw_vector") or rec.get("transaction_sig_vector")
                if not isinstance(raw, list) or len(raw) < 6:
                    continue
                try:
                    bayesian_pipeline._immune_memory.record(  # noqa: SLF001
                        tx_id=str(rec.get("tx_id_ref") or rec.get("record_id") or f"mongo_{loaded}"),
                        signal_vector=np.array(raw[:6], dtype=float),
                        posterior=float(rec.get("posterior_probability", 0.5)),
                        threat_level=str(rec.get("threat_classification", "MONITOR")),
                        mitigation_action=str(rec.get("mitigation_action", "LOG")),
                    )
                    loaded += 1
                except Exception:
                    continue
            return loaded
        except Exception:  # pragma: no cover
            return 0

    def sync_pipeline_memory(self, bayesian_pipeline: Any) -> int:
        """Persist current in-memory immune records to MongoDB."""
        if not self._enabled:
            return 0
        try:
            records = bayesian_pipeline.get_memory_records()
            upserts = 0
            now_iso = datetime.now(timezone.utc).isoformat()
            for rec in records:
                record_id = rec.get("record_id")
                if not record_id:
                    continue
                doc = {
                    "record_id": record_id,
                    "transaction_sig_vector": rec.get("transaction_sig_vector", []),
                    "raw_vector": rec.get("raw_vector", []),
                    "posterior_probability": rec.get("posterior_probability"),
                    "threat_classification": rec.get("threat_classification"),
                    "mitigation_action": rec.get("mitigation_action"),
                    "timestamp": rec.get("timestamp", time.time()),
                    "last_seen": rec.get("last_seen", time.time()),
                    "occurrence_count": rec.get("occurrence_count", 1),
                    "tx_id_ref": rec.get("tx_id_ref", ""),
                    "updated_at": now_iso,
                }
                self._memory_col.update_one({"record_id": record_id}, {"$set": doc}, upsert=True)
                upserts += 1
            return upserts
        except Exception:  # pragma: no cover
            return 0

    def log_event(self, source: str, attack_type: str | None, bayesian_result: dict, tx_id: str = "") -> None:
        if not self._enabled:
            return
        try:
            self._events_col.insert_one({
                "source": source,
                "attack_type": attack_type,
                "tx_id": tx_id,
                "timestamp": time.time(),
                "threat_level": bayesian_result.get("threat_level"),
                "posterior_probability": bayesian_result.get("posterior_probability"),
                "memory_similarity": bayesian_result.get("memory_similarity"),
                "memory_boosted_prior": bayesian_result.get("memory_boosted_prior"),
                "quarantine_status": bayesian_result.get("quarantine_status"),
            })
        except Exception:  # pragma: no cover
            return

    def status(self) -> dict:
        if not self._enabled:
            return {"enabled": False, "reason": self._reason}
        try:
            return {
                "enabled": True,
                "patterns_stored": int(self._memory_col.count_documents({})),
                "events_stored": int(self._events_col.count_documents({})),
            }
        except Exception as exc:  # pragma: no cover
            return {"enabled": False, "reason": f"Mongo read failed: {exc}"}

def _load_m6():
    """Load milestone6_complete.py from the portfolio folder."""
    portfolio_dir = os.path.dirname(__file__)
    project_dir   = os.path.join(portfolio_dir, "..")
    for folder in [portfolio_dir, project_dir]:
        path = os.path.normpath(os.path.join(folder, "milestone6_complete.py"))
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("milestone6_complete", path)
            mod  = types.ModuleType("milestone6_complete")
            mod.__spec__ = spec
            sys.modules["milestone6_complete"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "milestone6_complete.py not found in portfolio/ or project root."
    )

def get_virtual_trade_and_security(
    ticker: str,
    quantity: float,
    price: float,
    holdings: dict,
    current_prices: dict,
    total_value: float,
    risk_free_rate: float = 0.07,
    n_mc_paths: int = 1_000,
) -> dict:
    """Pipeline for M6."""
    result = {
        "ticker":             ticker,
        "quantity":           quantity,
        "price":              price,
        "trade_cost":         round(quantity * price, 2),
        "transaction":        None,
        "portfolio_impact":   None,
        "monte_carlo":        None,
        "risk_summary":       None,
        "security_pqc":       None,
        "security_bayesian":  None,
        "security_summary":   None,
        "bayesian_memory":    None,
        "error":              None,
    }
    try:
        m6 = _load_m6()
        import pandas as pd
        from portfolio.portfolio_complete import load_price_data, normalize_tickers_for_market_data
        from portfolio.portfolio_complete import compute_daily_returns

        all_tickers = normalize_tickers_for_market_data(list(set(list(holdings.keys()) + [ticker])))
        prices_df   = load_price_data(all_tickers, period="2y")
        if prices_df is None or prices_df.empty:
            result["error"] = "Could not load price data"
            return result

        daily_ret = compute_daily_returns(prices_df[all_tickers].dropna())
        pk, sk = m6.generate_keypair()

        sim = m6.run_virtual_trade_simulation(
            ticker, quantity, price, holdings, current_prices,
            daily_ret, total_value, risk_free_rate, n_mc_paths, (pk,sk)
        )

        tx = sim.get("transaction_record", {})
        result["transaction"] = {
            "tx_id": tx.get("tx_id", ""),
            "sha3_hash": tx.get("sha3_hash", ""),
            "verified": bool(tx.get("verification_status", False))
        }

        mc = sim.get("monte_carlo_projection", {})
        horizons_out = {}
        for h in ["1Y", "3Y", "5Y"]:
            hd = mc.get(h, {})
            if hd:
                horizons_out[h] = {
                    "real_value": round(float(hd.get("real",{}).get("expected_value",0)), 2),
                    "virtual_value": round(float(hd.get("virtual",{}).get("expected_value",0)), 2),
                    "prob_loss": round(float(hd.get("real",{}).get("downside_prob",0)), 4)
                }
        result["monte_carlo"] = {"horizons": horizons_out}

        memory_store = MongoBayesianMemory()
        sec = m6.SecurityEngine()
        bay = m6.BayesianSecurityPipeline(base_prior=memory_store.estimate_prior())
        memory_store.seed_pipeline_memory(bay)
        bay_r = bay.process_transaction_security(tx)
        memory_store.sync_pipeline_memory(bay)

        result["security_bayesian"] = {
            "threat_level": bay_r.threat_level,
            "posterior_probability": round(float(bay_r.posterior_probability), 4)
        }
        return result
    except Exception as e:
        result["error"] = str(e)
        return result

def get_security_attack_test(transaction: dict, attack_type: str) -> dict:
    """Test attack type."""
    try:
        m6 = _load_m6()
        sec = m6.SecurityEngine()
        res = sec.process_transaction_security(transaction)
        return {"attack_type": attack_type, "pqc": {"threat_level": res.threat_level}}
    except Exception as e:
        return {"error": str(e)}
