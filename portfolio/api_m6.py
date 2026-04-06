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


class MongoBayesianMemory:
    """
    Optional Mongo-backed persistent memory for the Bayesian immune layer.

    If MongoDB is not configured or unreachable, this class becomes a no-op
    and the API continues working with in-memory behavior.
    """

    def __init__(self):
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
            client = MongoClient(uri, serverSelectionTimeoutMS=1500)
            client.admin.command("ping")
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
    """
    Runs the full M6 pipeline: virtual trade simulation + security assessment.

    Parameters
    ----------
    ticker         : NSE ticker to virtually buy, e.g. "RELIANCE.NS"
    quantity       : number of shares to simulate buying
    price          : price per share in INR
    holdings       : {ticker: shares_held} — current real portfolio
    current_prices : {ticker: current_price_inr}
    total_value    : total current portfolio value in INR
    risk_free_rate : annual risk-free rate as decimal
    n_mc_paths     : number of Monte Carlo simulation paths (min 1000)

    Returns
    -------
    dict with keys:
        ticker          : ticker traded
        quantity        : shares bought
        price           : price per share
        trade_cost      : total cost in INR

        transaction     : {
            tx_id, sha3_hash, scheme, verified,
        }

        portfolio_impact : {
            expected_return_change, volatility_change, sharpe_change,
            cvar_change, diversification_change, hhi_change
        }

        monte_carlo      : {
            overall_verdict, best_horizon,
            horizons: {
                "1Y": {real_value, virtual_value, delta, median_cagr,
                        p5_value, p95_value, prob_loss},
                "3Y": {...},
                "5Y": {...}
            }
        }

        risk_summary     : str

        security_pqc     : {
            threat_level, entropy_score, anomaly_score,
            quarantine_status, actions_triggered,
            key_rotation, pattern_match, pattern_similarity,
            attack_type
        }

        security_bayesian : {
            threat_level, posterior_probability, threat_score,
            quarantine_status, actions_triggered,
            key_rotation_signal, memory_similarity,
            memory_boosted_prior
        }

        security_summary : {
            pqc_transactions_processed, pqc_quarantined,
            bayesian_transactions_processed, bayesian_quarantined,
            threat_patterns_learned
        }

        error            : None or error string
    """

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

        # ── Run the full virtual trade simulation pipeline ─────────────────────
        import numpy as np
        import pandas as pd
        from portfolio.portfolio_complete import load_price_data, normalize_tickers_for_market_data
        from portfolio.portfolio_complete   import compute_daily_returns

        # Fetch historical returns for all tickers (existing + new)
        all_tickers = normalize_tickers_for_market_data(list(set(list(holdings.keys()) + [ticker])))
        prices_df   = load_price_data(all_tickers, period="2y")
        if prices_df is None or prices_df.empty:
            result["error"] = "Could not load price data for simulation."
            return result

        valid       = [t for t in all_tickers if t in prices_df.columns]
        prices_df   = prices_df[valid].dropna()
        daily_ret   = compute_daily_returns(prices_df)

        # Generate keypair for this session
        pk, sk = m6.generate_keypair()

        sim = m6.run_virtual_trade_simulation(
            ticker         = ticker,
            quantity       = quantity,
            price          = price,
            real_holdings  = holdings,
            real_prices    = current_prices,
            daily_returns  = daily_ret,
            total_value    = total_value,
            risk_free_rate = risk_free_rate,
            n_mc_paths     = n_mc_paths,
            keypair        = (pk, sk),
        )

        # ── Transaction record ────────────────────────────────────────────────
        tx = sim.get("transaction_record", {})
        result["transaction"] = {
            "tx_id":    tx.get("tx_id", ""),
            "sha3_hash":tx.get("sha3_hash", ""),
            "scheme":   tx.get("signature", {}).get("scheme", ""),
            "verified": bool(tx.get("verification_status", False)),
        }

        # ── Portfolio impact ──────────────────────────────────────────────────
        imp = sim.get("portfolio_impact", {})
        result["portfolio_impact"] = {
            "expected_return_change": round(float(imp.get("expected_return_change", 0)), 4),
            "volatility_change":      round(float(imp.get("volatility_change",      0)), 4),
            "sharpe_change":          round(float(imp.get("sharpe_change",          0)), 4),
            "cvar_change":            round(float(imp.get("cvar_change",            0)), 4),
            "diversification_change": round(float(imp.get("diversification_change", 0)), 4),
            "hhi_change":             round(float(imp.get("hhi_change",             0)), 4),
        }

        # ── Monte Carlo projections ───────────────────────────────────────────
        mc = sim.get("monte_carlo_projection", {})
        horizons_out = {}
        for h in ["1Y", "3Y", "5Y"]:
            hd = mc.get(h, {})
            if hd:
                horizons_out[h] = {
                    "real_value":    round(float(hd.get("real",    {}).get("expected_value",  0)), 2),
                    "virtual_value": round(float(hd.get("virtual", {}).get("expected_value",  0)), 2),
                    "delta":         round(float(hd.get("deltas",  {}).get("expected_value_delta", 0)), 2),
                    "median_cagr":   round(float(hd.get("real",    {}).get("median_cagr",     0)), 4),
                    "p5_value":      round(float(hd.get("real",    {}).get("p5_value",         0)), 2),
                    "p95_value":     round(float(hd.get("real",    {}).get("p95_value",        0)), 2),
                    "prob_loss":     round(float(hd.get("real",    {}).get("prob_loss",        0)), 4),
                }

        result["monte_carlo"] = {
            "overall_verdict": mc.get("overall_verdict", ""),
            "best_horizon":    mc.get("best_horizon", ""),
            "horizons":        horizons_out,
        }

        result["risk_summary"] = str(sim.get("risk_summary", ""))

        # ── Security assessment on the actual transaction ─────────────────────
        memory_store = MongoBayesianMemory()
        dynamic_prior = memory_store.estimate_prior()
        sec = m6.SecurityEngine()
        bay = m6.BayesianSecurityPipeline(base_prior=dynamic_prior)
        memory_seeded = memory_store.seed_pipeline_memory(bay)

        pqc_r = sec.process_transaction_security(tx)
        bay_r = bay.process_transaction_security(tx)

        result["security_pqc"] = {
            "threat_level":      pqc_r.threat_level,
            "entropy_score":     round(float(pqc_r.entropy_score), 4),
            "anomaly_score":     round(float(pqc_r.anomaly_score), 4),
            "quarantine_status": pqc_r.quarantine_status,
            "actions_triggered": list(pqc_r.actions_triggered),
            "key_rotation":      bool(pqc_r.key_rotation),
            "pattern_match":     bool(pqc_r.pattern_match),
            "pattern_similarity":round(float(pqc_r.pattern_similarity), 4),
            "attack_type":       pqc_r.attack_type,
        }

        result["security_bayesian"] = {
            "threat_level":        bay_r.threat_level,
            "posterior_probability": round(float(bay_r.posterior_probability), 4),
            "threat_score":        round(float(bay_r.threat_score), 2),
            "quarantine_status":   bay_r.quarantine_status,
            "actions_triggered":   list(bay_r.actions_triggered),
            "key_rotation_signal": bool(bay_r.key_rotation_signal),
            "memory_similarity":   round(float(bay_r.memory_similarity), 4),
            "memory_boosted_prior":bool(bay_r.memory_boosted_prior),
        }

        ps = sec.system_status()
        bs = bay.system_status()
        memory_synced = memory_store.sync_pipeline_memory(bay)
        memory_store.log_event(
            source="simulate",
            attack_type=None,
            bayesian_result=bay_r.to_dict(),
            tx_id=str(tx.get("tx_id", "")),
        )
        result["security_summary"] = {
            "pqc_transactions_processed":      int(ps["transactions_processed"]),
            "pqc_quarantined":                 int(ps["response_engine"]["total_quarantined"]),
            "bayesian_transactions_processed": int(bs["transactions_processed"]),
            "bayesian_quarantined":            int(bs["quarantine_ledger"]["size"]),
            "threat_patterns_learned":         int(bs["memory_summary"]["total_patterns"]),
        }
        result["bayesian_memory"] = {
            "dynamic_base_prior": round(float(dynamic_prior), 4),
            "memory_seeded_records": int(memory_seeded),
            "memory_synced_records": int(memory_synced),
            **memory_store.status(),
        }

    except Exception as e:
        result["error"] = str(e)

    return result


def get_security_attack_test(
    transaction: dict,
    attack_type: str,
) -> dict:
    """
    Test a specific attack scenario against an existing transaction.

    Parameters
    ----------
    transaction : transaction dict from get_virtual_trade_and_security()["transaction"]
                  or a raw transaction dict with tx_id, sha3_hash, verification_status
    attack_type : one of:
                  "signature_forgery" | "weak_hash" | "replay" | "burst" | "combined"

    Returns
    -------
    dict with keys:
        attack_type       : attack tested
        pqc               : {threat_level, entropy_score, quarantine_status,
                              actions_triggered, key_rotation}
        bayesian          : {threat_level, posterior_probability,
                              quarantine_status, memory_boosted_prior,
                              key_rotation_signal}
        error             : None or error string
    """

    result = {
        "attack_type": attack_type,
        "pqc":         None,
        "bayesian":    None,
        "bayesian_memory": None,
        "error":       None,
    }

    try:
        m6 = _load_m6()

        # Build the corrupted transaction for this attack
        atk_tx = dict(transaction)

        if attack_type == "signature_forgery":
            atk_tx["verification_status"]     = False
            atk_tx["_entropy_composite_score"] = 0.72

        elif attack_type == "weak_hash":
            atk_tx["sha3_hash"]                = "00" * 32
            atk_tx["_entropy_composite_score"] = 0.003

        elif attack_type == "replay":
            atk_tx["signed_at"]                = time.time() + 0.1
            atk_tx["_entropy_composite_score"] = 0.72

        elif attack_type == "burst":
            atk_tx["tx_id"]                    = f"BURST_001_{transaction.get('tx_id','')[:8]}"
            atk_tx["signed_at"]                = time.time()
            atk_tx["_entropy_composite_score"] = 0.72

        elif attack_type == "combined":
            atk_tx["verification_status"]      = False
            atk_tx["sha3_hash"]                = "00" * 32
            atk_tx["_entropy_composite_score"] = 0.003

        memory_store = MongoBayesianMemory()
        dynamic_prior = memory_store.estimate_prior()
        sec   = m6.SecurityEngine()
        bay   = m6.BayesianSecurityPipeline(base_prior=dynamic_prior)
        memory_seeded = memory_store.seed_pipeline_memory(bay)

        # Seed with the benign original first (replay needs this)
        if attack_type == "replay":
            sec.process_transaction_security(dict(transaction))
            bay.process_transaction_security(dict(transaction))

        pqc_r = sec.process_transaction_security(atk_tx)
        bay_r = bay.process_transaction_security(atk_tx)

        memory_synced = memory_store.sync_pipeline_memory(bay)
        memory_store.log_event(
            source="security_test",
            attack_type=attack_type,
            bayesian_result=bay_r.to_dict(),
            tx_id=str(atk_tx.get("tx_id", "")),
        )

        result["pqc"] = {
            "threat_level":      pqc_r.threat_level,
            "entropy_score":     round(float(pqc_r.entropy_score), 4),
            "quarantine_status": pqc_r.quarantine_status,
            "actions_triggered": list(pqc_r.actions_triggered),
            "key_rotation":      bool(pqc_r.key_rotation),
        }

        result["bayesian"] = {
            "threat_level":          bay_r.threat_level,
            "posterior_probability": round(float(bay_r.posterior_probability), 4),
            "quarantine_status":     bay_r.quarantine_status,
            "memory_boosted_prior":  bool(bay_r.memory_boosted_prior),
            "key_rotation_signal":   bool(bay_r.key_rotation_signal),
        }
        result["bayesian_memory"] = {
            "dynamic_base_prior": round(float(dynamic_prior), 4),
            "memory_seeded_records": int(memory_seeded),
            "memory_synced_records": int(memory_synced),
            **memory_store.status(),
        }

    except Exception as e:
        result["error"] = str(e)

    return result
