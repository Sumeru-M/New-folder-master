import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from portfolio.api_m3 import get_portfolio_construction
from portfolio.api_m4 import get_scenario_analysis
from portfolio.api_m5 import get_institutional_optimisation
from portfolio.api_m6 import get_security_attack_test, get_virtual_trade_and_security
from portfolio.api_m7 import get_market_regime


ROOT_DIR = Path(__file__).resolve().parent.parent
FRONTEND_FILE = ROOT_DIR / "frontend" / "index.html"


def _cors_origins() -> list[str]:
    raw = os.getenv("CORS_ORIGINS", "*")
    origins = [o.strip() for o in raw.split(",") if o.strip()]
    return origins or ["*"]


app = FastAPI(title="ClearView Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class M3Request(BaseModel):
    tickers: list[str]
    current_weights: list[float] | None = None
    period: str = "2y"
    risk_free_rate: float = 0.07


class M4Request(BaseModel):
    tickers: list[str]
    portfolio_value: float = 1_000_000
    risk_free_rate: float = 0.07
    confidence_level: float = 0.95
    scenarios: str = "ALL"


class M5Request(BaseModel):
    tickers: list[str]
    current_weights: list[float] | None = None
    portfolio_value: float = 1_000_000
    risk_free_rate: float = 0.07
    max_weight: float = 0.40
    sector_cap: float = 0.60
    confidence_level: float = 0.95
    methods: str = "all"


class M6SimulateRequest(BaseModel):
    ticker: str
    quantity: float
    price: float
    holdings: dict[str, float]
    current_prices: dict[str, float]
    total_value: float
    risk_free_rate: float = 0.07
    n_mc_paths: int = 1_000


class M6SecurityTestRequest(BaseModel):
    transaction: dict[str, Any]
    attack_type: str


class M7Request(BaseModel):
    tickers: list[str]
    risk_free_rate: float = 0.07
    horizons: list[int] = Field(default_factory=lambda: [21, 63])
    risk_appetite: str = "balanced"
    hmm_restarts: int = 3
    hmm_max_iter: int = 150
    garch_n_sim: int = 300
    uncertainty_n_boot: int = 200


def _safe_json(result: dict[str, Any]) -> JSONResponse:
    return JSONResponse(content=result)


def _normalize_transaction_for_security(transaction: dict[str, Any]) -> dict[str, Any]:
    tx = dict(transaction or {})
    if "verification_status" not in tx and "verified" in tx:
        tx["verification_status"] = bool(tx["verified"])
    tx.setdefault("verification_status", True)
    tx.setdefault("signed_at", time.time())
    tx.setdefault("tx_id", f"TX_{int(time.time() * 1000)}")
    tx.setdefault("sha3_hash", "0" * 64)
    tx.setdefault("signature", {"scheme": "ML-DSA-III"})
    return tx


ATTACK_TYPE_MAP = {
    "adversarial_evasion": "signature_forgery",
    "sybil_trust_poisoning": "burst",
    "model_extraction": "weak_hash",
    "side_channel_timing_leaks": "replay",
    "side_channel_timing": "replay",
    "fault_injection": "combined",
    "triangular_fraud": "burst",
    "price_oracle_manipulation": "weak_hash",
    "front_running_sandwich_attack": "replay",
    "front_running": "replay",
}


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/m3/optimize")
def m3_optimize(req: M3Request) -> JSONResponse:
    try:
        result = get_portfolio_construction(
            tickers=req.tickers,
            current_weights=req.current_weights,
            period=req.period,
            risk_free_rate=req.risk_free_rate,
        )
        return _safe_json(result)
    except Exception as exc:  # pragma: no cover
        return _safe_json({"error": str(exc)})


@app.post("/api/m4/scenarios")
def m4_scenarios(req: M4Request) -> JSONResponse:
    try:
        result = get_scenario_analysis(
            tickers=req.tickers,
            portfolio_value=req.portfolio_value,
            risk_free_rate=req.risk_free_rate,
            confidence_level=req.confidence_level,
            scenarios=req.scenarios,
        )
        return _safe_json(result)
    except Exception as exc:  # pragma: no cover
        return _safe_json({"error": str(exc)})


@app.post("/api/m5/institutional")
def m5_institutional(req: M5Request) -> JSONResponse:
    try:
        result = get_institutional_optimisation(
            tickers=req.tickers,
            current_weights=req.current_weights,
            portfolio_value=req.portfolio_value,
            risk_free_rate=req.risk_free_rate,
            max_weight=req.max_weight,
            sector_cap=req.sector_cap,
            confidence_level=req.confidence_level,
            methods=req.methods,
        )
        return _safe_json(result)
    except Exception as exc:  # pragma: no cover
        return _safe_json({"error": str(exc)})


@app.post("/api/m6/simulate")
def m6_simulate(req: M6SimulateRequest) -> JSONResponse:
    try:
        result = get_virtual_trade_and_security(
            ticker=req.ticker,
            quantity=req.quantity,
            price=req.price,
            holdings=req.holdings,
            current_prices=req.current_prices,
            total_value=req.total_value,
            risk_free_rate=req.risk_free_rate,
            n_mc_paths=max(1000, int(req.n_mc_paths)),
        )

        # Keep compatibility with frontend and attack-test payload expectations.
        tx = result.get("transaction") or {}
        if tx:
            tx.setdefault("verification_status", tx.get("verified", True))
            tx.setdefault("signed_at", time.time())
            result["transaction"] = tx

        return _safe_json(result)
    except Exception as exc:  # pragma: no cover
        return _safe_json({"error": str(exc)})


@app.post("/api/m6/security/test")
def m6_security_test(req: M6SecurityTestRequest) -> JSONResponse:
    try:
        attack_type = ATTACK_TYPE_MAP.get(req.attack_type, req.attack_type)
        tx = _normalize_transaction_for_security(req.transaction)
        result = get_security_attack_test(transaction=tx, attack_type=attack_type)
        result["requested_attack_type"] = req.attack_type
        return _safe_json(result)
    except Exception as exc:  # pragma: no cover
        return _safe_json({"error": str(exc)})


@app.post("/api/m7/regime")
def m7_regime(req: M7Request) -> JSONResponse:
    try:
        result = get_market_regime(
            tickers=req.tickers,
            risk_free_rate=req.risk_free_rate,
            horizons=req.horizons,
            risk_appetite=req.risk_appetite,
            hmm_restarts=req.hmm_restarts,
            hmm_max_iter=req.hmm_max_iter,
            garch_n_sim=req.garch_n_sim,
            uncertainty_n_boot=req.uncertainty_n_boot,
        )
        return _safe_json(result)
    except Exception as exc:  # pragma: no cover
        return _safe_json({"error": str(exc)})


@app.get("/")
def serve_frontend():
    if FRONTEND_FILE.exists():
        return FileResponse(FRONTEND_FILE)
    return JSONResponse({"message": "Frontend file not found. Expected frontend/index.html"})


@app.get("/app")
def serve_frontend_app():
    return serve_frontend()
