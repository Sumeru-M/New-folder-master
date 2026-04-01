from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check
@app.get("/api/health")
def health():
    return {"status": "ok"}


# TEMP test route (to confirm everything works)
@app.get("/")
def root():
    return {"message": "Backend is running"}

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.portfolio_engine import PortfolioEngine  # adjust if needed

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Request schema =====
class M3Request(BaseModel):
    tickers: list[str]
    period: str
    risk_free_rate: float


# ===== Health =====
@app.get("/api/health")
def health():
    return {"status": "ok"}


# ===== M3 Endpoint =====
@app.post("/api/m3/optimize")
def optimize_portfolio(req: M3Request):
    try:
        engine = PortfolioEngine(
            tickers=req.tickers,
            period=req.period,
            risk_free_rate=req.risk_free_rate
        )

        result = engine.run_optimization()  # adjust to your function name

        return result

    except Exception as e:
        return {"error": str(e)}
