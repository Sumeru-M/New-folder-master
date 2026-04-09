# ClearView Analytics: Institutional Risk & Intelligence Engine

[![Vercel Deployment](https://img.shields.io/badge/Deployment-Live-success?style=flat-square&logo=vercel)](https://clearview-analytics-prod.vercel.app)
[![Tech Stack](https://img.shields.io/badge/Stack-FastAPI%20%7C%20Vanilla%20JS%20%7C%20Python-blue?style=flat-square)](https://fastapi.tiangolo.com/)
[![Security](https://img.shields.io/badge/Security-ML--DSA--III%20%2F%20PQC-blueviolet?style=flat-square)](https://csrc.nist.gov/projects/post-quantum-cryptography)

**ClearView Analytics** is a high-fidelity quantitative finance platform designed for institutional-grade portfolio optimization, regime-switching intelligence, and post-quantum secure risk management.

---

## Key Capabilities

### Quantitative Intelligence (M3–M7)
- **Regime-Switching Engine (M7):** Advanced Hidden Markov Models (HMM) coupled with GARCH(1,1) filters to detect market shifts and modulate risk and allocation dynamically.
- **Monte Carlo Projection (M6):** High-precision path modeling for 1Y/3Y/5Y horizons, accurately quantifying the **Probability of Loss** and downside risks.
- **Institutional Optimizer (M5):** Robust Black-Litterman and Markowitz frontiers for sophisticated multi-asset allocation.
- **Stress Testing (M4):** Macroeconomic scenario analysis covering inflation spikes, rate hikes, and historical crisis replays.

### Post-Quantum Immune Defense
- **ML-DSA-III (Crystals-Dilithium):** Implementation of NIST-standard post-quantum signatures for transaction integrity.
- **Bayesian Immune Layer:** An adaptive security pipeline that learns threat patterns using Neutral-point LLR Bayesian posteriors.
- **Episodic Threat Memory:** Cosine-similarity memory store for tracking and neutralizing adversarial anomalies in real-time.

---

## Tech Stack & Architecture

- **Backend:** [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11)
- **Frontend:** Pure Vanilla JS & CSS for maximum performance and a premium "Glassmorphism" aesthetic.
- **Libraries:** `NumPy`, `Pandas`, `SciPy`, `CVXPY`, `hmmlearn`, `Statsmodels`.
- **Cloud Infrastructure:** [Vercel](https://vercel.com/) (Serverless Lambda deployment).

---

## Local Setup

1. **Clone & Explore:**
   ```bash
   git clone https://github.com/Sumeru-M/New-folder-master.git
   cd New-folder-master
   ```

2. **Environment Ready:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ignite the Engine:**
   ```bash
   uvicorn src.main:app --reload
   ```
   Access the cockpit at `http://localhost:8000`.

---

## Cloud Deployment

The platform is fully optimized for **Vercel** via `app.py` and `vercel.json`.

### Required Environment Variables:
| Variable | Description |
|----------|-------------|
| `AUTH_SECRET` | Secure JWT key for the integrated identity system. |
| `CORS_ORIGINS` | Permitted domains for cross-origin resource sharing. |
| `MONGODB_URI` | *(Optional)* Connection string for persistent Bayesian memory storage. |

---

## Repository Structure
- `src/`: Core FastAPI server and business logic.
- `portfolio/`: High-performance quantitative engines (M3–M7).
- `frontend/`: Interactive Glassmorphism UI components.
- `docs/`: Expanded technical documentation and research notebooks.

---

Developed with precision for institutional risk management. **View the live demo:** [clearview-analytics-prod.vercel.app](https://clearview-analytics-prod.vercel.app)

## Disclaimer
This project is for educational and research purposes only.

-Not intended for real trading or investment
-No investment advice or guarantees provided
-Creator assumes no liability for financial losses
-Consult a financial advisor for investment decisions
-Past performance does not indicate future results
-By using this software, you agree to use it solely for learning purposes.
