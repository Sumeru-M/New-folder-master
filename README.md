# ClearView Analytics (Deployable Full Stack)

This repo now runs as a single deployable FastAPI product:

- Backend APIs: `/api/m3` to `/api/m7`
- Frontend: served from `frontend/index.html` at `/`
- Health check: `/api/health`

## What Was Wired

- Unified backend entrypoint: `src/main.py`
- Connected API routes:
  - `POST /api/m3/optimize`
  - `POST /api/m4/scenarios`
  - `POST /api/m5/institutional`
  - `POST /api/m6/simulate`
  - `POST /api/m6/security/test`
  - `POST /api/m7/regime`
- Frontend default API base now uses same-origin in deployed environments.
- Added attack-type mapping in M6 security test endpoint for frontend compatibility.

## Local Run

1. Create/activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start server:

```bash
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open:

- App: `http://localhost:8000/`
- Health: `http://localhost:8000/api/health`

## Docker Run

Build:

```bash
docker build -t clearview-analytics .
```

Run:

```bash
docker run --rm -p 8000:8000 clearview-analytics
```

## Render Deployment

This repo includes:

- `Procfile`
- `render.yaml`

You can deploy directly to Render from this repository. The app starts with:

```bash
uvicorn src.main:app --host 0.0.0.0 --port $PORT
```

## Notes

- `CORS_ORIGINS` can be set as a comma-separated env var. Default is `*`.
- Frontend can optionally be pointed to a custom backend by setting `window.__VITE_API_BASE_URL`.
- Mongo-backed Bayesian memory is optional:
  - Set `MONGODB_URI` to enable persistent immune memory.
  - The API stores learned Bayesian threat patterns and security events in MongoDB.
