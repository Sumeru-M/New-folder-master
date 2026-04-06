# ClearView Analytics (Deployable Full Stack)

This repo now runs as a single deployable FastAPI product:

- Backend APIs: `/api/m3` to `/api/m7`
- Frontend: served from `frontend/index.html` at `/`
- Health check: `/api/health`

## What Was Wired

- Unified backend entrypoint: `src/main.py`
- Username/password authentication:
  - `POST /api/auth/register`
  - `POST /api/auth/login`
  - `GET /api/auth/config`
  - `GET /api/auth/me`
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

## Vercel Deployment (Global)

This repo is Vercel-ready with a FastAPI entrypoint at `app.py`.

### Option A: Deploy from GitHub (recommended)

1. Push your latest code to GitHub.
2. In Vercel: `Add New Project` -> import this repo.
3. Framework preset: let Vercel auto-detect FastAPI/Python.
4. Set environment variables in Project Settings -> Environment Variables:
   - `CORS_ORIGINS=https://<your-project>.vercel.app`
   - `AUTH_SECRET=<your-jwt-secret>`
   - `AUTH_DB_PATH=<optional-custom-path-for-sqlite-auth-db>`
   - `AUTH_TOKEN_TTL_SECONDS=604800` (optional)
   - `MONGODB_URI=<your-mongodb-connection-string>` (optional but recommended for persistent Bayesian memory)
   - `MONGODB_DB=clearview_analytics` (optional)
   - `MONGODB_MEMORY_COLLECTION=bayesian_immune_memory` (optional)
   - `MONGODB_EVENTS_COLLECTION=bayesian_security_events` (optional)
5. Deploy.

### Option B: Deploy from CLI

```bash
npm i -g vercel
vercel login
vercel
vercel --prod
```

Then set env vars (example):

```bash
vercel env add CORS_ORIGINS production
vercel env add MONGODB_URI production
```

### Post-deploy checks

- App: `https://<your-project>.vercel.app/`
- Health: `https://<your-project>.vercel.app/api/health`

### Custom domain

In Vercel Project -> `Settings -> Domains`, add your domain and follow DNS instructions.

## Notes

- `CORS_ORIGINS` can be set as a comma-separated env var. Default is `*`.
- Frontend can optionally be pointed to a custom backend by setting `window.__VITE_API_BASE_URL`.
- `DATA_CACHE_DIR` can override market-data cache path. On Vercel, `/tmp/data_cache` is used automatically.
- Username/password auth is enabled by default.
  - Set `AUTH_SECRET` in production.
  - Optional: `AUTH_DB_PATH`, `AUTH_TOKEN_TTL_SECONDS`.
- Mongo-backed Bayesian memory is optional:
  - Set `MONGODB_URI` to enable persistent immune memory.
  - The API stores learned Bayesian threat patterns and security events in MongoDB.
