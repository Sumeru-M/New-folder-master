# Vercel Deployment Guide for ClearView Analytics

This guide walks you through deploying the ClearView Analytics FastAPI application to Vercel.

## Prerequisites

- GitHub account with this repository pushed
- Vercel account (free tier available at https://vercel.com)
- (Optional) MongoDB Atlas account for persistent Bayesian memory

## Deployment Steps

### Option A: Deploy from GitHub (Recommended)

1. **Push code to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Connect to Vercel**
   - Go to https://vercel.com/new
   - Click "Import Git Repository"
   - Select your GitHub repository
   - Vercel will auto-detect the Python framework

3. **Configure Environment Variables**
   - In the Vercel dashboard, go to Project Settings → Environment Variables
   - Add the following variables:

   **Required:**
   - `CORS_ORIGINS`: `https://<your-project>.vercel.app`
   - `AUTH_SECRET`: Generate a secure random string (e.g., `openssl rand -hex 32`)

   **Optional but Recommended:**
   - `MONGODB_URI`: Your MongoDB Atlas connection string (for persistent Bayesian memory)
   - `MONGODB_DB`: `clearview_analytics`
   - `MONGODB_MEMORY_COLLECTION`: `bayesian_immune_memory`
   - `MONGODB_EVENTS_COLLECTION`: `bayesian_security_events`

4. **Deploy**
   - Click "Deploy"
   - Wait for the build to complete (typically 2-3 minutes)

### Option B: Deploy from CLI

1. **Install Vercel CLI**
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```
   - Follow the prompts to link your project
   - For production deployment:
   ```bash
   vercel --prod
   ```

4. **Set Environment Variables**
   ```bash
   vercel env add CORS_ORIGINS production
   vercel env add AUTH_SECRET production
   vercel env add MONGODB_URI production
   ```

## Post-Deployment Verification

After deployment, verify your application is working:

1. **Health Check**
   ```bash
   curl https://<your-project>.vercel.app/api/health
   ```
   Expected response: `{"status":"ok"}`

2. **Frontend**
   - Visit `https://<your-project>.vercel.app/`
   - You should see the ClearView Analytics interface

3. **Authentication**
   ```bash
   curl -X POST https://<your-project>.vercel.app/api/auth/register \
     -H "Content-Type: application/json" \
     -d '{"username":"testuser","password":"testpass123","email":"test@example.com"}'
   ```

## Environment Variables Reference

| Variable | Required | Description | Example |
|----------|----------|-------------|---------|
| `CORS_ORIGINS` | Yes | Allowed CORS origins | `https://myapp.vercel.app` |
| `AUTH_SECRET` | Yes | JWT secret for authentication | `your-secret-key-here` |
| `AUTH_DB_PATH` | No | Path to SQLite auth database | `/tmp/auth.db` |
| `AUTH_TOKEN_TTL_SECONDS` | No | Token expiration time | `604800` (7 days) |
| `MONGODB_URI` | No | MongoDB connection string | `mongodb+srv://...` |
| `MONGODB_DB` | No | MongoDB database name | `clearview_analytics` |
| `MONGODB_MEMORY_COLLECTION` | No | Bayesian memory collection | `bayesian_immune_memory` |
| `MONGODB_EVENTS_COLLECTION` | No | Security events collection | `bayesian_security_events` |
| `DATA_CACHE_DIR` | No | Market data cache directory | `/tmp/data_cache` |

## API Endpoints

Once deployed, the following endpoints are available:

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and get JWT token
- `GET /api/auth/me` - Get current user info
- `GET /api/auth/config` - Get auth configuration

### Portfolio Analysis
- `POST /api/m3/optimize` - Portfolio optimization (M3)
- `POST /api/m4/scenarios` - Scenario analysis (M4)
- `POST /api/m5/institutional` - Institutional allocation (M5)
- `POST /api/m6/simulate` - Portfolio simulation (M6)
- `POST /api/m6/security/test` - Security testing (M6)
- `POST /api/m7/regime` - Market regime intelligence (M7)

### Health
- `GET /api/health` - Health check

## Troubleshooting

### Build Fails
- Check that `requirements.txt` is in the root directory
- Ensure all dependencies are compatible with Python 3.11
- Check Vercel build logs for specific errors

### Timeout Issues
- Vercel has a 60-second timeout for serverless functions
- For long-running operations, consider using background jobs or increasing timeout in `vercel.json`

### Data Cache Issues
- On Vercel, `/tmp` is ephemeral and cleared between deployments
- For persistent caching, use MongoDB or external storage

### Authentication Issues
- Ensure `AUTH_SECRET` is set in environment variables
- Check that `CORS_ORIGINS` matches your deployment URL

## Custom Domain

To add a custom domain:

1. In Vercel dashboard, go to Project Settings → Domains
2. Add your domain
3. Follow DNS configuration instructions
4. Update `CORS_ORIGINS` environment variable with your custom domain

## Monitoring & Logs

- View deployment logs in Vercel dashboard under Deployments
- Check function logs in the Logs tab
- Use the Analytics tab to monitor performance

## Rollback

To rollback to a previous deployment:

1. Go to Deployments tab in Vercel dashboard
2. Find the previous deployment
3. Click the three dots menu
4. Select "Promote to Production"

## Additional Resources

- [Vercel Python Documentation](https://vercel.com/docs/functions/python)
- [FastAPI Deployment Guide](https://fastapi.tiangolo.com/deployment/)
- [MongoDB Atlas Setup](https://www.mongodb.com/docs/atlas/getting-started/)
