# 🚀 ClearView Analytics - Vercel Deployment Guide

Welcome! This guide will help you deploy ClearView Analytics to Vercel in minutes.

## 📋 Table of Contents

1. [Quick Start (5 min)](#quick-start)
2. [Detailed Guide](#detailed-guide)
3. [Environment Variables](#environment-variables)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [Next Steps](#next-steps)

## Quick Start

### Prerequisites
- GitHub account with this repository
- Vercel account (free at https://vercel.com)

### Step 1: Push to GitHub
```bash
git add .
git commit -m "Deploy to Vercel"
git push origin main
```

### Step 2: Create Vercel Project
1. Go to https://vercel.com/new
2. Click "Import Git Repository"
3. Select your repository
4. Click "Import"

### Step 3: Generate AUTH_SECRET
```bash
openssl rand -hex 32
```
Copy the output - you'll need it next.

### Step 4: Set Environment Variables
In Vercel dashboard:
1. Go to Settings → Environment Variables
2. Add:
   - `CORS_ORIGINS`: `https://<your-project>.vercel.app`
   - `AUTH_SECRET`: (paste from Step 3)
3. Click "Save"

### Step 5: Redeploy
1. Go to Deployments
2. Click the three dots on latest deployment
3. Select "Redeploy"

### Step 6: Verify
```bash
curl https://<your-project>.vercel.app/api/health
```

✅ Done! Your app is live!

---

## Detailed Guide

### Option A: GitHub Deployment (Recommended)

**Advantages:**
- Automatic deployments on push
- Easy rollback
- No CLI installation needed
- Integrated with GitHub

**Steps:**

1. **Ensure code is on GitHub**
   ```bash
   git push origin main
   ```

2. **Create Vercel Project**
   - Visit https://vercel.com/new
   - Click "Import Git Repository"
   - Authorize GitHub if needed
   - Select your repository
   - Click "Import"

3. **Configure Project**
   - Framework: Python (auto-detected)
   - Build Command: `pip install -r requirements.txt`
   - Output Directory: `.`
   - Click "Deploy"

4. **Wait for Build**
   - Vercel will build and deploy
   - Takes 2-3 minutes
   - You'll see deployment progress

5. **Set Environment Variables**
   - After deployment, go to Settings → Environment Variables
   - Add required variables (see below)
   - Select "Production" environment
   - Click "Save"

6. **Redeploy with Variables**
   - Go to Deployments tab
   - Click three dots on latest deployment
   - Select "Redeploy"
   - Wait for new deployment

### Option B: CLI Deployment

**Advantages:**
- More control
- Can test locally first
- Faster for experienced users

**Steps:**

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```
   - Follow prompts
   - Link to existing project or create new
   - Confirm settings

4. **Set Environment Variables**
   ```bash
   vercel env add CORS_ORIGINS production
   vercel env add AUTH_SECRET production
   ```

5. **Deploy to Production**
   ```bash
   vercel --prod
   ```

---

## Environment Variables

### Required Variables

#### CORS_ORIGINS
- **Purpose**: Allow requests from your domain
- **Value**: `https://<your-project>.vercel.app`
- **Example**: `https://myapp.vercel.app`
- **Multiple origins**: `https://myapp.vercel.app,https://custom-domain.com`

#### AUTH_SECRET
- **Purpose**: JWT token signing secret
- **Generate**: `openssl rand -hex 32`
- **Length**: 64 characters (hex)
- **Security**: Keep this secret! Don't share or commit to git

### Optional Variables

#### MONGODB_URI
- **Purpose**: MongoDB connection for persistent storage
- **Value**: MongoDB Atlas connection string
- **Format**: `mongodb+srv://user:password@cluster.mongodb.net/`
- **Benefits**: Persistent data across deployments
- **Setup**: https://www.mongodb.com/docs/atlas/getting-started/

#### MONGODB_DB
- **Purpose**: Database name
- **Default**: `clearview_analytics`
- **Optional**: Only needed if using MongoDB

#### MONGODB_MEMORY_COLLECTION
- **Purpose**: Collection for Bayesian memory
- **Default**: `bayesian_immune_memory`
- **Optional**: Only needed if using MongoDB

#### MONGODB_EVENTS_COLLECTION
- **Purpose**: Collection for security events
- **Default**: `bayesian_security_events`
- **Optional**: Only needed if using MongoDB

#### AUTH_DB_PATH
- **Purpose**: SQLite database location
- **Default**: `/tmp/clearview_auth.db`
- **Note**: `/tmp` is ephemeral on Vercel
- **Recommendation**: Use MongoDB for production

#### AUTH_TOKEN_TTL_SECONDS
- **Purpose**: JWT token expiration time
- **Default**: `604800` (7 days)
- **Value**: Seconds
- **Example**: `86400` for 1 day

#### DATA_CACHE_DIR
- **Purpose**: Market data cache directory
- **Default**: `/tmp/data_cache`
- **Note**: `/tmp` is ephemeral on Vercel

---

## Verification

### 1. Health Check
```bash
curl https://<your-project>.vercel.app/api/health
```
Expected response:
```json
{"status":"ok"}
```

### 2. Frontend
Visit `https://<your-project>.vercel.app/` in your browser.
You should see the ClearView Analytics interface.

### 3. Authentication
```bash
# Register
curl -X POST https://<your-project>.vercel.app/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123",
    "email": "test@example.com"
  }'

# Login
curl -X POST https://<your-project>.vercel.app/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "password": "testpass123"
  }'
```

### 4. API Endpoints
Test each endpoint with your JWT token:
```bash
TOKEN="<your-jwt-token>"

# Portfolio Optimization
curl -X POST https://<your-project>.vercel.app/api/m3/optimize \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["RELIANCE.NS", "TCS.NS"]}'
```

---

## Troubleshooting

### Build Fails

**Error: "No module named 'numpy'"**
- Ensure `requirements.txt` is in root directory
- Check that all dependencies are listed
- Verify Python 3.11 compatibility

**Error: "Build command failed"**
- Check Vercel build logs for details
- Ensure no syntax errors in Python files
- Verify all imports are available

### Deployment Issues

**Error: "Function timed out"**
- Vercel has 60-second timeout
- Optimize long-running operations
- Consider using background jobs

**Error: "CORS error"**
- Ensure `CORS_ORIGINS` is set correctly
- Include `https://` protocol
- Check for typos in domain

**Error: "Unauthorized: invalid token"**
- Ensure `AUTH_SECRET` is set
- Check token hasn't expired
- Verify JWT format

### Data Issues

**Error: "Database not found"**
- On Vercel, `/tmp` is ephemeral
- Use MongoDB for persistent storage
- Set `MONGODB_URI` environment variable

**Error: "Cache not persisting"**
- `/tmp` is cleared between deployments
- Use external storage (MongoDB, S3, etc.)
- Consider caching strategy

### Performance Issues

**Slow cold starts**
- Normal for serverless functions (2-3 seconds)
- Subsequent requests are faster
- Consider upgrading Vercel plan

**High memory usage**
- Check for memory leaks
- Optimize data processing
- Consider splitting into multiple functions

---

## Next Steps

### 1. Add Custom Domain
1. Go to Vercel dashboard → Project Settings → Domains
2. Add your domain
3. Follow DNS configuration instructions
4. Update `CORS_ORIGINS` with custom domain

### 2. Enable MongoDB
1. Create MongoDB Atlas account
2. Create cluster and database
3. Get connection string
4. Set `MONGODB_URI` environment variable
5. Redeploy

### 3. Set Up Monitoring
1. Enable Vercel Analytics
2. Set up error alerts
3. Monitor function execution times
4. Review logs regularly

### 4. Configure CI/CD
1. Enable automatic deployments on push
2. Set up preview deployments for PRs
3. Configure deployment notifications
4. Set up automated testing

### 5. Optimize Performance
1. Enable caching for static assets
2. Optimize API response times
3. Use MongoDB for data persistence
4. Monitor cold start times

---

## API Endpoints

### Authentication
- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Get JWT token
- `GET /api/auth/me` - Get current user info
- `GET /api/auth/config` - Get auth configuration

### Portfolio Analysis
- `POST /api/m3/optimize` - Portfolio optimization
- `POST /api/m4/scenarios` - Scenario analysis
- `POST /api/m5/institutional` - Institutional allocation
- `POST /api/m6/simulate` - Portfolio simulation
- `POST /api/m6/security/test` - Security testing
- `POST /api/m7/regime` - Market regime intelligence

### Health
- `GET /api/health` - Health check

---

## Security Checklist

- ✅ `AUTH_SECRET` is strong and unique
- ✅ `CORS_ORIGINS` is restricted to your domain
- ✅ No secrets in code or `.env`
- ✅ HTTPS enforced (automatic on Vercel)
- ✅ Database credentials in environment variables
- ✅ JWT authentication enabled
- ✅ Regular dependency updates
- ✅ Monitoring and alerts configured

---

## Support

### Documentation
- `VERCEL_QUICK_START.md` - 5-minute quick start
- `DEPLOYMENT_CHECKLIST.md` - Verification checklist
- `DEPLOYMENT_SUMMARY.md` - Complete summary
- `README.md` - Project overview

### External Resources
- [Vercel Documentation](https://vercel.com/docs)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [MongoDB Atlas](https://www.mongodb.com/docs/atlas/)
- [Python on Vercel](https://vercel.com/docs/functions/python)

### Getting Help
- Check Vercel build logs
- Review function logs
- Check error messages
- Consult documentation

---

## Deployment Checklist

Before deploying:
- [ ] Code committed to GitHub
- [ ] `requirements.txt` updated
- [ ] `app.py` in root directory
- [ ] `frontend/index.html` exists
- [ ] No secrets in code

After deploying:
- [ ] Health check passes
- [ ] Frontend loads
- [ ] Can register user
- [ ] Can login
- [ ] API endpoints work
- [ ] Environment variables set
- [ ] Monitoring configured

---

## Rollback

If something goes wrong:

1. Go to Vercel dashboard → Deployments
2. Find the previous working deployment
3. Click the three dots menu
4. Select "Promote to Production"
5. Verify the rollback was successful

---

## Performance Metrics

- **Cold Start**: 2-3 seconds
- **Warm Start**: <100ms
- **Function Timeout**: 60 seconds
- **Memory**: 3008 MB
- **Max Payload**: 4.5 MB
- **Concurrent Executions**: Unlimited

---

## Pricing

Vercel offers:
- **Free Tier**: 100 GB bandwidth/month, unlimited deployments
- **Pro**: $20/month, 1 TB bandwidth/month
- **Enterprise**: Custom pricing

See https://vercel.com/pricing for details.

---

## FAQ

**Q: How long does deployment take?**
A: 2-3 minutes for initial build, <1 minute for subsequent deployments.

**Q: Can I use a custom domain?**
A: Yes! Add it in Vercel Project Settings → Domains.

**Q: Is my data persistent?**
A: `/tmp` is ephemeral. Use MongoDB for persistent storage.

**Q: How do I update my app?**
A: Push to GitHub and Vercel auto-deploys (if enabled).

**Q: Can I rollback?**
A: Yes! Go to Deployments and promote a previous deployment.

**Q: What's the cost?**
A: Free tier available. See https://vercel.com/pricing.

---

## Ready to Deploy?

1. Follow the **Quick Start** section above
2. Or run: `bash DEPLOY.sh`
3. Or see: `VERCEL_QUICK_START.md`

Your ClearView Analytics app will be live in minutes! 🎉

---

**Last Updated**: April 2026
**Version**: 1.0
**Status**: Ready for Production
