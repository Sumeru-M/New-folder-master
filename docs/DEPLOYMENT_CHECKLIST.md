# Vercel Deployment Checklist

## Pre-Deployment

- [ ] All code changes committed and pushed to GitHub
- [ ] `requirements.txt` is up to date with all dependencies
- [ ] `.env.example` documents all required environment variables
- [ ] `vercel.json` is configured correctly
- [ ] `app.py` exists in root directory and imports from `src.main`
- [ ] `frontend/index.html` exists for static frontend serving
- [ ] No sensitive data (API keys, passwords) in code or `.env`

## Environment Variables to Set in Vercel

### Required
- [ ] `CORS_ORIGINS` - Set to your Vercel deployment URL (e.g., `https://myapp.vercel.app`)
- [ ] `AUTH_SECRET` - Generate with: `openssl rand -hex 32`

### Optional but Recommended
- [ ] `MONGODB_URI` - MongoDB Atlas connection string (for persistent data)
- [ ] `MONGODB_DB` - Database name (default: `clearview_analytics`)
- [ ] `MONGODB_MEMORY_COLLECTION` - Collection name (default: `bayesian_immune_memory`)
- [ ] `MONGODB_EVENTS_COLLECTION` - Collection name (default: `bayesian_security_events`)

### Optional
- [ ] `AUTH_DB_PATH` - Custom SQLite database path
- [ ] `AUTH_TOKEN_TTL_SECONDS` - Token expiration (default: 604800 = 7 days)
- [ ] `DATA_CACHE_DIR` - Market data cache directory (default: `/tmp/data_cache`)

## Deployment Steps

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for Vercel deployment"
   git push origin main
   ```

2. **Create Vercel Project**
   - Go to https://vercel.com/new
   - Import your GitHub repository
   - Select Python framework
   - Click "Deploy"

3. **Configure Environment Variables**
   - After deployment starts, go to Project Settings → Environment Variables
   - Add all required variables from above
   - Redeploy with new environment variables

4. **Verify Deployment**
   - [ ] Health check: `curl https://<your-project>.vercel.app/api/health`
   - [ ] Frontend loads: Visit `https://<your-project>.vercel.app/`
   - [ ] Can register user: Test `/api/auth/register` endpoint
   - [ ] Can login: Test `/api/auth/login` endpoint

## Post-Deployment

- [ ] Monitor Vercel dashboard for errors
- [ ] Check function logs for any issues
- [ ] Test all API endpoints with real data
- [ ] Set up custom domain (if desired)
- [ ] Configure monitoring/alerts
- [ ] Document deployment URL for team

## Troubleshooting

### Build Fails
- Check Vercel build logs for specific errors
- Ensure Python 3.11 is compatible with all dependencies
- Verify `requirements.txt` has no conflicting versions

### Timeout Errors
- Vercel serverless functions have 60-second timeout
- Consider optimizing long-running operations
- Use background jobs for heavy computations

### Data Persistence Issues
- `/tmp` directory is ephemeral on Vercel
- Use MongoDB for persistent storage
- Set `MONGODB_URI` environment variable

### CORS Errors
- Ensure `CORS_ORIGINS` matches your deployment URL
- Include protocol (https://) in CORS_ORIGINS
- For multiple origins, use comma-separated list

### Authentication Issues
- Verify `AUTH_SECRET` is set and non-empty
- Check JWT token expiration with `AUTH_TOKEN_TTL_SECONDS`
- Ensure SQLite database path is writable (use `/tmp` on Vercel)

## Rollback Procedure

If deployment has issues:

1. Go to Vercel dashboard → Deployments
2. Find the previous working deployment
3. Click menu (three dots) → "Promote to Production"
4. Verify the rollback was successful

## Performance Optimization

- [ ] Enable caching for static assets
- [ ] Use MongoDB for data persistence instead of SQLite
- [ ] Optimize API response times
- [ ] Monitor cold start times
- [ ] Consider upgrading Vercel plan if needed

## Security Checklist

- [ ] `AUTH_SECRET` is strong and unique
- [ ] `CORS_ORIGINS` is restricted to your domain
- [ ] No API keys or secrets in code
- [ ] HTTPS is enforced (automatic on Vercel)
- [ ] Database credentials are in environment variables
- [ ] Regular security updates for dependencies

## Monitoring

- [ ] Set up Vercel Analytics
- [ ] Monitor error rates in Vercel dashboard
- [ ] Check function execution times
- [ ] Review logs regularly
- [ ] Set up alerts for failures

## Next Steps

- [ ] Add custom domain
- [ ] Set up CI/CD pipeline
- [ ] Configure automatic deployments on push
- [ ] Set up monitoring and alerting
- [ ] Document API for users
- [ ] Create user guide for frontend
