# Vercel Deployment - Quick Start

Get your ClearView Analytics app live on Vercel in 5 minutes.

## Step 1: Prepare Your Code (1 min)

```bash
# Make sure everything is committed
git add .
git commit -m "Ready for Vercel deployment"
git push origin main
```

## Step 2: Create Vercel Project (2 min)

1. Go to https://vercel.com/new
2. Click "Import Git Repository"
3. Select your GitHub repository
4. Click "Import"
5. Vercel will auto-detect Python framework
6. Click "Deploy"

## Step 3: Generate AUTH_SECRET (1 min)

Run this command to generate a secure secret:

```bash
bash generate_auth_secret.sh
```

Or manually:
```bash
openssl rand -hex 32
```

Copy the output - you'll need it in the next step.

## Step 4: Set Environment Variables (1 min)

After deployment starts:

1. Go to Vercel Dashboard → Your Project
2. Click "Settings" → "Environment Variables"
3. Add these variables:

| Name | Value | Environment |
|------|-------|-------------|
| `CORS_ORIGINS` | `https://<your-project>.vercel.app` | Production |
| `AUTH_SECRET` | (paste from Step 3) | Production |

4. Click "Save"
5. Go to "Deployments" tab
6. Click the three dots on the latest deployment
7. Select "Redeploy"

## Step 5: Verify Deployment (1 min)

Wait for redeployment to complete, then test:

```bash
# Health check
curl https://<your-project>.vercel.app/api/health

# Should return: {"status":"ok"}
```

Visit `https://<your-project>.vercel.app/` in your browser to see the frontend.

## Done! 🎉

Your app is now live on Vercel!

### Next Steps (Optional)

- **Add Custom Domain**: Settings → Domains → Add Domain
- **Enable MongoDB**: Set `MONGODB_URI` for persistent data
- **Monitor Performance**: Check Vercel Analytics dashboard
- **Set Up CI/CD**: Enable automatic deployments on push

## Troubleshooting

### Build Failed
- Check Vercel build logs for errors
- Ensure `requirements.txt` is in root directory
- Verify Python 3.11 compatibility

### Deployment URL Not Working
- Wait 2-3 minutes for deployment to complete
- Check that environment variables are set
- Verify `CORS_ORIGINS` matches your URL

### API Endpoints Return 500 Error
- Check Vercel function logs
- Ensure `AUTH_SECRET` is set
- Verify all environment variables are configured

### Need Help?
- See `VERCEL_DEPLOYMENT.md` for detailed guide
- Check `DEPLOYMENT_CHECKLIST.md` for verification steps
- Review Vercel documentation: https://vercel.com/docs

## Environment Variables Reference

```
CORS_ORIGINS=https://<your-project>.vercel.app
AUTH_SECRET=<generated-secret>
AUTH_DB_PATH=/tmp/clearview_auth.db
AUTH_TOKEN_TTL_SECONDS=604800
MONGODB_URI=<optional-mongodb-connection>
MONGODB_DB=clearview_analytics
DATA_CACHE_DIR=/tmp/data_cache
```

## API Endpoints

Once deployed, access these endpoints:

- **Health**: `GET /api/health`
- **Register**: `POST /api/auth/register`
- **Login**: `POST /api/auth/login`
- **Portfolio Optimization**: `POST /api/m3/optimize`
- **Scenario Analysis**: `POST /api/m4/scenarios`
- **Institutional Allocation**: `POST /api/m5/institutional`
- **Portfolio Simulation**: `POST /api/m6/simulate`
- **Security Testing**: `POST /api/m6/security/test`
- **Market Regime**: `POST /api/m7/regime`

## Deployment URL

Your app will be available at:
```
https://<your-project>.vercel.app
```

Replace `<your-project>` with your actual Vercel project name.

## Support

For more detailed information, see:
- `VERCEL_DEPLOYMENT.md` - Complete deployment guide
- `DEPLOYMENT_CHECKLIST.md` - Pre/post deployment checklist
- `README.md` - Project overview
