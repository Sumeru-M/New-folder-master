#!/bin/bash
# ClearView Analytics - Vercel Deployment Script

set -e

echo "🚀 ClearView Analytics - Vercel Deployment"
echo "=========================================="
echo ""

# Check if git is initialized
if [ ! -d .git ]; then
    echo "❌ Error: Not a git repository"
    echo "Please run: git init && git add . && git commit -m 'Initial commit'"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "❌ Error: requirements.txt not found"
    exit 1
fi

# Check if app.py exists
if [ ! -f app.py ]; then
    echo "❌ Error: app.py not found"
    exit 1
fi

echo "✅ Pre-flight checks passed"
echo ""

# Generate AUTH_SECRET
echo "📝 Generating AUTH_SECRET..."
AUTH_SECRET=$(openssl rand -hex 32)
echo "✅ AUTH_SECRET generated: $AUTH_SECRET"
echo ""

# Check if Vercel CLI is installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

echo "🔐 Environment Variables to Set:"
echo "================================"
echo "CORS_ORIGINS: https://<your-project>.vercel.app"
echo "AUTH_SECRET: $AUTH_SECRET"
echo ""

# Prompt for deployment method
echo "Choose deployment method:"
echo "1) GitHub (recommended) - Deploy from GitHub via Vercel dashboard"
echo "2) CLI - Deploy using Vercel CLI"
echo ""
read -p "Enter choice (1 or 2): " choice

case $choice in
    1)
        echo ""
        echo "📤 GitHub Deployment Instructions:"
        echo "1. Push your code to GitHub:"
        echo "   git push origin main"
        echo ""
        echo "2. Go to https://vercel.com/new"
        echo "3. Import your GitHub repository"
        echo "4. Vercel will auto-detect Python framework"
        echo "5. Click 'Deploy'"
        echo ""
        echo "6. After deployment, set environment variables:"
        echo "   - CORS_ORIGINS: https://<your-project>.vercel.app"
        echo "   - AUTH_SECRET: $AUTH_SECRET"
        echo ""
        echo "7. Redeploy with new environment variables"
        echo ""
        ;;
    2)
        echo ""
        echo "🔑 Logging into Vercel..."
        vercel login
        echo ""
        echo "📤 Deploying to Vercel..."
        vercel
        echo ""
        echo "⚙️  Setting environment variables..."
        echo ""
        echo "Run these commands to set environment variables:"
        echo "vercel env add CORS_ORIGINS production"
        echo "vercel env add AUTH_SECRET production"
        echo ""
        echo "Then redeploy:"
        echo "vercel --prod"
        echo ""
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac

echo "✅ Deployment setup complete!"
echo ""
echo "📚 For more information, see:"
echo "   - VERCEL_QUICK_START.md"
echo "   - VERCEL_DEPLOYMENT.md"
echo "   - DEPLOYMENT_CHECKLIST.md"
echo ""
echo "🎉 Your app will be live soon!"
