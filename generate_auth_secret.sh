#!/bin/bash
# Generate a secure AUTH_SECRET for Vercel deployment

echo "Generating secure AUTH_SECRET..."
AUTH_SECRET=$(openssl rand -hex 32)
echo ""
echo "Your AUTH_SECRET (copy this to Vercel environment variables):"
echo "================================"
echo "$AUTH_SECRET"
echo "================================"
echo ""
echo "Steps to add to Vercel:"
echo "1. Go to Vercel Dashboard → Project Settings → Environment Variables"
echo "2. Click 'Add New'"
echo "3. Name: AUTH_SECRET"
echo "4. Value: $AUTH_SECRET"
echo "5. Select 'Production' environment"
echo "6. Click 'Save'"
echo "7. Redeploy your project"
