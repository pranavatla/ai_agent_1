#!/bin/bash

# 🚀 Deploy to EC2 - Complete Deployment Script
# This script uploads all files and completes the deployment to blitz.atla.in

set -e

echo "=================================="
echo "🚀 Deploying to blitz.atla.in"
echo "=================================="
echo ""

# Configuration
EC2_USER="ubuntu"
EC2_IP="13.206.147.51"  # Your EC2 public IP
KEY_PATH="$HOME/.ssh/blitz-key.pem"  # Adjust path to your key
REMOTE_APP_DIR="/home/ubuntu/blitz-app"

# Check if key exists
if [ ! -f "$KEY_PATH" ]; then
    echo "❌ Error: SSH key not found at $KEY_PATH"
    echo "Please update KEY_PATH in this script with your actual key path"
    exit 1
fi

echo "📍 EC2 Configuration:"
echo "   IP: $EC2_IP"
echo "   User: $EC2_USER"
echo "   App Dir: $REMOTE_APP_DIR"
echo ""

# Step 1: Create app directory on remote server
echo "📁 Step 1: Creating app directory on EC2..."
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "mkdir -p $REMOTE_APP_DIR/pages $REMOTE_APP_DIR/styles"
echo "✓ Directory created"
echo ""

# Step 2: Copy all project files to remote server
echo "📤 Step 2: Uploading project files..."
scp -i "$KEY_PATH" -r package.json "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/"
scp -i "$KEY_PATH" -r next.config.js "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/"
scp -i "$KEY_PATH" -r tailwind.config.js "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/"
scp -i "$KEY_PATH" -r postcss.config.js "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/"
scp -i "$KEY_PATH" -r .env.local "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/"
scp -i "$KEY_PATH" -r pages-landing-page.jsx "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/pages/"
scp -i "$KEY_PATH" -r pages-marketing-dashboard.jsx "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/pages/"
scp -i "$KEY_PATH" -r pages-community-finder.jsx "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/pages/"
scp -i "$KEY_PATH" -r pages-14-week-curriculum.jsx "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/pages/"
scp -i "$KEY_PATH" -r pages/index.jsx "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/pages/"
scp -i "$KEY_PATH" -r pages/curriculum.jsx "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/pages/"
scp -i "$KEY_PATH" -r pages/dashboard.jsx "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/pages/"
scp -i "$KEY_PATH" -r pages/communities.jsx "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/pages/"
scp -i "$KEY_PATH" -r pages/_app.jsx "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/pages/"
scp -i "$KEY_PATH" -r pages/_document.jsx "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/pages/"
scp -i "$KEY_PATH" -r styles/globals.css "$EC2_USER@$EC2_IP:$REMOTE_APP_DIR/styles/"
echo "✓ Files uploaded"
echo ""

# Step 3: Install dependencies and build on remote server
echo "🔨 Step 3: Installing dependencies and building..."
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" << 'EOF'
cd /home/ubuntu/blitz-app
echo "Installing npm dependencies..."
npm install
echo "Building Next.js app..."
npm run build
echo "✓ Build completed successfully"
EOF
echo ""

# Step 4: Stop and restart PM2 app from correct directory
echo "🔄 Step 4: Restarting PM2..."
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" << 'EOF'
cd /home/ubuntu/blitz-app
pm2 stop blitz-app 2>/dev/null || true
pm2 delete blitz-app 2>/dev/null || true
pm2 start npm --name "blitz-app" -- start
pm2 save
pm2 status
EOF
echo "✓ PM2 restarted"
echo ""

# Step 5: Verify deployment
echo "✅ Step 5: Verifying deployment..."
sleep 3
ssh -i "$KEY_PATH" "$EC2_USER@$EC2_IP" "pm2 logs blitz-app | head -20"
echo ""

# Final message
echo "=================================="
echo "✅ Deployment Complete!"
echo "=================================="
echo ""
echo "🌐 Access your site at:"
echo "   https://blitz.atla.in"
echo ""
echo "📊 Monitor logs:"
echo "   ssh -i $KEY_PATH $EC2_USER@$EC2_IP"
echo "   pm2 logs blitz-app"
echo ""
echo "🔄 Restart app:"
echo "   pm2 restart blitz-app"
echo ""
echo "⚠️  If you see errors, check:"
echo "   - DNS propagation: nslookup blitz.atla.in"
echo "   - App status: pm2 status"
echo "   - Nginx config: sudo nginx -t"
echo ""
