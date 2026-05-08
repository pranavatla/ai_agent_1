#!/bin/bash

# 🚀 AIF App Deployment Script for AWS
# This script automates the deployment process
# Usage: bash deploy-to-aws.sh

set -e

echo "🚀 AIF App Deployment Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
DOMAIN="aif.atla.in"
APP_NAME="aif-app"
PORT="3000"
REPO_NAME="YOUR-REPO"  # Change this to your repo name

echo -e "${YELLOW}Configuration:${NC}"
echo "Domain: $DOMAIN"
echo "App Name: $APP_NAME"
echo "Port: $PORT"
echo ""

# Step 1: Update System
echo -e "${YELLOW}Step 1: Updating system...${NC}"
sudo apt update
sudo apt upgrade -y
echo -e "${GREEN}✓ System updated${NC}"

# Step 2: Install Node.js
echo -e "${YELLOW}Step 2: Installing Node.js...${NC}"
if ! command -v node &> /dev/null; then
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
    sudo apt-get install -y nodejs
    echo -e "${GREEN}✓ Node.js installed${NC}"
else
    echo -e "${GREEN}✓ Node.js already installed${NC}"
fi

# Step 3: Install Git
echo -e "${YELLOW}Step 3: Installing Git...${NC}"
sudo apt install -y git
echo -e "${GREEN}✓ Git installed${NC}"

# Step 4: Install PM2
echo -e "${YELLOW}Step 4: Installing PM2...${NC}"
if ! command -v pm2 &> /dev/null; then
    sudo npm install -g pm2
    echo -e "${GREEN}✓ PM2 installed${NC}"
else
    echo -e "${GREEN}✓ PM2 already installed${NC}"
fi

# Step 5: Install Nginx
echo -e "${YELLOW}Step 5: Installing Nginx...${NC}"
sudo apt install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx
echo -e "${GREEN}✓ Nginx installed and started${NC}"

# Step 6: Clone/Setup App
echo -e "${YELLOW}Step 6: Setting up application...${NC}"
cd /home/ubuntu

if [ -d "$REPO_NAME" ]; then
    echo "Repository already exists. Updating..."
    cd $REPO_NAME
    git pull
else
    echo "Cloning repository..."
    echo -e "${YELLOW}⚠️  Please enter your GitHub repo URL:${NC}"
    read REPO_URL
    git clone $REPO_URL $REPO_NAME
    cd $REPO_NAME
fi

# Step 7: Install Dependencies
echo -e "${YELLOW}Step 7: Installing dependencies...${NC}"
npm install
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Step 8: Build App
echo -e "${YELLOW}Step 8: Building Next.js app...${NC}"
npm run build
echo -e "${GREEN}✓ Build completed${NC}"

# Step 9: Start with PM2
echo -e "${YELLOW}Step 9: Starting app with PM2...${NC}"
pm2 stop $APP_NAME 2>/dev/null || true
pm2 start npm --name "$APP_NAME" -- start
pm2 startup
pm2 save
echo -e "${GREEN}✓ App started with PM2${NC}"

# Step 10: Configure Nginx
echo -e "${YELLOW}Step 10: Configuring Nginx...${NC}"
sudo tee /etc/nginx/sites-available/$DOMAIN > /dev/null <<EOF
upstream ${APP_NAME}_app {
    server 127.0.0.1:$PORT;
}

server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://${APP_NAME}_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_cache_bypass \$http_upgrade;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

sudo ln -sf /etc/nginx/sites-available/$DOMAIN /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
echo -e "${GREEN}✓ Nginx configured${NC}"

# Step 11: Install SSL
echo -e "${YELLOW}Step 11: Installing SSL certificate...${NC}"
sudo apt install -y certbot python3-certbot-nginx
echo -e "${YELLOW}⚠️  Running certbot. Follow the prompts...${NC}"
sudo certbot --nginx -d $DOMAIN
echo -e "${GREEN}✓ SSL certificate installed${NC}"

# Step 12: Create .env.production
echo -e "${YELLOW}Step 12: Creating .env.production...${NC}"
cat > /home/ubuntu/$REPO_NAME/.env.production <<EOF
NEXT_PUBLIC_API_URL=https://$DOMAIN
NODE_ENV=production
EOF

pm2 restart $APP_NAME
echo -e "${GREEN}✓ Environment variables set${NC}"

# Summary
echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}✓ Deployment Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "${YELLOW}Your app is now live at:${NC}"
echo -e "${GREEN}https://$DOMAIN${NC}"
echo ""
echo -e "${YELLOW}Quick Commands:${NC}"
echo "  View logs:      pm2 logs $APP_NAME"
echo "  Restart app:    pm2 restart $APP_NAME"
echo "  Stop app:       pm2 stop $APP_NAME"
echo "  App status:     pm2 status"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Make sure your DNS is pointing to this server's IP"
echo "  2. Test by visiting https://$DOMAIN"
echo "  3. Monitor logs: pm2 logs $APP_NAME"
echo ""
