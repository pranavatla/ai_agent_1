# 🚀 Final Deployment Steps for blitz.atla.in

## Status: Files Ready ✅

All your Next.js components and configuration files have been prepared and are ready to deploy to your EC2 instance.

---

## What's Prepared

✅ **4 React Components**
- `pages-landing-page.jsx` - Beautiful hero landing page
- `pages-marketing-dashboard.jsx` - Real-time dashboard
- `pages-community-finder.jsx` - Community discovery tool
- `pages-14-week-curriculum.jsx` - 14-week curriculum tracker

✅ **Next.js Configuration**
- `package.json` - Dependencies and scripts
- `next.config.js` - Next.js configuration
- `tailwind.config.js` - Tailwind CSS setup
- `postcss.config.js` - PostCSS configuration
- `.env.local` - Environment variables

✅ **Page Files**
- `pages/index.jsx` - Landing page route
- `pages/curriculum.jsx` - Curriculum route
- `pages/dashboard.jsx` - Dashboard route
- `pages/communities.jsx` - Communities route
- `pages/_app.jsx` - App wrapper
- `pages/_document.jsx` - HTML document
- `styles/globals.css` - Global styles

✅ **Deployment Scripts**
- `QUICK_DEPLOY.sh` - One-command deployment script

---

## How to Deploy

### Option 1: Use the Quick Deploy Script (Recommended)

```bash
# On your LOCAL machine, from the project directory:
cd ~/your/path/to/AI_Agent_1-develop

# Run the deployment script
bash QUICK_DEPLOY.sh /path/to/your/blitz-key.pem
```

**That's it!** The script will:
- ✅ Create directories on EC2
- ✅ Upload all files
- ✅ Install npm dependencies
- ✅ Build the Next.js app
- ✅ Restart PM2 with correct configuration
- ✅ Show you the logs

### Option 2: Manual Deployment (Step-by-Step)

If the script doesn't work, follow these manual steps:

#### Step 1: Copy Files to EC2

```bash
# Set variables
EC2_IP="13.206.147.51"
EC2_USER="ubuntu"
KEY_FILE="/path/to/your/blitz-key.pem"
REMOTE_DIR="/home/ubuntu/blitz-app"

# Create directories
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" "mkdir -p $REMOTE_DIR/pages $REMOTE_DIR/styles"

# Copy all files
cd /path/to/AI_Agent_1-develop

# Config files
scp -i "$KEY_FILE" package.json "$EC2_USER@$EC2_IP:$REMOTE_DIR/"
scp -i "$KEY_FILE" next.config.js "$EC2_USER@$EC2_IP:$REMOTE_DIR/"
scp -i "$KEY_FILE" tailwind.config.js "$EC2_USER@$EC2_IP:$REMOTE_DIR/"
scp -i "$KEY_FILE" postcss.config.js "$EC2_USER@$EC2_IP:$REMOTE_DIR/"
scp -i "$KEY_FILE" .env.local "$EC2_USER@$EC2_IP:$REMOTE_DIR/"

# Component files
scp -i "$KEY_FILE" pages-*.jsx "$EC2_USER@$EC2_IP:$REMOTE_DIR/pages/"

# Page files
scp -i "$KEY_FILE" pages/*.jsx "$EC2_USER@$EC2_IP:$REMOTE_DIR/pages/"

# Styles
scp -i "$KEY_FILE" styles/globals.css "$EC2_USER@$EC2_IP:$REMOTE_DIR/styles/"
```

#### Step 2: Build on EC2

```bash
ssh -i "$KEY_FILE" "$EC2_USER@$EC2_IP" << 'EOF'
cd /home/ubuntu/blitz-app

# Install dependencies
npm install

# Build Next.js app
npm run build

# Stop old PM2 process
pm2 stop blitz-app 2>/dev/null || true
pm2 delete blitz-app 2>/dev/null || true

# Start new process
pm2 start npm --name "blitz-app" -- start
pm2 save

# Show status
echo "✓ Deployment complete!"
pm2 status
EOF
```

---

## After Deployment: Verification Steps

### 1. Check DNS is working
```bash
nslookup blitz.atla.in
# Should return: 13.206.147.51
```

### 2. SSH into EC2 and verify
```bash
ssh -i "your-key.pem" ubuntu@13.206.147.51

# Check PM2 status
pm2 status
# Output should show: blitz-app online

# View logs
pm2 logs blitz-app | head -20

# Check if app is listening
curl http://localhost:3000
```

### 3. Test in browser
Open: **https://blitz.atla.in**

You should see:
- ✅ Green lock icon (HTTPS)
- ✅ Landing page with hero section
- ✅ Navigation bar with links
- ✅ Timeline and feature cards

### 4. Test all pages
Click through:
- [ ] Landing page (/) ✅
- [ ] Curriculum (/curriculum)
- [ ] Dashboard (/dashboard)
- [ ] Communities (/communities)

---

## Troubleshooting

### Site won't load or shows "502 Bad Gateway"

```bash
# SSH into EC2
ssh -i "your-key.pem" ubuntu@13.206.147.51

# Check if app is running
pm2 status

# Check logs for errors
pm2 logs blitz-app | tail -50

# Restart if needed
pm2 restart blitz-app

# Check Nginx is running
sudo systemctl status nginx

# Test Nginx config
sudo nginx -t
```

### DNS not resolving

```bash
# Check if DNS is propagated
nslookup blitz.atla.in

# If it shows wrong IP, verify Route 53 in AWS Console:
# - Go to Route 53
# - Find hosted zone: atla.in
# - Check A record for blitz.atla.in
# - Should point to: 13.206.147.51
# - If wrong, update it and wait 5-10 minutes
```

### npm install fails

```bash
# On EC2, try:
cd /home/ubuntu/blitz-app
rm -rf node_modules package-lock.json
npm cache clean --force
npm install
```

### Build fails

```bash
# On EC2, debug the build:
cd /home/ubuntu/blitz-app
npm run build -- --debug

# Check if all files are present
ls -la pages/
ls -la styles/
cat package.json
```

---

## Quick Reference Commands

```bash
# Monitor in real-time
pm2 monit

# View all logs
pm2 logs

# Restart app
pm2 restart blitz-app

# Stop app
pm2 stop blitz-app

# Start app
pm2 start blitz-app

# Check DNS
nslookup blitz.atla.in

# Test site
curl https://blitz.atla.in

# SSH to server
ssh -i "key.pem" ubuntu@13.206.147.51
```

---

## Your EC2 Info

| Item | Value |
|------|-------|
| **IP Address** | 13.206.147.51 |
| **User** | ubuntu |
| **Domain** | blitz.atla.in |
| **App Directory** | /home/ubuntu/blitz-app |
| **App Port** | 3000 |
| **Process Manager** | PM2 |
| **Reverse Proxy** | Nginx |
| **SSL** | Let's Encrypt (auto-renewing) |

---

## Expected Timeline

- ⏱️ **Copy files**: 1-2 minutes
- ⏱️ **npm install**: 2-3 minutes
- ⏱️ **npm run build**: 2-3 minutes
- ⏱️ **PM2 restart**: 1 minute
- ⏱️ **Total**: ~6-9 minutes

---

## Success Indicators ✅

When you see this, deployment is complete:

```
✅ https://blitz.atla.in loads with green lock
✅ Landing page displays perfectly
✅ Navigation works
✅ All 4 pages load (landing, curriculum, dashboard, communities)
✅ No console errors (F12 to check)
✅ PM2 shows blitz-app online
✅ Nginx logs show 200 status codes
```

---

## Next Steps After Live

1. **Test thoroughly** - Go through all 4 pages, click all buttons
2. **Check performance** - Pages should load in < 1 second
3. **Monitor logs** - `pm2 logs blitz-app` for any errors
4. **Share the URL** - https://blitz.atla.in is live!
5. **Plan updates** - Any changes? Update the code and redeploy

---

## Still Need Help?

If anything fails:

1. Check the logs: `pm2 logs blitz-app`
2. Verify files exist: `ls /home/ubuntu/blitz-app/pages/`
3. Test connection: `curl http://localhost:3000`
4. Check DNS: `nslookup blitz.atla.in`
5. Verify Nginx: `sudo nginx -t`

All these commands can be run after SSH'ing into your EC2 instance.

---

**Status**: ✅ Ready to Deploy!
**Next**: Run `bash QUICK_DEPLOY.sh /path/to/your/key.pem`

Let's go live! 🚀
