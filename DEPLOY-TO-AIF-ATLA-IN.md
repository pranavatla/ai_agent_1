# 🚀 Deployment to aif.atla.in - Your Checklist

## 📌 Project Details

- **App Name:** AIF - 14 Week AI Journey
- **Domain:** aif.atla.in
- **Platform:** AWS EC2
- **Framework:** Next.js + React
- **Created:** May 8, 2026

---

## ✅ Pre-Deployment Checklist

### AWS Setup
- [ ] AWS Account created
- [ ] EC2 instance launched (t2.micro)
- [ ] Ubuntu 22.04 LTS selected
- [ ] Security group configured:
  - [ ] Port 22 (SSH) - from your IP
  - [ ] Port 80 (HTTP) - 0.0.0.0/0
  - [ ] Port 443 (HTTPS) - 0.0.0.0/0
- [ ] .pem key file downloaded
- [ ] atla.in domain in Route 53

### Local Preparation
- [ ] All 4 JSX component files ready:
  - [ ] pages-landing-page.jsx
  - [ ] pages-marketing-dashboard.jsx
  - [ ] pages-community-finder.jsx
  - [ ] pages-14-week-curriculum.jsx
- [ ] Code pushed to GitHub (if using git deployment)
- [ ] .env variables prepared (if any)

---

## 🔑 Step 1: Connect to EC2 (5 min)

```bash
# Set permissions on key
chmod 400 your-key.pem

# SSH into server
ssh -i your-key.pem ubuntu@YOUR-EC2-PUBLIC-IP
```

**Verification:**
```bash
# You should see: ubuntu@ip-xxx-xxx-xxx-xxx:~$
```

- [ ] Successfully connected to EC2 instance

---

## 🔧 Step 2: Run Deployment Script (10 min)

### Option A: Automated (Recommended)

```bash
# Download script
wget https://raw.githubusercontent.com/YOUR-USERNAME/YOUR-REPO/main/deploy-to-aws.sh

# Or copy from your project folder

# Make it executable
chmod +x deploy-to-aws.sh

# Run the script
./deploy-to-aws.sh
```

**What the script does:**
- Installs Node.js
- Installs PM2
- Installs Nginx
- Clones your GitHub repo
- Builds your Next.js app
- Configures Nginx as reverse proxy
- Sets up SSL with Let's Encrypt
- Starts your app

### Option B: Manual (If Script Fails)

Follow these commands in order:

```bash
# 1. Update system
sudo apt update && sudo apt upgrade -y

# 2. Install Node.js
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# 3. Install git
sudo apt install -y git

# 4. Install PM2
sudo npm install -g pm2

# 5. Install Nginx
sudo apt install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx

# 6. Clone your repo
cd /home/ubuntu
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO

# 7. Install dependencies
npm install

# 8. Build
npm run build

# 9. Start with PM2
pm2 start npm --name "aif-app" -- start
pm2 startup
pm2 save

# 10. Configure Nginx
sudo tee /etc/nginx/sites-available/aif.atla.in > /dev/null <<'EOF'
upstream aif_app {
    server 127.0.0.1:3000;
}

server {
    listen 80;
    server_name aif.atla.in;

    location / {
        proxy_pass http://aif_app;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
EOF

# 11. Enable site
sudo ln -sf /etc/nginx/sites-available/aif.atla.in /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# 12. Install SSL
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d aif.atla.in
```

**Verification After Script:**
- [ ] Script ran without major errors
- [ ] No 502/500 errors in output
- [ ] PM2 shows app running: `pm2 status`
- [ ] Nginx restarted: `sudo systemctl status nginx`

---

## 🌐 Step 3: Update DNS (5 min)

### Get Your EC2 Public IP

On your EC2 instance:
```bash
curl -s http://169.254.169.254/latest/meta-data/public-ipv4
```

Or in AWS Console:
- EC2 Dashboard → Instances → Your instance → Public IPv4 address

### Update Route 53

1. Go to AWS Console
2. Route 53 → Hosted zones → atla.in
3. Create new record:
   - **Name:** aif.atla.in
   - **Type:** A
   - **Value:** Your EC2 public IP
   - **TTL:** 300
4. Click "Create records"

- [ ] DNS record created in Route 53
- [ ] Set to your EC2 public IP
- [ ] TTL set to 300

---

## ⏳ Step 4: Wait for DNS Propagation (5-10 min)

DNS propagation usually takes 5-10 minutes, sometimes up to 48 hours.

```bash
# Check if DNS is ready
nslookup aif.atla.in

# Should show your EC2 IP
# If it doesn't show your IP yet, wait a few more minutes
```

- [ ] `nslookup aif.atla.in` returns your EC2 IP
- [ ] DNS has propagated

---

## ✅ Step 5: Verify Deployment (5 min)

### Check Application Status

```bash
# SSH back in if disconnected
ssh -i your-key.pem ubuntu@YOUR-EC2-PUBLIC-IP

# Check PM2 status
pm2 status

# Should show:
# ┌─────┬──────────┬─────────┬─────────┬─────────┬──────────┐
# │ id  │ name     │ version │ pm_id   │ status  │ uptime   │
# ├─────┼──────────┼─────────┼─────────┼─────────┼──────────┤
# │ 0   │ aif-app  │ 1.0.0   │ 0       │ online  │ X mins   │
# └─────┴──────────┴─────────┴─────────┴─────────┴──────────┘
```

- [ ] `pm2 status` shows aif-app as "online"
- [ ] No errors in output

### Check Nginx

```bash
sudo nginx -t
# Should output: nginx: syntax is ok, nginx: configuration test is successful

sudo systemctl status nginx
# Should show: active (running)
```

- [ ] Nginx test passes
- [ ] Nginx is running

### Test SSL Certificate

```bash
sudo certbot certificates
# Should show your aif.atla.in certificate
```

- [ ] SSL certificate installed
- [ ] Valid for aif.atla.in

---

## 🌍 Step 6: Access Your Site (2 min)

### Test in Browser

Open: **https://aif.atla.in**

**What you should see:**
- ✅ Green lock icon (HTTPS)
- ✅ Landing page loads
- ✅ Navigation works
- ✅ All images/styles load
- ✅ No "404" or "502" errors

### Test All Pages

- [ ] Landing page (/) loads
- [ ] Curriculum page loads
- [ ] Dashboard page loads
- [ ] Communities page loads
- [ ] Navigation between pages works
- [ ] HTTPS (green lock) appears

### Test Responsiveness

- [ ] Works on desktop (full screen)
- [ ] Works on tablet (resize browser)
- [ ] Works on mobile (use browser dev tools)

---

## 🔍 Step 7: Verify All Components (5 min)

### Landing Page
- [ ] Hero section displays
- [ ] Timeline is visible
- [ ] Feature cards show 6 agents
- [ ] Impact section visible
- [ ] CTA buttons clickable

### Curriculum Page
- [ ] 14 weeks listed
- [ ] Weeks are expandable
- [ ] Projects show details
- [ ] Progress tracker works
- [ ] Can mark weeks complete

### Marketing Dashboard
- [ ] 4 metric cards display
- [ ] Approval status banner shows
- [ ] Tabs are clickable
- [ ] Performance data shows
- [ ] Charts/tables render

### Community Finder
- [ ] Search bar works
- [ ] Filter buttons work
- [ ] Community cards display
- [ ] Email copy button works
- [ ] Sorting works

- [ ] All 4 pages fully functional
- [ ] No missing content
- [ ] All styles loading
- [ ] All icons displaying

---

## 📊 Step 8: Check Logs (2 min)

```bash
# View app logs
pm2 logs aif-app

# View last 50 lines
pm2 logs aif-app | tail -50

# Check for errors
pm2 logs aif-app | grep -i error

# View Nginx logs
sudo tail -20 /var/log/nginx/access.log
sudo tail -20 /var/log/nginx/error.log
```

- [ ] No error messages in PM2 logs
- [ ] No error messages in Nginx logs
- [ ] Access logs show successful requests (200 status)

---

## 🎉 Step 9: Verify Deployment Complete (Final)

### Full URL Test
```bash
curl https://aif.atla.in
# Should return HTML of your landing page

curl -I https://aif.atla.in
# Should show: HTTP/2 200
```

- [ ] Site is accessible via HTTPS
- [ ] Returns 200 status code
- [ ] SSL certificate valid

### Performance Check

```bash
# Response time (should be < 1 second)
time curl https://aif.atla.in > /dev/null
```

- [ ] Response time acceptable
- [ ] No timeouts

---

## 📋 Post-Deployment Checklist

### Monitoring Setup
- [ ] Set up CloudWatch alarms (optional)
- [ ] Monitor PM2 logs regularly
- [ ] Check SSL certificate expiry

### Documentation
- [ ] Saved EC2 IP address
- [ ] Saved SSH key securely
- [ ] Documented deployment steps
- [ ] Noted any customizations

### Backups
- [ ] Code backed up to GitHub
- [ ] .env file saved securely
- [ ] Database backup plan (if needed)

### Security
- [ ] SSH restricted to your IP only
- [ ] SSH password authentication disabled
- [ ] Firewall rules verified
- [ ] SSL auto-renewal enabled

---

## 🚨 Troubleshooting During Deployment

### Can't connect via SSH?
```bash
# Check if instance is running
# In AWS console: EC2 → Instances → check status

# Verify security group allows port 22
# AWS Console → Security Groups → check inbound rules

# Check key permissions
chmod 400 your-key.pem
```

### App won't start?
```bash
# Check logs
pm2 logs aif-app

# Check Node.js
node --version

# Check npm
npm --version

# Try manual start
cd /home/ubuntu/YOUR-REPO
npm run build
npm start
```

### Nginx errors?
```bash
# Test config
sudo nginx -t

# Check error log
sudo tail -f /var/log/nginx/error.log

# Restart nginx
sudo systemctl restart nginx
```

### DNS not working?
```bash
# Check Route 53 in AWS console
# Make sure A record points to your EC2 IP
# Wait for propagation (5-10 minutes usually)

nslookup aif.atla.in
# Should eventually show your IP
```

### 502 Bad Gateway?
```bash
# Check if Node app is running
pm2 status

# Check Nginx config
sudo nginx -t

# Restart both
pm2 restart aif-app
sudo systemctl restart nginx
```

---

## 📈 Maintenance Schedule

### Daily
- [ ] Check that https://aif.atla.in is accessible
- [ ] Quickly scan logs for errors: `pm2 logs aif-app | head -20`

### Weekly
- [ ] Review PM2 logs: `pm2 logs aif-app`
- [ ] Check disk space: `df -h`
- [ ] Check SSL certificate: `sudo certbot certificates`

### Monthly
- [ ] Update dependencies: `npm update`
- [ ] Review security groups in AWS
- [ ] Check SSL auto-renewal worked

### Quarterly
- [ ] Update OS packages: `sudo apt update && sudo apt upgrade -y`
- [ ] Review AWS costs
- [ ] Plan feature updates

---

## 🎯 Success Indicators

✅ **Deployment is successful when:**

1. **DNS Works**
   - `nslookup aif.atla.in` returns your EC2 IP

2. **HTTPS Loads**
   - Browser shows green lock
   - https://aif.atla.in loads without errors

3. **All Pages Work**
   - Landing, Curriculum, Dashboard, Communities all load
   - Navigation between pages works

4. **No Errors**
   - Console (F12) shows no errors
   - PM2 logs show no error messages
   - Nginx logs show only 200/300 status codes

5. **Performance is Good**
   - Page loads in < 2 seconds
   - No timeout errors
   - Responsive on all devices

---

## 📞 After Deployment

### Share Your Site
- [ ] Share https://aif.atla.in on social media
- [ ] Send to your email list
- [ ] Tell your network
- [ ] Add to portfolio

### Gather Feedback
- [ ] Ask users for feedback
- [ ] Monitor analytics
- [ ] Collect error reports
- [ ] Plan improvements

### Keep it Running
- [ ] Monitor PM2 logs daily
- [ ] Update dependencies weekly
- [ ] Review errors immediately
- [ ] Plan feature updates

---

## 🎉 You Did It!

When you see this in your browser:

```
✅ https://aif.atla.in 
✅ Green lock (HTTPS)
✅ Landing page loads perfectly
✅ All 4 pages work
✅ Navigation is smooth
✅ No errors in console
```

**Your deployment is COMPLETE!** 🚀

---

## 📊 Final Status

| Item | Status |
|------|--------|
| **Domain** | aif.atla.in |
| **HTTPS** | ✅ Active |
| **App Status** | ✅ Running |
| **SSL** | ✅ Valid |
| **DNS** | ✅ Configured |
| **Pages** | ✅ All Working |
| **Deployment** | ✅ Complete |

---

**Deployed:** [Date]
**Status:** LIVE ✅
**Domain:** https://aif.atla.in

Congratulations! Your 14-week AI journey is now live to the world! 🎉🚀
