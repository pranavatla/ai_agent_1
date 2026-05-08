# 🚀 AWS Deployment Guide - Deploy to aif.atla.in

## Overview
You're deploying a Next.js app to AWS and connecting it to your custom domain `aif.atla.in`.

**Domain:** aif.atla.in
**Platform:** AWS
**Framework:** Next.js + React

---

## 📋 Prerequisites

Before you start, make sure you have:
- [ ] AWS Account with access to EC2, Route 53, and RDS (optional)
- [ ] atla.in domain registered (in Route 53 or external registrar)
- [ ] SSH key pair created in AWS
- [ ] Your Next.js app ready (you have this!)

---

## 🎯 Step 1: Set Up AWS EC2 Instance

### 1.1 Launch an EC2 Instance

1. Go to AWS Console → EC2 → Instances
2. Click "Launch Instances"
3. Choose AMI:
   - Select **Ubuntu Server 22.04 LTS** (free tier eligible)
4. Instance Type:
   - Select **t2.micro** (free tier)
5. Security Group:
   - Open ports: 22 (SSH), 80 (HTTP), 443 (HTTPS)
   - Source: 0.0.0.0/0 (or your IP)

```
Inbound Rules:
- Port 22: SSH (your IP only for security)
- Port 80: HTTP (0.0.0.0/0)
- Port 443: HTTPS (0.0.0.0/0)
```

6. Storage: 30 GB (free tier)
7. Launch and download your `.pem` key file

### 1.2 Connect to Your Instance

```bash
# Change permissions on your key
chmod 400 your-key.pem

# SSH into your instance
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

---

## 🔧 Step 2: Set Up Your Server

### 2.1 Update System

```bash
sudo apt update
sudo apt upgrade -y
```

### 2.2 Install Node.js & npm

```bash
# Install Node.js (LTS)
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs

# Verify installation
node --version
npm --version
```

### 2.3 Install Git

```bash
sudo apt install -y git
```

### 2.4 Install PM2 (Process Manager)

```bash
sudo npm install -g pm2
```

### 2.5 Install Nginx (Reverse Proxy)

```bash
sudo apt install -y nginx
sudo systemctl start nginx
sudo systemctl enable nginx
```

---

## 📦 Step 3: Deploy Your Next.js App

### 3.1 Clone Your Repository

```bash
cd /home/ubuntu
git clone https://github.com/YOUR-USERNAME/YOUR-REPO.git
cd YOUR-REPO
```

Or if not in git yet, upload your files:
```bash
# On your local machine
scp -i your-key.pem -r /path/to/your/app ubuntu@your-ec2-ip:/home/ubuntu/my-app
```

### 3.2 Install Dependencies

```bash
npm install
```

### 3.3 Build the Next.js App

```bash
npm run build
```

### 3.4 Start with PM2

```bash
# Start the app
pm2 start npm --name "aif-app" -- start

# Make it restart on reboot
pm2 startup
pm2 save
```

---

## 🌐 Step 4: Configure Nginx as Reverse Proxy

### 4.1 Create Nginx Config

```bash
sudo nano /etc/nginx/sites-available/aif.atla.in
```

### 4.2 Add This Configuration

```nginx
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
```

### 4.3 Enable the Site

```bash
sudo ln -s /etc/nginx/sites-available/aif.atla.in /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl restart nginx
```

---

## 🔒 Step 5: Set Up SSL Certificate (HTTPS)

### 5.1 Install Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### 5.2 Get SSL Certificate

```bash
sudo certbot --nginx -d aif.atla.in
```

This will:
- Verify domain ownership
- Install SSL certificate
- Auto-update Nginx config
- Auto-renewal enabled

---

## 🔗 Step 6: Configure DNS (Route 53)

### 6.1 Get Your EC2 Public IP

```bash
# From your EC2 instance or AWS console
echo $your-ec2-public-ip
```

### 6.2 Update Route 53

1. Go to AWS Console → Route 53
2. Select your hosted zone for **atla.in**
3. Create a new record:
   - **Name:** aif.atla.in
   - **Type:** A
   - **Value:** Your EC2 public IP
   - **TTL:** 300
4. Click "Create records"

Wait 5-10 minutes for DNS to propagate.

---

## ✅ Step 7: Verify Deployment

### 7.1 Test DNS Resolution

```bash
nslookup aif.atla.in
# Should show your EC2 IP
```

### 7.2 Visit Your Site

Open browser and go to: **https://aif.atla.in**

You should see your landing page! 🎉

### 7.3 Check Logs

```bash
# View PM2 logs
pm2 logs aif-app

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## 🔄 Step 8: Environment Variables

### 8.1 Create .env.production

```bash
cd /home/ubuntu/YOUR-REPO
nano .env.production
```

### 8.2 Add Your Variables

```env
NEXT_PUBLIC_API_URL=https://aif.atla.in
NODE_ENV=production
# Add any other env vars your app needs
```

### 8.3 Restart App

```bash
pm2 restart aif-app
```

---

## 📊 Step 9: Set Up Monitoring

### 9.1 Monitor with PM2

```bash
# Dashboard
pm2 monit

# Status
pm2 status

# Logs
pm2 logs
```

### 9.2 Set Up CloudWatch (Optional)

In AWS Console:
1. EC2 → Instances
2. Select your instance
3. CloudWatch → Monitoring tab
4. Enable detailed monitoring

---

## 🚨 Troubleshooting

### Site Not Loading?

```bash
# Check if app is running
pm2 status

# Restart if needed
pm2 restart aif-app

# Check Nginx
sudo systemctl status nginx
sudo nginx -t
```

### DNS Not Working?

```bash
# Flush local DNS (macOS)
sudo dscacheutil -flushcache

# Check Route 53 records in AWS console
# Wait for DNS propagation (up to 48 hours, usually 5-10 min)
```

### SSL Certificate Issues?

```bash
# Check certificate
sudo certbot certificates

# Renew manually
sudo certbot renew --dry-run

# Fix Nginx config
sudo certbot --nginx -d aif.atla.in --force-renewal
```

### Port Already in Use?

```bash
# Find process using port 3000
lsof -i :3000

# Kill process
kill -9 <PID>

# Restart app
pm2 restart aif-app
```

---

## 🔐 Security Best Practices

### 8.1 Update Firewall Rules

```bash
# Only allow SSH from your IP
# In AWS Security Group, restrict port 22 to your IP
```

### 8.2 Disable Root SSH

```bash
sudo nano /etc/ssh/sshd_config
# Set: PermitRootLogin no
# Set: PasswordAuthentication no
sudo systemctl restart sshd
```

### 8.3 Set Up Fail2Ban (Optional)

```bash
sudo apt install -y fail2ban
sudo systemctl enable fail2ban
```

---

## 📈 Deployment Checklist

- [ ] EC2 instance running
- [ ] Node.js & npm installed
- [ ] App cloned/uploaded
- [ ] Dependencies installed
- [ ] App built successfully
- [ ] PM2 running the app
- [ ] Nginx configured
- [ ] SSL certificate installed
- [ ] DNS records updated in Route 53
- [ ] https://aif.atla.in loads
- [ ] All pages accessible
- [ ] HTTPS working (green lock)
- [ ] Monitoring set up
- [ ] Backups configured

---

## 🚀 Post-Deployment

### Keep App Updated

```bash
cd /home/ubuntu/YOUR-REPO
git pull
npm install
npm run build
pm2 restart aif-app
```

### Monitor Performance

```bash
# Check server stats
top
free -h
df -h

# Check app logs
pm2 logs aif-app
```

### Regular Maintenance

- [ ] Weekly: Check logs for errors
- [ ] Monthly: Update dependencies (`npm update`)
- [ ] Monthly: Check certificate expiry
- [ ] Quarterly: Review security groups
- [ ] Quarterly: Update OS packages

---

## 📊 Your Deployment Summary

| Item | Details |
|------|---------|
| **Domain** | aif.atla.in |
| **Platform** | AWS EC2 |
| **Instance** | t2.micro (free tier) |
| **OS** | Ubuntu 22.04 LTS |
| **Framework** | Next.js |
| **Reverse Proxy** | Nginx |
| **Process Manager** | PM2 |
| **SSL** | Let's Encrypt (auto-renewing) |
| **DNS** | AWS Route 53 |

---

## 💡 Quick Commands Reference

```bash
# SSH into server
ssh -i your-key.pem ubuntu@your-ec2-ip

# Check app status
pm2 status

# View logs
pm2 logs aif-app

# Restart app
pm2 restart aif-app

# Stop app
pm2 stop aif-app

# Start app
pm2 start aif-app

# Restart Nginx
sudo systemctl restart nginx

# Check Nginx status
sudo systemctl status nginx

# View Nginx logs
sudo tail -f /var/log/nginx/access.log
```

---

## 🎉 Success!

When you see this in your browser:
```
✅ https://aif.atla.in loads with HTTPS (green lock)
✅ Landing page displays perfectly
✅ All components work
✅ Navigation works
```

**Your deployment is complete!** 🚀

---

## 📞 Need Help?

**App not starting?**
→ Check logs: `pm2 logs aif-app`

**Can't reach the site?**
→ Check DNS: `nslookup aif.atla.in`
→ Check Nginx: `sudo nginx -t`

**SSL certificate issues?**
→ Check cert: `sudo certbot certificates`
→ Renew: `sudo certbot renew`

**Nginx 502 Bad Gateway?**
→ Check if app running: `pm2 status`
→ Restart: `pm2 restart aif-app`

---

**Deployment Date:** May 8, 2026
**Status:** Ready to Deploy
**Estimated Time:** 30-45 minutes
**Difficulty:** Intermediate

Good luck! Your app will be live soon! 🚀
