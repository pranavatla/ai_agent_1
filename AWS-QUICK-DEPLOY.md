# ⚡ AWS Deployment - Quick Reference

**Domain:** `aif.atla.in`
**Status:** Ready to deploy ✅

---

## 🚀 Deploy in 3 Steps

### Step 1: SSH into Your EC2 Instance
```bash
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

### Step 2: Run Automated Deployment Script
```bash
# Copy the script to your server, then run:
bash deploy-to-aws.sh

# The script will:
# ✓ Update system
# ✓ Install Node.js
# ✓ Install Nginx
# ✓ Install PM2
# ✓ Clone your repo
# ✓ Build your Next.js app
# ✓ Configure Nginx
# ✓ Set up SSL (Let's Encrypt)
# ✓ Start your app
```

### Step 3: Update DNS in Route 53
```
Domain: aif.atla.in
Type: A Record
Value: Your EC2 Public IP
TTL: 300
```

**Wait 5-10 minutes for DNS to propagate**

---

## ✅ Verify Deployment

```bash
# Check DNS
nslookup aif.atla.in

# Check app status
pm2 status

# View logs
pm2 logs aif-app

# Test the site
curl https://aif.atla.in
```

---

## 🔧 Common Commands

```bash
# Connect to server
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

# Check Nginx
sudo nginx -t
sudo systemctl status nginx

# SSH logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

---

## 🆘 Troubleshooting

### Site not loading?
```bash
# Check if app running
pm2 status

# Check logs
pm2 logs aif-app

# Restart
pm2 restart aif-app
```

### DNS not working?
```bash
# Verify DNS record in AWS Route 53
# Check in AWS console: Route 53 → Hosted zones → atla.in
# Make sure aif.atla.in points to your EC2 IP

nslookup aif.atla.in  # Should show your EC2 IP
```

### SSL certificate error?
```bash
# Check certificate
sudo certbot certificates

# Renew
sudo certbot renew

# Or force renew
sudo certbot --nginx -d aif.atla.in --force-renewal
```

### 502 Bad Gateway?
```bash
# Check if Node app running
pm2 status

# Restart Node app
pm2 restart aif-app

# Check Nginx error logs
sudo tail -f /var/log/nginx/error.log
```

---

## 📊 Deployment Checklist

- [ ] EC2 instance launched (t2.micro, free tier)
- [ ] Security group opened (ports 22, 80, 443)
- [ ] SSH key downloaded and configured
- [ ] SSH connection working
- [ ] Deployment script run successfully
- [ ] App built without errors
- [ ] PM2 showing app running (`pm2 status`)
- [ ] Nginx configured and running
- [ ] SSL certificate installed (green lock)
- [ ] DNS updated in Route 53
- [ ] aif.atla.in resolves to your IP
- [ ] Can access https://aif.atla.in
- [ ] All pages load correctly
- [ ] No console errors

---

## 📈 Monitor Your App

```bash
# Real-time monitoring
pm2 monit

# CPU & memory usage
top

# Disk space
df -h

# View all logs
pm2 logs

# Filter logs for errors
pm2 logs | grep -i error
```

---

## 🔄 Update Your App

```bash
cd /home/ubuntu/YOUR-REPO
git pull              # Get latest code
npm install           # Update dependencies
npm run build         # Build app
pm2 restart aif-app   # Restart
```

---

## 📁 Important Directories

```
/home/ubuntu/YOUR-REPO/          # Your app
/etc/nginx/sites-available/      # Nginx configs
/var/log/nginx/                  # Nginx logs
~/.pm2/logs/                      # PM2 logs
```

---

## 🎯 Your Deployment Info

| Item | Value |
|------|-------|
| **Domain** | aif.atla.in |
| **EC2 IP** | Your public IP here |
| **Instance** | t2.micro (free tier) |
| **OS** | Ubuntu 22.04 LTS |
| **App Port** | 3000 |
| **Status** | Deploying... |

---

## 📞 Need Help?

**Can't SSH?**
- Check security group allows port 22
- Verify key permissions: `chmod 400 your-key.pem`
- Check IP is correct in EC2 console

**App won't start?**
- Check logs: `pm2 logs aif-app`
- Try: `pm2 restart aif-app`
- Check Node installed: `node --version`

**Site won't load?**
- Check DNS: `nslookup aif.atla.in`
- Check Nginx: `sudo nginx -t`
- Check app running: `pm2 status`
- Check SSL: `sudo certbot certificates`

**Getting 502 error?**
- Check Node app: `pm2 status`
- Restart: `pm2 restart aif-app`
- Check Nginx config: `sudo nginx -t`
- View error log: `sudo tail -f /var/log/nginx/error.log`

---

## 🎉 Success Criteria

When this is done:
✅ https://aif.atla.in loads in browser
✅ Green lock icon (HTTPS working)
✅ Landing page displays
✅ Navigation works
✅ All 4 pages accessible:
   - Landing page (/)
   - Curriculum (/curriculum)
   - Dashboard (/dashboard/marketing)
   - Communities (/dashboard/communities)

---

## 📝 Notes

- SSL certificate auto-renews (no manual action needed)
- PM2 auto-restarts on reboot
- App logs available: `pm2 logs aif-app`
- Keep your .pem key safe!
- Backup your app code to GitHub

---

**Status:** ✅ Ready to Deploy

**Time to Live:** ~30 minutes

**Domain:** aif.atla.in

**Let's do this!** 🚀
