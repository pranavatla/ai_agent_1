# 🚀 Deployment to blitz.atla.in - Your 14-Week AI Journey

## 📌 Project Details

- **App Name:** 14-Week AI Blitz Journey
- **Domain:** blitz.atla.in
- **Platform:** AWS EC2
- **Framework:** Next.js + React
- **Status:** Ready to deploy ✅

---

## ✅ Pre-Deployment Checklist

### AWS Setup
- [ ] AWS Account ready
- [ ] EC2 instance available (t2.micro)
- [ ] atla.in domain in Route 53
- [ ] Security group configured (ports 22, 80, 443)
- [ ] .pem key file ready

### Local Preparation
- [ ] All 4 JSX components ready:
  - [ ] pages-landing-page.jsx
  - [ ] pages-marketing-dashboard.jsx
  - [ ] pages-community-finder.jsx
  - [ ] pages-14-week-curriculum.jsx
- [ ] Code ready to deploy

---

## 🔑 Step 1: Connect to EC2 (2 min)

```bash
# Set permissions on key
chmod 400 your-key.pem

# SSH into server
ssh -i your-key.pem ubuntu@YOUR-EC2-PUBLIC-IP
```

- [ ] Successfully connected to EC2

---

## 🚀 Step 2: Run Deployment Script (15 min)

### Quick Deployment

```bash
# Download the script
wget https://raw.githubusercontent.com/YOUR-USERNAME/YOUR-REPO/main/deploy-to-aws.sh

# Make it executable
chmod +x deploy-to-aws.sh

# Run it
./deploy-to-aws.sh
```

When prompted:
- **App name:** blitz-app
- **Domain:** blitz.atla.in
- **Port:** 3000

### What the script does:
✅ Installs Node.js & npm
✅ Installs PM2 (process manager)
✅ Installs Nginx (reverse proxy)
✅ Clones your GitHub repo
✅ Builds your Next.js app
✅ Configures Nginx for blitz.atla.in
✅ Sets up SSL with Let's Encrypt
✅ Starts your app with PM2

- [ ] Deployment script completed successfully
- [ ] No major errors in output

---

## 🌐 Step 3: Verify App Running (2 min)

```bash
# Check PM2 status
pm2 status

# Should show blitz-app as "online"
```

Expected output:
```
┌─────┬──────────────┬─────────┬─────────┬─────────┬──────────┐
│ id  │ name         │ version │ pm_id   │ status  │ uptime   │
├─────┼──────────────┼─────────┼─────────┼─────────┼──────────┤
│ 0   │ blitz-app    │ 1.0.0   │ 0       │ online  │ X mins   │
└─────┴──────────────┴─────────┴─────────┴─────────┴──────────┘
```

- [ ] blitz-app shows as "online"
- [ ] No error messages

---

## 🔗 Step 4: Update DNS in Route 53 (5 min)

### Get Your EC2 Public IP

```bash
# On your EC2 instance
curl -s http://169.254.169.254/latest/meta-data/public-ipv4
```

### Create DNS Record

1. Go to AWS Console → Route 53
2. Select hosted zone: **atla.in**
3. Create new record:
   - **Name:** blitz.atla.in
   - **Type:** A
   - **Value:** Your EC2 Public IP
   - **TTL:** 300
4. Click "Create records"

**Wait 5-10 minutes for DNS to propagate**

- [ ] A record created for blitz.atla.in
- [ ] Points to your EC2 IP

---

## ⏳ Step 5: Verify DNS (3 min)

```bash
# Check DNS resolution
nslookup blitz.atla.in

# Should show your EC2 IP
```

- [ ] nslookup returns your EC2 IP
- [ ] DNS has propagated

---

## ✅ Step 6: Access Your Site (3 min)

### Visit in Browser

Open: **https://blitz.atla.in**

You should see:
- ✅ Green lock icon (HTTPS)
- ✅ Landing page loads
- ✅ Navigation works
- ✅ No "404" or "502" errors

### Test All Pages

Click through:
- [ ] Landing page (/) - loads instantly
- [ ] Curriculum page - 14 weeks visible
- [ ] Dashboard page - metrics display
- [ ] Communities page - search works

### Check Styling

- [ ] Dark theme displays correctly
- [ ] Blue/cyan accents visible
- [ ] Images load properly
- [ ] Icons from lucide-react show
- [ ] Responsive design works

- [ ] All 4 pages load successfully
- [ ] HTTPS with green lock
- [ ] No console errors

---

## 🔍 Step 7: Component Verification (5 min)

### Landing Page
- [ ] Hero section with CTA buttons
- [ ] Timeline showing 4 phases
- [ ] 6 feature cards for agents
- [ ] Impact section visible
- [ ] Professional footer

### Curriculum Page
- [ ] 14 weeks listed
- [ ] Can expand each week
- [ ] Projects show details
- [ ] Progress tracker visible
- [ ] Can mark weeks complete

### Dashboard Page
- [ ] 4 metric cards displayed
- [ ] Weekly trends chart
- [ ] Community tracking table
- [ ] 3 interactive tabs
- [ ] KPI analytics section

### Communities Page
- [ ] Search functionality works
- [ ] Filtering by category works
- [ ] Sorting by engagement works
- [ ] Community cards display data
- [ ] Email copy button works

- [ ] All components fully functional
- [ ] No missing content
- [ ] All interactive elements work

---

## 📊 Step 8: Check Logs (2 min)

```bash
# View app logs
pm2 logs blitz-app

# Check for errors
pm2 logs blitz-app | grep -i error

# View Nginx
sudo tail -20 /var/log/nginx/access.log
```

- [ ] No error messages in PM2 logs
- [ ] No error messages in Nginx logs
- [ ] Only 200/300 status codes in access log

---

## 🎯 Step 9: Performance Check (2 min)

```bash
# Check response time
time curl https://blitz.atla.in > /dev/null

# Should complete in < 1 second
```

- [ ] Response time under 1 second
- [ ] No timeouts
- [ ] Fast page load

---

## 🔐 Step 10: Security Verification (2 min)

```bash
# Check SSL certificate
sudo certbot certificates

# Should show valid cert for blitz.atla.in
```

- [ ] SSL certificate valid
- [ ] Auto-renewal enabled
- [ ] HTTPS working (green lock)

---

## ✨ Final Verification Checklist

- [ ] Domain: blitz.atla.in
- [ ] HTTPS: ✅ Green lock visible
- [ ] Landing page: ✅ Loads correctly
- [ ] Curriculum: ✅ All 14 weeks visible
- [ ] Dashboard: ✅ Metrics displayed
- [ ] Communities: ✅ Search works
- [ ] Navigation: ✅ Links work
- [ ] Performance: ✅ < 1 second
- [ ] Security: ✅ SSL valid
- [ ] No errors: ✅ Console clean

---

## 📋 Post-Deployment

### Monitoring
```bash
# Watch app status
pm2 monit

# View logs in real-time
pm2 logs blitz-app

# Check updates needed
npm outdated
```

### Updates
```bash
# Keep dependencies updated
npm update

# Check for security issues
npm audit

# Keep OS updated
sudo apt update && sudo apt upgrade -y
```

### Maintenance Schedule

**Daily:**
- [ ] Quick check that https://blitz.atla.in loads

**Weekly:**
- [ ] Review PM2 logs for errors
- [ ] Check disk space: `df -h`

**Monthly:**
- [ ] Update dependencies: `npm update`
- [ ] Check SSL: `sudo certbot certificates`

**Quarterly:**
- [ ] Update OS: `sudo apt upgrade -y`
- [ ] Review AWS security groups
- [ ] Check performance metrics

---

## 🚨 Troubleshooting

### Can't connect via SSH?
```bash
# Check security group allows port 22
# AWS Console → Security Groups → Verify inbound rules

# Check key permissions
chmod 400 your-key.pem

# Check instance is running
# AWS Console → EC2 → Instances
```

### Site won't load?
```bash
# Check DNS
nslookup blitz.atla.in

# Check PM2
pm2 status

# Check Nginx
sudo nginx -t
sudo systemctl status nginx

# Check logs
pm2 logs blitz-app
sudo tail -f /var/log/nginx/error.log
```

### 502 Bad Gateway?
```bash
# Restart app
pm2 restart blitz-app

# Restart Nginx
sudo systemctl restart nginx

# Check app is running
pm2 status
```

### SSL certificate error?
```bash
# Check cert
sudo certbot certificates

# Renew
sudo certbot renew

# Force renew
sudo certbot --nginx -d blitz.atla.in --force-renewal
```

---

## 📞 Quick Reference

### Essential Commands

```bash
# SSH to server
ssh -i your-key.pem ubuntu@YOUR-EC2-IP

# Check app
pm2 status

# View logs
pm2 logs blitz-app

# Restart
pm2 restart blitz-app

# Check DNS
nslookup blitz.atla.in

# Check Nginx
sudo nginx -t
sudo systemctl restart nginx

# Check cert
sudo certbot certificates
```

---

## 🎉 Success Indicators

✅ **Deployment is successful when:**

1. **DNS Works**
   - `nslookup blitz.atla.in` returns your EC2 IP

2. **HTTPS Loads**
   - Browser shows green lock
   - https://blitz.atla.in loads

3. **All Pages Work**
   - Landing, Curriculum, Dashboard, Communities all load
   - Navigation between pages works

4. **No Errors**
   - Console (F12) shows no errors
   - PM2 logs show no errors
   - Nginx logs show only 200/300

5. **Performance Good**
   - Pages load in < 2 seconds
   - No timeouts
   - Responsive on all devices

---

## 📈 After Deployment

### Share Your Site
- [ ] Post on social media
- [ ] Send to email list
- [ ] Tell your network
- [ ] Add to portfolio

### Gather Feedback
- [ ] Ask users for feedback
- [ ] Monitor analytics
- [ ] Collect feature requests
- [ ] Plan improvements

### Keep It Running
- [ ] Monitor logs daily
- [ ] Update dependencies weekly
- [ ] Plan feature updates

---

## 🎯 Your Deployment Summary

| Item | Details |
|------|---------|
| **Domain** | blitz.atla.in |
| **Platform** | AWS EC2 |
| **Instance** | t2.micro (free tier) |
| **Framework** | Next.js + React |
| **Reverse Proxy** | Nginx |
| **Process Manager** | PM2 |
| **SSL** | Let's Encrypt (auto-renewing) |
| **Time to Deploy** | ~30 minutes |

---

## 🚀 You're Ready!

Everything you need is prepared. Follow this checklist and your 14-week AI blitz journey will be live at **blitz.atla.in** in about 30 minutes.

**Status:** ✅ READY TO DEPLOY

**Next Step:** SSH into your EC2 and run the deployment script!

Let's do this! 🚀
