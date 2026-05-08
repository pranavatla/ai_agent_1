# 🚀 Webpage Integration Guide for Your 14-Week AI Journey

## Overview

You now have **3 production-ready React components** ready to integrate into your Next.js application:

1. **Landing Page** (`pages-landing-page.jsx`) - Showcase your 14-week AI mastery journey
2. **Marketing Dashboard** (`pages-marketing-dashboard.jsx`) - Real-time tracking of your AIF agents
3. **Community Finder** (`pages-community-finder.jsx`) - Interactive tool to discover and track communities

This guide walks you through integrating them into your Next.js app with a full deployment strategy.

---

## 📁 Project Structure

Your Next.js app should have this structure:

```
my-app/
├── pages/
│   ├── index.js                    # Home/Landing page
│   ├── dashboard/
│   │   ├── marketing.js            # Marketing Dashboard
│   │   └── communities.js          # Community Finder
│   ├── api/                        # Backend API routes
│   └── _app.js                     # App wrapper
├── components/
│   ├── Navbar.jsx                  # Shared navigation
│   ├── Footer.jsx                  # Shared footer
│   └── ...other components
├── styles/
│   └── globals.css                 # Global Tailwind CSS
├── public/                         # Static assets
├── next.config.js
├── tailwind.config.js
└── package.json
```

---

## 🔧 Step 1: Set Up Your Next.js Project

### If you don't have a Next.js app yet:

```bash
npx create-next-app@latest my-app --typescript --tailwind --eslint
cd my-app
npm install lucide-react
```

### If you already have a Next.js app, ensure you have Tailwind CSS:

```bash
npm install -D tailwindcss postcss autoprefixer lucide-react
npx tailwindcss init -p
```

Update your `tailwind.config.js`:

```javascript
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx}',
    './components/**/*.{js,ts,jsx,tsx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

---

## 📄 Step 2: Create Pages

### Option A: Using the Components as Full Pages

Copy the JSX components into your `pages/` directory:

```bash
cp pages-landing-page.jsx pages/index.js
cp pages-marketing-dashboard.jsx pages/dashboard/marketing.js
cp pages-community-finder.jsx pages/dashboard/communities.js
```

Then import in each file:

```javascript
// pages/index.js
import AIJourneyLanding from './pages-landing-page'
export default AIJourneyLanding

// pages/dashboard/marketing.js
import MarketingDashboard from './pages-marketing-dashboard'
export default MarketingDashboard

// pages/dashboard/communities.js
import CommunityFinder from './pages-community-finder'
export default CommunityFinder
```

### Option B: Creating Separate Components (Recommended)

Create component files and import them into pages:

```bash
mkdir -p components/pages
cp pages-landing-page.jsx components/pages/AIJourneyLanding.jsx
cp pages-marketing-dashboard.jsx components/pages/MarketingDashboard.jsx
cp pages-community-finder.jsx components/pages/CommunityFinder.jsx
```

Then in your pages:

```javascript
// pages/index.js
import AIJourneyLanding from '../components/pages/AIJourneyLanding'

export default function Home() {
  return <AIJourneyLanding />
}

// pages/dashboard/marketing.js
import MarketingDashboard from '../../components/pages/MarketingDashboard'

export default function Marketing() {
  return <MarketingDashboard />
}

// pages/dashboard/communities.js
import CommunityFinder from '../../components/pages/CommunityFinder'

export default function Communities() {
  return <CommunityFinder />
}
```

---

## 🎯 Step 3: Add Navigation

Create a `components/Navbar.jsx` to link between pages:

```javascript
import Link from 'next/link'
import { Brain } from 'lucide-react'

export default function Navbar() {
  return (
    <nav className="fixed top-0 w-full bg-slate-900/80 backdrop-blur-md border-b border-slate-700 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4 flex justify-between items-center">
        <Link href="/" className="flex items-center gap-2">
          <Brain className="w-8 h-8 text-blue-400" />
          <span className="text-xl font-bold">AIF Blitz</span>
        </Link>
        <div className="hidden md:flex gap-8">
          <Link href="/" className="hover:text-blue-400 transition">Home</Link>
          <Link href="/dashboard/marketing" className="hover:text-blue-400 transition">Dashboard</Link>
          <Link href="/dashboard/communities" className="hover:text-blue-400 transition">Communities</Link>
        </div>
        <button className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-lg transition">
          Launch App
        </button>
      </div>
    </nav>
  )
}
```

Add it to your `pages/_app.js`:

```javascript
import Navbar from '../components/Navbar'
import '../styles/globals.css'

function MyApp({ Component, pageProps }) {
  return (
    <>
      <Navbar />
      <Component {...pageProps} />
    </>
  )
}

export default MyApp
```

---

## 🔗 Step 4: Connect to Your Backend APIs

### Create API routes to serve dynamic data:

**pages/api/marketing/metrics.js**
```javascript
export default function handler(req, res) {
  // Fetch from your actual data store
  const metrics = {
    today: {
      researchCount: 5,
      emailsComposed: 5,
      emailsSent: 3,
      totalReach: 2500000,
      engagementRate: 4.2,
      approvalStatus: 'APPROVED',
    },
    // ... more data
  }
  
  res.status(200).json(metrics)
}
```

**pages/api/communities/list.js**
```javascript
export default function handler(req, res) {
  // Fetch communities from your database
  const communities = [
    {
      id: 1,
      name: 'r/aws',
      platform: 'Reddit',
      // ... more fields
    },
    // ... more communities
  ]
  
  res.status(200).json(communities)
}
```

### Update your components to fetch data:

```javascript
// In your Marketing Dashboard component
import { useEffect, useState } from 'react'

export default function MarketingDashboard() {
  const [metrics, setMetrics] = useState(null)
  
  useEffect(() => {
    fetch('/api/marketing/metrics')
      .then(res => res.json())
      .then(data => setMetrics(data))
  }, [])
  
  if (!metrics) return <div>Loading...</div>
  
  return (
    // ... render with metrics
  )
}
```

---

## 📦 Step 5: Deploy Your App

### Deploy to Vercel (Recommended for Next.js)

```bash
npm i -g vercel
vercel
```

Or connect your GitHub repo to Vercel and it auto-deploys on push.

### Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_API_URL=https://your-backend.com
DATABASE_URL=your_database_url
AGENT_API_KEY=your_api_key
```

---

## 🎨 Step 6: Customization

### Update Colors

Edit `tailwind.config.js` to customize the color scheme:

```javascript
theme: {
  extend: {
    colors: {
      // Your custom colors
      brand: {
        primary: '#0066CC',
        secondary: '#00CCCC',
      }
    }
  }
}
```

### Add Your Branding

- Replace logos in the Navbar
- Update footer with your info
- Customize button text and CTAs
- Update color gradients in components

### Link to Your Actual Services

Update the CTA buttons to point to:
- Your app signup URL
- Your documentation
- Your GitHub repository
- Your contact form

---

## 🚀 Step 7: Advanced Features

### Add Authentication

```bash
npm install next-auth
```

Protect dashboard routes:

```javascript
// pages/dashboard/marketing.js
import { getSession } from 'next-auth/react'

export async function getServerSideProps(context) {
  const session = await getSession(context)
  
  if (!session) {
    return {
      redirect: {
        destination: '/login',
        permanent: false,
      },
    }
  }
  
  return {
    props: { session },
  }
}
```

### Add Database Integration

Connect to your marketing agent data:

```javascript
// pages/api/communities/list.js
import { connectDB } from '../../lib/mongodb'
import Community from '../../models/Community'

export default async function handler(req, res) {
  await connectDB()
  const communities = await Community.find({})
  res.status(200).json(communities)
}
```

### Real-Time Updates with WebSockets

```javascript
// For live dashboard updates
import { useEffect, useState } from 'react'

export default function MarketingDashboard() {
  const [metrics, setMetrics] = useState(null)
  
  useEffect(() => {
    const ws = new WebSocket('wss://your-server.com/metrics')
    
    ws.onmessage = (event) => {
      setMetrics(JSON.parse(event.data))
    }
    
    return () => ws.close()
  }, [])
  
  // ... render metrics
}
```

---

## 📊 Step 8: Monitor & Analytics

Add analytics to track user behavior:

```bash
npm install gtag
```

Add to `pages/_app.js`:

```javascript
import { useEffect } from 'react'
import { useRouter } from 'next/router'
import * as gtag from '../lib/gtag'

function MyApp({ Component, pageProps }) {
  const router = useRouter()
  
  useEffect(() => {
    const handleRouteChange = (url) => {
      gtag.pageview(url)
    }
    
    router.events.on('routeChangeComplete', handleRouteChange)
    return () => router.events.off('routeChangeComplete', handleRouteChange)
  }, [router.events])
  
  return <Component {...pageProps} />
}

export default MyApp
```

---

## ✅ Quick Checklist

- [ ] Next.js app created with Tailwind CSS
- [ ] Components copied to `components/pages/`
- [ ] Pages created in `pages/` directory
- [ ] Navbar component created and integrated
- [ ] Navigation links working between pages
- [ ] API routes created for dynamic data
- [ ] Components updated to fetch from APIs
- [ ] Environment variables configured
- [ ] Deployed to hosting (Vercel recommended)
- [ ] Custom branding applied
- [ ] Analytics integrated
- [ ] Authentication added (if needed)

---

## 🤝 Support

Need help integrating these components? Here's what to check:

1. **Tailwind CSS not loading** → Check `tailwind.config.js` content paths
2. **Components not rendering** → Check import paths and file extensions
3. **API errors** → Verify API routes exist and return correct data
4. **Styling issues** → Make sure `globals.css` has `@tailwind` directives

---

## 📚 Next Steps

1. **Deploy your site** - Get it live on Vercel or your hosting platform
2. **Connect real data** - Wire up your marketing agent APIs
3. **Add authentication** - Protect sensitive pages with login
4. **Monitor performance** - Track metrics and optimize
5. **Gather feedback** - Share with your community and iterate

---

**Created:** May 8, 2026  
**Status:** Production-Ready Components  
**Next Update:** After your first deployment
