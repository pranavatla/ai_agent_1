# Getting Started: atla.in Multi-Agent System

## Overview

You're building **atla.in** - a production-grade multi-agent system that showcases:
- 5 specialized AI agents (Portfolio, Technical, Strategic, Analytics, Coordinator)
- Agent orchestration and communication
- Integration with obs.atla.in for live data
- Interactive frontend for exploring agents and multi-agent scenarios

This is the **portfolio-defining project** of your 14-week AI journey.

---

## Phase 1: Backend Setup (Today - 2 hours)

### 1.1 Initialize Backend Project

```bash
# Navigate to your atla-in-next directory
cd ~/Projects/AI_Agent_1-atla-in-next

# Create backend directory structure
mkdir -p backend/{agents,services,routers,models,database}
cd backend

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Create requirements.txt
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
pydantic-settings==2.1.0
python-multipart==0.0.6
sqlalchemy==2.0.23
aiosqlite==0.19.0
python-dotenv==1.0.0
httpx==0.25.1
EOF

# Install dependencies
pip install -r requirements.txt
```

### 1.2 Create Backend Structure

Copy the `01-backend-main.py` file I provided and save it as:

```bash
# In backend/ directory
cat > main.py << 'CONTENT'
# [PASTE CONTENT FROM 01-backend-main.py]
CONTENT
```

### 1.3 Test Backend

```bash
# In backend/ directory
uvicorn main:app --reload --port 8001

# In another terminal, test the API:
curl http://localhost:8001/health
# Expected response: {"status":"ok","version":"1.0.0","agents":5}

# List agents:
curl http://localhost:8001/agents
# Shows all 5 agents with metadata

# Chat with coordinator:
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user1","query":"Tell me about your career","conversation_id":"conv1"}'
```

**Success Criteria:**
- ✅ Backend starts without errors
- ✅ `/health` returns status ok
- ✅ `/agents` lists 5 agents
- ✅ `/chat` processes queries and routes to agents

---

## Phase 2: Frontend Setup (2-3 hours)

### 2.1 Initialize Next.js Project

```bash
# Navigate to atla-in-next directory (sibling to backend)
cd ~/Projects/AI_Agent_1-atla-in-next

# Create frontend
npx create-next-app@latest frontend \
  --typescript \
  --tailwind \
  --eslint \
  --app \
  --no-src-dir \
  --import-alias '@/*'

cd frontend

# Install additional dependencies
npm install -D shadcn-ui
npm install recharts lucide-react clsx tailwind-merge
npm install zustand react-query

# Update tsconfig.json for strict mode (recommended)
```

### 2.2 Create Frontend Structure

```bash
# Create directories
mkdir -p app/{agents,playground,portfolio}
mkdir -p components/{ui,agents,chat,layout}
mkdir -p lib

# Copy home page
cat > app/page.tsx << 'CONTENT'
# [PASTE CONTENT FROM 02-nextjs-app-page.tsx]
CONTENT
```

### 2.3 Create Environment File

```bash
# frontend/.env.local
NEXT_PUBLIC_API_URL=http://localhost:8001
NEXT_PUBLIC_APP_NAME=atla.in
```

### 2.4 Test Frontend

```bash
# In frontend/ directory
npm run dev

# Open browser to http://localhost:3000
# You should see the home page with agent cards
```

**Success Criteria:**
- ✅ Frontend starts on port 3000
- ✅ Home page loads with gradient background
- ✅ Agent cards display with icons
- ✅ Clicking agent card would navigate (page not built yet)

---

## Phase 3: Connect Frontend to Backend (1 hour)

### 3.1 Create API Client

```typescript
// lib/api-client.ts
export const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8001';

export async function fetchAgents() {
  const res = await fetch(`${API_URL}/agents`);
  if (!res.ok) throw new Error('Failed to fetch agents');
  return res.json();
}

export async function chatWithAgent(userId: string, query: string, agentId?: string) {
  const endpoint = agentId ? `/chat/${agentId}` : '/chat';
  
  const res = await fetch(`${API_URL}${endpoint}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      user_id: userId,
      query,
      conversation_id: `conv_${Date.now()}`,
      agent_id: agentId
    })
  });

  if (!res.ok) throw new Error('Failed to get response');
  return res.json();
}
```

### 3.2 Test Integration

```bash
# Start both services
# Terminal 1 (Backend):
cd ~/Projects/AI_Agent_1-atla-in-next/backend
source venv/bin/activate
uvicorn main:app --reload --port 8001

# Terminal 2 (Frontend):
cd ~/Projects/AI_Agent_1-atla-in-next/frontend
npm run dev

# Browser: http://localhost:3000
# Should see agents loading from backend
```

---

## Phase 4: Build Individual Agent Pages (3 hours)

### 4.1 Create Agent Detail Page

```bash
# app/agents/[agent_id]/page.tsx
```

**Features:**
- Show agent details (name, role, description, capabilities)
- Chat interface for talking to specific agent
- Display conversation history
- Show response formatting with proper styling

### 4.2 Create Agents Directory

```bash
# app/agents/page.tsx
```

**Features:**
- Grid of all agents
- Filter by role
- Search agents
- Link to individual agent pages

---

## Phase 5: Build Multi-Agent Playground (4 hours)

### 5.1 Playground Features

```bash
# app/playground/page.tsx
```

**Features:**
- Free-form chat interface
- Shows which agent(s) responded
- Display agent reasoning chain (when available)
- Pre-built scenarios:
  - "Design a startup tech stack" → Strategic + Technical agents
  - "Review my portfolio" → Portfolio + Analytics agents
  - "System architecture deep dive" → Technical agent with follow-ups

### 5.2 Scenario System

```typescript
// lib/scenarios.ts
export const scenarios = [
  {
    id: 'startup-stack',
    title: 'Design Startup Tech Stack',
    description: 'Ask agents to recommend architecture for a startup',
    agents: ['strategic', 'technical'],
    initialQuery: 'Design a tech stack for a Series A startup with B2B SaaS product...'
  },
  // ... more scenarios
];
```

---

## Quick Reference: API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Health check |
| GET | `/agents` | List all agents |
| GET | `/agents/{agent_id}` | Get specific agent |
| POST | `/chat` | Chat with coordinator |
| POST | `/chat/{agent_id}` | Chat with specific agent |

---

## Development Checklist

### Week 6-7 Tasks

- [ ] Backend initialization (FastAPI, agents, message bus)
- [ ] Frontend initialization (Next.js, TypeScript, Tailwind)
- [ ] API client integration
- [ ] Test backend-frontend communication
- [ ] Create agent detail page
- [ ] Create agents directory page
- [ ] Create playground interface
- [ ] Implement scenario system
- [ ] Add conversation history/memory display

### Week 8-9 Tasks

- [ ] Agent memory system (conversation context)
- [ ] Message bus implementation (inter-agent communication)
- [ ] Orchestration patterns (sequential, parallel, hierarchical)
- [ ] Agent reasoning transparency (show thinking steps)
- [ ] Integration with obs.atla.in for live data queries
- [ ] Analytics agent enhanced with real metrics

### Week 10-11 Tasks

- [ ] Portfolio agent enrichment (resume, projects, skills data)
- [ ] Technical agent knowledge base (system design docs, code snippets)
- [ ] Strategic agent planning capabilities
- [ ] Advanced UI: drag-drop scenario builder
- [ ] Real-time agent communication visualization
- [ ] Performance metrics dashboard

### Week 12-14 Tasks

- [ ] Polish and refinement
- [ ] Error handling and edge cases
- [ ] Deployment (Vercel for frontend, Railway for backend)
- [ ] Documentation and README
- [ ] Demo scenarios creation
- [ ] Final portfolio narrative

---

## File Structure (Target)

```
AI_Agent_1-atla-in-next/
├── backend/
│   ├── venv/
│   ├── main.py                      # FastAPI app
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base.py                  # Agent base class
│   │   ├── portfolio.py             # Portfolio agent
│   │   ├── technical.py             # Technical agent
│   │   ├── strategic.py             # Strategic agent
│   │   ├── analytics.py             # Analytics agent
│   │   └── coordinator.py           # Coordinator agent
│   ├── services/
│   │   ├── orchestration.py         # Agent orchestration
│   │   ├── message_bus.py           # Inter-agent communication
│   │   └── memory.py                # Conversation memory
│   ├── routers/
│   │   ├── agents.py                # Agent endpoints
│   │   ├── chat.py                  # Chat endpoints
│   │   └── health.py                # Health check
│   ├── models/
│   ├── database/
│   ├── requirements.txt
│   └── .env
│
├── frontend/
│   ├── app/
│   │   ├── layout.tsx               # Root layout
│   │   ├── page.tsx                 # Home
│   │   ├── agents/
│   │   │   ├── page.tsx             # Agents directory
│   │   │   └── [agent_id]/
│   │   │       └── page.tsx         # Agent detail page
│   │   ├── playground/
│   │   │   ├── page.tsx             # Multi-agent playground
│   │   │   └── scenarios/
│   │   └── portfolio/
│   │       └── page.tsx             # Portfolio showcase
│   │
│   ├── components/
│   │   ├── ui/                      # shadcn components
│   │   ├── agents/                  # Agent-specific components
│   │   ├── chat/                    # Chat interface
│   │   └── layout/
│   │
│   ├── lib/
│   │   ├── api-client.ts           # Backend communication
│   │   ├── scenarios.ts            # Scenario definitions
│   │   ├── utils.ts
│   │   └── types.ts                # TypeScript types
│   │
│   ├── styles/
│   │   └── globals.css             # Global styles
│   │
│   ├── .env.local
│   ├── package.json
│   ├── tsconfig.json
│   └── tailwind.config.ts
│
├── NEXT_STEPS_MULTI_AGENT.md        # Strategy document
└── README.md                        # Project overview
```

---

## Common Issues & Solutions

### Issue: Backend not found from frontend

**Solution:**
```typescript
// Check CORS is enabled in FastAPI
// Verify NEXT_PUBLIC_API_URL in .env.local matches backend port
// In backend, ensure allow_origins=["*"]
```

### Issue: Agent not responding

**Solution:**
```python
# Check agent.process_query() is implemented
# Verify agent is registered in orchestrator._initialize_agents()
# Test endpoint directly with curl
```

### Issue: Frontend components not loading

**Solution:**
```bash
# Clear Next.js cache
rm -rf .next
npm run dev

# Check TypeScript errors
npm run type-check
```

---

## Testing Commands

### Test Backend Endpoints

```bash
# Test health
curl http://localhost:8001/health

# Test agents list
curl http://localhost:8001/agents

# Test chat with portfolio agent
curl -X POST http://localhost:8001/chat/portfolio \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "query": "Tell me about your projects",
    "conversation_id": "test1"
  }'

# Test coordinator routing
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "test",
    "query": "Design a scalable system",
    "conversation_id": "test2"
  }'
```

### Test Frontend Integration

```javascript
// In browser console
fetch('http://localhost:8001/agents')
  .then(r => r.json())
  .then(d => console.log(d))
```

---

## Next Steps

1. **Set up backend** (30 min)
   - Copy main.py, install deps, start server
   
2. **Set up frontend** (30 min)
   - Initialize Next.js, create basic structure
   
3. **Connect them** (30 min)
   - Create API client, test communication
   
4. **Build first page** (1 hour)
   - Create agents directory page with real data from backend
   
5. **Test integration** (30 min)
   - Verify data flows correctly end-to-end

**Total time: 3.5 hours**

After this, you'll have a working foundation to build on!

---

## Support Files Provided

1. **NEXT_STEPS_MULTI_AGENT.md** - Full strategic roadmap (read first!)
2. **01-backend-main.py** - Complete FastAPI backend with 5 agents
3. **02-nextjs-app-page.tsx** - Next.js home page component
4. **GETTING_STARTED_MULTI_AGENT.md** - This file

---

## Questions?

Key questions to answer as you build:

1. Should agents query obs.atla.in live or cache data?
2. How much reasoning transparency should users see?
3. What scenarios best showcase agent collaboration?
4. Should agents learn/improve from conversations?

These will shape your implementation decisions!

Good luck! 🚀

