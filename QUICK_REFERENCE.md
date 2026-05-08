# Quick Reference: atla.in Multi-Agent System

## In 60 Seconds

**What**: You're building atla.in - a multi-agent AI system that showcases your skills
**Why**: Portfolio piece + technical demonstration of AI orchestration
**How**: 5 agents (Portfolio, Technical, Strategic, Analytics, Coordinator) working together
**When**: Weeks 6-14 of your 14-week AI journey
**Where**: `AI_Agent_1-atla-in-next` branch

---

## Files You Have

### Documents
```
AI_Agent_1-develop/
├── NEXT_STEPS_MULTI_AGENT.md        ← Full strategy (READ THIS FIRST)
├── MULTI_AGENT_SUMMARY.md           ← This overview
└── QUICK_REFERENCE.md               ← You are here
```

### Boilerplate Code (in outputs folder)
```
outputs/
├── 01-backend-main.py               ← FastAPI backend (ready to use)
├── 02-nextjs-app-page.tsx           ← Next.js home page (ready to use)
└── GETTING_STARTED_MULTI_AGENT.md   ← Step-by-step setup guide
```

---

## 3-Hour Quick Start

```bash
# Terminal 1: Backend
cd ~/Projects/AI_Agent_1-atla-in-next/backend
cat > main.py << 'EOF'
# Copy content from 01-backend-main.py
EOF
python -m venv venv && source venv/bin/activate
pip install fastapi uvicorn pydantic
uvicorn main:app --reload --port 8001

# Terminal 2: Frontend
cd ~/Projects/AI_Agent_1-atla-in-next/frontend
npx create-next-app@latest . --typescript --tailwind --app
npm install lucide-react
cat > app/page.tsx << 'EOF'
# Copy content from 02-nextjs-app-page.tsx
EOF
npm run dev

# Browser: http://localhost:3000
# Shows agents fetched from backend ✅
```

---

## The 5 Agents

| Agent | Role | Specialty | Example |
|-------|------|-----------|---------|
| **Portfolio** | Career Brand | "Tell me about your projects" | Showcases work |
| **Technical** | Deep Engineering | "How do you design systems?" | Architecture |
| **Strategic** | Planning | "What's your roadmap?" | Vision |
| **Analytics** | Data Insights | "Show me metrics" | Performance |
| **Coordinator** | Routing | Directs to right agent | Orchestration |

---

## Key Endpoints

```bash
# GET /health
curl http://localhost:8001/health
→ {"status":"ok","agents":5}

# GET /agents
curl http://localhost:8001/agents
→ {"agents":[...], "total":5}

# POST /chat (any agent)
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user1","query":"Hi"}'
→ {"conversation_id":"...", "response":"...", "agent_id":"coordinator"}

# POST /chat/{agent_id} (specific agent)
curl -X POST http://localhost:8001/chat/technical \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user1","query":"Design a system"}'
→ Response from technical agent
```

---

## Development Phases

### Phase 1 (Week 6-7): Foundation
- [x] Backend infrastructure
- [x] 5 agents implemented
- [x] Frontend home page
- [ ] Agent pages
- [ ] Chat interface

### Phase 2 (Week 8-9): Communication
- [ ] Inter-agent messaging
- [ ] Conversation memory
- [ ] Multi-agent routing
- [ ] Reasoning display

### Phase 3 (Week 10-11): Experiences
- [ ] Playground
- [ ] Scenarios
- [ ] Portfolio showcase
- [ ] obs.atla.in integration

### Phase 4 (Week 12-14): Polish
- [ ] Error handling
- [ ] Performance
- [ ] Deployment
- [ ] Documentation

---

## Architecture Layers

```
┌──────────────────────┐
│   Frontend UI        │  ← React, Next.js, TypeScript
│  (Home, Directory)   │
└──────────┬───────────┘
           │ HTTP/JSON
           ↓
┌──────────────────────┐
│  Backend API         │  ← FastAPI, Async Python
│  (FastAPI Routes)    │
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│  Agent Layer         │  ← 5 Specialized Agents
│  (Orchestrator)      │
└──────────┬───────────┘
           │
           ↓
┌──────────────────────┐
│  Message Bus         │  ← Inter-agent communication
│  (Communication)     │
└──────────────────────┘
```

---

## Quick Decisions

**Q: Port conflicts?**
→ Backend: 8001, Frontend: 3000. Change in code if needed.

**Q: Want to modify agents?**
→ Edit the agent class in main.py. Subclass `Agent` base class.

**Q: Add more agents?**
→ Create new class extending `Agent`. Add to `Orchestrator._initialize_agents()`.

**Q: Connect to obs.atla.in?**
→ In Analytics Agent, add HTTP client to query obs API. Do this in Phase 3.

**Q: Deploy it?**
→ Backend → Railway, Frontend → Vercel. Week 14 task.

---

## Testing Checklist

```
Backend:
☐ pip install works
☐ uvicorn starts
☐ /health responds
☐ /agents returns 5 agents
☐ /chat accepts queries

Frontend:
☐ npm install works
☐ npm run dev starts
☐ http://localhost:3000 loads
☐ Agent cards visible
☐ Agents fetch from backend
☐ No console errors

Integration:
☐ Both running simultaneously
☐ Frontend shows real agents
☐ Click agent navigates (TBD)
☐ No CORS errors
```

---

## File Copying Cheat Sheet

```bash
# 1. Get boilerplate
ls -la ~/Downloads/outputs/

# 2. Copy backend
cp ~/Downloads/outputs/01-backend-main.py \
   ~/Projects/AI_Agent_1-atla-in-next/backend/main.py

# 3. Copy frontend page
cp ~/Downloads/outputs/02-nextjs-app-page.tsx \
   ~/Projects/AI_Agent_1-atla-in-next/frontend/app/page.tsx

# 4. Copy guides
cp ~/Downloads/outputs/GETTING_STARTED_MULTI_AGENT.md \
   ~/Projects/AI_Agent_1-develop/

# 5. Start building!
```

---

## Key Concepts

**Agent**: Specialized AI persona (Portfolio, Technical, etc.)

**Orchestrator**: Manages all agents, routes requests

**Message Bus**: Enables agents to communicate with each other

**Coordinator**: Agent that decides which agent should handle a query

**Scenario**: Pre-built use case showing agents collaborating

**Reasoning Chain**: Transparent thinking steps shown to user

---

## Success Looks Like

**Week 6 End**: Backend starts, 5 agents respond to queries
**Week 7 End**: Frontend home page loads, displays agents
**Week 8 End**: Can chat with individual agents
**Week 9 End**: Agents can communicate with each other
**Week 10 End**: Multi-agent playground works
**Week 11 End**: Integration with obs.atla.in
**Week 12-14**: Polish, optimize, deploy

---

## Pro Tips

1. **Don't Overthink**: Use provided code, modify as you go
2. **Build Incrementally**: Get each piece working, then add features
3. **Test Often**: Run both servers, test endpoints with curl
4. **Document As You Go**: Add comments explaining your changes
5. **Version Control**: Commit after each working feature
6. **Use TypeScript**: It catches bugs, frontend already typed
7. **Log Everything**: Add print statements for debugging

---

## When You Get Stuck

1. **Backend error?** → Check terminal output, curl the endpoint
2. **Frontend error?** → Check browser console, npm run dev output
3. **CORS error?** → Ensure backend has correct CORS middleware
4. **Type error?** → TypeScript is helping! Read the error carefully
5. **Can't connect?** → Verify both servers running on correct ports

---

## Resources

- FastAPI docs: https://fastapi.tiangolo.com
- Next.js docs: https://nextjs.org/docs
- Tailwind docs: https://tailwindcss.com/docs
- TypeScript docs: https://www.typescriptlang.org/docs
- Pydantic docs: https://docs.pydantic.dev

---

## The Big Picture

You're building **3 interconnected systems**:

1. **obs.atla.in** (Weeks 1-5)
   - Cloud ops platform
   - Real-time metrics
   - SQLite persistence
   ✅ COMPLETE

2. **atla.in** (Weeks 6-14)
   - Multi-agent showcase
   - Portfolio narrative
   - Strategic planning
   ← YOU ARE HERE

3. **Integration** (Week 12-14)
   - atla.in queries obs.atla.in
   - Live data in agents
   - Complete ecosystem

**Together**: A full-stack AI system that gets you hired.

---

## One More Thing

**You've got this.** 

The foundation is solid. The code is ready. The path is clear.

Just start with GETTING_STARTED_MULTI_AGENT.md and follow the phases.

By Week 14, you'll have built something that will genuinely impress people.

Now go build. 🚀

