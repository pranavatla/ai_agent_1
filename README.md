🚀 Atla AI Agent

Serverless + Containerized AI Backend | RAG | Persistent Memory | Multi-Cloud Deployment

Atla AI Agent — atla.in

A production-grade conversational AI agent with persistent memory, Retrieval-Augmented Generation (RAG), and a modern glassmorphism UI.

Live demo: https://atla.in

This project demonstrates:

FastAPI-based AI backend

Persistent vector memory (ChromaDB)

Claude API (production) + Ollama (local dev)

Dual deployment: Vercel (serverless) + Railway (container runtime)

Git-based CI/CD automation

✨ What It Does

Answers questions using a local knowledge base (RAG via ChromaDB)

Remembers conversations across server restarts

Routes to Claude API in production

Supports Ollama + Llama 3 locally

Clean animated dark UI with real-time “Thinking…” state

Deployable across multiple cloud runtimes from a single repo

🏗 Architecture Overview
AI_Agent_1/
│
├── api/
│   └── index.py          # Primary FastAPI application (Vercel runtime)
│
├── app1.py               # Railway entrypoint (imports api.index)
├── ingest.py             # Ingest single file into vector store
├── ingest_all.py         # Bulk ingest knowledge base
├── memory_test.py        # ChromaDB smoke test
│
├── knowledge_base/       # Drop .txt files here
├── chroma_db/            # Persistent vector database (gitignored)
│
├── index.html            # Frontend UI
├── requirements.txt
├── vercel.json           # Vercel routing config
└── README.md
☁️ Deployment Model
Platform	Role
Vercel	Static frontend + serverless backend
Railway	Persistent container backend
Netlify (optional)	Static frontend pointing to Railway
How It Works

api/index.py → primary FastAPI app

app1.py → thin wrapper for Railway (from api.index import app)

Same backend logic deployed to two environments

Push to Git → auto redeploy on both platforms

🧠 Tech Stack
Layer	Technology
Backend	FastAPI + Uvicorn
Vector DB	ChromaDB (persistent)
LLM (prod)	Claude API (claude-3-5-haiku)
LLM (dev)	Ollama + Llama 3
Frontend	Vanilla HTML / CSS / JS
Serverless	Vercel
Container Runtime	Railway
💻 Quick Start (Local)
1️⃣ Clone & Setup
git clone https://github.com/saipranavAtla/atla-agent.git
cd atla-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
2️⃣ Optional: Run with Ollama (Local Dev)

In a separate terminal:

ollama serve
ollama pull llama3
3️⃣ Optional: Use Claude API
export ANTHROPIC_API_KEY=sk-ant-...
4️⃣ Ingest Knowledge Base
python ingest_all.py

This creates the persistent chroma_db/ directory.

5️⃣ Start Backend
uvicorn app1:app --reload --port 8000
6️⃣ Open UI

Either open index.html directly
or serve it:

python -m http.server 3000

Visit:

http://localhost:3000
📡 API Endpoints
Method	Endpoint	Description
POST	/generate/	Chat with the AI agent
GET	/health	Backend health check
GET	/	Root info
Example Request
curl -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What services do you offer?"}'
📚 Adding Knowledge

Drop any .txt file into:

knowledge_base/

Then run:

python ingest_all.py

The AI will incorporate the new content into retrieval responses.

🚂 Deploy to Railway

Push repo to GitHub

Go to railway.app → New Project → Deploy from GitHub

Set environment variable:

ANTHROPIC_API_KEY = sk-ant-...

Start command:

uvicorn app1:app --host 0.0.0.0 --port $PORT

Railway auto deploys

Update frontend API_BASE if using Railway URL

Point DNS (atla.in) to Railway

🚀 Deploy to Vercel

Connect GitHub repo to Vercel

Ensure vercel.json exists

Set required environment variables

Push to main

Auto deployment triggers

Vercel handles:

Static frontend hosting

Serverless FastAPI backend via /api/index.py

🔐 Environment Variables
Variable	Required	Description
ANTHROPIC_API_KEY	Yes (Claude mode)	Production LLM
OPENAI_API_KEY	Optional	If using OpenAI mode
OLLAMA_BASE_URL	Optional	Local dev mode

Must be configured separately per platform.

⚙️ CI/CD Workflow

Every git push:

Vercel rebuilds frontend + serverless backend

Railway redeploys backend

No manual build steps required

⚠ Limitations

Current architecture is suitable for:

Personal AI agents

Small-scale SaaS

Portfolio projects

Experimental deployments

For enterprise scale:

Replace in-memory rate limiting with Redis

Add distributed caching

Add structured logging

Add observability (Prometheus / OpenTelemetry)

Add horizontal scaling strategy

🛠 Future Roadmap

Streaming responses

Advanced memory layering

Redis-backed distributed rate limiting

Vector DB cloud migration

Custom domain API subdomain (api.atla.in)

Structured logging & monitoring

👤 Built By

Sai Pranav Atla
AI Engineer — Bengaluru

LinkedIn

GitHub

https://atla.in

🔥 What This Project Demonstrates

Retrieval-Augmented Generation (RAG)

Persistent vector memory

Dual runtime deployment (serverless + container)

Clean Git-based CI/CD

Multi-cloud architecture from one codebase

This repository is both a functional AI system and an infrastructure learning project.
