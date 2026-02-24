# Atla AI Agent — atla.in

A production-grade conversational AI agent with persistent memory, RAG, and a clean glassmorphism UI.

**Live demo:** [atla.in](https://atla.in)

---

## What it does

- Answers questions using a local knowledge base (RAG via ChromaDB)
- Remembers conversations permanently across server restarts
- Routes to **Claude API** in production or **Ollama (Llama 3)** locally
- Clean, animated dark UI with real-time "Thinking…" state

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Uvicorn |
| Vector DB | ChromaDB (persistent) |
| LLM (prod) | Claude API (claude-3-5-haiku) |
| LLM (dev) | Ollama + Llama 3 (local) |
| Frontend | Vanilla HTML/CSS/JS |
| Deploy | Railway / Render |

---

## Project Structure

```
AI_Agent_1/
├── app1.py           # FastAPI backend — main entry point
├── ingest.py         # Ingest a single .txt file into memory
├── ingest_all.py     # Bulk ingest all files in /knowledge_base
├── memory_test.py    # Quick ChromaDB smoke test
├── knowledge_base/   # Drop .txt files here to expand AI knowledge
│   └── company_info.txt
├── chroma_db/        # Auto-created vector store (gitignored)
├── index.html        # Frontend UI
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Quick Start (Local)

### 1. Clone and set up environment

```bash
git clone https://github.com/saipranavAtla/atla-agent.git
cd atla-agent
python3.13 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. (Optional) Run with Ollama locally

```bash
# In a separate terminal
ollama serve
ollama pull llama3
```

### 3. (Optional) Use Claude API

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Ingest your knowledge base

```bash
python ingest_all.py
```

### 5. Start the server

```bash
uvicorn app1:app --reload --port 8000
```

### 6. Open the UI

Open `index.html` directly in your browser, or serve it:

```bash
python -m http.server 3000
```
Then visit: [http://localhost:3000](http://localhost:3000)

---

## Deploy to Railway (10 minutes)

1. Push this repo to GitHub
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub
3. Set environment variable: `ANTHROPIC_API_KEY = sk-ant-...`
4. Railway auto-detects FastAPI and deploys
5. Get your URL (e.g. `https://atla-agent.up.railway.app`)
6. Update `API_BASE` in `index.html` with this URL
7. Point `atla.in` DNS to Railway

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/generate/` | Chat with the AI agent |
| `GET`  | `/health`    | Backend health + active LLM backend |
| `GET`  | `/`          | Root info |

**Example request:**
```bash
curl -X POST http://localhost:8000/generate/ \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What services do you offer?"}'
```

---

## Adding Knowledge

Drop any `.txt` file into the `knowledge_base/` folder and run:

```bash
python ingest_all.py
```

The AI will now use this content when answering relevant questions.

---

## Built by

**Sai Pranav Atla** — AI Engineer, Bengaluru

- [LinkedIn](https://linkedin.com/in/saipranavAtla)
- [GitHub](https://github.com/saipranavAtla)
- [atla.in](https://atla.in)
