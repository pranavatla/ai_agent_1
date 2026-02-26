# Atla AI Agent

Production-grade conversational AI webapp with a FastAPI backend and a minimal chat UI.

## Overview

This repo serves a static frontend (`index.html`) and a serverless FastAPI backend (`api/index.py`).
It is designed to deploy cleanly on Vercel.

## Architecture

AI_Agent_1/
├── api/index.py        # FastAPI app (Vercel serverless runtime)
├── index.html          # Frontend UI
├── requirements.txt
├── vercel.json
├── ingest.py
├── ingest_all.py
├── memory_test.py
├── knowledge_base/
├── chroma_db/          # gitignored
└── README.md

## Core Features

- FastAPI + OpenAI SDK
- Conversation trimming and summarization for cost control
- Single-repo deployment (static frontend + serverless backend)

## Local Development

1. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Set env var

```bash
export OPENAI_API_KEY=sk-...
```

3. Run backend

```bash
uvicorn app1:app --reload --port 8000
```

4. Serve frontend

```bash
python -m http.server 3000
```

Open:
`http://localhost:3000`

## API Endpoints

- `POST /generate/`  → Chat with the assistant
- `GET /health`      → Backend health check
- `GET /`            → Root info

## Vercel Deployment

1. Connect this repo to Vercel
2. Set environment variables:

- `OPENAI_API_KEY`

3. Deploy

Vercel serves:
- `index.html` as the frontend
- `api/index.py` as serverless backend (see `vercel.json` routes)

## Docker (Single URL)

The container serves the UI at `/` and the API at `/generate/` and `/health`.

Build:

```bash
docker build -t atla-agent .
```

Run:

```bash
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... atla-agent
```

Open:
`http://localhost:8000`

## Notes

- `index.html` uses same-origin API calls in production, so no extra CORS setup is needed.
- The backend supports long conversations by summarizing when token usage is high.
