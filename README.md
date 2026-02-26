# Atla AI Agent

Production-ready AI chat webapp built with FastAPI, OpenAI, and a lightweight frontend.

Live demo: [https://ai-agent-1-six.vercel.app/](https://ai-agent-1-six.vercel.app/)

## Why this project

This project demonstrates:
- API-first backend engineering with FastAPI
- Token-aware conversation management (trim + summarize)
- Dual deployment strategy from one repo
- Serverless deployment (Vercel) and container deployment (Railway/Docker)

## Stack

- Backend: FastAPI, Pydantic, Uvicorn
- LLM: OpenAI Chat Completions (`gpt-4o-mini`)
- Token accounting: `tiktoken`
- Frontend: Vanilla HTML/CSS/JS
- Deploy: Vercel (serverless) + Railway (Docker)

## Architecture

- `api/index.py`: Main FastAPI app
- `app1.py`: Runtime entrypoint (`from api.index import app`)
- `index.html`: Chat UI
- `vercel.json`: Vercel routing/build config
- `Dockerfile`: Container build
- `railway.toml`: Railway Docker deploy config

## API Endpoints

- `GET /`: Serves `index.html` when available
- `GET /health`: Health check
- `POST /generate/`: Chat completion endpoint

Request body for `POST /generate/`:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello"}
  ]
}
```

## Local Development

1. Create environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Set environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

3. Run the app:

```bash
uvicorn app1:app --reload --port 8000
```

4. Open:

`http://localhost:8000`

## Deploy on Vercel (Current live setup)

1. Connect repository in Vercel.
2. Set environment variable:
- `OPENAI_API_KEY`
3. Deploy.

Routing behavior comes from `vercel.json`:
- `/` -> `index.html`
- `/health` -> `api/index.py`
- `/generate/` -> `api/index.py`

## Deploy on Railway (Docker, single URL)

This repo is already configured for Railway Docker deployments via `railway.toml`.

1. Create a new Railway project from this repo.
2. Add environment variable:
- `OPENAI_API_KEY`
3. Deploy.

Railway will:
- Build with `Dockerfile`
- Start with `python run.py` (reads `PORT` safely in Python)
- Run health checks on `/health`

Result: one Railway URL serves both UI (`/`) and API (`/generate/`).

## Run with Docker locally

```bash
docker build -t atla-agent .
docker run -p 8000:8000 -e OPENAI_API_KEY=sk-... atla-agent
```

Open:
`http://localhost:8000`

## Environment Variables

- `OPENAI_API_KEY` (required)

## Notes

- The frontend uses same-origin calls in production.
- CORS is enabled in backend for compatibility across environments.
- Long conversations are protected with token limits and summarization.
