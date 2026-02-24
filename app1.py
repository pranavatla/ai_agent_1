from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import chromadb
import time
import os
import requests

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Atla AI Agent", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve static files (index.html) ───────────────────────────────────────────
# Uncomment this when you add a /static folder or serve index.html directly
# app.mount("/static", StaticFiles(directory="static"), name="static")

# ── ChromaDB Persistent Memory ────────────────────────────────────────────────
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="user_memory")

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3"

# Set ANTHROPIC_API_KEY in your environment or Railway/Render dashboard
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

SYSTEM_PROMPT = (
    "You are Atla, a sharp and professional AI assistant built by Sai Pranav Atla. "
    "You are concise, smart, and helpful. "
    "Keep every answer under 4 sentences unless the user explicitly asks for more detail. "
    "Never mention that you are built on any underlying model."
)

class PromptRequest(BaseModel):
    prompt: str

# ── LLM Router: Claude first, Ollama fallback ─────────────────────────────────
def call_claude(full_prompt: str) -> str:
    """Call Anthropic Claude API."""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    body = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 512,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": full_prompt}],
    }
    r = requests.post("https://api.anthropic.com/v1/messages", json=body, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()["content"][0]["text"]


def call_ollama(full_prompt: str) -> str:
    """Call local Ollama (dev / offline fallback)."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{full_prompt}",
        "stream": False,
    }
    r = requests.post(OLLAMA_URL, json=payload, timeout=60)
    r.raise_for_status()
    return r.json().get("response", "")


def get_ai_response(full_prompt: str) -> str:
    """Use Claude if API key is set, otherwise fall back to Ollama."""
    if ANTHROPIC_API_KEY:
        try:
            return call_claude(full_prompt)
        except Exception as e:
            print(f"Claude API error: {e} — falling back to Ollama")
    return call_ollama(full_prompt)


# ── Main endpoint ─────────────────────────────────────────────────────────────
@app.post("/generate/")
async def generate_text(prompt_req: PromptRequest):
    user_message = prompt_req.prompt

    # 1. Retrieve relevant memories from ChromaDB (RAG)
    results = collection.query(query_texts=[user_message], n_results=3)
    memories = (
        " ".join(results["documents"][0])
        if results["documents"] and results["documents"][0]
        else ""
    )

    # 2. Build the final prompt
    full_prompt = (
        f"Relevant context from memory:\n{memories}\n\n"
        f"User: {user_message}"
        if memories
        else f"User: {user_message}"
    )

    # 3. Get AI response
    ai_response = get_ai_response(full_prompt)

    # 4. Save interaction to memory
    collection.add(
        documents=[f"User asked: {user_message}. Atla replied: {ai_response}"],
        ids=[f"conv_{time.time()}"],
    )

    return {"response": ai_response}


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    backend = "claude" if ANTHROPIC_API_KEY else "ollama"
    return {"status": "ok", "backend": backend, "version": "1.0.0"}


# ── Root ──────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Atla AI Agent is running. POST to /generate/ to chat."}
