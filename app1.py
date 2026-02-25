from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import chromadb
import time
import os
from collections import defaultdict

from openai import OpenAI, RateLimitError, APIError, AuthenticationError


# ── App Init ────────────────────────────────────────────────────────────────
app = FastAPI(title="Atla AI Agent", version="3.0.0")

# ── CORS Configuration ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://reliable-choux-3bbc18.netlify.app",
        "https://genuine-churros-be1013.netlify.app",
        "https://atla.in",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── ChromaDB Persistent Memory ──────────────────────────────────────────────
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="user_memory")


# ── OpenAI Setup ─────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)


# ── Rate Limiting Setup ─────────────────────────────────────────────────────
request_log = defaultdict(list)
RATE_LIMIT = 10
WINDOW = 60  # seconds


SYSTEM_PROMPT = (
    "You are Atla, a sharp and professional AI assistant built by Sai Pranav Atla. "
    "You are concise, smart, and helpful. "
    "Keep every answer under 4 sentences unless the user explicitly asks for more detail. "
    "Never mention that you are built on any underlying model."
)


class PromptRequest(BaseModel):
    prompt: str


# ── LLM Call With Fallback ──────────────────────────────────────────────────
def get_ai_response(full_prompt: str) -> str:
    if not OPENAI_API_KEY:
        return "AI service configuration issue."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.7,
            max_tokens=500,
        )
        return response.choices[0].message.content

    except RateLimitError:
        # Fallback model
        try:
            fallback = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.7,
                max_tokens=500,
            )
            return fallback.choices[0].message.content
        except Exception:
            return "Service temporarily busy. Please try again shortly."

    except AuthenticationError:
        return "AI service authentication failed."

    except APIError:
        return "AI service currently unavailable."

    except Exception as e:
        print("Unexpected LLM error:", str(e))
        return "Unexpected AI service error."


# ── Main Endpoint ───────────────────────────────────────────────────────────
@app.post("/generate/")
async def generate_text(request: Request, prompt_req: PromptRequest):

    # ── Rate Limiting ──
    client_ip = request.client.host
    current_time = time.time()

    request_log[client_ip] = [
        t for t in request_log[client_ip]
        if current_time - t < WINDOW
    ]

    if len(request_log[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please slow down."
        )

    request_log[client_ip].append(current_time)

    user_message = prompt_req.prompt

    # ── Retrieve Memory (RAG) ──
    results = collection.query(query_texts=[user_message], n_results=3)

    memories = (
        " ".join(results["documents"][0])
        if results.get("documents") and results["documents"][0]
        else ""
    )

    full_prompt = (
        f"Relevant context from memory:\n{memories}\n\nUser: {user_message}"
        if memories
        else f"User: {user_message}"
    )

    # ── Generate Response ──
    ai_response = get_ai_response(full_prompt)

    # ── Store Interaction ──
    collection.add(
        documents=[f"User asked: {user_message}. Atla replied: {ai_response}"],
        ids=[f"conv_{time.time()}"],
    )

    return {"response": ai_response}


# ── Health Endpoint ─────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "openai" if OPENAI_API_KEY else "none",
        "rate_limit": f"{RATE_LIMIT} req/{WINDOW}s",
        "version": "3.0.0",
    }


# ── Root Endpoint ───────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Atla AI Agent is running. POST to /generate/ to chat."}