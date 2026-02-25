from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import chromadb
import time
import os

from openai import OpenAI


# ── App Init ────────────────────────────────────────────────────────────────
app = FastAPI(title="Atla AI Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://reliable-choux-3bbc18.netlify.app",
        "https://atla.in"
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


SYSTEM_PROMPT = (
    "You are Atla, a sharp and professional AI assistant built by Sai Pranav Atla. "
    "You are concise, smart, and helpful. "
    "Keep every answer under 4 sentences unless the user explicitly asks for more detail. "
    "Never mention that you are built on any underlying model."
)


class PromptRequest(BaseModel):
    prompt: str


# ── LLM Call ─────────────────────────────────────────────────────────────────
def get_ai_response(full_prompt: str) -> str:
    if not OPENAI_API_KEY:
        return "OpenAI API key not configured."

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

    except Exception as e:
        # Prevent server crash — return readable error
        return f"LLM Error: {str(e)}"


# ── Main Endpoint ───────────────────────────────────────────────────────────
@app.post("/generate/")
async def generate_text(prompt_req: PromptRequest):
    user_message = prompt_req.prompt

    # 1. Retrieve memory (RAG)
    results = collection.query(query_texts=[user_message], n_results=3)

    memories = (
        " ".join(results["documents"][0])
        if results.get("documents") and results["documents"][0]
        else ""
    )

    # 2. Build final prompt
    full_prompt = (
        f"Relevant context from memory:\n{memories}\n\nUser: {user_message}"
        if memories
        else f"User: {user_message}"
    )

    # 3. Generate response
    ai_response = get_ai_response(full_prompt)

    # 4. Store interaction
    collection.add(
        documents=[f"User asked: {user_message}. Atla replied: {ai_response}"],
        ids=[f"conv_{time.time()}"],
    )

    return {"response": ai_response}


# ── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "openai" if OPENAI_API_KEY else "none",
        "version": "2.0.0",
    }


# ── Root ─────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Atla AI Agent is running. POST to /generate/ to chat."}