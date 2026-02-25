from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, RateLimitError, APIError, AuthenticationError
import time
import os
from collections import defaultdict
from typing import List

app = FastAPI(title="Atla AI Agent", version="3.2.0")

# ── CORS ─────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://reliable-choux-3bbc18.netlify.app",
        "https://atla-dev.netlify.app",
        "https://atla.in",
        "http://localhost:8000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── OpenAI Setup ─────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

# ── Rate Limiting ────────────────────────────────────
request_log = defaultdict(list)
RATE_LIMIT = 10
WINDOW = 60

SYSTEM_PROMPT = (
    "You are Atla, a sharp and professional AI assistant built by Sai Pranav Atla. "
    "You are concise, smart, and helpful. "
    "Keep every answer under 4 sentences unless the user explicitly asks for more detail. "
    "Never mention that you are built on any underlying model."
)

# ── Models ───────────────────────────────────────────
class Message(BaseModel):
    role: str
    content: str

class ConversationRequest(BaseModel):
    messages: List[Message]

# ── AI Response Function ─────────────────────────────
def get_ai_response(messages: List[Message]) -> str:
    if not OPENAI_API_KEY:
        return "AI service configuration issue."

    try:
        openai_messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for msg in messages:
            if msg.role in ["user", "assistant"]:
                openai_messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=openai_messages,
            temperature=0.7,
            max_tokens=400,
        )

        return response.choices[0].message.content

    except RateLimitError:
        return "AI service rate limited. Try again shortly."
    except AuthenticationError:
        return "AI service authentication failed."
    except APIError:
        return "AI service currently unavailable."
    except Exception as e:
        print("Unexpected error:", str(e))
        return "Unexpected AI service error."

# ── Endpoint ─────────────────────────────────────────
@app.post("/generate/")
async def generate_text(request: Request, convo: ConversationRequest):
    client_ip = request.client.host
    current_time = time.time()

    request_log[client_ip] = [
        t for t in request_log[client_ip]
        if current_time - t < WINDOW
    ]

    if len(request_log[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests.")

    request_log[client_ip].append(current_time)

    ai_response = get_ai_response(convo.messages)
    return {"response": ai_response}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "openai" if OPENAI_API_KEY else "none",
        "rate_limit": f"{RATE_LIMIT} req/{WINDOW}s",
        "version": "3.2.0",
    }

@app.get("/")
def root():
    return {"message": "Atla AI Agent running."}