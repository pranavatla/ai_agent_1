from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI, RateLimitError, APIError, AuthenticationError
import time
import os
from collections import defaultdict

app = FastAPI(title="Atla AI Agent", version="3.1.0")

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

class PromptRequest(BaseModel):
    prompt: str

def get_ai_response(user_prompt: str) -> str:
    if not OPENAI_API_KEY:
        return "AI service configuration issue."

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
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

@app.post("/generate/")
async def generate_text(request: Request, prompt_req: PromptRequest):
    client_ip = request.client.host
    current_time = time.time()

    request_log[client_ip] = [
        t for t in request_log[client_ip]
        if current_time - t < WINDOW
    ]

    if len(request_log[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests.")

    request_log[client_ip].append(current_time)

    ai_response = get_ai_response(prompt_req.prompt)
    return {"response": ai_response}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "openai" if OPENAI_API_KEY else "none",
        "rate_limit": f"{RATE_LIMIT} req/{WINDOW}s",
        "version": "3.1.0",
    }

@app.get("/")
def root():
    return {"message": "Atla AI Agent running."}