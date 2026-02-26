import os
import tiktoken
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

# =====================================
# App Setup
# =====================================

app = FastAPI()

# =====================================
# CORS (THIS FIXES VERCEL)
# =====================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use "*" for now to eliminate CORS errors
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================
# OpenAI Setup
# =====================================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS_ALLOWED = 12000
SUMMARIZE_TRIGGER = 9000

encoding = tiktoken.encoding_for_model(MODEL_NAME)

# =====================================
# Request Schema
# =====================================

class ChatRequest(BaseModel):
    messages: list

# =====================================
# Token Estimation
# =====================================

def estimate_tokens(messages):
    total = 0
    for msg in messages:
        total += len(encoding.encode(msg["content"]))
    return total

# =====================================
# Summarization (Cost Protection)
# =====================================

def summarize_messages(messages):
    summary_prompt = [
        {"role": "system", "content": "Summarize this conversation briefly."},
        {"role": "user", "content": str(messages)}
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=summary_prompt,
        temperature=0.3,
    )

    summary_text = response.choices[0].message.content

    return [
        {"role": "system", "content": "Conversation summary so far:"},
        {"role": "system", "content": summary_text}
    ]

# =====================================
# Health Route
# =====================================

@app.get("/")
def health():
    return {"message": "Atla AI Agent running."}

# =====================================
# Generate Route
# =====================================

@app.post("/generate/")
def generate(request: ChatRequest):

    messages = request.messages

    if not messages:
        raise HTTPException(status_code=400, detail="Messages required")

    token_count = estimate_tokens(messages)

    if token_count > MAX_TOKENS_ALLOWED:
        raise HTTPException(
            status_code=413,
            detail="Conversation too long. Refresh chat."
        )

    if token_count > SUMMARIZE_TRIGGER:
        messages = summarize_messages(messages)

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.7,
        )

        return {"response": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))