from fastapi import FastAPI, Request
from pydantic import BaseModel
import requests
import chromadb
import time
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 1. Enable CORS so your website can talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Initialize Permanent Memory (ChromaDB)
# This creates a folder called 'chroma_db' in your project directory
db_client = chromadb.PersistentClient(path="./chroma_db")
collection = db_client.get_or_create_collection(name="user_memory")

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate_text(prompt_req: PromptRequest):
    user_message = prompt_req.prompt

    # A. SEARCH: Now it will find the "CloudStream" info you just ingested
    results = collection.query(
        query_texts=[user_message],
        n_results=2
    )
    
    memories = " ".join(results['documents'][0]) if results['documents'] and results['documents'][0] else ""
    
    # NEW: Instruction to keep the AI professional and brief
    system_instruction = "You are a professional assistant for CloudStream Solutions. Keep answers under 3 sentences."
    
    full_prompt = f"{system_instruction}\nContext: {memories}\nUser: {user_message}"

    # C. LLM CALL: Send the context + question to your local Llama 3
    payload = {
        "model": MODEL_NAME,
        "prompt": full_prompt,
        "stream": False
    }
    
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    ai_response = data.get("response", "")

    # D. SAVE: Store this new interaction with a unique timestamp ID
    # This ensures the 'chroma_db' folder gets created and updated
    collection.add(
        documents=[f"User said: {user_message}. AI replied: {ai_response}"],
        ids=[f"id_{time.time()}"] 
    )

    return {"response": ai_response}