from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import math
import os
import shutil
import time

import chromadb
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import APIError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel


app = FastAPI(title="AtlaOps AI Agent", version="4.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


CHROMA_PATH = Path("./chroma_db")


def init_collection():
    try:
        client_local = chromadb.PersistentClient(path=str(CHROMA_PATH))
        return client_local.get_or_create_collection(name="user_memory")
    except Exception as exc:
        backup_dir = Path(f"./chroma_db_legacy_{int(time.time())}")
        if CHROMA_PATH.exists():
            shutil.move(str(CHROMA_PATH), str(backup_dir))
        client_local = chromadb.PersistentClient(path=str(CHROMA_PATH))
        coll = client_local.get_or_create_collection(name="user_memory")
        print(f"ChromaDB reset after init failure: {exc}. Backup: {backup_dir}")
        return coll


collection = init_collection()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
client = OpenAI(api_key=OPENAI_API_KEY)

request_log = defaultdict(list)
RATE_LIMIT = 20
WINDOW = 60

SYSTEM_PROMPT = (
    "You are AtlaOps Guru, an AI cloud operations assistant built by Sai Pranav Atla. "
    "Answer with operational clarity, reference architecture tradeoffs, and keep replies concise "
    "unless the user asks for a deep dive."
)


class PromptRequest(BaseModel):
    prompt: str


class IncidentRequest(BaseModel):
    incident_type: str


PROJECT_ROOT = Path(__file__).resolve().parent
INDEX_HTML = PROJECT_ROOT / "index.html"

ALLOWED_INCIDENTS = {"normal", "traffic_spike", "db_errors", "recovery"}

ops_state = {
    "incident": "normal",
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "timeline": [
        {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": "System initialized in healthy state.",
        }
    ],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def push_timeline(event: str) -> None:
    ops_state["timeline"].insert(0, {"time": utc_now(), "event": event})
    ops_state["timeline"] = ops_state["timeline"][:30]
    ops_state["updated_at"] = utc_now()


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
            temperature=0.5,
            max_tokens=450,
        )
        return response.choices[0].message.content
    except RateLimitError:
        return "LLM rate limit reached. Please retry in a few seconds."
    except AuthenticationError:
        return "AI service authentication failed."
    except APIError:
        return "AI service currently unavailable."
    except Exception:
        return "Unexpected AI service error."


def generate_metrics() -> dict:
    t = time.time() / 6.0
    incident = ops_state["incident"]

    cpu = 42 + 8 * math.sin(t)
    memory = 58 + 6 * math.cos(t / 2)
    latency = 120 + 20 * math.sin(t / 1.3)
    error_rate = 0.4 + 0.2 * abs(math.sin(t / 1.1))
    pods = 6 + int(abs(math.sin(t / 1.8)) * 2)
    rps = 190 + int(abs(math.sin(t)) * 35)

    if incident == "traffic_spike":
        cpu += 35
        latency += 120
        error_rate += 1.4
        pods += 5
        rps += 340
    elif incident == "db_errors":
        cpu += 12
        memory += 10
        latency += 85
        error_rate += 3.2
    elif incident == "recovery":
        cpu -= 8
        latency -= 20
        error_rate -= 0.2

    cpu = max(5, min(99, round(cpu, 1)))
    memory = max(10, min(99, round(memory, 1)))
    latency = max(40, round(latency, 1))
    error_rate = max(0.0, round(error_rate, 2))

    checkout_status = "degraded" if incident == "db_errors" else "healthy"
    api_status = "degraded" if incident in {"traffic_spike", "db_errors"} else "healthy"

    services = [
        {"name": "api-gateway", "status": api_status, "latency_ms": round(latency * 0.9, 1)},
        {"name": "orders-service", "status": checkout_status, "latency_ms": latency},
        {"name": "payments-worker", "status": checkout_status, "latency_ms": round(latency * 1.1, 1)},
        {"name": "ops-guru-rag", "status": "healthy", "latency_ms": round(latency * 0.75, 1)},
    ]

    return {
        "timestamp": utc_now(),
        "incident": incident,
        "metrics": {
            "cpu_percent": cpu,
            "memory_percent": memory,
            "latency_p95_ms": latency,
            "error_rate_percent": error_rate,
            "pod_count": pods,
            "requests_per_min": rps,
        },
        "services": services,
    }


def generate_logs(limit: int) -> list[dict]:
    incident = ops_state["incident"]
    base = [
        "INFO api-gateway request completed route=/health status=200",
        "INFO orders-service cache hit ratio=0.93",
        "INFO payments-worker batch settled count=21",
        "INFO ops-guru-rag context chunks=4 retrieval_ms=41",
    ]

    if incident == "traffic_spike":
        base.extend(
            [
                "WARN autoscaler scale_out pods=+3 reason=cpu_above_threshold",
                "WARN api-gateway latency elevated p95=290ms",
                "ALERT cloudwatch HighRequestRate triggered",
            ]
        )
    elif incident == "db_errors":
        base.extend(
            [
                "ERROR orders-db timeout query=SELECT * FROM orders",
                "ERROR payments-worker retry exhausted payment_id=py_8172",
                "ALERT cloudwatch DatabaseErrorRate triggered",
            ]
        )
    elif incident == "recovery":
        base.extend(
            [
                "INFO incident-automation remediation playbook completed",
                "INFO api-gateway latency recovered p95=128ms",
                "RESOLVED cloudwatch alarms back to normal",
            ]
        )

    logs = []
    for i in range(limit):
        entry = base[i % len(base)]
        logs.append({
            "time": utc_now(),
            "line": entry,
        })
    return logs


def build_ops_context() -> str:
    metrics_payload = generate_metrics()
    metrics = metrics_payload["metrics"]
    recent_logs = generate_logs(4)
    recent_events = ops_state["timeline"][:3]

    log_lines = " | ".join(log["line"] for log in recent_logs)
    timeline_lines = " | ".join(event["event"] for event in recent_events)

    return (
        f"Incident mode: {ops_state['incident']}. "
        f"CPU={metrics['cpu_percent']}%, Memory={metrics['memory_percent']}%, "
        f"P95 Latency={metrics['latency_p95_ms']}ms, Errors={metrics['error_rate_percent']}%, "
        f"Pods={metrics['pod_count']}, RPM={metrics['requests_per_min']}. "
        f"Recent timeline: {timeline_lines}. "
        f"Recent logs: {log_lines}."
    )


@app.get("/")
def root():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return {"message": "AtlaOps backend is running."}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "backend": "openai" if OPENAI_API_KEY else "none",
        "rate_limit": f"{RATE_LIMIT} req/{WINDOW}s",
        "version": "4.0.0",
        "incident": ops_state["incident"],
    }


@app.get("/ops/metrics")
def ops_metrics():
    return generate_metrics()


@app.get("/ops/logs")
def ops_logs(limit: int = Query(default=20, ge=5, le=100)):
    return {
        "incident": ops_state["incident"],
        "logs": generate_logs(limit),
    }


@app.get("/ops/incidents")
def ops_incidents():
    return {
        "incident": ops_state["incident"],
        "updated_at": ops_state["updated_at"],
        "timeline": ops_state["timeline"],
    }


@app.post("/ops/incidents/trigger")
def trigger_incident(payload: IncidentRequest):
    incident_type = payload.incident_type.strip().lower()
    if incident_type not in ALLOWED_INCIDENTS:
        raise HTTPException(status_code=400, detail="Unsupported incident type")

    ops_state["incident"] = incident_type
    if incident_type == "traffic_spike":
        push_timeline("Traffic spike simulation started. Autoscaling initiated.")
    elif incident_type == "db_errors":
        push_timeline("Database error burst simulated. Checkout degradation detected.")
    elif incident_type == "recovery":
        push_timeline("Recovery workflow simulated. Services stabilizing.")
    else:
        push_timeline("System returned to normal baseline.")

    return {
        "ok": True,
        "incident": ops_state["incident"],
        "updated_at": ops_state["updated_at"],
    }


@app.get("/ops/architecture")
def ops_architecture():
    return {
        "nodes": [
            "Route53",
            "CloudFront",
            "S3 Frontend",
            "API Gateway",
            "Lambda AtlaOps API",
            "OpenAI LLM",
            "Vector Store",
            "CloudWatch",
        ],
        "edges": [
            ["Route53", "CloudFront"],
            ["CloudFront", "S3 Frontend"],
            ["CloudFront", "API Gateway"],
            ["API Gateway", "Lambda AtlaOps API"],
            ["Lambda AtlaOps API", "OpenAI LLM"],
            ["Lambda AtlaOps API", "Vector Store"],
            ["Lambda AtlaOps API", "CloudWatch"],
        ],
    }


@app.post("/generate/")
async def generate_text(request: Request, prompt_req: PromptRequest):
    client_ip = request.client.host if request.client else "unknown"
    current_time = time.time()

    request_log[client_ip] = [
        t for t in request_log[client_ip]
        if current_time - t < WINDOW
    ]

    if len(request_log[client_ip]) >= RATE_LIMIT:
        raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")

    request_log[client_ip].append(current_time)

    user_message = prompt_req.prompt
    results = collection.query(query_texts=[user_message], n_results=3)

    memories = (
        " ".join(results["documents"][0])
        if results.get("documents") and results["documents"][0]
        else ""
    )

    incident_context = build_ops_context()
    full_prompt = (
        f"{incident_context}\nRelevant context from memory:\n{memories}\n\nUser: {user_message}"
        if memories
        else f"{incident_context}\nUser: {user_message}"
    )

    ai_response = get_ai_response(full_prompt)

    collection.add(
        documents=[f"User asked: {user_message}. AtlaOps Guru replied: {ai_response}"],
        ids=[f"conv_{time.time()}"],
    )

    return {"response": ai_response}
