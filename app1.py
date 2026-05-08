from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import os
import shutil
import sqlite3
import time
import re
from threading import Lock

try:
    import chromadb
except Exception:
    chromadb = None

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from mangum import Mangum
from openai import APIError, AuthenticationError, OpenAI, RateLimitError
from pydantic import BaseModel


app = FastAPI(title="AtlaOps AI Agent", version="4.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
EMBEDDING_MODEL = "text-embedding-3-small"

PROJECT_ROOT = Path(__file__).resolve().parent
CHROMA_PATH = Path(os.environ.get("CHROMA_PATH", "/tmp/atlaops_chroma_db"))
DB_PATH = PROJECT_ROOT / "obs_backend.db"
INDEX_HTML = PROJECT_ROOT / "index.html"
KB_SOURCE_DIRS = [PROJECT_ROOT / "docs" / "atlaops-kb", PROJECT_ROOT / "knowledge_base"]

request_log = defaultdict(list)
RATE_LIMIT = 20
WINDOW = 60
db_lock = Lock()

ALLOWED_INCIDENTS = {"normal", "traffic_spike", "db_errors", "recovery"}
ALLOWED_SITES = {"obs", "aif"}

SYSTEM_PROMPT = (
    "You are AtlaOps Guru, an AI cloud operations assistant built by Sai Pranav Atla. "
    "Give concise, technically precise answers. "
    "When knowledge-base context is provided, ground your answer in it and reference evidence briefly."
)

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

site_states = {
    "obs": {
        "incident": "normal",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "timeline": [
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "event": "OBS system initialized in healthy state.",
            }
        ],
    },
    "aif": {
        "incident": "normal",
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "timeline": [
            {
                "time": datetime.now(timezone.utc).isoformat(),
                "event": "AIF system initialized in healthy state.",
            }
        ],
    },
}

# Track incident start time for duration calculation per site
incident_start_times = {"obs": time.time(), "aif": time.time()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def embed_texts(texts: list[str]) -> list[list[float]]:
    if openai_client is None:
        return []
    try:
        response = openai_client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [item.embedding for item in response.data]
    except Exception:
        return []


def init_collections():
    if chromadb is None:
        return None, None

    try:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        user_memory = chroma_client.get_or_create_collection(name="user_memory")
        kb_docs = chroma_client.get_or_create_collection(name="atlaops_kb")
        return user_memory, kb_docs
    except Exception as exc:
        backup_dir = CHROMA_PATH.with_name(f"{CHROMA_PATH.name}_legacy_{int(time.time())}")
        if CHROMA_PATH.exists():
            shutil.move(str(CHROMA_PATH), str(backup_dir))
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        user_memory = chroma_client.get_or_create_collection(name="user_memory")
        kb_docs = chroma_client.get_or_create_collection(name="atlaops_kb")
        print(f"ChromaDB reset after init failure: {exc}. Backup: {backup_dir}")
        return user_memory, kb_docs


user_collection, kb_collection = init_collections()


def init_database():
    """Create SQLite database and schema if not present (idempotent)."""
    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            # Historical metrics table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS historical_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                incident_type TEXT NOT NULL,
                cpu_percent REAL,
                memory_percent REAL,
                latency_p95_ms REAL,
                error_rate_percent REAL,
                pod_count INTEGER,
                requests_per_min INTEGER,
                services_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_metrics_timestamp
              ON historical_metrics(timestamp)
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_historical_metrics_incident
              ON historical_metrics(incident_type)
            """)

            # Incident logs table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS incident_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                incident_type TEXT NOT NULL,
                event_type TEXT NOT NULL,
                event_message TEXT NOT NULL,
                triggered_by TEXT,
                related_metric_name TEXT,
                metric_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_incident_logs_timestamp
              ON incident_logs(timestamp)
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_incident_logs_incident_type
              ON incident_logs(incident_type)
            """)

            # Conversations table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_prompt TEXT NOT NULL,
                ai_response TEXT NOT NULL,
                incident_state TEXT NOT NULL,
                cpu_at_time REAL,
                latency_at_time REAL,
                error_rate_at_time REAL,
                kb_sources_json TEXT,
                response_ms INTEGER,
                tokens_used INTEGER,
                model TEXT DEFAULT 'gpt-4o-mini',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_timestamp
              ON conversations(timestamp)
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_incident
              ON conversations(incident_state)
            """)

            # Performance analytics table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                incident_type TEXT NOT NULL,
                cpu_avg REAL,
                cpu_max REAL,
                cpu_min REAL,
                memory_avg REAL,
                memory_max REAL,
                latency_avg REAL,
                latency_p99_ms REAL,
                error_rate_avg REAL,
                error_rate_max REAL,
                service_count_healthy INTEGER,
                service_count_degraded INTEGER,
                slo_latency_met INTEGER,
                slo_error_rate_met INTEGER,
                duration_seconds INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_timestamp
              ON performance_analytics(timestamp)
            """)
            cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_analytics_incident
              ON performance_analytics(incident_type)
            """)

            conn.commit()
            conn.close()
            print(f"Database initialized at {DB_PATH}")
        except Exception as exc:
            print(f"Database init error: {exc}")


def log_metrics_to_db(metrics_dict: dict):
    """Persist metrics snapshot to database."""
    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO historical_metrics (
                timestamp, incident_type, cpu_percent, memory_percent,
                latency_p95_ms, error_rate_percent, pod_count, requests_per_min,
                services_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                metrics_dict["timestamp"],
                metrics_dict["incident"],
                metrics_dict["metrics"]["cpu_percent"],
                metrics_dict["metrics"]["memory_percent"],
                metrics_dict["metrics"]["latency_p95_ms"],
                metrics_dict["metrics"]["error_rate_percent"],
                metrics_dict["metrics"]["pod_count"],
                metrics_dict["metrics"]["requests_per_min"],
                json.dumps(metrics_dict["services"])
            ))

            conn.commit()
            conn.close()
        except Exception as exc:
            print(f"Metrics logging error: {exc}")


def log_incident_event_to_db(event_message: str, triggered_by: str = None,
                            metric_name: str = None, metric_value: float = None):
    """Persist incident event to database."""
    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO incident_logs (
                timestamp, incident_type, event_type, event_message,
                triggered_by, related_metric_name, metric_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                utc_now(),
                ops_state["incident"],
                "milestone",
                event_message,
                triggered_by,
                metric_name,
                metric_value
            ))

            conn.commit()
            conn.close()
        except Exception as exc:
            print(f"Event logging error: {exc}")


def log_conversation_to_db(user_prompt: str, ai_response: str, metrics_snapshot: dict,
                          kb_sources: list, response_ms: int):
    """Persist conversation pair to database."""
    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            cursor.execute("""
            INSERT INTO conversations (
                timestamp, user_prompt, ai_response, incident_state,
                cpu_at_time, latency_at_time, error_rate_at_time,
                kb_sources_json, response_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                utc_now(),
                user_prompt,
                ai_response,
                ops_state["incident"],
                metrics_snapshot.get("cpu_percent"),
                metrics_snapshot.get("latency_p95_ms"),
                metrics_snapshot.get("error_rate_percent"),
                json.dumps(kb_sources),
                response_ms
            ))

            conn.commit()
            conn.close()
        except Exception as exc:
            print(f"Conversation logging error: {exc}")


def cleanup_old_data(retention_days: int = 30):
    """Remove metrics older than retention_days."""
    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()

            cursor.execute("DELETE FROM historical_metrics WHERE timestamp < ?", (cutoff,))
            cursor.execute("DELETE FROM incident_logs WHERE timestamp < ?", (cutoff,))

            # Keep conversations longer for audit (90 days)
            conv_cutoff = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
            cursor.execute("DELETE FROM conversations WHERE timestamp < ?", (conv_cutoff,))

            conn.commit()
            conn.close()
        except Exception as exc:
            print(f"Cleanup error: {exc}")


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not normalized:
        return []

    chunks = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        chunks.append(normalized[start:end])
        if end == len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def load_file_kb_chunks() -> list[dict]:
    chunks = []
    for kb_dir in KB_SOURCE_DIRS:
        if not kb_dir.exists():
            continue
        for path in sorted(kb_dir.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".md", ".txt"}:
                continue
            try:
                rel = path.relative_to(PROJECT_ROOT).as_posix()
                for idx, chunk in enumerate(chunk_text(path.read_text(encoding="utf-8"))):
                    chunks.append({"text": chunk, "source": rel, "chunk_index": idx})
            except Exception:
                continue
    return chunks


FILE_KB_CHUNKS = load_file_kb_chunks()
FILE_KB_EMBEDDINGS: list[list[float]] | None = None


def tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9][a-z0-9_-]{2,}", text.lower())}


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def retrieve_file_kb_context(query: str, top_k: int = 4):
    if not FILE_KB_CHUNKS:
        return "", []

    global FILE_KB_EMBEDDINGS

    query_embedding = embed_texts([query])
    if query_embedding:
        if FILE_KB_EMBEDDINGS is None:
            FILE_KB_EMBEDDINGS = embed_texts([item["text"] for item in FILE_KB_CHUNKS])
        if FILE_KB_EMBEDDINGS:
            scored = [
                (cosine_similarity(query_embedding[0], embedding), item)
                for item, embedding in zip(FILE_KB_CHUNKS, FILE_KB_EMBEDDINGS)
            ]
        else:
            scored = []
    else:
        query_tokens = tokenize(query)
        scored = [
            (len(query_tokens & tokenize(item["text"])), item)
            for item in FILE_KB_CHUNKS
        ]

    top = [item for score, item in sorted(scored, key=lambda row: row[0], reverse=True)[:top_k] if score > 0]
    if not top:
        top = FILE_KB_CHUNKS[:top_k]

    contexts = [
        f"[{item['source']}#chunk-{item['chunk_index']}] {item['text']}"
        for item in top
    ]
    sources = [
        {"source": item["source"], "chunk": item["chunk_index"]}
        for item in top
    ]
    return "\n\n".join(contexts), sources


def reset_user_collection():
    global user_collection

    if chromadb is None:
        user_collection = None
        return

    client_local = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client_local.delete_collection("user_memory")
    except Exception:
        pass
    user_collection = client_local.get_or_create_collection(name="user_memory")


class PromptRequest(BaseModel):
    prompt: str


class IncidentRequest(BaseModel):
    incident_type: str


def push_timeline(event: str, site: str = "obs") -> None:
    state = get_site_state(site)
    state["timeline"].insert(0, {"time": utc_now(), "event": event})
    state["timeline"] = state["timeline"][:30]
    state["updated_at"] = utc_now()

    # Also persist to database
    log_incident_event_to_db(event)


def get_ai_response(full_prompt: str) -> str:
    if openai_client is None:
        return "AI service configuration issue."

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": full_prompt},
            ],
            temperature=0.4,
            max_tokens=500,
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


# ============================================================================
# STEP 3: METRIC CORRELATION - Realistic metric interdependencies
# ============================================================================
def get_site_state(site: str) -> dict:
    return site_states.get(site, site_states["obs"])


def generate_metrics(site: str = "obs") -> dict:
    """
    Generate correlated metrics where incidents create cascading effects.

    Correlation logic:
    - traffic_spike: Traffic increases → CPU/Memory increase → Pods scale up →
                     Latency increases → Error rate increases
    - db_errors: Database errors → Memory spikes more than CPU →
                 Latency increases significantly → Error rate very high
    - recovery: All metrics smoothly return to baseline
    """
    t = time.time() / 6.0
    incident = get_site_state(site)["incident"]

    # Baseline sinusoidal curves (normal state)
    base_cpu = 42 + 8 * math.sin(t)
    base_memory = 58 + 6 * math.cos(t / 2)
    base_latency = 120 + 20 * math.sin(t / 1.3)
    base_error_rate = 0.4 + 0.2 * abs(math.sin(t / 1.1))
    base_pods = 6 + int(abs(math.sin(t / 1.8)) * 2)
    base_rps = 190 + int(abs(math.sin(t)) * 35)

    # Start with baselines
    cpu = base_cpu
    memory = base_memory
    latency = base_latency
    error_rate = base_error_rate
    pods = base_pods
    rps = base_rps

    if incident == "traffic_spike":
        # CORRELATION CHAIN 1: RPS spike → CPU → Memory → Pod scaling → Latency → Errors
        rps_multiplier = 2.8  # RPS increases significantly
        rps += 340

        # Traffic spike directly increases CPU and Memory
        cpu += 35  # Traffic handling overhead
        memory += 8  # RPS increases memory usage less than CPU

        # High CPU triggers pod scaling, which increases RPS but reduces per-pod load slightly
        pods += 5  # Autoscaler adds pods

        # Even with scaling, latency increases due to network contention and queueing
        latency += 120  # Network bottleneck, request queueing

        # Under load, error rate increases (timeouts, circuit breakers)
        error_rate += 1.4  # Timeout errors, failed connections

    elif incident == "db_errors":
        # CORRELATION CHAIN 2: DB errors → Memory thrashing → CPU increases → Latency → Errors
        # Database errors cause connection pool exhaustion and memory pressure

        cpu += 12   # CPU needed for retry logic and error handling
        memory += 10  # Retry buffers, error logs, connection queue memory > CPU increase

        # DB errors don't trigger autoscaling the same way (fewer pods helps sometimes)
        pods += 1   # Minimal scaling; the issue is backend, not frontend load

        # Checkout service becomes severely latent due to DB timeouts
        latency += 85  # DB query timeouts (not as high as traffic spike)

        # Error rate spikes much more than traffic spike (database is critical path)
        error_rate += 3.2  # Payment failures, checkout aborts

    elif incident == "recovery":
        # Smooth decay back to baseline
        recovery_factor = 0.15  # Reduce the delta by 15% each call

        cpu = base_cpu + (cpu - base_cpu) * (1 - recovery_factor)
        memory = base_memory + (memory - base_memory) * (1 - recovery_factor)
        latency = base_latency + (latency - base_latency) * (1 - recovery_factor)
        error_rate = max(base_error_rate, base_error_rate + (error_rate - base_error_rate) * (1 - recovery_factor))
        pods = max(base_pods, int(pods * (1 - recovery_factor * 0.5)))  # Pods scale down more slowly
        rps = base_rps + (rps - base_rps) * (1 - recovery_factor)

    # Clamp to realistic ranges
    cpu = max(5, min(99, round(cpu, 1)))
    memory = max(10, min(99, round(memory, 1)))
    latency = max(40, round(latency, 1))
    error_rate = max(0.0, round(error_rate, 2))
    pods = max(3, int(pods))
    rps = max(100, int(rps))

    # Service status reflects the incident state
    checkout_status = "degraded" if incident == "db_errors" else "healthy"
    api_status = "degraded" if incident in {"traffic_spike", "db_errors"} else "healthy"

    # Service latencies scale with overall latency
    services = [
        {"name": "api-gateway", "status": api_status, "latency_ms": round(latency * 0.9, 1)},
        {"name": "orders-service", "status": checkout_status, "latency_ms": latency},
        {"name": "payments-worker", "status": checkout_status, "latency_ms": round(latency * 1.1, 1)},
        {"name": "ops-guru-rag", "status": "healthy", "latency_ms": round(latency * 0.75, 1)},
    ]

    metrics_dict = {
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

    # STEP 4: Log metrics to database for persistence & analytics
    log_metrics_to_db(metrics_dict)

    return metrics_dict


def generate_logs(limit: int, site: str = "obs") -> list[dict]:
    incident = get_site_state(site)["incident"]
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
        logs.append({"time": utc_now(), "line": entry})
    return logs


def build_ops_context(site: str = "obs") -> str:
    metrics_payload = generate_metrics(site)
    metrics = metrics_payload["metrics"]
    state = get_site_state(site)
    recent_logs = generate_logs(4, site)
    recent_events = state["timeline"][:3]

    log_lines = " | ".join(log["line"] for log in recent_logs)
    timeline_lines = " | ".join(event["event"] for event in recent_events)

    return (
        f"Site: {site}. Incident mode: {state['incident']}. "
        f"CPU={metrics['cpu_percent']}%, Memory={metrics['memory_percent']}%, "
        f"P95 Latency={metrics['latency_p95_ms']}ms, Errors={metrics['error_rate_percent']}%, "
        f"Pods={metrics['pod_count']}, RPM={metrics['requests_per_min']}. "
        f"Recent timeline: {timeline_lines}. "
        f"Recent logs: {log_lines}."
    )


def retrieve_kb_context(query: str, top_k: int = 4):
    if kb_collection is None or kb_collection.count() == 0:
        return retrieve_file_kb_context(query, top_k=top_k)

    try:
        query_embedding = embed_texts([query])
        if not query_embedding:
            return "", []
        results = kb_collection.query(query_embeddings=query_embedding, n_results=top_k)
    except Exception:
        return retrieve_file_kb_context(query, top_k=top_k)
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    contexts = []
    sources = []

    for idx, doc in enumerate(documents):
        meta = metadatas[idx] if idx < len(metadatas) and metadatas[idx] else {}
        source = meta.get("source", "unknown")
        chunk_index = meta.get("chunk_index", "?")
        contexts.append(f"[{source}#chunk-{chunk_index}] {doc}")
        sources.append({"source": source, "chunk": chunk_index})

    context_block = "\n\n".join(contexts)
    dedup = []
    seen = set()
    for source in sources:
        key = (source["source"], str(source["chunk"]))
        if key in seen:
            continue
        seen.add(key)
        dedup.append(source)

    return context_block, dedup


def retrieve_memory_context(query: str, top_k: int = 2) -> str:
    if user_collection is None or user_collection.count() == 0:
        return ""
    try:
        query_embedding = embed_texts([query])
        if not query_embedding:
            return ""
        results = user_collection.query(query_embeddings=query_embedding, n_results=top_k)
    except Exception:
        return ""
    documents = results.get("documents", [[]])[0]
    if not documents:
        return ""
    return "\n".join(documents)


@app.on_event("startup")
async def startup_event():
    """Initialize database and other resources on app startup."""
    init_database()
    print("AtlaOps backend v4.2.0 started with database persistence enabled")


@app.get("/")
def root():
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return {"message": "AtlaOps backend is running."}


@app.get("/health")
def health():
    chroma_kb_count = kb_collection.count() if kb_collection is not None else 0
    return {
        "status": "ok",
        "backend": "openai" if OPENAI_API_KEY else "none",
        "rate_limit": f"{RATE_LIMIT} req/{WINDOW}s",
        "version": "4.2.0",
        "incident": ops_state["incident"],
        "kb_chunks": chroma_kb_count or len(FILE_KB_CHUNKS),
        "database": "sqlite" if DB_PATH.exists() else "not_initialized",
    }


@app.get("/ops/metrics")
def ops_metrics(site: str = Query("obs", description="Site key: obs or aif")):
    if site not in ALLOWED_SITES:
        raise HTTPException(status_code=400, detail="Unsupported site")
    return generate_metrics(site)


@app.get("/ops/logs")
def ops_logs(site: str = Query("obs", description="Site key: obs or aif"), limit: int = Query(default=20, ge=5, le=100)):
    if site not in ALLOWED_SITES:
        raise HTTPException(status_code=400, detail="Unsupported site")
    state = get_site_state(site)
    return {"incident": state["incident"], "site": site, "logs": generate_logs(limit, site)}


@app.get("/ops/incidents")
def ops_incidents(site: str = Query("obs", description="Site key: obs or aif")):
    if site not in ALLOWED_SITES:
        raise HTTPException(status_code=400, detail="Unsupported site")
    state = get_site_state(site)
    return {
        "incident": state["incident"],
        "site": site,
        "updated_at": state["updated_at"],
        "timeline": state["timeline"],
    }


@app.post("/ops/incidents/trigger")
def trigger_incident(payload: IncidentRequest, site: str = Query("obs", description="Site key: obs or aif")):
    global incident_start_times
    if site not in ALLOWED_SITES:
        raise HTTPException(status_code=400, detail="Unsupported site")

    incident_type = payload.incident_type.strip().lower()
    if incident_type not in ALLOWED_INCIDENTS:
        raise HTTPException(status_code=400, detail="Unsupported incident type")

    state = get_site_state(site)
    state["incident"] = incident_type
    state["updated_at"] = utc_now()
    incident_start_times[site] = time.time()

    if incident_type == "traffic_spike":
        push_timeline("Traffic spike simulation started. Autoscaling initiated.", site)
    elif incident_type == "db_errors":
        push_timeline("Database error burst simulated. Checkout degradation detected.", site)
    elif incident_type == "recovery":
        push_timeline("Recovery workflow simulated. Services stabilizing.", site)
    else:
        push_timeline("System returned to normal baseline.", site)

    return {"ok": True, "incident": state["incident"], "site": site, "updated_at": state["updated_at"]}


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


@app.get("/ops/kb/status")
def kb_status():
    chroma_kb_count = kb_collection.count() if kb_collection is not None else 0
    return {
        "kb_chunks": chroma_kb_count or len(FILE_KB_CHUNKS),
        "file_kb_chunks": len(FILE_KB_CHUNKS),
        "memory_chunks": user_collection.count() if user_collection is not None else 0,
    }


@app.get("/ops/incidents/rca")
def incident_rca(site: str = Query("obs", description="Site key: obs or aif")):
    if site not in ALLOWED_SITES:
        raise HTTPException(status_code=400, detail="Unsupported site")
    state = get_site_state(site)
    incident = state["incident"]
    metrics = generate_metrics(site)["metrics"]
    recent_logs = [entry["line"] for entry in generate_logs(8, site)]
    recent_events = [entry["event"] for entry in state["timeline"][:4]]

    if incident == "traffic_spike":
        summary = "Traffic surge caused latency amplification and autoscaling pressure."
        likely_root_cause = "Request rate exceeded baseline, saturating API gateway and service pods."
        mitigation = [
            "Scale out stateless services and verify autoscaler thresholds.",
            "Apply temporary rate limiting for abusive clients.",
            "Tune cache and edge TTL for high-read paths.",
        ]
    elif incident == "db_errors":
        summary = "Checkout path degradation driven by database timeouts."
        likely_root_cause = "Orders database query latency and retries increased error propagation."
        mitigation = [
            "Investigate slow queries and connection pool saturation.",
            "Enable circuit-breaker behavior for failing DB dependencies.",
            "Shift read-heavy paths to cache and validate retry/backoff config.",
        ]
    elif incident == "recovery":
        summary = "System is in recovery mode after mitigation workflow."
        likely_root_cause = "Prior incident signals are stabilizing after remediation actions."
        mitigation = [
            "Keep elevated monitoring until latency and error trends fully normalize.",
            "Run post-incident validation checks on dependent services.",
            "Document timeline and finalize postmortem actions.",
        ]
    else:
        summary = "No active incident detected; platform is operating at baseline."
        likely_root_cause = "N/A"
        mitigation = [
            "Maintain baseline observability and alert hygiene.",
            "Run periodic failure drills to validate runbooks.",
            "Review capacity thresholds before peak traffic windows.",
        ]

    return {
        "incident": incident,
        "site": site,
        "generated_at": utc_now(),
        "summary": summary,
        "likely_root_cause": likely_root_cause,
        "signals": {
            "cpu_percent": metrics["cpu_percent"],
            "memory_percent": metrics["memory_percent"],
            "latency_p95_ms": metrics["latency_p95_ms"],
            "error_rate_percent": metrics["error_rate_percent"],
            "pod_count": metrics["pod_count"],
            "requests_per_min": metrics["requests_per_min"],
        },
        "recent_events": recent_events,
        "recent_logs": recent_logs,
        "mitigation_plan": mitigation,
    }


@app.post("/generate/")
async def generate_text(request: Request, prompt_req: PromptRequest):
    try:
        if not OPENAI_API_KEY:
            return {
                "response": "OPENAI_API_KEY is not configured in this running server process.",
                "sources": [],
                "incident": ops_state["incident"],
            }

        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()

        request_log[client_ip] = [t for t in request_log[client_ip] if current_time - t < WINDOW]
        if len(request_log[client_ip]) >= RATE_LIMIT:
            raise HTTPException(status_code=429, detail="Too many requests. Please slow down.")
        request_log[client_ip].append(current_time)

        user_message = prompt_req.prompt.strip()
        if not user_message:
            raise HTTPException(status_code=400, detail="Prompt is required.")

        start_time = time.time()

        ops_context = build_ops_context()
        kb_context, sources = retrieve_kb_context(user_message, top_k=4)
        memory_context = retrieve_memory_context(user_message, top_k=2)

        full_prompt = (
            f"Current ops state:\n{ops_context}\n\n"
            f"Knowledge base context:\n{kb_context or 'No KB chunks found.'}\n\n"
            f"Conversation memory:\n{memory_context or 'No prior memory found.'}\n\n"
            f"User question: {user_message}\n\n"
            "Instructions: use the context above when relevant, be explicit about incident signals, "
            "and avoid claims not supported by the provided context."
        )

        ai_response = get_ai_response(full_prompt)
        response_ms = int((time.time() - start_time) * 1000)

        # Get current metrics for context snapshot
        current_metrics = generate_metrics()["metrics"]

        # Log conversation to database
        log_conversation_to_db(user_message, ai_response, current_metrics, sources, response_ms)

        # Store in vector memory as well
        memory_doc = f"User asked: {user_message}. AtlaOps Guru replied: {ai_response}"
        memory_embeddings = embed_texts([memory_doc])
        if user_collection is not None and memory_embeddings:
            try:
                user_collection.add(
                    documents=[memory_doc],
                    embeddings=memory_embeddings,
                    metadatas=[{"source": "conversation", "time": utc_now()}],
                    ids=[f"conv_{time.time()}"],
                )
            except Exception as exc:
                if "dimensionality" in str(exc).lower():
                    reset_user_collection()
                    if user_collection is not None:
                        user_collection.add(
                            documents=[memory_doc],
                            embeddings=memory_embeddings,
                            metadatas=[{"source": "conversation", "time": utc_now()}],
                            ids=[f"conv_{time.time()}"],
                        )

        return {
            "response": ai_response,
            "sources": sources,
            "incident": ops_state["incident"],
        }
    except HTTPException:
        raise
    except Exception as exc:
        return {
            "response": f"Ops Guru backend error: {type(exc).__name__}: {exc}",
            "sources": [],
            "incident": ops_state["incident"],
        }


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.get("/analytics/metrics-history")
def metrics_history(
    start: str = Query(..., description="ISO 8601 start timestamp"),
    end: str = Query(..., description="ISO 8601 end timestamp"),
    incident_type: str | None = Query(None, description="Filter by incident type")
):
    """Return historical metric snapshots for charting/analysis."""
    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM historical_metrics WHERE timestamp BETWEEN ? AND ?"
            params = [start, end]

            if incident_type:
                query += " AND incident_type = ?"
                params.append(incident_type)

            query += " ORDER BY timestamp DESC LIMIT 1000"

            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return {"metrics": rows, "count": len(rows)}
        except Exception as exc:
            return {"metrics": [], "count": 0, "error": str(exc)}


@app.get("/analytics/incident-timeline")
def incident_timeline(
    start: str = Query(..., description="ISO 8601 start timestamp"),
    end: str = Query(..., description="ISO 8601 end timestamp"),
    incident_type: str | None = Query(None, description="Filter by incident type")
):
    """Return structured incident events with context."""
    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            query = "SELECT * FROM incident_logs WHERE timestamp BETWEEN ? AND ?"
            params = [start, end]

            if incident_type:
                query += " AND incident_type = ?"
                params.append(incident_type)

            query += " ORDER BY timestamp DESC"

            cursor.execute(query, params)
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()

            return {"events": rows, "count": len(rows)}
        except Exception as exc:
            return {"events": [], "count": 0, "error": str(exc)}


@app.get("/analytics/slo-report")
def slo_report(
    start: str = Query(..., description="ISO 8601 start timestamp"),
    end: str = Query(..., description="ISO 8601 end timestamp")
):
    """Return SLO compliance metrics."""
    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            cursor.execute("""
            SELECT
                COUNT(*) as total_hours,
                SUM(CASE WHEN slo_latency_met = 1 THEN 1 ELSE 0 END) as latency_met_hours,
                SUM(CASE WHEN slo_error_rate_met = 1 THEN 1 ELSE 0 END) as error_rate_met_hours
            FROM performance_analytics
            WHERE timestamp BETWEEN ? AND ?
            """, (start, end))

            row = cursor.fetchone()
            conn.close()

            total = row[0] or 0
            latency_met = row[1] or 0
            error_rate_met = row[2] or 0

            return {
                "slo_latency_compliance_percent": (latency_met / total * 100) if total > 0 else 0,
                "slo_error_rate_compliance_percent": (error_rate_met / total * 100) if total > 0 else 0,
                "total_hours_tracked": total
            }
        except Exception as exc:
            return {"error": str(exc)}


@app.get("/analytics/conversation-stats")
def conversation_stats(
    start: str = Query(..., description="ISO 8601 start timestamp"),
    end: str = Query(..., description="ISO 8601 end timestamp")
):
    """Return conversation statistics and KB usage metrics."""
    with db_lock:
        try:
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()

            cursor.execute("""
            SELECT
                COUNT(*) as conversation_count,
                AVG(response_ms) as avg_response_ms,
                MIN(response_ms) as min_response_ms,
                MAX(response_ms) as max_response_ms,
                incident_state
            FROM conversations
            WHERE timestamp BETWEEN ? AND ?
            GROUP BY incident_state
            """, (start, end))

            rows = cursor.fetchall()
            conn.close()

            stats = []
            for row in rows:
                stats.append({
                    "incident_state": row[4],
                    "conversation_count": row[0],
                    "avg_response_ms": round(row[1], 2) if row[1] else 0,
                    "min_response_ms": row[2] or 0,
                    "max_response_ms": row[3] or 0
                })

            return {"stats": stats}
        except Exception as exc:
            return {"stats": [], "error": str(exc)}


handler = Mangum(app)
