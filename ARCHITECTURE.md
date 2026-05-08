# obs.atla.in Architecture v4.2.0

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FRONTEND (index.html)                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Dark Theme Dashboard with Real-time Metrics                     │   │
│  │ ┌─────────────────────────────────────────────────────────────┐ │   │
│  │ │ Metric Gauges: CPU, Memory, Latency, Error Rate, Pods, RPS │ │   │
│  │ │ Service Status: api-gateway, orders-service, payments      │ │   │
│  │ │ Scenario Triggers: traffic_spike, db_errors, recovery      │ │   │
│  │ │ Timeline Events: Real-time incident progression            │ │   │
│  │ │ Log Viewer: Live system logs                               │ │   │
│  │ │ AI Chat: AtlaOps Guru conversation interface               │ │   │
│  │ └─────────────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
                        (HTTP Requests & Responses)
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                    BACKEND (app1.py) - FastAPI                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Core Metrics Engine (STEP 3: CORRELATED METRICS)                │   │
│  │ ┌────────────────────────────────────────────────────────────┐  │   │
│  │ │ generate_metrics()                                         │  │   │
│  │ │ - Baseline sinusoidal curves (normal operation)            │  │   │
│  │ │ - Incident-based multipliers:                             │  │   │
│  │ │   * traffic_spike: RPS→CPU→Memory→Pods→Latency→Errors   │  │   │
│  │ │   * db_errors: Memory thrashing > CPU; high error impact  │  │   │
│  │ │   * recovery: Smooth exponential decay to baseline        │  │   │
│  │ │ - Service status reflection                               │  │   │
│  │ └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ API Endpoints                                                    │   │
│  │ ┌────────────────────────────────────────────────────────────┐  │   │
│  │ │ /ops/metrics              → generate_metrics()             │  │   │
│  │ │ /ops/logs                 → incident-specific logs         │  │   │
│  │ │ /ops/incidents            → current incident state         │  │   │
│  │ │ /ops/incidents/trigger    → trigger incident scenario      │  │   │
│  │ │ /ops/incidents/rca        → root cause analysis            │  │   │
│  │ │ /ops/architecture         → system architecture graph      │  │   │
│  │ │                                                             │  │   │
│  │ │ [NEW] /generate/          → AI conversation endpoint       │  │   │
│  │ │ [NEW] /analytics/metrics-history → historical metrics      │  │   │
│  │ │ [NEW] /analytics/incident-timeline → event timeline        │  │   │
│  │ │ [NEW] /analytics/slo-report → SLO compliance stats        │  │   │
│  │ │ [NEW] /analytics/conversation-stats → AI usage metrics    │  │   │
│  │ └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                    ↓                                     │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Data Persistence Layer (STEP 4: DATABASE INTEGRATION)           │   │
│  │ ┌────────────────────────────────────────────────────────────┐  │   │
│  │ │ log_metrics_to_db()       → historical_metrics table       │  │   │
│  │ │ log_incident_event_to_db()→ incident_logs table           │  │   │
│  │ │ log_conversation_to_db()  → conversations table           │  │   │
│  │ │ compute_hourly_analytics()→ performance_analytics table   │  │   │
│  │ │                                                             │  │   │
│  │ │ Thread-safe database access with db_lock                  │  │   │
│  │ │ Graceful error handling (logs errors, continues)          │  │   │
│  │ │ Automatic data retention cleanup (30/90 days)             │  │   │
│  │ └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│              DATA LAYER (obs_backend.db) - SQLite 3.x                   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │ Table: historical_metrics (Time-Series)                         │   │
│  │ ┌────────────────────────────────────────────────────────────┐  │   │
│  │ │ Columns: timestamp, incident_type, cpu%, memory%,         │  │   │
│  │ │          latency_p95_ms, error_rate%, pod_count, rps     │  │   │
│  │ │ Index: timestamp, incident_type                           │  │   │
│  │ │ Growth: ~1 row / 2-3 seconds (when /ops/metrics called)  │  │   │
│  │ │ Retention: 30 days (auto-cleanup)                         │  │   │
│  │ │ Use Case: Historical metric queries, charting             │  │   │
│  │ └────────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │ Table: incident_logs (Event Timeline)                           │   │
│  │ ┌────────────────────────────────────────────────────────────┐  │   │
│  │ │ Columns: timestamp, incident_type, event_type,           │  │   │
│  │ │          event_message, triggered_by, metric_name, value  │  │   │
│  │ │ Index: timestamp, incident_type                           │  │   │
│  │ │ Growth: ~1 row per timeline event (10-50/day)            │  │   │
│  │ │ Retention: 30 days (auto-cleanup)                         │  │   │
│  │ │ Use Case: Incident replay, causality analysis             │  │   │
│  │ └────────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │ Table: conversations (Audit Trail)                              │   │
│  │ ┌────────────────────────────────────────────────────────────┐  │   │
│  │ │ Columns: timestamp, user_prompt, ai_response,            │  │   │
│  │ │          incident_state, cpu_at_time, latency_at_time,   │  │   │
│  │ │          kb_sources, response_ms, tokens_used            │  │   │
│  │ │ Index: timestamp, incident_state                          │  │   │
│  │ │ Growth: ~1 row per AI conversation (0-100/day)           │  │   │
│  │ │ Retention: 90 days (longer for audit)                    │  │   │
│  │ │ Use Case: Conversation replay, KB effectiveness tracking  │  │   │
│  │ └────────────────────────────────────────────────────────────┘  │   │
│  │                                                                  │   │
│  │ Table: performance_analytics (Aggregates)                       │   │
│  │ ┌────────────────────────────────────────────────────────────┐  │   │
│  │ │ Columns: timestamp (hour-level), incident_type,          │  │   │
│  │ │          cpu_avg/max/min, latency_avg/p99,               │  │   │
│  │ │          slo_latency_met, slo_error_rate_met             │  │   │
│  │ │ Index: timestamp, incident_type                           │  │   │
│  │ │ Growth: ~1 row per hour (24 rows/day)                    │  │   │
│  │ │ Retention: 30 days (auto-cleanup)                         │  │   │
│  │ │ Use Case: Fast SLO reporting, hourly aggregates           │  │   │
│  │ └────────────────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────┐
│                  EXTERNAL SERVICES (Optional)                           │
│  ├─ OpenAI GPT-4o-mini (AI responses)                                  │
│  ├─ ChromaDB (Vector embeddings for RAG)                               │
│  └─ Knowledge Base (Markdown docs in /docs/atlaops-kb)                 │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow: Incident Trigger

```
User clicks "Traffic Spike" button
    ↓
Frontend: fetch(/ops/incidents/trigger, {incident_type: "traffic_spike"})
    ↓
Backend: /ops/incidents/trigger endpoint
    ↓
    ├─ Update ops_state["incident"] = "traffic_spike"
    ├─ Call push_timeline("Traffic spike simulation started...")
    │  ├─ Add to ops_state["timeline"] (in-memory)
    │  └─ Call log_incident_event_to_db() [THREAD-SAFE]
    │     └─ INSERT into incident_logs table
    ├─ Record incident_start_time = now()
    └─ Return {"ok": true, ...}
    ↓
Frontend: Display timeline event + highlight scenario button
    ↓
Next /ops/metrics call:
    ├─ generate_metrics() detects incident == "traffic_spike"
    │  ├─ Base metrics: CPU=42%, Memory=58%, Latency=120ms
    │  ├─ APPLY CORRELATION MULTIPLIERS:
    │  │  ├─ RPS += 340 (traffic surge)
    │  │  ├─ CPU += 35 (handling overhead)
    │  │  ├─ Memory += 8 (buffering)
    │  │  ├─ Pods += 5 (autoscaling)
    │  │  ├─ Latency += 120 (queueing)
    │  │  └─ Errors += 1.4 (timeouts)
    │  ├─ CLAMP values to realistic ranges
    │  └─ Calculate service statuses
    ├─ Metrics: {CPU: 77%, Memory: 65%, Latency: 245ms, Errors: 1.8%, ...}
    └─ Call log_metrics_to_db(metrics_dict) [THREAD-SAFE]
       └─ INSERT into historical_metrics table
    ↓
Frontend: Fetch /ops/metrics, update gauges, animate values
    ↓
(Every 2-3 seconds, repeat metrics generation + logging)
    ↓
User asks AI: "Why is latency so high?"
    ↓
Backend: /generate endpoint
    ├─ build_ops_context() (includes current metrics)
    ├─ retrieve_kb_context() (search knowledge base)
    ├─ retrieve_memory_context() (prior conversations)
    ├─ Call get_ai_response() (OpenAI GPT-4o-mini)
    ├─ Get response: "Traffic spike detected. RPS=520. Pods scaled 5→11. Network saturated."
    └─ Call log_conversation_to_db(prompt, response, metrics_snapshot, sources, ms)
       └─ INSERT into conversations table
          └─ Also store in ChromaDB vectors for RAG
    ↓
Frontend: Display AI response with sources
    ↓
User triggers recovery:
    ├─ frontend fetch(/ops/incidents/trigger, {incident_type: "recovery"})
    ├─ Backend: ops_state["incident"] = "recovery"
    ├─ push_timeline() + log_incident_event_to_db()
    └─ Next /ops/metrics applies decay factor to all metrics
       ├─ CPU = 42 + (77-42) * 0.85 ≈ 71% (smooth reduction)
       ├─ Memory, Latency, Errors similarly decay
       ├─ Pods scale down slowly to prevent thrashing
       └─ Metrics converge to baseline over ~7 calls (15 seconds)
```

---

## Metric Correlation Example: Traffic Spike

```
Time: T0
  RPS: 190 → 530  (280 req/min increase) ← INITIAL TRIGGER
  CPU: 42 → 77    (35% increase from request handling)
  Memory: 58 → 66 (8% increase for buffering)
  Pods: 6 → 11    (5 pod auto-scale)
  Latency: 120 → 245 (120ms increase from queueing despite scaling)
  Errors: 0.4 → 1.8 (1.4% from timeouts under load)
  
  Services:
    api-gateway: degraded (latency 220ms)
    orders-service: degraded (latency 245ms)
    payments-worker: degraded (latency 270ms)

Time: T0 + 3s (Next metrics call, still in traffic_spike)
  RPS: 520 → 560  (continuing high traffic)
  CPU: 77 → 78    (high, stable with pods added)
  Memory: 66 → 67 (buffering stabilized)
  Pods: 11 → 11   (stayed at max)
  Latency: 245 → 250 (still high due to network saturation)
  Errors: 1.8 → 1.9 (still elevated)

Time: T0 + 12s (User triggers recovery)
  Recovery decay applied:
  RPS: 560 - (560-190) * 0.15 ≈ 515  (15% reduction per call)
  CPU: 78 - (78-42) * 0.15 ≈ 73     (smooth descent)
  Memory: 67 - (67-58) * 0.15 ≈ 66  (smooth descent)
  Pods: 11 * 0.925 ≈ 10             (slower scale-down)
  Latency: 250 - (250-120) * 0.15 ≈ 231
  Errors: 1.9 - (1.9-0.4) * 0.15 ≈ 1.76

Time: T0 + 15s (Another recovery cycle)
  All metrics 15% closer to baseline
  
Time: T0 + 21s (Recovery complete)
  Back to approximately:
  RPS: 190, CPU: 42%, Memory: 58%, Pods: 6, Latency: 120ms, Errors: 0.4%
```

---

## Database Query Examples

### 1. What was the peak CPU during traffic_spike incidents?

```sql
SELECT MAX(cpu_percent), AVG(cpu_percent)
FROM historical_metrics
WHERE incident_type = 'traffic_spike'
```

**Result:** Peak CPU 92%, Avg CPU 78%

---

### 2. How long did the incident last?

```sql
SELECT 
  MIN(timestamp) as incident_start,
  MAX(timestamp) as incident_end,
  ROUND((julianday(MAX(timestamp)) - julianday(MIN(timestamp))) * 86400) as duration_seconds
FROM incident_logs
WHERE incident_type = 'traffic_spike'
ORDER BY timestamp DESC
LIMIT 1
```

**Result:** Started 2026-04-24 15:30:15, ended 15:50:42 = 20min 27sec

---

### 3. Was SLO met (p95 latency < 200ms)?

```sql
SELECT 
  COUNT(*) as total_samples,
  SUM(CASE WHEN latency_p95_ms < 200 THEN 1 ELSE 0 END) as slo_met_count,
  ROUND(100.0 * SUM(CASE WHEN latency_p95_ms < 200 THEN 1 ELSE 0 END) / COUNT(*), 2) as slo_compliance
FROM historical_metrics
WHERE incident_type = 'traffic_spike'
```

**Result:** 45 samples, 2 met SLO = 4.4% compliance (SLO violation!)

---

### 4. Did KB help answer questions?

```sql
SELECT 
  incident_state,
  COUNT(*) as conversations,
  AVG(response_ms) as avg_response_time_ms,
  SUM(CASE WHEN kb_sources_json IS NOT NULL THEN 1 ELSE 0 END) as kb_used_count
FROM conversations
WHERE timestamp > datetime('now', '-24 hours')
GROUP BY incident_state
```

**Result:** During traffic_spike, 5 conversations, avg 1420ms response, KB used 4 times

---

## Performance Characteristics

| Operation | Complexity | Time | Scaling |
|-----------|-----------|------|---------|
| INSERT metrics | O(1) | ~1ms | Append-only |
| Query 1-hour metrics | O(log n) | ~5ms | Indexed on timestamp |
| Query full day | O(n) | ~20ms | 43,200 rows scanned |
| SLO report (1 month) | O(log n) | ~10ms | Pre-aggregated |
| Conversation search | O(n) | ~15ms | 100s of rows/month |

**Index Usage:**
- `idx_historical_metrics_timestamp` — Used for range queries (`start < timestamp < end`)
- `idx_historical_metrics_incident` — Used for incident type filtering
- `idx_incident_logs_timestamp` — Timeline queries
- `idx_conversations_timestamp` — Conversation history lookups

---

## Thread Safety & Concurrency

```python
# Thread-safe database writes via lock
db_lock = Lock()

def log_metrics_to_db(metrics_dict):
    with db_lock:  # Only one thread writes at a time
        conn = sqlite3.connect(str(DB_PATH))
        # [INSERT operation]
        conn.close()  # Lock released after write
```

**Safety Guarantees:**
- Multiple API requests → don't corrupt database
- Concurrent writes → serialized via lock
- Lock timeout → ~10ms (minimal impact)
- If lock fails → error logged, API continues (graceful degradation)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.0.0 | 2026-04-23 | Initial FastAPI backend |
| 4.1.0 | 2026-04-23 | Added scenario system, dynamic metrics |
| 4.2.0 | 2026-04-24 | **Step 3:** Metric correlation; **Step 4:** SQLite persistence |

---

## Next Architectural Enhancements

1. **Read Replicas**: Query-only SQLite connection (faster analytics without locking writes)
2. **Time-Series Compression**: Archive old metrics to compressed format
3. **Streaming Aggregation**: Real-time SLO calculation (vs. hourly batch)
4. **Multi-Incident Support**: Track multiple simultaneous scenarios
5. **Distributed Tracing**: Correlation IDs across services
6. **Custom Alerts**: Rule engine for SLO violations

