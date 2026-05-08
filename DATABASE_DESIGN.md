# AtlaOps Database Design & Integration Guide

## Overview

This document provides a **comprehensive technical explanation** of the SQLite database integration for obs.atla.in's backend. The database will persist historical metrics, incident logs, conversation history, and performance analytics—enabling analytics, trend analysis, and temporal insights beyond what in-memory state can provide.

---

## Part 1: Why Database Integration?

### Current Limitations (In-Memory Only)

The current `app1.py` stores data in memory via the `ops_state` dictionary:
- **Metrics** are computed on-the-fly using sinusoidal functions; no historical record exists
- **Timeline events** are capped at 30 entries; older events are lost
- **Conversations** stored in ChromaDB vector store, but no structured audit log
- **No analytics**: Cannot answer "What was CPU at 3:47pm?" or "What was the average latency during traffic spike?"

### Benefits of SQLite Integration

1. **Persistence**: Data survives service restarts
2. **Analytics**: Query historical trends, SLO compliance, MTTR, incident duration
3. **Audit Trail**: Full conversation history with timestamps, user inputs, and system responses
4. **Time-Series Analysis**: Plot metrics over 30-minute, 1-hour, daily windows
5. **Incident Correlation**: Link metrics snapshots to specific timeline events
6. **Performance Insights**: Track which scenarios cause biggest performance impact

---

## Part 2: Schema Design

### Database: `obs_backend.db` (SQLite 3.x)

#### Table 1: `historical_metrics`
Stores snapshots of all metrics at regular intervals.

```sql
CREATE TABLE historical_metrics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,           -- UTC ISO 8601
  incident_type TEXT NOT NULL,       -- "normal", "traffic_spike", "db_errors", "recovery"
  
  -- Core metrics
  cpu_percent REAL,
  memory_percent REAL,
  latency_p95_ms REAL,
  error_rate_percent REAL,
  pod_count INTEGER,
  requests_per_min INTEGER,
  
  -- Service statuses (JSON array for flexibility)
  services_json TEXT,                -- JSON: [{name, status, latency_ms}, ...]
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_historical_metrics_timestamp 
  ON historical_metrics(timestamp);
CREATE INDEX idx_historical_metrics_incident 
  ON historical_metrics(incident_type);
```

**Why this structure?**
- **Timestamp as primary query key**: Most analytics queries filter by time range
- **Incident type included**: Enables "what metrics changed during traffic_spike?" queries
- **Services as JSON**: Avoids normalization complexity while remaining queryable
- **Indexes on timestamp & incident**: Fast lookups for the two most common filters

---

#### Table 2: `incident_logs`
Structured event timeline with causality and context.

```sql
CREATE TABLE incident_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,           -- UTC ISO 8601
  incident_type TEXT NOT NULL,       -- Current incident state when event occurred
  event_type TEXT NOT NULL,          -- "incident_start", "incident_end", "milestone"
  event_message TEXT NOT NULL,
  
  -- Optional context for RCA
  triggered_by TEXT,                 -- e.g., "cpu_above_threshold", "user_action"
  related_metric_name TEXT,          -- e.g., "cpu_percent", "latency_p95_ms"
  metric_value REAL,                 -- The value that triggered the event
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_incident_logs_timestamp 
  ON incident_logs(timestamp);
CREATE INDEX idx_incident_logs_incident_type 
  ON incident_logs(incident_type);
```

**Why separate from historical_metrics?**
- Metrics are **dense** (every 2-3 seconds) and quantitative
- Events are **sparse** (one per significant state change) and qualitative
- Separating allows efficient queries: "all events in incident_type='traffic_spike'" is fast

---

#### Table 3: `conversations`
Full audit of all interactions with the AI system.

```sql
CREATE TABLE conversations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,           -- UTC ISO 8601 when message was sent
  user_prompt TEXT NOT NULL,         -- User's question/request
  ai_response TEXT NOT NULL,         -- Full AI response (up to 500 tokens)
  
  -- Context snapshot
  incident_state TEXT NOT NULL,      -- The incident_type at conversation time
  cpu_at_time REAL,                  -- Snapshot of key metrics when responded
  latency_at_time REAL,
  error_rate_at_time REAL,
  
  -- Sources & retrieval context
  kb_sources_json TEXT,              -- JSON array of sources used
  
  -- Metadata
  response_ms INTEGER,               -- How long LLM took to respond
  tokens_used INTEGER,               -- Prompt + completion tokens (if available)
  model TEXT DEFAULT 'gpt-4o-mini',
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_conversations_timestamp 
  ON conversations(timestamp);
CREATE INDEX idx_conversations_incident 
  ON conversations(incident_state);
```

**Why this structure?**
- **Full context snapshot**: When answering "what did I say about latency?", we can see what latency was when the question was asked
- **Response metrics**: Track if LLM got slower under high load
- **Sources linked to response**: "Did the KB help answer that question?" analytics

---

#### Table 4: `performance_analytics`
Pre-computed/aggregated metrics for quick reporting.

```sql
CREATE TABLE performance_analytics (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT NOT NULL,           -- UTC ISO 8601 (hour-level granularity)
  incident_type TEXT NOT NULL,       -- Incident active during this hour
  
  -- Aggregated metrics for the hour
  cpu_avg REAL,
  cpu_max REAL,
  cpu_min REAL,
  
  memory_avg REAL,
  memory_max REAL,
  
  latency_avg REAL,
  latency_p99_ms REAL,              -- Percentile from samples
  
  error_rate_avg REAL,
  error_rate_max REAL,
  
  -- Availability metrics
  service_count_healthy INTEGER,
  service_count_degraded INTEGER,
  
  -- SLO tracking
  slo_latency_met INTEGER,           -- 1=true, 0=false (p95 < 200ms)
  slo_error_rate_met INTEGER,        -- 1=true, 0=false (error_rate < 1%)
  
  -- Duration
  duration_seconds INTEGER,          -- How long this incident lasted
  
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_analytics_timestamp 
  ON performance_analytics(timestamp);
CREATE INDEX idx_analytics_incident 
  ON performance_analytics(incident_type);
```

**Why pre-aggregation?**
- Real-time queries (dashboards) shouldn't scan millions of rows
- SLO queries ("was p95 latency <200ms this hour?") are pre-computed
- Reports ("incident X lasted 45 minutes, SLO breach 22 minutes") are instant

---

## Part 3: Integration Points in Code

### 3.1 Initialization

At app startup, ensure database and tables exist:

```python
def init_database():
    """Create SQLite database and schema if not present."""
    db_path = PROJECT_ROOT / "obs_backend.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Create tables (idempotent; IF NOT EXISTS prevents errors)
    cursor.executescript("""
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
    );
    
    CREATE INDEX IF NOT EXISTS idx_historical_metrics_timestamp 
      ON historical_metrics(timestamp);
    CREATE INDEX IF NOT EXISTS idx_historical_metrics_incident 
      ON historical_metrics(incident_type);
    
    -- [incident_logs, conversations, performance_analytics tables...]
    """)
    
    conn.commit()
    conn.close()
```

Called once at app startup (idempotent, safe to call repeatedly).

---

### 3.2 Writing Historical Metrics

In `generate_metrics()` function, after computing metrics, write to DB:

```python
def log_metrics_to_db(metrics_dict):
    """Persist metrics snapshot to database."""
    db_path = PROJECT_ROOT / "obs_backend.db"
    conn = sqlite3.connect(str(db_path))
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
```

Called after every `/ops/metrics` request, or every ~2-3 seconds.

---

### 3.3 Writing Incident Events

In `push_timeline()`, also write to incident_logs:

```python
def log_incident_event_to_db(event_message, triggered_by=None, metric_name=None, metric_value=None):
    """Persist incident event to database."""
    db_path = PROJECT_ROOT / "obs_backend.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cursor.execute("""
    INSERT INTO incident_logs (
        timestamp, incident_type, event_type, event_message,
        triggered_by, related_metric_name, metric_value
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        utc_now(),
        ops_state["incident"],
        "milestone",  # or "incident_start", "incident_end"
        event_message,
        triggered_by,
        metric_name,
        metric_value
    ))
    
    conn.commit()
    conn.close()
```

Called alongside `push_timeline()` to maintain parallel records.

---

### 3.4 Writing Conversations

In `generate_text()` endpoint, after AI response:

```python
def log_conversation_to_db(user_prompt, ai_response, metrics_snapshot, kb_sources, response_ms):
    """Persist conversation pair to database."""
    db_path = PROJECT_ROOT / "obs_backend.db"
    conn = sqlite3.connect(str(db_path))
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
        metrics_snapshot["cpu_percent"],
        metrics_snapshot["latency_p95_ms"],
        metrics_snapshot["error_rate_percent"],
        json.dumps(kb_sources),
        response_ms
    ))
    
    conn.commit()
    conn.close()
```

---

### 3.5 Computing Analytics

Periodically (e.g., every 60 seconds), aggregate the last hour's metrics:

```python
def compute_hourly_analytics():
    """Compute and store hourly aggregates for fast reporting."""
    db_path = PROJECT_ROOT / "obs_backend.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get last hour of metrics
    one_hour_ago = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    cursor.execute("""
    SELECT 
        AVG(cpu_percent), MAX(cpu_percent), MIN(cpu_percent),
        AVG(memory_percent), MAX(memory_percent),
        AVG(latency_p95_ms), AVG(error_rate_percent), MAX(error_rate_percent),
        incident_type,
        COUNT(*) as sample_count
    FROM historical_metrics
    WHERE timestamp > ?
    GROUP BY incident_type
    """, (one_hour_ago,))
    
    row = cursor.fetchone()
    if row:
        cursor.execute("""
        INSERT INTO performance_analytics (
            timestamp, incident_type, cpu_avg, cpu_max, cpu_min,
            memory_avg, memory_max, latency_avg, error_rate_avg, error_rate_max,
            slo_latency_met, slo_error_rate_met
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            utc_now(),
            row[8],  # incident_type
            row[0], row[1], row[2],  # cpu avg, max, min
            row[3], row[4],  # memory avg, max
            row[5],  # latency avg
            row[6], row[7],  # error_rate avg, max
            1 if row[5] < 200 else 0,  # SLO check: p95 < 200ms
            1 if row[6] < 1.0 else 0   # SLO check: error_rate < 1%
        ))
    
    conn.commit()
    conn.close()
```

---

## Part 4: New API Endpoints for Analytics

### Endpoint: `GET /analytics/metrics-history`
Query historical metrics over a time range.

```python
@app.get("/analytics/metrics-history")
def metrics_history(
    start: str = Query(..., description="ISO 8601 start timestamp"),
    end: str = Query(..., description="ISO 8601 end timestamp"),
    incident_type: str | None = Query(None, description="Filter by incident type")
):
    """Return metric history for charting/analysis."""
    db_path = PROJECT_ROOT / "obs_backend.db"
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row  # Return dicts instead of tuples
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
```

### Endpoint: `GET /analytics/incident-timeline`
Retrieve structured incident events.

```python
@app.get("/analytics/incident-timeline")
def incident_timeline(
    start: str = Query(...),
    end: str = Query(...),
    incident_type: str | None = Query(None)
):
    """Return incident events with context."""
    db_path = PROJECT_ROOT / "obs_backend.db"
    conn = sqlite3.connect(str(db_path))
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
```

### Endpoint: `GET /analytics/slo-report`
SLO compliance report for a time period.

```python
@app.get("/analytics/slo-report")
def slo_report(
    start: str = Query(...),
    end: str = Query(...)
):
    """Return SLO compliance metrics."""
    db_path = PROJECT_ROOT / "obs_backend.db"
    conn = sqlite3.connect(str(db_path))
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
        "slo_latency_compliance": (latency_met / total * 100) if total > 0 else 0,
        "slo_error_rate_compliance": (error_rate_met / total * 100) if total > 0 else 0,
        "total_hours_tracked": total
    }
```

---

## Part 5: Query Patterns & Examples

### Pattern 1: "What was the average latency during the traffic spike?"

```python
cursor.execute("""
SELECT 
    AVG(latency_p95_ms) as avg_latency,
    MAX(latency_p95_ms) as peak_latency
FROM historical_metrics
WHERE incident_type = 'traffic_spike'
""")
```

### Pattern 2: "Show me CPU and memory trend over the last 30 minutes"

```python
thirty_min_ago = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()

cursor.execute("""
SELECT timestamp, cpu_percent, memory_percent
FROM historical_metrics
WHERE timestamp > ?
ORDER BY timestamp ASC
""", (thirty_min_ago,))
```

### Pattern 3: "How long did the last incident last?"

```python
cursor.execute("""
SELECT 
    MIN(timestamp) as incident_start,
    MAX(timestamp) as incident_end
FROM incident_logs
WHERE incident_type = 'traffic_spike'
ORDER BY timestamp DESC
LIMIT 1
""")
```

### Pattern 4: "Did the KB help answer user questions during high-load incidents?"

```python
cursor.execute("""
SELECT 
    c.incident_state,
    COUNT(*) as conversation_count,
    AVG(CASE WHEN c.kb_sources_json NOT NULL THEN 1 ELSE 0 END) as kb_usage_rate
FROM conversations c
GROUP BY c.incident_state
""")
```

---

## Part 6: Data Retention & Cleanup

To prevent unbounded database growth, implement a retention policy:

```python
def cleanup_old_data(retention_days: int = 30):
    """Remove metrics older than retention_days."""
    db_path = PROJECT_ROOT / "obs_backend.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention_days)).isoformat()
    
    cursor.execute("DELETE FROM historical_metrics WHERE timestamp < ?", (cutoff,))
    cursor.execute("DELETE FROM incident_logs WHERE timestamp < ?", (cutoff,))
    
    # Keep conversations longer for audit
    cursor.execute("DELETE FROM conversations WHERE timestamp < ?", 
                   ((datetime.now(timezone.utc) - timedelta(days=90)).isoformat(),))
    
    conn.commit()
    conn.close()
```

Run as a scheduled background task (e.g., daily at midnight).

---

## Part 7: Implementation Checklist

- [ ] Add `import sqlite3` and `import json` to app1.py
- [ ] Create `init_database()` function and call it at app startup
- [ ] Modify `generate_metrics()` to call `log_metrics_to_db()`
- [ ] Modify `push_timeline()` to call `log_incident_event_to_db()`
- [ ] Modify `generate_text()` to call `log_conversation_to_db()`
- [ ] Create background task `compute_hourly_analytics()` (runs every 60 seconds)
- [ ] Add 4 new GET endpoints for analytics
- [ ] Add cleanup task to run daily
- [ ] Test: trigger an incident, then query `/analytics/metrics-history` to verify data is persisted

---

## Conclusion

This design provides:
1. **Structured persistence** without over-normalizing (balancing ACID properties with query simplicity)
2. **Fast analytics** via pre-aggregated `performance_analytics` table
3. **Full audit trail** of conversations and incidents
4. **Temporal analysis** to understand causality and impact
5. **SLO tracking** for compliance and reporting

The integration points are minimal and non-intrusive—all database writes happen alongside existing code, with proper error handling to ensure DB issues don't break the API.

