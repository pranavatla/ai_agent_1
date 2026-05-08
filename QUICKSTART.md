# Quick Start: obs.atla.in v4.2.0

## Run the Backend

```bash
cd /Users/pranav/Projects/AI_Agent_1-develop/

# Install dependencies (if not already done)
pip install fastapi uvicorn mangum openai chromadb pydantic

# Start the server
python app1.py

# Or with auto-reload for development:
uvicorn app1:app --reload --host 0.0.0.0 --port 8000
```

The server will:
1. Create `obs_backend.db` (SQLite database)
2. Initialize 4 tables if they don't exist
3. Start accepting requests at `http://localhost:8000`

## Access the UI

Open your browser:
```
http://localhost:8000/
```

You'll see the index.html dashboard with:
- Metric gauges (CPU, Memory, Latency, Error Rate, Pods, RPS)
- Service status cards
- Timeline of events
- Log viewer
- AI chat interface

## Trigger an Incident

### Via Browser Console
```javascript
fetch('/ops/incidents/trigger', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ incident_type: 'traffic_spike' })
})
```

### Via curl
```bash
curl -X POST http://localhost:8000/ops/incidents/trigger \
  -H "Content-Type: application/json" \
  -d '{"incident_type": "traffic_spike"}'
```

### Available Incidents
- `normal` — Baseline healthy state
- `traffic_spike` — High request volume (RPS ↑, CPU ↑, Latency ↑, Errors ↑)
- `db_errors` — Database timeouts (Memory ↑↑, Latency ↑, Errors ↑↑)
- `recovery` — Smooth decay back to normal

## Watch Metrics Change

Once incident is triggered:
1. **RPS** jumps to 500+ requests/min
2. **CPU** spikes to 70-80%
3. **Memory** increases to 60-70%
4. **Latency** shoots to 200-250ms
5. **Pods** scale up to 11 (from 6)
6. **Error Rate** climbs to 1.4-1.8%
7. **Timeline** adds events describing cascade
8. **Logs** show WARN/ALERT entries

Let it run for 15-30 seconds, then trigger `recovery` to watch metrics smooth back down.

## Query Analytics

### Historical Metrics (Last 24 Hours)
```bash
YESTERDAY=$(date -u -d '1 day ago' +%Y-%m-%dT%H:%M:%SZ)
NOW=$(date -u +%Y-%m-%dT%H:%M:%SZ)

curl "http://localhost:8000/analytics/metrics-history?start=$YESTERDAY&end=$NOW"
```

### Incident Timeline
```bash
curl "http://localhost:8000/analytics/incident-timeline?start=$YESTERDAY&end=$NOW"
```

### SLO Compliance Report
```bash
curl "http://localhost:8000/analytics/slo-report?start=$YESTERDAY&end=$NOW"
```

### Conversation Statistics
```bash
curl "http://localhost:8000/analytics/conversation-stats?start=$YESTERDAY&end=$NOW"
```

## Chat with AtlaOps Guru

In the UI, type questions about the incident:
- "What's causing high latency?"
- "Are pods scaling fast enough?"
- "What's the error rate trend?"
- "Should we scale out more?"

The AI will:
1. Analyze current metrics
2. Search the knowledge base
3. Consider conversation history
4. Return a technical answer
5. Log the entire conversation to the database

## Check Database

```bash
# Verify database exists and is growing
ls -lh obs_backend.db

# Inspect with sqlite3 CLI
sqlite3 obs_backend.db

# Count rows per table
> SELECT 'historical_metrics', COUNT(*) FROM historical_metrics
> UNION ALL SELECT 'incident_logs', COUNT(*) FROM incident_logs
> UNION ALL SELECT 'conversations', COUNT(*) FROM conversations;

# View latest metrics
> SELECT timestamp, cpu_percent, latency_p95_ms, error_rate_percent 
  FROM historical_metrics ORDER BY timestamp DESC LIMIT 5;

# View timeline
> SELECT timestamp, event_message FROM incident_logs ORDER BY timestamp DESC LIMIT 5;
```

## Testing the 14-Week Journey Demo

```bash
# 1. Start server
python app1.py

# 2. Open browser, verify dashboard loads
open http://localhost:8000/

# 3. Wait 5 seconds, watch auto-trigger of traffic_spike
# (frontend triggers it automatically)

# 4. Watch metrics cascade:
#    - RPS spikes first
#    - CPU follows
#    - Pods scale
#    - Latency climbs
#    - Errors rise

# 5. Ask AI about the incident via chat

# 6. Trigger recovery
curl -X POST http://localhost:8000/ops/incidents/trigger \
  -d '{"incident_type": "recovery"}'

# 7. Watch metrics decay smoothly

# 8. Query analytics to see historical data
curl http://localhost:8000/analytics/metrics-history?start=2026-04-24T00:00:00Z&end=2026-04-25T00:00:00Z | jq .

# 9. Check database
sqlite3 obs_backend.db "SELECT COUNT(*) FROM historical_metrics;"
```

## File Structure

```
/Users/pranav/Projects/AI_Agent_1-develop/
├── index.html                 # Frontend dashboard (dark theme, scenarios)
├── app1.py                    # Backend API (FastAPI + SQLite persistence)
├── obs_backend.db             # SQLite database (auto-created)
├── DATABASE_DESIGN.md         # Comprehensive DB schema & design docs
├── IMPLEMENTATION_SUMMARY.md  # Step 3 & 4 technical details
├── QUICKSTART.md              # This file
└── docs/
    └── atlaops-kb/            # Knowledge base for RAG
```

## Endpoints Reference

### Core Ops Endpoints
- `GET /` — Dashboard HTML
- `GET /health` — Health check
- `GET /ops/metrics` — Current metrics snapshot
- `GET /ops/logs?limit=20` — Recent logs
- `GET /ops/incidents` — Current incident state + timeline
- `POST /ops/incidents/trigger` — Change incident state
- `GET /ops/incidents/rca` — Root cause analysis
- `GET /ops/architecture` — System architecture graph

### Analytics Endpoints
- `GET /analytics/metrics-history?start=...&end=...` — Historical metrics
- `GET /analytics/incident-timeline?start=...&end=...` — Event timeline
- `GET /analytics/slo-report?start=...&end=...` — SLO compliance
- `GET /analytics/conversation-stats?start=...&end=...` — Conversation metrics

### AI Endpoints
- `POST /generate/` — Chat with AtlaOps Guru
  ```bash
  curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"prompt": "What is causing high latency?"}'
  ```

## Troubleshooting

**Port 8000 already in use:**
```bash
lsof -i :8000
kill -9 <PID>
# Then restart
```

**Database locked:**
- Restart the server
- SQLite will auto-recover

**Missing OPENAI_API_KEY:**
- AI won't respond, but rest of system works
- Export your key: `export OPENAI_API_KEY=sk-...`
- Restart server

**No response from /analytics endpoints:**
- Run the server for at least 10 seconds first (needs data to query)
- Trigger an incident so there's data to persist

## Next Enhancements

1. **Frontend Charts**: Plot historical metrics from `/analytics/metrics-history`
2. **Real-time SLO Widget**: Display `/analytics/slo-report` on dashboard
3. **Incident Replay**: Load past timeline from database
4. **Export**: CSV/JSON downloads of metrics
5. **Alerts**: Notify when SLO violations occur

## Key Features

✅ **Realistic Metrics** — Traffic spike causes cascading failures (CPU → pods → latency → errors)
✅ **Persistent Storage** — SQLite with 4 normalized tables
✅ **Analytics Ready** — Query historical trends with time-range filters
✅ **SLO Tracking** — Pre-computed compliance metrics
✅ **Audit Trail** — Full conversation & incident history
✅ **Production-Ready** — Thread-safe, indexed, error-handled

---

**Version:** 4.2.0  
**Last Updated:** 2026-04-24  
**Backend:** FastAPI + SQLite  
**Frontend:** Vanilla JS + CSS Variables  
**AI:** OpenAI GPT-4o-mini  
**Status:** ✅ Steps 3 & 4 Complete
