# API Reference

Start server: `make serve` → `http://localhost:8000`

## Endpoints

### OpenAI-Compatible

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat (supports `stream=true` with SSE progress events) |
| `GET` | `/v1/models` | List models |

### Core

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest` | Queue content for asynchronous ingestion (returns 202) |
| `GET` | `/ingest/{job_id}` | Check ingest job status |
| `GET` | `/beliefs` | All beliefs |
| `GET` | `/beliefs/{topic}` | Single belief |
| `GET` | `/health` | Agent health |

## Schemas

### POST /ingest

```json
{"text": "...", "topic_override": ""}
```

Response (202 Accepted):
```json
{
  "job_id": "abc123",
  "queue_depth": 3
}
```

### GET /ingest/{job_id}

Response (completed):
```json
{
  "status": "done",
  "score": 0.65,
  "reasoning_type": "empirical_data",
  "belief_update_recommended": true,
  "urgency": "standard",
  "topics": ["climate"],
  "summary": "..."
}
```

### POST /v1/chat/completions

```json
{
  "model": "sonality",
  "messages": [{"role": "user", "content": "..."}],
  "stream": true
}
```

When `stream=true`, SSE events include progress (`event: thinking`, `event: tool_call`, `event: tool_result`, `event: reviewing`, `event: done`) interleaved with standard `data:` content chunks.

### GET /beliefs

```json
[
  {"topic": "climate_change", "valence": 0.8, "confidence": 0.7, "evidence_count": 5, "uncertainty": 0.3, "belief_text": "..."}
]
```

### GET /health

```json
{
  "belief_count": 12,
  "snapshot_version": 5,
  "uptime_seconds": 3600.0,
  "version": "0.1.0"
}
```
