# API Reference

Start server: `make serve` → `http://localhost:8000`

## Endpoints

### OpenAI-Compatible

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat (supports `stream=true`) |
| `GET` | `/v1/models` | List models |

### Core

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Chat → response + ESS |
| `POST` | `/ingest` | Ingest content with ESS classification |
| `GET` | `/beliefs` | All beliefs |
| `GET` | `/beliefs/{topic}` | Single belief |
| `GET` | `/health` | Agent health |

## Schemas

### POST /chat

```json
{"message": "...", "context": []}
```

Response:
```json
{
  "response": "...",
  "ess_score": 0.65,
  "reasoning_type": "empirical_data",
  "topics": ["climate"]
}
```

### POST /ingest

```json
{"text": "...", "topic_override": ""}
```

Response:
```json
{
  "success": true,
  "score": 0.65,
  "reasoning_type": "empirical_data",
  "belief_update_recommended": true,
  "urgency": "routine",
  "topics": ["climate"],
  "summary": "..."
}
```

### POST /v1/chat/completions

```json
{
  "model": "sonality",
  "messages": [{"role": "user", "content": "..."}],
  "stream": false
}
```

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
