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
| `POST` | `/ingest` | Ingest content (no response) |
| `GET` | `/beliefs` | All beliefs |
| `GET` | `/beliefs/{topic}` | Single belief |
| `GET` | `/health` | Agent health |

## Schemas

### POST /chat

```json
{"messages": [{"role": "user", "content": "..."}]}
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
{"text": "...", "topic_override": "optional"}
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
{
  "beliefs": [
    {"topic": "climate_change", "valence": 0.8, "confidence": 0.7, "evidence_count": 5}
  ]
}
```

### GET /health

```json
{
  "status": "healthy",
  "belief_count": 12,
  "snapshot_version": 5
}
```
