# Configuration

Configuration is environment-driven via `.env` (see `.env.example`).

## Required Runtime Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `SONALITY_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio, etc.) |
| `SONALITY_MODEL` | `gpt-4.1-mini` | Main response model |
| `SONALITY_ESS_MODEL` | Same as `SONALITY_MODEL` | ESS and reflection model |
| `SONALITY_API_KEY` | `OPENAI_API_KEY` | API key (optional for local providers) |

## Database Variables

| Variable | Default |
|----------|---------|
| `SONALITY_NEO4J_URL` | `bolt://localhost:7687` |
| `SONALITY_NEO4J_USER` | `neo4j` |
| `SONALITY_NEO4J_PASSWORD` | `sonality_password` |
| `SONALITY_NEO4J_DATABASE` | `neo4j` |
| `SONALITY_QDRANT_URL` | `http://localhost:6333` |

Both Neo4j and Qdrant are required at runtime.

## Embedding Configuration

Embeddings are computed locally using **FastEmbed** with the `BAAI/bge-large-en-v1.5` model (1024 dimensions). No external embedding API is required.

| Variable | Default | Notes |
|----------|---------|-------|
| `SONALITY_EMBEDDING_MAX_CHARS` | `4096` | Max characters per embedding input |

## Retrieval Tuning

| Variable | Default | Notes |
|----------|---------|-------|
| `SONALITY_RETRIEVAL_MAX_ITERATIONS` | `3` | Max chain retrieval iterations |
| `SONALITY_RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.8` | Sufficiency confidence threshold |
| `SONALITY_RETRIEVAL_OVER_FETCH_FACTOR` | `3` | Over-fetch multiplier for reranking |
| `SONALITY_MAX_RERANK_CANDIDATES` | `50` | Max candidates for LLM reranking |

## Qdrant Search Tuning

| Variable | Default | Notes |
|----------|---------|-------|
| `SONALITY_QDRANT_SEARCH_EF` | `128` | HNSW ef parameter |
| `SONALITY_QDRANT_RESCORE` | `true` | Enable quantization rescoring |

## LLM Configuration

| Variable | Default | Notes |
|----------|---------|-------|
| `SONALITY_AGENT_TEMPERATURE` | `0.6` | Response temperature |
| `SONALITY_LLM_MAX_TOKENS` | `8192` | Max output tokens |
| `SONALITY_LLM_TIMEOUT` | `300` | HTTP request timeout (seconds) |
| `SONALITY_ASYNC_TIMEOUT` | `1500` | Async operation timeout (seconds) |
| `SONALITY_FAST_LLM_MODEL` | Same as `ESS_MODEL` | Model for fast scoring tasks |

## Runtime Artifacts

| Path | Purpose |
|------|---------|
| `data/ess_log.jsonl` | ESS classification audit log |
| `data/teaching_bench/` | Benchmark output artifacts |

Graph data (episodes, beliefs, personality) is stored in **Neo4j**.
Vector data (derivatives, semantic features, knowledge) is stored in **Qdrant**.

## Environment File Example

```bash
# .env
SONALITY_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
SONALITY_MODEL=gpt-4o
SONALITY_ESS_MODEL=gpt-4o-mini

# Database (use Docker Compose defaults)
SONALITY_NEO4J_URL=bolt://localhost:7687
SONALITY_NEO4J_PASSWORD=sonality_password
SONALITY_QDRANT_URL=http://localhost:6333
```

## Docker Compose

The `docker-compose.yml` provides all required services:

```bash
docker compose up -d
```

Services:
- `neo4j`: Graph database (ports 7474, 7687)
- `qdrant`: Vector database (ports 6333, 6334)
- `speaches`: STT/TTS for Telegram (port 8001)
