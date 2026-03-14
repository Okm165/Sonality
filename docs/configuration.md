# Configuration

Configuration is environment-driven via `.env` (see `.env.example`).

## Required Runtime Variables

| Variable | Default | Notes |
|---|---|---|
| `SONALITY_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio, etc.) |
| `SONALITY_MODEL` | `gpt-4.1-mini` | Main response model |
| `SONALITY_ESS_MODEL` | `SONALITY_MODEL` | ESS and reflection model |
| `SONALITY_EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-8B` | Embedding model on the same provider API |

`SONALITY_API_KEY` is optional for local providers that do not require auth.

## Path A Database Variables

| Variable | Default |
|---|---|
| `SONALITY_NEO4J_URL` | `bolt://localhost:7687` |
| `SONALITY_NEO4J_USER` | `neo4j` |
| `SONALITY_NEO4J_PASSWORD` | `sonality_password` |
| `SONALITY_NEO4J_DATABASE` | `neo4j` |
| `SONALITY_POSTGRES_URL` | `postgresql://sonality:sonality_password@localhost:5432/sonality` |
| `SONALITY_PG_POOL_MIN_SIZE` | `2` |
| `SONALITY_PG_POOL_MAX_SIZE` | `10` |

Path A storage is required at runtime.

## Retrieval and Reflection Tuning

| Variable | Default |
|---|---|
| `SONALITY_EPISODIC_RETRIEVAL_COUNT` | `3` |
| `SONALITY_SEMANTIC_RETRIEVAL_COUNT` | `2` |
| `SONALITY_RETRIEVAL_MAX_ITERATIONS` | `3` |
| `SONALITY_RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.8` |
| `SONALITY_RETRIEVAL_OVER_FETCH_FACTOR` | `3` |
| `SONALITY_REFLECTION_EVERY` | `20` |
| `SONALITY_OPINION_COOLING_PERIOD` | `3` |

## Runtime Artifacts

- `data/sponge.json`
- `data/sponge_history/`
- `data/ess_log.jsonl`

Graph/vector data is stored in Neo4j/PostgreSQL, not in local Chroma files.
