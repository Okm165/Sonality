# Configuration

Configuration is environment-driven via `.env` (see `.env.example`).

## Required Runtime Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `SONALITY_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible endpoint (OpenAI, Ollama, vLLM, LM Studio, etc.) |
| `SONALITY_MODEL` | `gpt-4.1-mini` | Main response model |
| `SONALITY_STRUCTURED_MODEL` | Same as `SONALITY_MODEL` | Structured output / ESS classification (**must support tool_choice**) |
| `SONALITY_AGENT_MODEL` | Same as `SONALITY_MODEL` | Agentic loop model (**must support tool_choice**) |
| `SONALITY_API_KEY` | `OPENAI_API_KEY` | API key (optional for local providers) |

**Tool-capable models (Ollama):** qwen2.5, qwen3, llama3.1, llama3.2, mistral, mixtral

**Non-tool models:** phi4, gemma, deepseek — use only for `SONALITY_FAST_MODEL` or `SONALITY_REASONING_MODEL`

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
| `SONALITY_LLM_MAX_TOKENS` | `8192` | Max output tokens per LLM call |
| `SONALITY_LLM_TIMEOUT` | `600` | HTTP request timeout (seconds) |
| `SONALITY_ASYNC_TIMEOUT` | `3000` | Async operation timeout (seconds) |
| `SONALITY_FAST_MODEL` | Same as `SONALITY_MODEL` | Model for fast scoring / lightweight tasks |

## Runtime Artifacts

| Path | Purpose |
|------|---------|
| `data/ess_log.jsonl` | ESS classification audit log |
| `data/teaching_bench/` | Benchmark output artifacts |

Graph data (episodes, beliefs, personality) is stored in **Neo4j**.
Vector data (derivatives, semantic features — including knowledge propositions) is stored in **Qdrant**.

## Environment File Example

```bash
# .env
SONALITY_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
SONALITY_MODEL=gpt-4o
SONALITY_STRUCTURED_MODEL=gpt-4o-mini

# Database (use Docker Compose defaults)
SONALITY_NEO4J_URL=bolt://localhost:7687
SONALITY_NEO4J_PASSWORD=sonality_password
SONALITY_QDRANT_URL=http://localhost:6333
```

## Web Search

| Variable | Default | Notes |
|----------|---------|-------|
| `SONALITY_FATHOM_URL` | `http://localhost:8010` | Fathom research service URL |
| `SONALITY_WEB_SEARCH_ENABLED` | `true` | Master toggle for web access |
| `SONALITY_REFLECTION_WEB_QUERIES` | `3` | Max web queries per reflection enrichment step |

The agent uses a unified agentic loop where the LLM decides ALL cognitive steps via tool calls: recall_memory, web_search, web_extract, synthesize, and integrate_knowledge. All tools are always available — there are no per-turn limits. Pipeline enforcement overrides premature "finish" handoffs when substantive recall/search was done but synthesize or integrate_knowledge was skipped. Stall detection and deduplication prevent infinite loops. Web search and extraction are delegated to the Fathom research service, which uses Playwright for JavaScript rendering and DuckDuckGo for search.

## HTTP API Authentication

| Variable | Default | Notes |
|----------|---------|-------|
| `SONALITY_HTTP_API_KEY` | *(unset)* | If set, all API requests require this key |

## Logging

| Variable | Default | Notes |
|----------|---------|-------|
| `SONALITY_LOG_LEVEL` | `INFO` | Python log level (DEBUG for full pipeline visibility) |

Log format uses `%(asctime)s | %(levelname)-8s | %(name)s | %(message)s` with %-style lazy evaluation.

## Docker Compose

The `docker-compose.yml` provides all required services:

```bash
docker compose up -d
```

Services:
- `sonality`: Agent API server (port 8000)
- `fathom`: Research + search service (port 8010)
- `neo4j`: Graph database for both sonality and fathom (ports 7474, 7687)
- `qdrant`: Vector database (ports 6333, 6334)
- `speaches`: STT/TTS for Telegram (port 8020)
