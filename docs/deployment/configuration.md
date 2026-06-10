# Configuration

All configuration is managed through a single `.env` file (template at `.env.example`). The system uses pydantic-settings for type-safe configuration loading with sensible defaults.

---

## Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible chat completions endpoint |
| `SONALITY_API_KEY` | *(empty)* | API key; leave empty for local servers |
| `SONALITY_MODEL` | `gpt-4.1-mini` | Primary reasoning model |
| `SONALITY_STRUCTURED_MODEL` | same as MODEL | Model for structured JSON output (ESS, routing) |
| `SONALITY_FAST_MODEL` | same as MODEL | Model for consolidation and simple tasks |
| `SONALITY_REASONING_MODEL` | same as MODEL | Model for deep reflection |

**Model tier strategy:** For cloud deployments, assign different models to different tiers based on cost/capability:
- Agent/Reasoning tier: most capable model (handles nuanced reasoning)
- Structured tier: reliable JSON output model (handles classification)
- Fast tier: smallest viable model (handles consolidation, routing)

For local inference (single llama.cpp instance), all tiers typically point to the same model.

---

## Token Budget

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_CHAT_INPUT_TOKEN_BUDGET` | `150000` | Max input tokens for chat context |
| `SONALITY_LLM_MAX_TOKENS` | `8192` | Max generation tokens for main response |
| `SONALITY_ESS_MAX_TOKENS` | `512` | Max tokens for ESS classification |
| `SONALITY_STRUCTURED_JSON_MAX_TOKENS` | `256` | Max tokens for routing/sufficiency |
| `SONALITY_EXTRACTION_MAX_TOKENS` | `1024` | Max tokens for knowledge/feature extraction |
| `SONALITY_RERANK_MAX_TOKENS` | `512` | Max tokens for episode reranking |

The input budget (150K) is configured for a 262K context window at 65% utilization. Adjust proportionally for smaller context models.

---

## Concurrency and Performance

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_LLM_CONCURRENCY` | `1` | Max concurrent LLM requests |
| `SONALITY_EMBEDDING_CONCURRENCY` | `4` | Max concurrent embedding requests |
| `SONALITY_LLM_MAX_RETRIES` | `3` | Retry attempts on transient failure |
| `SONALITY_LLM_BACKOFF_BASE` | `2.0` | Exponential backoff multiplier |
| `SONALITY_ASYNC_TIMEOUT` | `1500` | Seconds before async operations timeout |

For single-slot local inference, keep `LLM_CONCURRENCY=1`. For cloud providers, increase to 4-8 for better throughput on parallel bookkeeping tasks.

---

## Database

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `SONALITY_NEO4J_USER` | `neo4j` | Neo4j username |
| `SONALITY_NEO4J_PASSWORD` | `sonality_password` | Neo4j password |
| `SONALITY_NEO4J_MAX_POOL_SIZE` | `50` | Connection pool size |
| `SONALITY_NEO4J_CONNECTION_TIMEOUT` | `30.0` | Connection timeout (seconds) |
| `SONALITY_QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `SONALITY_EMBEDDING_URL` | `http://localhost:8090` | Embedding server |

---

## Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_EMBEDDING_CACHE_SIZE` | `10000` | LRU cache entries for query embeddings |
| `SONALITY_EMBEDDING_MAX_CHARS` | `4096` | Max characters per embedding input |
| `SONALITY_EMBEDDING_DIM` | `2560` | Embedding dimension (must match model) |

---

## Fathom Research Engine

| Variable | Default | Description |
|----------|---------|-------------|
| `FATHOM_BASE_URL` | same as SONALITY_BASE_URL | LLM endpoint for Fathom |
| `FATHOM_API_KEY` | same as SONALITY_API_KEY | API key for Fathom |
| `FATHOM_MODEL` | same as SONALITY_MODEL | Model for research analysis |
| `FATHOM_BROWSER_URL` | `http://localhost:8030` | Browserless CDP endpoint |
| `FATHOM_MAX_SESSIONS` | `2` | Max concurrent research sessions |

Fathom inherits Sonality's LLM configuration unless explicitly overridden with `FATHOM_*` variables. This simplifies single-model deployments while allowing independent scaling.

---

## API Security

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_HTTP_API_KEY` | *(empty)* | Bearer token for API authentication |

When set, all API requests must include `Authorization: Bearer <key>`. When empty, the API is unauthenticated (suitable for local development).

---

## Chat Clients

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_SONALITY_URL` | `http://localhost:8000` | Sonality API endpoint for clients |
| `CHAT_TELEGRAM_TOKEN` | *(empty)* | Telegram bot token |
| `CHAT_SPEACHES_URL` | `http://localhost:8020` | STT/TTS service URL |

---

## Runtime Overrides

Model selection can be overridden at runtime without editing `.env`:

```bash
uv run sonality --model "anthropic/claude-sonnet-4" --ess-model "anthropic/claude-3.7-sonnet"
make run ARGS='--model "openrouter/meta-llama/llama-4-maverick"'
```

---

## Thinking Model Support

For models with chain-of-thought reasoning (Qwen3, DeepSeek-R1, Mistral-3.1):
- Thinking is left enabled for the main response generation
- Chain-of-thought is automatically stripped from user-visible output
- Structured calls (ESS, routing, extraction) always disable thinking for reliable JSON output

No configuration is needed --- the system detects thinking-capable models automatically and handles them correctly.

---

## Capability Modes

Different configurations produce different system capabilities:

| Configuration | Capability | What You Lose |
|---------------|-----------|---------------|
| Full stack (`docker compose up`) | Complete: chat, research, memory, voice | Nothing |
| No Fathom | Chat + memory + belief revision | No web research; agent relies on existing knowledge |
| No embedding server | Chat + graph memory | No vector retrieval; retrieval falls back to graph-only |
| No Browserless | Chat + memory + search-only research | No page fetching; research limited to search snippets |
| No Speaches | Full minus voice | Terminal and API still work; Telegram loses voice |
| Cloud LLM only | Full capability via API | Requires internet; costs per token; higher concurrency possible |
| Local LLM only | Full capability offline | Limited by GPU VRAM; typically single-request throughput |

The system detects missing services at startup and adjusts its behavior. For example, if Fathom is unreachable, the `web_research` tool reports a service unavailable error and the agent continues with other tools.
