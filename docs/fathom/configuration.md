# Fathom Configuration

All Fathom settings use the `FATHOM_` prefix and are loaded via `pydantic-settings`.

## LLM Endpoint

Fathom uses the same OpenAI-compatible provider pattern as Sonality:

| Variable | Default | Description |
|----------|---------|-------------|
| `FATHOM_BASE_URL` | **(required)** | OpenAI-compatible API base |
| `FATHOM_API_KEY` | `""` | API key (optional for local models) |
| `FATHOM_MODEL` | **(required)** | Model for all LLM calls |
| `FATHOM_LLM_MAX_TOKENS` | `8192` | Max tokens per LLM response |
| `FATHOM_LLM_MAX_RETRIES` | `3` | Retry count on failure |
| `FATHOM_LLM_BACKOFF_BASE` | `2.0` | Exponential backoff base (seconds) |
| `FATHOM_LLM_CONCURRENCY` | `1` | Max parallel LLM calls |
| `FATHOM_LLM_REQUEST_TIMEOUT` | `600` | Per-request timeout (seconds) |

## Research Session

| Variable | Default | Description |
|----------|---------|-------------|
| `FATHOM_N` | `4` | Pages fetched per round (Playwright batch) |
| `FATHOM_MAX_PAGES` | `80` | Hard stop — total pages per session |
| `FATHOM_SEARCH_CONCURRENCY` | `8` | Parallel DuckDuckGo searches |

## Neo4j (shared with Sonality)

Fathom shares the Neo4j instance with Sonality. Research data uses separate
node labels (`ResearchSession`, `FrontierURL`, `ResearchFact`, `ChecklistQuestion`)
to avoid collision with Sonality's `Episode`, `Belief`, etc.

| Variable | Default | Description |
|----------|---------|-------------|
| `FATHOM_NEO4J_URL` | `bolt://localhost:7687` | Neo4j bolt URL |
| `FATHOM_NEO4J_USER` | `neo4j` | Neo4j username |
| `FATHOM_NEO4J_PASSWORD` | `sonality_password` | Neo4j password |
| `FATHOM_NEO4J_DATABASE` | `neo4j` | Neo4j database name |

## Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `FATHOM_LOG_LEVEL` | `INFO` | Log level |
