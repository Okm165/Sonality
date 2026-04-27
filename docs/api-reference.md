# API Reference

Concise reference for active runtime modules and HTTP endpoints.

## HTTP API

Start the server: `make run` or `sonality-server` (exposes on `http://localhost:8000`).

### OpenAI-Compatible Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (supports `stream=true` for SSE) |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/{model_id}` | Get model info |

### Chat and Ingestion

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/chat` | Simple chat — returns response + ESS score, reasoning type, topics |
| `POST` | `/ingest` | Ingest content without generating a response (news, articles) |

### Belief State

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/beliefs` | All current beliefs, sorted by absolute valence |
| `GET` | `/beliefs/{topic}` | Single belief by topic (404 if not found) |

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Agent health: belief count, snapshot version |
| `GET` | `/v1/health` | Alias for `/health` |

---

## Request/Response Schemas

### Chat Completions

**Request** (`POST /v1/chat/completions`):

```json
{
  "model": "sonality",
  "messages": [
    {"role": "user", "content": "What do you think about climate change?"}
  ],
  "temperature": 0.7,
  "max_tokens": 1024,
  "stream": false
}
```

**Response**:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1714123456,
  "model": "sonality",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "..."},
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 50,
    "total_tokens": 60
  }
}
```

### Simple Chat

**Request** (`POST /chat`):

```json
{
  "message": "What's your view on renewable energy?",
  "context": []
}
```

**Response**:

```json
{
  "response": "...",
  "ess_score": 0.45,
  "reasoning_type": "logical_argument",
  "topics": ["energy", "environment"]
}
```

### Ingest

**Request** (`POST /ingest`):

```json
{
  "text": "Solar capacity grew 25% in 2024...",
  "topic_override": "renewable_energy"
}
```

**Response**:

```json
{
  "success": true,
  "score": 0.65,
  "reasoning_type": "empirical_data",
  "belief_update_recommended": true,
  "urgency": "standard",
  "topics": ["renewable_energy", "solar"],
  "summary": "Solar energy capacity statistics for 2024"
}
```

### Beliefs

**Response** (`GET /beliefs`):

```json
[
  {
    "topic": "climate_change",
    "valence": 0.75,
    "confidence": 0.85,
    "evidence_count": 12,
    "uncertainty": 0.15,
    "belief_text": "Climate change is a significant threat requiring action"
  }
]
```

---

## Python Modules

### `sonality.agent`

Core agent orchestration.

```python
class SonalityAgent:
    def __init__(self, model: str = config.MODEL, ess_model: str = config.ESS_MODEL) -> None
    def respond(self, messages: list[dict[str, str]]) -> str
    def respond_stream(self, messages: list[dict[str, str]]) -> Iterator[StreamChunk]
    def ingest(self, text: str, *, topic_override: str = "") -> ESSResult
    def get_all_beliefs(self) -> list[BeliefNode]
    def get_belief(self, topic: str) -> BeliefNode | None
    def get_snapshot(self) -> PersonalitySnapshot
    def get_health(self) -> tuple[int, int]
    def shutdown(self) -> None
```

### `sonality.provider`

Unified OpenAI-compatible transport for all model calls.

```python
def chat_completion(...) -> ChatResult
def chat_completion_stream(...) -> Iterator[StreamChunk]
def strip_thinking_trace(text: str) -> str
```

### `sonality.ess`

Evidence Strength Score classification.

```python
def classify(
    user_message: str,
    snapshot_text: str,
    model: str = config.ESS_MODEL,
    tracked_topics: str = "",
) -> ESSResult

def classifier_exception_fallback(user_message: str) -> ESSResult
```

Key enums: `ReasoningType`, `SourceReliability`, `OpinionDirection`, `KnowledgeDensity`, `UrgencyLevel`

### `sonality.memory.graph`

Neo4j graph persistence and traversal.

```python
class MemoryGraph:
    async def store_episode_atomically(...)
    async def get_episodes(uids: list[str]) -> list[EpisodeNode]
    async def find_belief_related_episodes(query: str, limit: int) -> list[EpisodeNode]
    async def find_topic_related_episodes(query: str, limit: int) -> list[EpisodeNode]
    async def traverse_temporal_context(uid: str) -> list[EpisodeNode]
    async def get_forgetting_candidates(limit: int) -> list[EpisodeNode]
    async def upsert_belief(topic: str, ...) -> None
    async def get_all_beliefs() -> list[BeliefNode]
    async def get_personality_snapshot() -> PersonalitySnapshot
    async def upsert_personality_snapshot(text: str) -> None
```

### `sonality.memory.dual_store`

Dual-store (Neo4j + Qdrant) coordination.

```python
class DualEpisodeStore:
    async def store(...) -> StoredEpisode
    async def vector_search(query: str, top_k: int) -> list[VectorSearchHit]
    async def archive_derivatives(episode_uid: str) -> None
    async def delete_derivatives(episode_uid: str) -> None
```

### `sonality.memory.semantic_features`

Background semantic feature extraction.

```python
class SemanticIngestionWorker:
    def start(self) -> None
    def stop(self) -> None
    def enqueue(self, episode_uid: str, content: str, categories: tuple = ()) -> None
```

### `sonality.memory.retrieval`

Query routing and retrieval strategies.

```python
# router.py
def route_query(query: str) -> RouteDecision

# chain.py
async def chain_retrieve(store, graph, query, base_n) -> list[EpisodeNode]

# split.py
async def split_retrieve(store, graph, query, n_per_sub) -> list[EpisodeNode]

# reranker.py
def rerank_episodes(query: str, candidates: list[EpisodeNode], top_k: int) -> list[EpisodeNode]
```

---

## Configuration

Key environment variables (see `sonality/config.py`):

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | LLM API key (required) |
| `OPENAI_BASE_URL` | `https://api.openai.com/v1` | LLM API base URL |
| `SONALITY_MODEL` | `gpt-4o` | Main chat model |
| `SONALITY_ESS_MODEL` | `gpt-4o-mini` | ESS classifier model |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `password` | Neo4j password |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant connection |
