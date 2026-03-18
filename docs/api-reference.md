# API Reference

Concise reference for active runtime modules and HTTP endpoints.

## HTTP API

Start the server: `make run` (exposes on `http://localhost:8000`).

### OpenAI-Compatible

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | Chat completions (last user message processed; `stream=true` returns 501) |
| `GET` | `/v1/models` | List available models |
| `GET` | `/v1/models/{model_id}` | Get model info |
| `POST` | `/v1/embeddings` | Create text embeddings |

### Chat and Ingestion

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Simple chat — returns response + ESS score, reasoning type, topics |
| `POST` | `/ingest` | Ingest content without generating a response (news, reports) |

### Belief State

| Method | Path | Description |
|---|---|---|
| `GET` | `/beliefs` | All current belief states, sorted by position strength |
| `GET` | `/beliefs/{topic}` | Single belief (returns defaults if topic unknown) |
| `POST` | `/beliefs/{topic}/probability` | Calibrated probability estimate (Platt scaling) |
| `GET` | `/beliefs/{topic}/correlations` | Belief correlations for a topic |

### Compatibility aliases

| Method | Path | Alias for |
|---|---|---|
| `GET` | `/probability/{topic}` | `/beliefs/{topic}/probability` |
| `GET` | `/correlations/{topic}` | `/beliefs/{topic}/correlations` |

### Health

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Agent health: sponge version, interaction count, belief count |

## `sonality.agent`

### `SonalityAgent`

- `respond(user_message: str) -> str`  
  Main synchronous entrypoint.
- `shutdown() -> None`  
  Stops background workers and closes DB resources.

The agent requires Path A storage (Neo4j + Qdrant).

## `sonality.provider`

Unified OpenAI-compatible transport for all model calls.

- `chat_completion(...) -> ChatResult`
- `embed(model: str, texts: list[str], dimensions: int = 0) -> list[list[float]]`
- `extract_tool_call_arguments(...) -> dict[str, object]`
- `parse_json_object(text: str) -> dict[str, object]`

## `sonality.ess`

Evidence classification and coercion-safe parsing.

- `classify(client, user_message, sponge_snapshot, model=config.ESS_MODEL) -> ESSResult`
- `classifier_exception_fallback(user_message: str) -> ESSResult`

Key enums:

- `ReasoningType`
- `SourceReliability`
- `OpinionDirection`

## `sonality.memory.graph`

Graph persistence and traversal.

Key methods:

- `store_episode_atomically(...)`
- `find_belief_related_episodes(...)`
- `find_topic_related_episodes(...)`
- `traverse_temporal_context(...)`
- `update_utility(...)`
- `get_forgetting_candidates(...)`
- `list_recent_episode_context(...)`

## `sonality.memory.dual_store`

Dual-store orchestration.

- `store(...) -> StoredEpisode`
- `vector_search(...) -> list[tuple[str, str, float]]`
- `verify_consistency() -> list[str]`
- `archive_derivatives(episode_uid: str) -> None`

## `sonality.memory.semantic_features`

Background semantic feature ingestion.

- `SemanticIngestionWorker.start()`
- `SemanticIngestionWorker.stop()`
- `SemanticIngestionWorker.enqueue(episode_uid: str, content: str)`

## `sonality.memory.sponge`

Persistent personality model.

- `update_opinion(...)`
- `stage_opinion_update(...)`
- `apply_due_staged_updates() -> list[str]`
- `record_shift(...)`
- `save(path, history_dir)`
- `load(path) -> SpongeState`
