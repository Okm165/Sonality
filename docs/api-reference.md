# API Reference

Concise reference for active runtime modules.

## `sonality.agent`

### `SonalityAgent`

- `respond(user_message: str) -> str`  
  Main synchronous entrypoint.
- `shutdown() -> None`  
  Stops background workers and closes DB resources.

The agent requires Path A storage (Neo4j + PostgreSQL/pgvector).

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
