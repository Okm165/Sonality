# System Overview

Sonality runs on one memory architecture only:

- `Neo4j` for graph structure (episodes, derivatives, topics, segments, beliefs)
- `PostgreSQL + pgvector` for semantic/vector retrieval
- one OpenAI-compatible provider for both chat and embeddings

Legacy Chroma/EpisodeStore flow is removed from runtime.

## Runtime Loop

`SonalityAgent.respond()` orchestrates:

1. Route and retrieve memory context (vector + graph + semantic features)
2. Build markdown system prompt
3. Generate response via provider chat completion
4. Run ESS classification
5. Store episode in dual store (atomic graph write, then pgvector write with rollback)
6. Update beliefs/insights, run optional reflection, persist sponge state

## Core Components

- `sonality/agent.py` - orchestration and lifecycle
- `sonality/provider.py` - OpenAI-compatible chat/embedding transport
- `sonality/memory/dual_store.py` - dual-store write + consistency checks
- `sonality/memory/graph.py` - graph operations and traversal
- `sonality/memory/semantic_features.py` - background semantic feature ingestion
- `sonality/memory/sponge.py` - personality state, staged updates, persistence

## Retrieval Model

Query routing is LLM-first (`QueryRouter`) and supports:

- `TEMPORAL` / `AGGREGATION` -> `ChainOfQueryAgent`
- `MULTI_ENTITY` -> `SplitQueryAgent`
- `BELIEF_QUERY` -> belief-edge traversal + topic traversal + vector retrieval
- other categories -> vector retrieval + topic traversal

All paths can use listwise reranking. Semantic feature lookup is included when the
router sets `semantic_memory="SEARCH"`.

## Reflection and Health

Reflection is periodic or event-driven. It performs:

- LLM-guided belief decay decisions
- entrenchment detection
- optional segment consolidation and forgetting passes
- snapshot rewrite with validation guardrails

Health diagnostics are logged every interaction, with reflection-level checks for
coherence and consistency.
