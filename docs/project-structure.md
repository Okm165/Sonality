# Project Structure

This page explains what each major repository area owns, why it exists, and where new code should go.
It is the structural reference for contributors who need to keep Sonality small, auditable, and coherent.

## Top-Level Layout

| Path | Purpose | Why It Exists |
|---|---|---|
| `sonality/` | Runtime package | Keeps production behavior in one focused package with minimal dependencies. |
| `tests/` | Correctness and invariant tests | Protects runtime logic and memory behavior without API access. |
| `benches/` | Evaluation and benchmark harnesses | Isolates heavy validation and release-gating logic from runtime code. |
| `docs/` | User-facing project documentation | Stores architecture, concepts, tutorials, and references in one place. |
| `data/` | Runtime state (gitignored) | Persists personality state, memory store, and audit traces across sessions. |

## Runtime Package Responsibilities

`sonality/` is intentionally flat at the top level. Each module has one clear responsibility. The `memory/` and `llm/` subpackages group related concerns.

| Module | Owns | Why Separate |
|---|---|---|
| `sonality/agent.py` | End-to-end interaction orchestration (`respond`) + async bridge | Keeps the full runtime control flow in one canonical entrypoint. |
| `sonality/cli.py` | REPL UX, command dispatch, startup checks | Separates operator interaction concerns from runtime cognition logic. |
| `sonality/config.py` | Environment/config constants and client wiring | Ensures one canonical source of runtime configuration. |
| `sonality/prompts.py` | Prompt templates and prompt assembly helpers | Prevents prompt drift across call sites. |
| `sonality/ess.py` | Evidence Strength Score classification + coercion fallback | Isolates schema handling and retries from the core agent loop. |
| `sonality/llm/caller.py` | Structured JSON LLM calls with Pydantic response models | Centralizes LLM invocation pattern for memory subsystem tasks. |
| `sonality/llm/prompts.py` | Prompt templates for memory LLM tasks (chunking, forgetting, routing, etc.) | Separates memory-subsystem prompts from agent-level prompts. |
| `sonality/memory/sponge.py` | `SpongeState`, belief math, decay, persistence | Centralizes personality-state mechanics and serialization. |
| `sonality/memory/episodes.py` | Episode storage/retrieval over ChromaDB (legacy fallback) | Keeps legacy memory indexing available as fallback. |
| `sonality/memory/updater.py` | Update magnitude, snapshot validation, insight extraction | Isolates update-policy mechanics from both storage and orchestration. |
| `sonality/memory/stm.py` | Short-Term Memory: bounded deque + PostgreSQL persistence | Separates short-term conversation buffer from long-term storage. |
| `sonality/memory/stm_consolidator.py` | Background LLM summarization of evicted STM messages | Keeps async summarization logic out of the STM buffer. |
| `sonality/memory/graph.py` | Neo4j graph model: `EpisodeNode`, `DerivativeNode`, `EdgeType` | Centralizes graph schema and CRUD operations. |
| `sonality/memory/dual_store.py` | `DualEpisodeStore`: Neo4j graph + pgvector derivative storage | Coordinates graph and vector storage into a single store API. |
| `sonality/memory/derivatives.py` | LLM-based semantic chunking into derivative embeddings | Isolates chunking logic from storage logic. |
| `sonality/memory/embedder.py` | External embedding provider (OpenAI/OpenRouter) | Keeps embedding API calls separate from retrieval logic. |
| `sonality/memory/db.py` | Database connection management (Neo4j + PostgreSQL pools) | Single factory for all database connections. |
| `sonality/memory/segmentation.py` | LLM event boundary detection for conversation segmentation | Separates conversation segmentation from episode storage. |
| `sonality/memory/consolidation.py` | Consolidation engine for merging related episodes | Keeps episode merging logic separate from reflection. |
| `sonality/memory/forgetting.py` | LLM-based importance assessment with soft archival | Isolates forgetting decisions from reflection orchestration. |
| `sonality/memory/health.py` | LLM memory health assessment | Keeps health diagnostics separate from maintenance actions. |
| `sonality/memory/belief_provenance.py` | Belief-episode provenance linking via Neo4j graph edges | Tracks which episodes support which beliefs. |
| `sonality/memory/semantic_features.py` | Background semantic feature extraction worker | Extracts structured features from episodes asynchronously. |
| `sonality/memory/retrieval/router.py` | `QueryRouter`: classify queries by category (factual/personal/recent/meta) | Routes queries to the optimal retrieval strategy. |
| `sonality/memory/retrieval/chain.py` | `ChainOfQueryAgent`: iterative retrieval with refinement | Handles factual queries that may need multiple search passes. |
| `sonality/memory/retrieval/split.py` | `SplitQueryAgent`: parallel sub-query decomposition | Handles complex queries by splitting into independent sub-queries. |
| `sonality/memory/retrieval/reranker.py` | LLM listwise reranking of retrieval results | Replaces heuristic reranking with LLM judgment. |
| `sonality/memory/__init__.py` | Memory subsystem public exports | Keeps import surface stable for callers. |
| `sonality/__main__.py` | `python -m sonality` entrypoint | Provides a standard module-execution path. |
| `sonality/__init__.py` | Package metadata | Keeps package-level metadata in one location. |

## Evaluation Layer Responsibilities

`benches/` contains evaluation logic that is intentionally decoupled from runtime.

| Module | Owns | Why Separate |
|---|---|---|
| `benches/scenario_contracts.py` | Declarative step expectations and scenario contracts | Keeps benchmark expectations typed and reusable across packs. |
| `benches/teaching_scenarios.py` and `benches/live_scenarios.py` | Scenario definitions | Keeps behavioral/evaluation content data-driven and auditable. |
| `benches/scenario_runner.py` | Deterministic scenario execution and step-level checks | Reuses one runner across many benchmark packs. |
| `benches/teaching_harness.py` | Multi-pack orchestration, traces, gate outcomes, summaries | Centralizes release-gating artifact generation in one place. |
| `benches/test_*.py` | Non-live/live benchmark validation | Verifies evaluation contracts without mixing into `tests/`. |

## Test Layer Responsibilities

`tests/` protects runtime correctness and memory behavior:

- Runtime behavior and math invariants (`tests/test_behavioral.py`, `tests/test_sponge.py`, `tests/test_ess_parsing.py`).
- Storage/retrieval and memory behavior (`tests/test_episodes.py`).

## Runtime Data Artifacts

All persistent runtime artifacts live under `data/` (legacy ChromaDB) and external databases (new architecture):

| Path / Service | Meaning |
|---|---|
| `data/sponge.json` | Current personality state snapshot and structured belief state |
| `data/sponge_history/` | Versioned historical sponge states for rollback and inspection |
| `data/chromadb/` | Episode vector database files (legacy fallback) |
| `data/ess_log.jsonl` | Structured runtime audit events |
| Neo4j | Episode graph nodes, derivative nodes, provenance edges, temporal edges |
| PostgreSQL (pgvector) | Derivative embedding vectors for similarity search |
| PostgreSQL (tables) | STM state, semantic features |

## Placement Rules for New Code

- Add production runtime logic to `sonality/` only.
- Add release/evaluation logic to `benches/`, not to runtime modules.
- Add correctness/invariant checks to `tests/`; keep benchmark-specific tests in `benches/`.
- Add architectural explanations, tutorials, and references to `docs/`; avoid development-history logs.

