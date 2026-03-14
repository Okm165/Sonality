# Project Structure

## Top Level

| Path | Purpose |
|---|---|
| `sonality/` | Runtime package |
| `tests/` | Deterministic unit tests |
| `benches/` | Benchmark and live-eval harnesses |
| `docs/` | User-facing documentation |

## Runtime Modules

| Module | Responsibility |
|---|---|
| `sonality/agent.py` | Main orchestration (`respond`) |
| `sonality/provider.py` | Unified OpenAI-compatible API client |
| `sonality/ess.py` | ESS classification and coercion-safe parsing |
| `sonality/prompts.py` | System prompt construction |
| `sonality/config.py` | Configuration and defaults |
| `sonality/memory/graph.py` | Neo4j graph storage/traversal |
| `sonality/memory/dual_store.py` | Dual-store write and consistency |
| `sonality/memory/semantic_features.py` | Feature ingestion worker |
| `sonality/memory/sponge.py` | Personality state and persistence |
| `sonality/memory/retrieval/` | Router, chain/split agents, reranker |

## Runtime Data

| Path | Meaning |
|---|---|
| `data/sponge.json` | Current personality state |
| `data/sponge_history/` | Archived snapshots |
| `data/ess_log.jsonl` | Audit events |

Persistent episodic/semantic memory lives in Neo4j/PostgreSQL, not local Chroma files.
