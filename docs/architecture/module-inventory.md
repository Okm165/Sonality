# Module Inventory

> **Last verified:** April 2026  
> **Total Python modules:** 75

This document catalogs all Python modules in the Sonality codebase with their verified purposes.

## Core Package (`sonality/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | - | Package initialization |
| `__main__.py` | - | CLI entry point |
| `agent.py` | ~590 | Core agent orchestration, reflection, response generation |
| `api.py` | ~330 | FastAPI endpoints: /chat, /ingest, /health, /beliefs |
| `cli.py` | ~150 | Command-line interface (typer) |
| `config.py` | ~80 | Environment configuration variables |
| `ess.py` | ~520 | Evidence Strength Score classifier |
| `prompts.py` | ~730 | All prompt templates (17 constants) |
| `provider.py` | ~470 | LLM abstraction layer, streaming, JSON parsing |
| `schema.py` | ~150 | Neo4j/Qdrant schema definitions, enums |

## Memory Subsystem (`sonality/memory/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | - | Exports key classes |
| `belief_provenance.py` | ~230 | LLM-based belief evidence assessment |
| `consolidation.py` | ~200 | Segment summary generation |
| `db.py` | ~150 | DatabaseConnections class, schema init |
| `derivatives.py` | ~180 | Episode chunking operations |
| `dual_store.py` | ~300 | Atomic Neo4j + Qdrant writes |
| `embedder.py` | ~80 | FastEmbed wrapper (BAAI/bge-large-en-v1.5) |
| `forgetting.py` | ~400 | Memory lifecycle, archiving, pruning |
| `graph.py` | ~730 | Neo4j operations, EdgeType enum |
| `knowledge_extract.py` | ~500 | SLIDE-inspired proposition extraction |
| `segmentation.py` | ~120 | EventBoundaryDetector for conversations |
| `semantic_features.py` | ~550 | Personality/preference extraction worker |

## Retrieval Subsystem (`sonality/memory/retrieval/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | - | Exports retrieval router |
| `chain.py` | ~200 | Multi-hop chain-of-thought retrieval |
| `reranker.py` | ~200 | LLM-based listwise reranking |
| `router.py` | ~350 | Query classification, strategy selection |
| `split.py` | ~150 | Multi-query split retrieval |

## LLM Utilities (`sonality/llm/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | - | Exports llm_call |
| `caller.py` | ~250 | Structured LLM calls with retry, JSON repair |

## Chat Clients (`chat/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | - | Package initialization |
| `__main__.py` | - | CLI entry point for chat module |
| `audio.py` | ~220 | AudioProcessor: STT/TTS via Speaches |
| `client.py` | ~140 | SonalityClient: async API client |
| `config.py` | ~50 | Chat-specific configuration |
| `llm.py` | ~40 | LLM utility for TTS optimization |
| `telegram.py` | ~240 | Telegram bot (aiogram) |
| `terminal.py` | ~180 | Interactive REPL interface |

## Feed Scripts (`scripts/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `_helpers.py` | ~60 | Shared utilities for feed scripts |
| `feed.py` | ~310 | RSS/GNews feed ingestion |
| `memory_diagnostics.py` | ~100 | Debug utilities for memory inspection |
| `x_feed.py` | ~420 | X (Twitter) API feed ingestion |

## Benchmarks (`benches/`)

| Module | Lines | Purpose |
|--------|-------|---------|
| `__init__.py` | - | Package initialization |
| `conftest.py` | - | Pytest fixtures for benchmarks |
| `composed_scenarios.py` | - | Complex multi-step scenarios |
| `integrated_harness.py` | ~350 | Full pipeline benchmark harness |
| `knowledge_accumulation_scenarios.py` | - | Knowledge learning scenarios |
| `knowledge_harness.py` | ~200 | Knowledge benchmark harness |
| `live_scenarios.py` | - | Live environment scenarios |
| `psych_harness.py` | ~200 | Psychological profile benchmark |
| `scenario_contracts.py` | ~300 | Scenario/step dataclasses, metrics |
| `scenario_runner.py` | ~250 | Generic scenario execution |
| `teaching_harness.py` | ~250 | Knowledge teaching benchmark |
| `teaching_scenarios.py` | - | Teaching test cases |
| `test_*.py` | - | Various benchmark test files |

## Tests (`tests/`)

| Module | Purpose |
|--------|---------|
| `conftest.py` | Shared pytest fixtures |
| `containers.py` | Docker container management for tests |
| `test_api.py` | API endpoint tests |
| `test_ess_parsing.py` | ESS output parsing tests |
| `test_live_graduated.py` | Graduated live integration tests |
| `test_provider_timeout.py` | Provider timeout handling tests |
| `memory/test_*.py` | Memory subsystem unit tests |
| `memory/retrieval/test_*.py` | Retrieval strategy tests |

## Module Dependency Graph

```mermaid
flowchart TB
    subgraph core["Core"]
        agent["agent.py"]
        api["api.py"]
        prompts["prompts.py"]
        ess["ess.py"]
        config["config.py"]
        provider["provider.py"]
        schema["schema.py"]
    end

    subgraph memory["Memory"]
        db["db.py"]
        dual_store["dual_store.py"]
        graph["graph.py"]
        forgetting["forgetting.py"]
        embedder["embedder.py"]
        derivatives["derivatives.py"]
        consolidation["consolidation.py"]
        segmentation["segmentation.py"]
        knowledge["knowledge_extract.py"]
        semantic["semantic_features.py"]
        provenance["belief_provenance.py"]
    end

    subgraph retrieval["Retrieval"]
        router["router.py"]
        chain["chain.py"]
        split["split.py"]
        reranker["reranker.py"]
    end

    subgraph llm["LLM"]
        caller["caller.py"]
    end

    %% Core dependencies
    api --> agent
    agent --> ess
    agent --> prompts
    agent --> provider
    agent --> db

    %% Memory dependencies
    agent --> dual_store
    agent --> forgetting
    dual_store --> graph
    dual_store --> embedder
    dual_store --> derivatives
    db --> schema
    forgetting --> graph

    %% Retrieval dependencies
    agent --> router
    router --> chain
    router --> split
    router --> reranker

    %% LLM dependencies
    ess --> caller
    router --> caller
    knowledge --> caller
    semantic --> caller
    provenance --> caller
    caller --> provider

    %% Background workers
    agent --> semantic
    agent --> knowledge
    agent --> provenance
    agent --> segmentation
    agent --> consolidation
```

## Key Design Patterns

### 1. Structured LLM Calls
All LLM interactions that require structured output use `llm_call()` from `sonality/llm/caller.py`, which provides:
- Pydantic model validation
- JSON repair on parse failures
- Exponential backoff retry
- Graceful fallbacks

### 2. Dual Database Writes
`dual_store.py` ensures atomicity between Neo4j and Qdrant:
- Single transaction boundary
- Rollback on any failure
- UID-based correlation

### 3. Background Processing
Long-running tasks are handled by background workers:
- `SemanticIngestionWorker`: Feature extraction
- Knowledge extraction triggered post-ingest
- Reflection triggered by significant interactions

### 4. Prompt Centralization
All prompts live in `prompts.py`:
- Single source of truth
- Easy version control
- Consistent formatting

### 5. Configuration Layering
Environment → `config.py` → Default values:
- No hardcoded credentials
- All tunables externalized
- Sensible defaults for development

---

**See also:**
- `docs/architecture/verified-data-flow.md` - Complete data pipeline
- `docs/architecture/database-schema.md` - Storage schemas
- `docs/architecture/prompts-reference.md` - Prompt documentation
