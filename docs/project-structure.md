# Project Structure

Complete inventory of Sonality's codebase organization.

## Directory Layout

```
sonality/
├── sonality/                 # Core runtime package
│   ├── llm/                  # LLM utilities
│   └── memory/               # Memory subsystem
│       └── retrieval/        # Retrieval pipeline
├── chat/                     # API clients and UIs
├── scripts/                  # Feed ingestion CLIs
├── tests/                    # Unit and integration tests
│   └── memory/               # Memory-specific tests
├── benches/                  # Benchmark harnesses
├── docs/                     # Documentation (Zensical)
│   ├── architecture/
│   ├── concepts/
│   └── research/
├── data/                     # Runtime data (gitignored)
└── .github/workflows/        # CI configuration
```

## Top-Level Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package metadata, dependencies, tool configuration |
| `Makefile` | Developer workflows and commands |
| `Dockerfile` | Container build (Python 3.13 + uv + FastEmbed) |
| `docker-compose.yml` | Multi-service stack (app + Neo4j + Qdrant + Speaches) |
| `zensical.toml` | Documentation site configuration |
| `.env.example` | Environment variable template |
| `conftest.py` | Root pytest configuration and logging |
| `README.md` | Project overview and quick start |

## Core Package (`sonality/`)

### Top-Level Modules

| Module | Lines | Responsibility |
|--------|-------|----------------|
| `__init__.py` | — | Package version (`0.1.0`) |
| `__main__.py` | — | Entry point for `python -m sonality` → CLI |
| `agent.py` | ~500 | **Core orchestration**: respond, ingest, reflect, post-processing |
| `api.py` | ~330 | **FastAPI server**: OpenAI-compatible + custom endpoints |
| `cli.py` | ~200 | **Interactive REPL** with inspection commands |
| `config.py` | ~150 | Environment loading, constants, tuning parameters |
| `ess.py` | ~200 | **ESS classifier**: score, reasoning type, topics, belief recommendation |
| `prompts.py` | ~400 | All agent-level prompt templates |
| `provider.py` | ~300 | **LLM HTTP client**: chat, streaming, retry, JSON repair |
| `schema.py` | ~150 | Qdrant collection schemas, Neo4j schema statements |

### LLM Utilities (`sonality/llm/`)

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | Package exports |
| `caller.py` | `llm_call()`: structured Pydantic outputs, retry loop, JSON repair |

### Memory Subsystem (`sonality/memory/`)

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | Public exports for memory operations |
| `db.py` | **`DatabaseConnections`**: Neo4j driver + Qdrant client lifecycle |
| `graph.py` | **`MemoryGraph`**: Cypher for episodes, beliefs, topics, segments, edges |
| `dual_store.py` | **Episode pipeline**: chunk → embed → Neo4j + Qdrant with rollback |
| `embedder.py` | **FastEmbed** wrapper with query cache |
| `derivatives.py` | LLM semantic chunking into derivative units |
| `knowledge_extract.py` | Knowledge proposition extraction to Qdrant |
| `segmentation.py` | Event boundary detection for conversation segments |
| `consolidation.py` | Segment readiness and summary generation |
| `forgetting.py` | Batch decisions: KEEP / ARCHIVE / FORGET |
| `belief_provenance.py` | Evidence-belief assessment with AGM contraction |
| `semantic_features.py` | **`SemanticIngestionWorker`**: async feature extraction |

### Retrieval Pipeline (`sonality/memory/retrieval/`)

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | Re-exports: `route_query`, `chain_retrieve`, `split_retrieve`, `rerank_episodes` |
| `router.py` | **Query routing** LLM: category + flags |
| `chain.py` | **Iterative retrieval** with sufficiency checks |
| `split.py` | **Query decomposition** for multi-entity questions |
| `reranker.py` | **Listwise LLM reranking** of episode candidates |

## Chat Clients (`chat/`)

| Module | Responsibility |
|--------|----------------|
| `__init__.py` | Package init |
| `__main__.py` | Dispatcher for `python -m chat terminal|telegram` |
| `config.py` | Chat-specific env: Telegram token, API URLs, Speaches config |
| `client.py` | **`SonalityClient`**: async httpx client with history management |
| `terminal.py` | **Rich TUI** for local chat sessions |
| `telegram.py` | **aiogram bot**: voice + text messaging |
| `audio.py` | **Speaches integration**: STT/TTS over HTTP |
| `llm.py` | LLM wrapper for TTS text optimization |

## Feed Scripts (`scripts/`)

| Script | Responsibility |
|--------|----------------|
| `feed.py` | **RSS + GNews** aggregation → `/ingest` loop |
| `x_feed.py` | **X API v2** search → `/ingest` loop |
| `memory_diagnostics.py` | Neo4j + Qdrant consistency checks, orphan cleanup |
| `_helpers.py` | Shared Rich printing utilities |

## Test Suite (`tests/`)

| File | Coverage |
|------|----------|
| `test_api.py` | FastAPI endpoint tests (mocked agent) |
| `test_live_graduated.py` | Infrastructure validation (L0–L3x) |
| `test_provider_timeout.py` | Provider timeout behavior |
| `test_ess_parsing.py` | ESS response parsing |
| `containers.py` | Testcontainers for Neo4j/Qdrant |
| `conftest.py` | Test fixtures and session setup |
| `memory/test_derivatives.py` | Chunk/derivative tests |
| `memory/test_forgetting.py` | Memory forgetting tests |
| `memory/test_segmentation.py` | Episode segmentation tests |
| `memory/retrieval/test_*.py` | Retrieval pipeline tests (chain, split, router, reranker) |

## Benchmark Suite (`benches/`)

| File | Purpose |
|------|---------|
| `teaching_harness.py` | Core benchmark runner |
| `teaching_scenarios.py` | Scenario pack definitions |
| `scenario_runner.py` | Pack execution engine |
| `scenario_contracts.py` | Contract verification |
| `integrated_harness.py` | Multi-scenario integration |
| `live_scenarios.py` | Live API scenarios |
| `knowledge_harness.py` | Knowledge acquisition tests |
| `psych_harness.py` | Psychological stability tests |
| `test_teaching_suite_live.py` | 60-pack teaching suite |
| `test_knowledge_*.py` | Knowledge battery tests |
| `conftest.py` | Benchmark fixtures |

## Documentation (`docs/`)

| Directory/File | Content |
|----------------|---------|
| `index.md` | Landing page with architecture snapshot |
| `getting-started.md` | Quick start guide |
| `configuration.md` | Environment variable reference |
| `api-reference.md` | HTTP endpoint documentation |
| `testing.md` | Test suite overview |
| `architecture/` | System design documents |
| `concepts/` | Core concepts (ESS, reflection, opinion dynamics) |
| `research/` | Academic background and references |

## Entry Points

Declared in `pyproject.toml [project.scripts]`:

| Command | Target | Purpose |
|---------|--------|---------|
| `sonality` | `sonality.cli:main` | Interactive REPL |
| `sonality-server` | `sonality.api:serve` | FastAPI server (uvicorn) |
| `sonality-chat` | `chat.terminal:main` | Rich terminal client |
| `sonality-telegram` | `chat.telegram:main` | Telegram bot |

## Dependency Groups

| Group | Purpose | Key packages |
|-------|---------|--------------|
| *(default)* | Runtime | fastembed, fastapi, uvicorn, neo4j, qdrant-client, pydantic |
| `scripts` | Feed ingestion | openai, httpx, rich, feedparser, gnews, aiogram |
| `dev` | Development | ruff, pytest, mypy, playwright, testcontainers |
| `docs` | Documentation | zensical |

## Runtime Data (`data/`)

| Path | Purpose |
|------|---------|
| `ess_log.jsonl` | ESS classification audit trail |
| `teaching_bench/` | Benchmark output artifacts |

Note: Personality state is stored in Neo4j (`PersonalitySnapshot` node), not local files.

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| `sonality` | 8000 | FastAPI application |
| `neo4j` | 7474, 7687 | Graph database |
| `qdrant` | 6333, 6334 | Vector database |
| `speaches` | 8001 | STT/TTS (Whisper + Kokoro) |

## CI/CD (`.github/workflows/`)

| Workflow | Triggers | Actions |
|----------|----------|---------|
| `ci.yml` | push/PR to main | ruff, mypy, pytest (tests + non-live benches) |
| `docs.yml` | push to main | zensical build → GitHub Pages |
