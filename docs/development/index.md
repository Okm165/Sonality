# Development

This section covers the development workflow for contributing to Sonality — a Python 3.12+ monorepo managed with uv, type-checked with Pyright, and formatted with Ruff.

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker and Docker Compose (for databases and services)
- Optional: AMD ROCm or NVIDIA CUDA (for local LLM inference)

## Setup

```bash
# Clone and install
git clone <repository-url>
cd Sonality
make install-dev   # installs all extras including dev tools

# Configure environment
cp .env.example .env
# Edit .env: set SONALITY_BASE_URL and SONALITY_API_KEY for your LLM provider

# Start databases
make db-up

# Verify setup
make check
```

## Project Layout

```
src/
├── sonality/       # Personality engine (28 files)
├── fathom/         # Web research service (14 files)
├── shared/         # Cross-service infrastructure (12 files)
└── chat/           # Client implementations (8 files)
tests/              # Unit tests
scripts/            # Utility scripts (feeds, diagnostics, model download)
docker/             # Service Dockerfiles
```

All source packages live under `src/` and are configured as a single hatchling wheel with four packages. Dependencies are managed in `pyproject.toml`.

## Quality Tooling

| Tool | Purpose | Command |
|------|---------|---------|
| **Ruff** | Lint + format (replaces flake8, isort, black) | `make lint` / `make format` |
| **Pyright** | Static type checking (standard mode) | `make typecheck` |
| **pytest** | Unit tests | `make test` |

### Running All Checks

```bash
make check      # lint + typecheck + test
make check-ci   # adds format-check + docs build (mirrors CI exactly)
```

### Formatting

```bash
make format         # Auto-format all source
make format-check   # Check without modifying (CI uses this)
```

Ruff is configured for Python 3.12 with 100-character line length. Selected rule sets: E, F, W, I, N, UP, B, SIM, RUF.

## Code Conventions

### Type Safety

- No `Any`, `object`, untyped dicts, bare `tuple`, bare `list`
- Prefer frozen dataclasses and `Final` for immutability
- Return structured types (dataclass/Pydantic model), never raw tuples or dicts
- All LLM outputs are parsed into typed Pydantic models

### Function Design

- Single responsibility, narrow context, few parameters
- Pure functions preferred; side effects isolated to explicit I/O boundaries
- One-liners when clarity is preserved
- No prop drilling — structured contexts over parameter chains

### Naming

- Match domain terminology exactly across the entire codebase
- If one module calls it `evidence_strength`, all modules call it `evidence_strength`
- Variable names match paper/spec terminology where applicable

### Module Structure

- Deep modules with small interfaces
- `shared` provides infrastructure; domain packages provide behavior
- No circular imports; dependency direction is always toward `shared`
- New files require strong justification — prefer extending existing modules

## Testing

See [Testing](testing.md) for the test architecture and philosophy.

## Continuous Integration

GitHub Actions (`.github/workflows/ci.yml`) runs on every push and PR:

1. Format check (ruff format --check)
2. Lint (ruff check)
3. Type check (pyright)
4. Unit tests (pytest, `not live` marker)
5. Docs build (zensical build --clean)

The CI does not require API keys — all tests that need external services are marked `live` and excluded.

## Documentation

Documentation is built with [Zensical](https://zensical.org) and deployed to GitHub Pages:

```bash
make docs          # Build static site to site/
make docs-serve    # Live-reload development server
```

The docs workflow (`.github/workflows/docs.yml`) automatically deploys on push to main/master when files in `docs/`, `zensical.toml`, or `README.md` change. Manual deployment is also available via workflow dispatch.

## Common Workflows

| Task | Command |
|------|---------|
| Run the agent interactively | `make run` |
| Start the HTTP API | `make serve` |
| Start Fathom research service | `make fathom-serve` |
| Feed news articles | `make feed` |
| Feed X/Twitter posts | `make x-feed` |
| Inspect current beliefs | `make beliefs` |
| Run memory health diagnostics | `make memory-diagnostics` |
| Reset personality (fresh start) | `make reset` |
| Full cleanup | `make nuke` |

### Content Ingestion

The feed scripts enable autonomous belief formation from external sources:

**`make feed`** — Fetches news articles from topic-organized RSS feeds (BBC, France24, DW, VOA) and GNews, then ingests each article through the full Sonality pipeline. Articles are ESS-classified and only high-quality content triggers belief updates. This enables the agent to form opinions about current events without manual conversation.

**`make x-feed`** — Fetches recent posts from X/Twitter by topic, ingests them with lower quality expectations (social media content typically scores lower on ESS). Useful for tracking emerging opinions and discourse patterns.

Both scripts use the `/ingest` API endpoint, meaning content passes through ESS classification and belief provenance assessment identically to interactive conversation.

### Memory Diagnostics

**`make memory-diagnostics`** — Runs comprehensive health checks on the dual-store:

- Cross-store consistency (Neo4j episodes have matching Qdrant vectors)
- Orphan detection (derivatives without parent episodes)
- Temporal chain integrity (no broken TEMPORAL_NEXT links)
- Payload completeness (vectors have all required metadata fields)
- Isolated node detection (graph nodes with no connections)

**`make memory-diagnostics-fix`** — Same checks with automatic repair of orphaned derivatives.
