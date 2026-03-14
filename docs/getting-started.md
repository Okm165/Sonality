# Getting Started

## Prerequisites

- Python 3.12+
- `uv`
- running Neo4j and PostgreSQL/pgvector (for Path A)
- OpenAI-compatible model endpoint

## Setup

```bash
git clone <repository-url>
cd sonality
make install
cp .env.example .env
```

Edit `.env`:

- provider: `SONALITY_BASE_URL`, `SONALITY_API_KEY`
- models: `SONALITY_MODEL`, `SONALITY_ESS_MODEL`, `SONALITY_EMBEDDING_MODEL`
- databases: `SONALITY_NEO4J_*`, `SONALITY_POSTGRES_URL`

## Run

```bash
make run
```

Docker:

```bash
docker compose up -d postgres neo4j
docker compose run --rm sonality
```

## Useful REPL Commands

- `/models`
- `/snapshot`
- `/beliefs`
- `/shifts`
- `/health`
- `/reset`
- `/quit`

## Verify Installation

```bash
make check
```

This runs lint, type-checking, and tests.
