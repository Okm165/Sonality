# Getting Started

## Prerequisites

- Python 3.12+
- `uv`
- running Neo4j and Qdrant (for Path A)
- OpenAI-compatible model endpoint

## Setup

```bash
git clone <repository-url>
cd sonality
make install
cp .env.example .env
```

Edit `.env`:

- `SONALITY_BASE_URL` — OpenAI-compatible endpoint
- `SONALITY_API_KEY` — API key (optional for local providers)
- `SONALITY_MODEL` — Main model (default: `gpt-4.1-mini`)

## Run

```bash
make run
```

Docker:

```bash
docker compose up -d qdrant neo4j
docker compose run --rm sonality
```

## Useful REPL Commands

- `/beliefs` - show current belief states
- `/health` - show agent health (belief count, snapshot version)
- `/clear` - clear conversation history
- `/help` - show available commands
- `/quit` - exit REPL

## Verify Installation

```bash
make check
```

This runs lint, type-checking, and tests.
