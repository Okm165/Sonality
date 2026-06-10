# Quick Start

This guide covers getting the Sonality LLM personality agent running locally in under five minutes. Sonality requires Python 3.12+, Docker for infrastructure services, and optionally a GPU for local LLM inference.

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Docker and Docker Compose (for infrastructure services)

---

## Cloud Provider Setup

The fastest path uses a cloud LLM provider (OpenAI, Anthropic via proxy, OpenRouter):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/sonality/sonality.git && cd sonality
make install
cp .env.example .env
```

Edit `.env` to set your provider:

```bash
SONALITY_BASE_URL=https://api.openai.com/v1
SONALITY_API_KEY=sk-...
SONALITY_MODEL=gpt-4.1-mini
```

Start the databases and run:

```bash
make db-up    # Start Neo4j + Qdrant
make run      # Launch interactive REPL
```

---

## Full Local Stack

For completely local operation (no external API calls), use the Docker Compose stack:

```bash
cp .env.example .env
docker compose up -d
```

This starts all services:

| Service | Port | Function |
|---------|------|----------|
| Sonality | 8000 | Agent API |
| Fathom | 8010 | Research engine |
| Speaches | 8020 | STT/TTS (optional) |
| Browserless | 8030 | Web fetching |
| llama-cpp | 8080 | LLM inference (ROCm/CUDA) |
| llama-cpp-embed | 8090 | Embedding server |
| Neo4j | 7474/7687 | Graph database |
| Qdrant | 6333/6334 | Vector database |

The LLM server uses a 262K context window with a quantized model. First startup downloads model weights (~4GB for the default model).

---

## Standalone Agent

Run Sonality against your own LLM endpoint (local llama.cpp, Ollama, vLLM, etc.):

```bash
cp .env.example .env
# Edit .env: set SONALITY_BASE_URL to your endpoint
make install
make db-up
make run
```

---

## Verify Installation

```bash
make preflight-live        # Validate config and model availability
make preflight-live-probe  # Send a tiny real request to verify access
make check                 # Run lint + typecheck + tests (no API needed)
```

---

## REPL Commands

Once running, the interactive REPL provides:

| Command | Description |
|---------|-------------|
| `/snapshot` | Display current personality narrative |
| `/beliefs` | Show all belief vectors with confidence scores |
| `/models` | Show active model configuration |
| `/quit` | Exit |

---

## API Access

For programmatic access, start the server:

```bash
make serve  # Starts on :8000
```

The API follows the OpenAI chat completions contract:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "What do you think about renewable energy?"}]}'
```

---

## Chat Client

For a richer interactive experience with streaming progress display:

```bash
make chat  # Rich terminal TUI with tool progress visibility
```

---

## Next Steps

- [Configuration](configuration.md) --- All environment variables and their effects
- [Docker Stack](docker.md) --- Detailed infrastructure setup and tuning
