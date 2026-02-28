# Getting Started

## Prerequisites

- **Python 3.12+** (the project targets 3.12–3.13)
- **[uv](https://docs.astral.sh/uv/)** — fast Python package manager
- **LLM API key** — an API key for the configured model provider (see [Model Considerations](model-considerations.md))

## Installation

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd sonality

# Install dependencies (creates .venv automatically)
make install

# Configure
cp .env.example .env
# Edit .env — add your SONALITY_API_KEY
```

## Running

### Local

```bash
make run
```

This starts an interactive REPL where you can chat with the agent and observe its personality evolving in real-time.

### Docker

```bash
cp .env.example .env
# Edit .env — add your API key
docker compose run --rm sonality
```

## REPL Commands

Once running, the REPL supports introspection commands:

| Command | Description |
|---|---|
| `/sponge` | Full personality state dump (JSON) |
| `/snapshot` | Current narrative snapshot text |
| `/beliefs` | Opinion vectors with confidence and evidence counts |
| `/insights` | Pending insights awaiting next reflection |
| `/staged` | Staged opinion updates awaiting cooling-period commit |
| `/topics` | Topic engagement counts |
| `/shifts` | Recent personality shifts with magnitudes |
| `/health` | Personality health and maturation metrics |
| `/diff` | Text diff of last sponge snapshot change |
| `/reset` | Reset to seed personality |
| `/quit` | Exit |

## First Interaction

When you start Sonality for the first time, it begins with a **seed personality**:

> *I'm a new mind, still forming my views. I'm analytically inclined — I find myself drawn to structural explanations and evidence-based reasoning over ideology or emotional appeals. I'm genuinely curious about ideas I haven't encountered yet...*

The first 10 interactions are in the **bootstrap phase** — opinion updates receive 0.5× dampening to prevent early conversations from having outsized influence on the agent's long-term personality.

Try these to see the system in action:

1. **Casual chat** (low ESS): *"Hey, how's it going?"* — should score near 0.0, no personality change.
2. **Bare assertion** (low ESS): *"I think AI is overhyped."* — should score < 0.15, no change.
3. **Structured argument** (high ESS): *"Studies on automation show that while AI displaces routine tasks, it historically creates more jobs than it eliminates — the Bureau of Labor Statistics data from 2010–2023 shows net job creation in every sector that adopted automation."* — should score > 0.5, triggers opinion update.

After each interaction, the REPL displays the ESS score, any topic updates, and whether the sponge was modified.

## Observing Evolution

Use `/shifts` to see the recent personality changes:

```
Recent shifts:
  #3 (mag 0.042): ESS 0.67: Agent showed stronger analytical lens when evaluating...
  #7 (mag 0.035): ESS 0.55: Developed nuanced view distinguishing empirical claims...
```

Use `/beliefs` to see the opinion vectors:

```
Beliefs:
  ai_automation: +0.04 (confidence=0.23, evidence=1, last=#3)
  technology_regulation: -0.08 (confidence=0.35, evidence=2, last=#7)
```

Use `/diff` after a reflection cycle (every 20 interactions) to see how the personality narrative changed.

## Makefile Commands

```
make install       Install dependencies (creates .venv)
make install-dev   Install with dev tools (ruff, pytest, mypy)
make run           Start the Sonality REPL
make lint          Lint code with ruff
make format        Format code with ruff
make typecheck     Type-check with mypy
make test          Run pytest suite
make check         Run all quality checks (lint + typecheck + test)
make docker-build  Build Docker image
make docker-run    Run in Docker (interactive)
make sponge        Print current sponge state
make shifts        Print recent personality shifts
make reset         Reset sponge to seed state
make docs          Build documentation site (output in site/)
make docs-serve    Serve documentation locally with live reload
make clean         Remove caches
make nuke          Full reset (remove .venv, data, caches)
```

## Project Structure

```
sonality/
├── pyproject.toml              Dependencies and tool config
├── Makefile                    Dev workflows
├── Dockerfile                  Container build
├── docker-compose.yml          Container orchestration
├── .env.example                Configuration template
├── sonality/                   Python package
│   ├── __init__.py             Package version
│   ├── __main__.py             python -m sonality
│   ├── cli.py                  Terminal REPL
│   ├── agent.py                Core loop: context → LLM → post-process
│   ├── config.py               Environment + defaults
│   ├── prompts.py              Prompt templates
│   ├── ess.py                  Evidence Strength Score classifier
│   └── memory/                 Memory subsystem
│       ├── __init__.py         Re-exports
│       ├── sponge.py           SpongeState model, persistence
│       ├── episodes.py         ChromaDB episode storage + retrieval
│       └── updater.py          Magnitude computation, snapshot validation
├── tests/                      Test suite (166 tests)
└── data/                       Runtime data (gitignored)
    ├── sponge.json             Current personality state
    ├── sponge_history/         Archived versions
    ├── chromadb/               Episode vector store
    └── ess_log.jsonl           Audit trail
```
