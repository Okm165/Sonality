.DEFAULT_GOAL := help
SHELL := /bin/bash

# --- Help ---

.PHONY: help
help: ## Show available commands
	@echo ""
	@echo "  Zensical — Sonality + Fathom + Chat monorepo"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# --- Setup ---

.PHONY: install install-dev
install: ## Install dependencies with uv (creates local .venv)
	uv sync

install-dev: ## Install with dev dependencies (ruff, pytest, pyright)
	uv sync --all-extras

# --- Database ---

.PHONY: db-up db-down db-reset db-clear
db-up: ## Start database containers (Neo4j + Qdrant — shared by sonality and fathom)
	docker compose up -d neo4j qdrant

db-down: ## Stop database containers
	docker compose down

db-reset: db-down ## Reset databases (delete all data and restart)
	docker volume rm -f sonality_neo4j_data sonality_qdrant_data sonality_neo4j_logs 2>/dev/null || true
	docker compose up -d neo4j qdrant
	@echo "Databases reset. Schema is applied automatically on next startup."

db-clear: ## Clear all data from databases while preserving schema
	@echo "Clearing Qdrant collections..."
	curl -s -X DELETE "http://localhost:6333/collections/derivatives" || true
	curl -s -X DELETE "http://localhost:6333/collections/semantic_features" || true
	@echo "Clearing Neo4j data..."
	docker compose exec -T neo4j cypher-shell -u neo4j -p sonality_password "MATCH (n) DETACH DELETE n"
	@echo "Databases cleared (Qdrant collections will be recreated on next startup)"

# --- Run ---

.PHONY: run serve chat telegram feed x-feed fathom fathom-serve up down preflight-live preflight-live-probe
run: ## Start the Sonality REPL agent
	uv run sonality $(ARGS)

serve: ## Start the Sonality API server (OpenAI-compatible at /v1/chat/completions)
	uv run sonality-server --host 0.0.0.0 --port 8000 $(ARGS)

fathom-serve: ## Start Fathom search + research service (port 8010)
	uv run fathom-server --host 0.0.0.0 --port 8010 $(ARGS)

up: ## Start all services (databases + fathom + sonality)
	docker compose up -d

down: ## Stop all services
	docker compose down

chat: ## Interactive terminal chat with Sonality
	uv run --extra chat sonality-chat

telegram: ## Start Telegram bot (requires CHAT_TELEGRAM_TOKEN)
	uv run --extra chat sonality-telegram

# GNEWS_LIMIT=10 RSS_ENTRIES=10 FEED_THROTTLE=5 make feed
feed: ## Feed news articles to Sonality for belief formation
	uv run --extra scripts python scripts/feed.py

# X_FEED_MAX_RESULTS=10 X_FEED_SORT_ORDER=relevancy make x-feed
x-feed: ## Feed X (Twitter) posts to Sonality for belief formation
	uv run --extra scripts python scripts/x_feed.py

preflight-live: ## Validate live API config and selected models
	@uv run python -c "exec('''from sonality import config\nimport sys\nimport json\nfrom urllib.request import Request, urlopen\nfrom urllib.error import URLError, HTTPError\n\nmissing = list(config.missing_live_api_config())\nif missing:\n    print(\"Missing required config: \" + \", \".join(missing))\n    sys.exit(1)\n\nprint(\"Live config OK\")\nprint(\"  Base URL:   \" + config.BASE_URL)\nprint(\"  Model:      \" + config.MODEL)\nprint(\"  Structured: \" + config.STRUCTURED_MODEL)\n\nif config.MODEL == config.STRUCTURED_MODEL:\n    print(\"  Note: main and structured model are the same\")\n\nbase = config.BASE_URL.rstrip(\"/\")\nbase_root = base.rsplit(\"/v1\", 1)[0] if base.endswith(\"/v1\") else base\navailable_models = []\ntry:\n    req = Request(base_root + \"/api/tags\", method=\"GET\")\n    with urlopen(req, timeout=10) as resp:\n        data = json.loads(resp.read())\n        available_models = [m[\"name\"] for m in data.get(\"models\", [])]\nexcept (URLError, HTTPError, json.JSONDecodeError, KeyError):\n    try:\n        models_url = base + \"/models\" if base.endswith(\"/v1\") else base + \"/v1/models\"\n        req = Request(models_url, method=\"GET\")\n        if config.API_KEY:\n            req.add_header(\"Authorization\", \"Bearer \" + config.API_KEY)\n        with urlopen(req, timeout=10) as resp:\n            data = json.loads(resp.read())\n            available_models = [m[\"id\"] for m in data.get(\"data\", [])]\n    except (URLError, HTTPError, json.JSONDecodeError, KeyError):\n        pass\n\nif available_models:\n    print(\"  Provider OK (\" + str(len(available_models)) + \" models available)\")\n    model_ok = any(config.MODEL in m or m in config.MODEL for m in available_models)\n    structured_ok = any(config.STRUCTURED_MODEL in m or m in config.STRUCTURED_MODEL for m in available_models)\n    if not model_ok:\n        print(\"  ERROR: Model '\" + config.MODEL + \"' not found on provider\")\n        print(\"  Available: \" + \", \".join(available_models[:5]) + (\"...\" if len(available_models) > 5 else \"\"))\n        sys.exit(1)\n    if not structured_ok:\n        print(\"  ERROR: Structured model '\" + config.STRUCTURED_MODEL + \"' not found on provider\")\n        sys.exit(1)\n    print(\"  Models verified on provider\")\nelse:\n    print(\"  Warning: Could not list models from provider\")\n''')"

preflight-live-probe: preflight-live ## Run a tiny live call to verify endpoint/model/policy access
	@uv run python -c "exec('''import json\nimport sys\nfrom sonality import config\nfrom sonality.caller import provider as default_provider\n\ntry:\n    result = default_provider.chat_completion(\n        model=config.MODEL,\n        max_tokens=8,\n        messages=({\"role\": \"user\", \"content\": \"Reply with OK only.\"},),\n        temperature=0.0,\n    )\n    text = result.text.strip()\n    print(\"Live probe OK\")\n    print(\"  Probe model: \" + config.MODEL)\n    print(\"  Probe response preview: \" + (text[:40] if text else \"<empty>\"))\nexcept Exception as exc:\n    print(f\"Live probe failed: {exc.__class__.__name__}\")\n    print(f\"  {exc}\")\n    if isinstance(exc, RuntimeError):\n        print(\"  Hint: verify SONALITY_BASE_URL, SONALITY_API_KEY, and model IDs.\")\n    sys.exit(1)\n''')"

# --- Quality ---

.PHONY: lint format format-check typecheck test check check-ci
lint: ## Lint code (ruff check)
	uv run ruff check src/ tests/

format: ## Format code (ruff format)
	uv run ruff format src/ tests/
	uv run ruff check --fix src/ tests/

format-check: ## Check formatting without writing changes (CI parity)
	uv run ruff format --check src/ tests/

typecheck: ## Type-check code (pyright)
	uv run pyright src/sonality/ src/fathom/ src/shared/ src/chat/

test: ## Run tests (pytest, skip live API tests)
	uv run pytest tests -m "not live" -v -s

test-report: ## Run tests with JSON report and summary table
	uv run pytest tests -m "not live" -v --tb=short --json-report --json-report-file=test-report.json 2>/dev/null || \
		uv run pytest tests -m "not live" -v --tb=short --junitxml=test-report.xml
	@echo ""
	@echo "Test report written to test-report.json (or test-report.xml)"

check: lint typecheck test ## Run all no-key quality checks

check-ci: format-check check docs ## Run local checks equivalent to CI

# --- Diagnostics ---

.PHONY: memory-diagnostics memory-diagnostics-fix
memory-diagnostics: ## Run Neo4j + Qdrant memory health diagnostics
	uv run python scripts/memory_diagnostics.py

memory-diagnostics-fix: ## Run memory diagnostics and auto-fix orphan derivatives
	uv run python scripts/memory_diagnostics.py --fix-orphans

# --- Docker ---

.PHONY: docker-build docker-build-sonality docker-build-fathom docker-run
docker-build: docker-build-sonality docker-build-fathom ## Build all Docker images

docker-build-sonality: ## Build Sonality Docker image
	docker build -f docker/sonality.Dockerfile -t sonality .

docker-build-fathom: ## Build Fathom Docker image
	docker build -f docker/fathom.Dockerfile -t fathom .

docker-run: ## Run Sonality agent in Docker (interactive)
	docker compose run --rm sonality

# --- Inspect ---

.PHONY: beliefs
beliefs: ## Show current beliefs from graph
	@uv run python -c "import asyncio; from sonality.memory import DatabaseConnections, MemoryGraph; \
		async def _show(): \
			db = await DatabaseConnections.create(); g = MemoryGraph(db.neo4j_driver); \
			beliefs = await g.get_all_beliefs(); \
			[print(f'  {b.topic}: val={b.valence:+.2f} conf={b.confidence:.2f} — {b.belief_text[:60]}') for b in beliefs] if beliefs else print('  No beliefs yet.'); \
			await db.close(); \
		asyncio.run(_show())"

# --- Docs ---

.PHONY: docs docs-serve
docs: ## Build documentation site (output in site/)
	uv run --group docs zensical build --clean

docs-serve: ## Serve documentation locally with live reload
	uv run --group docs zensical serve

# --- Utility ---

.PHONY: reset clean nuke
reset: ## Reset personality snapshot (preserves .venv)
	@uv run python -c "import asyncio; from sonality.memory import DatabaseConnections, MemoryGraph; \
		async def _reset(): \
			db = await DatabaseConnections.create(); g = MemoryGraph(db.neo4j_driver); \
			await g.upsert_personality_snapshot(''); await db.close(); \
		asyncio.run(_reset())" 2>/dev/null || true
	@echo "Personality reset. Next run starts from seed."

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .mypy_cache/ .ruff_cache/ .pytest_cache/ htmlcov/ .coverage
	@echo "Cleaned."

nuke: clean reset ## Full reset — remove .venv, data, and all caches
	rm -rf .venv/
	@echo "Nuked. Run 'make install' to start fresh."
