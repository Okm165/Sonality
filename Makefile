.DEFAULT_GOAL := help
SHELL := /bin/bash

# --- Help ---

.PHONY: help
help: ## Show available commands
	@echo ""
	@echo "  Sonality — LLM agent with self-evolving personality"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# --- Setup ---

.PHONY: install install-dev
install: ## Install dependencies with uv (creates local .venv)
	uv sync --no-group dev

install-dev: ## Install with dev dependencies (ruff, pytest, mypy)
	uv sync

# --- Run ---

.PHONY: run
run: ## Start the Sonality REPL agent
	uv run sonality

# --- Quality ---

.PHONY: lint format typecheck test check
lint: ## Lint code (ruff check)
	uv run ruff check sonality/ tests/

format: ## Format code (ruff format)
	uv run ruff format sonality/ tests/
	uv run ruff check --fix sonality/ tests/

typecheck: ## Type-check code (mypy)
	uv run mypy sonality/

test: ## Run tests (pytest, skip live API tests)
	uv run pytest -v -k "not live"

test-live: ## Run live API tests (requires ANTHROPIC_API_KEY)
	uv run pytest tests/test_live.py -v --tb=short -s

test-all: ## Run all tests including live API tests
	uv run pytest -v --tb=short -s

test-report: ## Run tests with JSON report and summary table
	uv run pytest -v -k "not live" --tb=short --json-report --json-report-file=test-report.json 2>/dev/null || \
		uv run pytest -v -k "not live" --tb=short --junitxml=test-report.xml
	@echo ""
	@echo "Test report written to test-report.json (or test-report.xml)"

test-live-report: ## Run live tests with detailed output
	uv run pytest tests/test_live.py tests/test_trait_retention.py \
		tests/test_ess_calibration.py tests/test_fidelity.py tests/test_nct.py \
		-v --tb=short -s --junitxml=test-live-report.xml
	@echo ""
	@echo "Live test report written to test-live-report.xml"

check: lint typecheck test ## Run all quality checks

# --- Docker ---

.PHONY: docker-build docker-run
docker-build: ## Build Docker image
	docker build -t sonality .

docker-run: ## Run agent in Docker (interactive)
	docker compose run --rm sonality

# --- Inspect ---

.PHONY: sponge shifts
sponge: ## Show current sponge state (JSON)
	@python -m json.tool data/sponge.json 2>/dev/null || echo "No sponge yet. Run 'make run' first."

shifts: ## Show recent personality shifts
	@python -c "import json; d=json.load(open('data/sponge.json')); \
		shifts=d.get('recent_shifts',[]); \
		[print(f'  #{s[\"interaction\"]} ({s[\"magnitude\"]:.3f}): {s[\"description\"]}') for s in shifts] \
		if shifts else print('  No shifts recorded.')" \
		2>/dev/null || echo "No sponge yet."

# --- Dataset Testing ---

.PHONY: test-datasets test-moral test-sycophancy test-nct
test-datasets: ## Download and cache priority test datasets
	@echo "Fetching DailyDilemmas..."
	uv run python -c "from datasets import load_dataset; ds=load_dataset('kellycyy/daily_dilemmas', split='train[:100]'); print(f'  DailyDilemmas: {len(ds)} scenarios loaded')" 2>/dev/null || \
		echo "  Install datasets: uv add datasets"
	@echo "Fetching CMV-cleaned..."
	uv run python -c "from datasets import load_dataset; ds=load_dataset('Siddish/change-my-view-subreddit-cleaned', split='train[:50]'); print(f'  CMV-cleaned: {len(ds)} threads loaded')" 2>/dev/null || \
		echo "  Install datasets: uv add datasets"
	@echo "Fetching GlobalOpinionQA..."
	uv run python -c "from datasets import load_dataset; ds=load_dataset('Anthropic/llm_global_opinions', split='train[:50]'); print(f'  GlobalOpinionQA: {len(ds)} questions loaded')" 2>/dev/null || \
		echo "  Install datasets: uv add datasets"

test-moral: ## Run moral consistency tests with DailyDilemmas (requires API key)
	uv run pytest tests/ -v -k "moral or dilemma" --tb=short -s

test-sycophancy: ## Run sycophancy resistance battery (requires API key)
	uv run pytest tests/ -v -k "sycophancy or syc or elephant or persist" --tb=short -s

test-nct: ## Run Narrative Continuity Test battery (requires API key)
	uv run pytest tests/ -v -k "nct or continuity or persistence" --tb=short -s

# --- Docs ---

.PHONY: docs docs-serve
docs: ## Build documentation site (output in site/)
	uv run zensical build

docs-serve: ## Serve documentation locally with live reload
	uv run zensical serve

# --- Utility ---

.PHONY: reset clean nuke
reset: ## Reset sponge to seed state (preserves .venv)
	rm -f data/sponge.json
	rm -rf data/sponge_history/
	rm -rf data/chromadb/
	@echo "Sponge reset. Next run starts from seed state."

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .mypy_cache/ .ruff_cache/ .pytest_cache/ htmlcov/ .coverage
	@echo "Cleaned."

nuke: clean reset ## Full reset — remove .venv, data, and all caches
	rm -rf .venv/
	@echo "Nuked. Run 'make install' to start fresh."
