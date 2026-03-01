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
	uv run ruff check sonality/ tests/ benches/

format: ## Format code (ruff format)
	uv run ruff format sonality/ tests/ benches/
	uv run ruff check --fix sonality/ tests/ benches/

typecheck: ## Type-check code (mypy)
	uv run mypy sonality/

test: ## Run tests (pytest, skip live API tests)
	uv run pytest -v -k "not live"

test-live: ## Run live API tests (requires SONALITY_API_KEY)
	uv run pytest benches -m "bench and live" -v --tb=short -s

test-all: ## Run all tests including live API tests
	uv run pytest tests benches -v --tb=short -s

test-report: ## Run tests with JSON report and summary table
	uv run pytest -v -k "not live" --tb=short --json-report --json-report-file=test-report.json 2>/dev/null || \
		uv run pytest -v -k "not live" --tb=short --junitxml=test-report.xml
	@echo ""
	@echo "Test report written to test-report.json (or test-report.xml)"

test-live-report: ## Run live tests with detailed output
	uv run pytest benches -m "bench and live" -v --tb=short -s --junitxml=test-live-report.xml
	@echo ""
	@echo "Live test report written to test-live-report.xml"

.PHONY: bench-teaching bench-teaching-lean bench-teaching-high bench-memory bench-personality
bench-teaching: ## Run teaching benchmark suite (default profile, API required)
	uv run pytest benches/test_teaching_harness.py benches/test_teaching_suite_live.py \
		-m bench -v --tb=short -s --bench-profile default

bench-teaching-lean: ## Run teaching benchmark suite (lean profile)
	uv run pytest benches/test_teaching_harness.py benches/test_teaching_suite_live.py \
		-m bench -v --tb=short -s --bench-profile lean

bench-teaching-high: ## Run teaching benchmark suite (high_assurance profile)
	uv run pytest benches/test_teaching_harness.py benches/test_teaching_suite_live.py \
		-m bench -v --tb=short -s --bench-profile high_assurance

bench-memory: ## Run memory-structure and memory-leakage benchmark slices
	uv run pytest benches/test_teaching_harness.py benches/test_teaching_suite_live.py \
		-m bench -v --tb=short -s --bench-profile default \
		-k "memory_structure or memory_leakage"

bench-personality: ## Run personality-development benchmark slices
	uv run pytest benches/test_teaching_harness.py benches/test_teaching_suite_live.py \
		-m bench -v --tb=short -s --bench-profile default \
		-k "selective_revision or misinformation_cie or source_vigilance or source_reputation_transfer or identity_threat_resilience or counterfactual_recovery or consensus_pressure_resilience or delayed_regrounding or cross_session_reconciliation or source_memory_integrity or cross_topic_ledger_consistency or belief_decay_retention or spacing_durability or recency_quality_tradeoff or causal_replacement_fidelity or inoculation_booster_durability or motivated_skepticism_resilience or source_tag_decay_resilience or base_rate_anecdote_resilience or interference_partition_retention or source_rehabilitation_hysteresis or framing_invariance_resilience or countermyth_causal_chain_consistency or majority_trust_repair_conflict or contradictory_confidence_regrounding or provenance_conflict_arbitration or value_priority_conflict_stability or long_delay_identity_consistency or cross_domain_provenance_transfer_boundary or false_balance_weight_of_evidence_resilience or outgroup_source_derogation_resilience or commitment_consistency_pressure_resilience or authority_bias_evidence_priority_resilience or anchoring_adjustment_resilience or status_quo_default_resilience or sunk_cost_escalation_resilience or outcome_bias_process_fidelity_resilience or hindsight_certainty_resilience or omission_bias_action_inaction_resilience or endowment_effect_ownership_resilience or ambiguity_aversion_evidence_priority_resilience or belief_perseverance_debiasing_resilience or correspondence_bias_situational_resilience or conjunction_fallacy_probability_resilience or longmem or perturbation or argument_defense or prebunking or narrative_identity or contradiction_resolution or value_coherence or epistemic_calibration or trajectory_drift or revision_fidelity or sycophancy or continuity"

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
	uv run python -c "from datasets import load_dataset; ds=load_dataset('kellycyy/daily_dilemmas', split='test[:100]'); print(f'  DailyDilemmas: {len(ds)} scenarios loaded')" 2>/dev/null || \
		echo "  DailyDilemmas unavailable (check dataset access or split)"
	@echo "Fetching CMV-cleaned..."
	uv run python -c "from datasets import load_dataset; ds=load_dataset('Siddish/change-my-view-subreddit-cleaned', split='train[:50]'); print(f'  CMV-cleaned: {len(ds)} threads loaded')" 2>/dev/null || \
		echo "  Install datasets: uv add datasets"
	@echo "Fetching GlobalOpinionQA..."
	uv run python -c "from datasets import load_dataset; ds=load_dataset('Anthropic/llm_global_opinions', split='train[:50]'); print(f'  GlobalOpinionQA: {len(ds)} questions loaded')" 2>/dev/null || \
		echo "  Install datasets: uv add datasets"

test-moral: ## Run moral consistency tests with DailyDilemmas (requires API key)
	uv run pytest tests/ -v -k "moral or dilemma" --tb=short -s

test-sycophancy: ## Run sycophancy resistance battery (requires API key)
	uv run pytest benches/ -m "bench and live" -v -k "sycophancy or syc or elephant or persist" --tb=short -s

test-nct: ## Run Narrative Continuity Test battery (requires API key)
	uv run pytest benches/ -m "bench and live" -v -k "nct or continuity or persistence" --tb=short -s

# --- Docs ---

.PHONY: docs docs-serve
docs: ## Build documentation site (output in site/)
	uv run --with zensical zensical build

docs-serve: ## Serve documentation locally with live reload
	uv run --with zensical zensical serve

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
