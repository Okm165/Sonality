.DEFAULT_GOAL := help
SHELL := /bin/bash
BENCH_PROGRESS ?= pack
BENCH_PACK_GROUP ?= all
BENCH_PACKS ?=
BENCH_OUTPUT_ROOT ?= data/teaching_bench
BENCH_SIGNAL_MIN_PACK_PASS_RATE ?= 0.85
BENCH_SIGNAL_MAX_ESS_DEFAULT_RATE ?= 0.05
BENCH_SIGNAL_MAX_ESS_RETRY_RATE ?= 0.15
BENCH_HOTSPOT_PACKS ?= narrative_identity,trajectory_drift,revision_fidelity,value_coherence,epistemic_calibration,cross_session_reconciliation,long_delay_identity_consistency
BENCH_AUTO_HOTSPOT_PROFILE ?= rapid
BENCH_AUTO_HOTSPOT_MIN_PASS_RATE ?= 1.0
BENCH_AUTO_HOTSPOT_MAX_PACKS ?= 8
BENCH_AUTO_HOTSPOT_FALLBACK ?= $(BENCH_HOTSPOT_PACKS)
BENCH_REQUIRE_PROBE ?= 0
BENCH_ITERATE_STAGES ?= pulse rapid hotspots-auto
BENCH_PACK_OFFSET ?= 0
BENCH_PACK_LIMIT ?= 0
BENCH_FIRST_SIGNAL_PACK_LIMIT ?= 6
BENCH_SEGMENT_PROFILE ?= rapid
BENCH_SEGMENT_SIZE ?= 6
BENCH_SEGMENT_MAX_SEGMENTS ?= 0
BENCH_SEGMENT_ORDER ?= declared
BENCH_CONTEXT_GROUPS ?= identity revision misinformation provenance bias memory safety development
BENCH_FAILURE_RERUN_PROFILE ?= rapid
BENCH_FAILURE_SELECT_MAX_PACKS ?= 12
BENCH_FAILURE_SELECT_FALLBACK ?= $(BENCH_HOTSPOT_PACKS)
BENCH_TEACHING_BASE = uv run pytest benches/test_teaching_suite_live.py \
	-m bench -v --tb=short -s --bench-progress $(BENCH_PROGRESS) \
	--bench-output-root "$(BENCH_OUTPUT_ROOT)" \
	--bench-pack-offset $(BENCH_PACK_OFFSET) --bench-pack-limit $(BENCH_PACK_LIMIT)

# --- Help ---

.PHONY: help
help: ## Show available commands
	@echo ""
	@echo "  Sonality — LLM agent with self-evolving personality"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""

# --- Setup ---

.PHONY: install install-dev schema-scripts
install: ## Install dependencies with uv (creates local .venv)
	uv sync --no-group dev

install-dev: ## Install with dev dependencies (ruff, pytest, mypy)
	uv sync

schema-scripts: ## Regenerate database init scripts from sonality/schema.py
	@uv run python -c "from sonality.schema import write_all_init_scripts; pg, neo = write_all_init_scripts(); print(f'Generated {pg} and {neo}')"

# --- Database ---

.PHONY: db-up db-down db-reset db-init-neo4j db-init-postgres db-clear
db-up: ## Start database containers (Neo4j + PostgreSQL)
	docker compose up -d neo4j postgres

db-down: ## Stop database containers
	docker compose down

db-reset: db-down ## Reset databases (delete all data and restart)
	docker volume rm -f sonality_neo4j_data sonality_postgres_data sonality_neo4j_logs 2>/dev/null || true
	docker compose up -d neo4j postgres
	@echo "Waiting for databases to be ready..."
	@sleep 10
	@$(MAKE) db-init-neo4j

db-init-neo4j: ## Initialize Neo4j schema using cypher-shell (run after db-up)
	@echo "Initializing Neo4j schema..."
	docker compose exec -T neo4j cypher-shell -u neo4j -p sonality_password --file /scripts/init_neo4j.cypher || \
		echo "Neo4j init failed or already initialized (schema is idempotent)"

db-init-postgres: ## Re-run PostgreSQL init script (requires fresh container)
	@echo "PostgreSQL is auto-initialized via docker-entrypoint-initdb.d/"
	@echo "To re-initialize, run: make db-reset"

db-clear: ## Clear all data from databases while preserving schema
	@echo "Clearing PostgreSQL data..."
	docker compose exec -T postgres psql -U sonality -d sonality -c \
		"TRUNCATE derivatives, semantic_features RESTART IDENTITY CASCADE; DELETE FROM stm_state WHERE session_id != 'default'; UPDATE stm_state SET running_summary = '', message_buffer = '[]'::jsonb WHERE session_id = 'default';"
	@echo "Clearing Neo4j data..."
	docker compose exec -T neo4j cypher-shell -u neo4j -p sonality_password "MATCH (n) DETACH DELETE n"
	@echo "Databases cleared (schema preserved)"

# --- Run ---

.PHONY: run preflight-live preflight-live-probe
run: ## Start the Sonality REPL agent
	uv run sonality $(ARGS)

preflight-live: ## Validate live API config and selected models
	@uv run python -c "from sonality import config; import sys; missing=list(config.missing_live_api_config()); \
	(missing and (print('Missing required config: ' + ', '.join(missing)) or sys.exit(1))) or None; \
	print('Live config OK'); \
	print('  Base URL:   ' + config.BASE_URL); \
	print('  Model:      ' + config.MODEL); \
	print('  ESS model:  ' + config.ESS_MODEL); \
	(config.MODEL == config.ESS_MODEL) and print('  Warning: main and ESS model are identical; set SONALITY_ESS_MODEL to reduce self-judge coupling'); \
	print('  Tip: ensure SONALITY_MODEL matches your OpenAI-compatible endpoint'); \
	print('  Tip: run make preflight-live-probe to verify live endpoint access with a tiny request')"

preflight-live-probe: preflight-live ## Run a tiny live call to verify endpoint/model/policy access
	@uv run python -c "exec('''import json\nimport sys\nfrom sonality import config\nfrom sonality.provider import chat_completion\n\ntry:\n    result = chat_completion(\n        model=config.MODEL,\n        max_tokens=8,\n        messages=({\"role\": \"user\", \"content\": \"Reply with OK only.\"},),\n        temperature=0.0,\n    )\n    text = result.text.strip()\n    print(\"Live probe OK\")\n    print(\"  Probe model: \" + config.MODEL)\n    print(\"  Probe response preview: \" + (text[:40] if text else \"<empty>\"))\nexcept Exception as exc:\n    print(f\"Live probe failed: {exc.__class__.__name__}\")\n    print(f\"  {exc}\")\n    if isinstance(exc, RuntimeError):\n        print(\"  Hint: verify SONALITY_BASE_URL, SONALITY_API_KEY, and model IDs.\")\n    sys.exit(1)\n''')"

# --- Quality ---

.PHONY: lint format format-check typecheck test bench-contracts check check-ci
lint: ## Lint code (ruff check)
	uv run ruff check sonality/ tests/ benches/

format: ## Format code (ruff format)
	uv run ruff format sonality/ tests/ benches/
	uv run ruff check --fix sonality/ tests/ benches/

format-check: ## Check formatting without writing changes (CI parity)
	uv run ruff format --check sonality/ tests/ benches/

typecheck: ## Type-check code (mypy)
	uv run mypy sonality/

test: ## Run tests (pytest, skip live API tests)
	uv run pytest tests -m "not live" -v

bench-contracts: ## Run non-live benchmark contracts (no API)
	uv run pytest benches -m "bench and not live" -q

test-live: ## Run live API tests (requires SONALITY_API_KEY)
	uv run pytest benches -m "bench and live" -v --tb=short -s

test-all: ## Run all tests including live API tests
	uv run pytest tests benches -v --tb=short -s

test-report: ## Run tests with JSON report and summary table
	uv run pytest tests -m "not live" -v --tb=short --json-report --json-report-file=test-report.json 2>/dev/null || \
		uv run pytest tests -m "not live" -v --tb=short --junitxml=test-report.xml
	@echo ""
	@echo "Test report written to test-report.json (or test-report.xml)"

test-live-report: ## Run live tests with detailed output
	uv run pytest benches -m "bench and live" -v --tb=short -s --junitxml=test-live-report.xml
	@echo ""
	@echo "Live test report written to test-live-report.xml"

.PHONY: bench-teaching bench-teaching-lean bench-teaching-high bench-teaching-profile bench-teaching-smoke bench-teaching-pulse bench-teaching-rapid bench-teaching-first-signal bench-plan-segments bench-teaching-segmented bench-select-failures-last bench-teaching-failures-last bench-teaching-contextual bench-teaching-hotspots bench-select-hotspots-last bench-teaching-hotspots-auto bench-teaching-safety bench-teaching-development bench-teaching-iterate bench-memory bench-personality bench-report-last bench-report-failures-last bench-report-root bench-report-memory-root bench-report-beliefs-last bench-report-insights-root bench-report-delta-last bench-signal-gate-last
bench-teaching: ## Run teaching benchmark suite (default profile, API required)
	$(BENCH_TEACHING_BASE) --bench-profile default \
		--bench-pack-group $(BENCH_PACK_GROUP) --bench-packs "$(BENCH_PACKS)"

bench-teaching-lean: ## Run teaching benchmark suite (lean profile)
	$(BENCH_TEACHING_BASE) --bench-profile lean \
		--bench-pack-group $(BENCH_PACK_GROUP) --bench-packs "$(BENCH_PACKS)"

bench-teaching-high: ## Run teaching benchmark suite (high_assurance profile)
	$(BENCH_TEACHING_BASE) --bench-profile high_assurance \
		--bench-pack-group $(BENCH_PACK_GROUP) --bench-packs "$(BENCH_PACKS)"

bench-teaching-profile: preflight-live ## Run teaching suite with BENCH_SEGMENT_PROFILE and current pack selectors
	$(BENCH_TEACHING_BASE) --bench-profile "$(BENCH_SEGMENT_PROFILE)" \
		--bench-pack-group "$(BENCH_PACK_GROUP)" --bench-packs "$(BENCH_PACKS)"

bench-teaching-smoke: ## Run a fast smoke slice (3 packs, lean profile)
	$(BENCH_TEACHING_BASE) --bench-profile lean --bench-pack-group smoke

bench-teaching-pulse: preflight-live ## Run ultra-fast 2-pack pulse to detect obvious breakage
	$(BENCH_TEACHING_BASE) --bench-profile rapid --bench-pack-group pulse

bench-teaching-rapid: preflight-live ## Run single-replicate triage slice for very fast signal
	$(BENCH_TEACHING_BASE) --bench-profile rapid --bench-pack-group triage

bench-teaching-first-signal: BENCH_PACK_OFFSET = 0
bench-teaching-first-signal: BENCH_PACK_LIMIT = $(BENCH_FIRST_SIGNAL_PACK_LIMIT)
bench-teaching-first-signal: preflight-live ## Run first-N packs for immediate go/no-go signal
	$(BENCH_TEACHING_BASE) --bench-profile rapid --bench-pack-group all

bench-plan-segments: ## Print deterministic segment plan for current group/pack selection
	@uv run python -c "exec('''import json\nfrom pathlib import Path\n\nfrom benches.teaching_harness import resolve_benchmark_packs\n\npack_group = \"$(BENCH_PACK_GROUP)\"\npack_keys = tuple(key.strip() for key in \"$(BENCH_PACKS)\".split(\",\") if key.strip())\nsegment_size = int(\"$(BENCH_SEGMENT_SIZE)\")\nsegment_order = \"$(BENCH_SEGMENT_ORDER)\".strip() or \"declared\"\nif segment_size <= 0:\n    raise SystemExit(\"BENCH_SEGMENT_SIZE must be > 0\")\nif segment_order not in {\"declared\", \"weak_first\"}:\n    raise SystemExit(\"BENCH_SEGMENT_ORDER must be one of: declared, weak_first\")\n\npacks = resolve_benchmark_packs(pack_group=pack_group, pack_keys=pack_keys)\nordered_keys = [pack.key for pack in packs]\nif segment_order == \"weak_first\":\n    root = Path(\"$(BENCH_OUTPUT_ROOT)\")\n    runs = sorted(\n        (run for run in root.iterdir() if run.is_dir()),\n        key=lambda run: run.stat().st_mtime,\n        reverse=True,\n    ) if root.exists() else []\n    completed = [run for run in runs if (run / \"run_summary.json\").exists()]\n    if completed:\n        summary = json.loads((completed[0] / \"run_summary.json\").read_text(encoding=\"utf-8\"))\n        rates: dict[str, list[float]] = {}\n        for row in summary.get(\"pack_results\", []):\n            if not isinstance(row, dict):\n                continue\n            pack = row.get(\"pack\")\n            pass_rate = row.get(\"pass_rate\")\n            if not isinstance(pack, str):\n                continue\n            if not isinstance(pass_rate, (int, float)) or isinstance(pass_rate, bool):\n                continue\n            rates.setdefault(pack, []).append(float(pass_rate))\n\n        def order_key(key: str) -> tuple[int, float, float, str]:\n            values = rates.get(key)\n            if values:\n                return (0, min(values), sum(values) / len(values), key)\n            return (1, 1.0, 1.0, key)\n\n        ordered_keys = sorted(ordered_keys, key=order_key)\n\nprint(f\"Selected packs: {len(ordered_keys)}\")\nprint(f\"Selection source: group={pack_group} explicit_keys={bool(pack_keys)}\")\nprint(f\"Segment order: {segment_order}\")\nfor index, start in enumerate(range(0, len(ordered_keys), segment_size), start=1):\n    segment = \",\".join(ordered_keys[start:start + segment_size])\n    print(f\"  segment {index}: offset={start} limit={segment_size} packs={segment}\")\n''')"

bench-teaching-segmented: ## Run selected packs in deterministic segments with gate checks between segments
	@set -euo pipefail; \
		$(MAKE) preflight-live; \
		if [ "$(BENCH_REQUIRE_PROBE)" = "1" ]; then \
			$(MAKE) preflight-live-probe; \
		fi; \
		segment_size=$(BENCH_SEGMENT_SIZE); \
		if [ "$$segment_size" -le 0 ]; then \
			echo "BENCH_SEGMENT_SIZE must be > 0"; \
			exit 2; \
		fi; \
		selected_packs="$$(uv run python -c "exec('''import json\nfrom pathlib import Path\n\nfrom benches.teaching_harness import resolve_benchmark_packs\n\npack_group = \"$(BENCH_PACK_GROUP)\"\npack_keys = tuple(key.strip() for key in \"$(BENCH_PACKS)\".split(\",\") if key.strip())\nsegment_order = \"$(BENCH_SEGMENT_ORDER)\".strip() or \"declared\"\nif segment_order not in {\"declared\", \"weak_first\"}:\n    raise SystemExit(\"BENCH_SEGMENT_ORDER must be one of: declared, weak_first\")\n\npacks = resolve_benchmark_packs(pack_group=pack_group, pack_keys=pack_keys)\nordered_keys = [pack.key for pack in packs]\nif segment_order == \"weak_first\":\n    root = Path(\"$(BENCH_OUTPUT_ROOT)\")\n    runs = sorted(\n        (run for run in root.iterdir() if run.is_dir()),\n        key=lambda run: run.stat().st_mtime,\n        reverse=True,\n    ) if root.exists() else []\n    completed = [run for run in runs if (run / \"run_summary.json\").exists()]\n    if completed:\n        summary = json.loads((completed[0] / \"run_summary.json\").read_text(encoding=\"utf-8\"))\n        rates: dict[str, list[float]] = {}\n        for row in summary.get(\"pack_results\", []):\n            if not isinstance(row, dict):\n                continue\n            pack = row.get(\"pack\")\n            pass_rate = row.get(\"pass_rate\")\n            if not isinstance(pack, str):\n                continue\n            if not isinstance(pass_rate, (int, float)) or isinstance(pass_rate, bool):\n                continue\n            rates.setdefault(pack, []).append(float(pass_rate))\n\n        def order_key(key: str) -> tuple[int, float, float, str]:\n            values = rates.get(key)\n            if values:\n                return (0, min(values), sum(values) / len(values), key)\n            return (1, 1.0, 1.0, key)\n\n        ordered_keys = sorted(ordered_keys, key=order_key)\n\nprint(\",\".join(ordered_keys))\n''')")"; \
		total_packs="$$(awk -F',' '{print NF}' <<< "$$selected_packs")"; \
		if [ "$$total_packs" -le 0 ]; then \
			echo "No packs selected for segmented run."; \
			exit 2; \
		fi; \
		echo "Segment order mode: $(BENCH_SEGMENT_ORDER) | total packs: $$total_packs"; \
		segment_total=$$(( (total_packs + segment_size - 1) / segment_size )); \
		max_segments=$(BENCH_SEGMENT_MAX_SEGMENTS); \
		if [ "$$max_segments" -gt 0 ] && [ "$$segment_total" -gt "$$max_segments" ]; then \
			segment_total="$$max_segments"; \
		fi; \
		segment_idx=1; \
		offset=0; \
		while [ "$$segment_idx" -le "$$segment_total" ]; do \
			echo "=== Segment $$segment_idx/$$segment_total | offset=$$offset limit=$$segment_size profile=$(BENCH_SEGMENT_PROFILE) ==="; \
			$(MAKE) bench-teaching-profile \
				BENCH_SEGMENT_PROFILE="$(BENCH_SEGMENT_PROFILE)" \
				BENCH_PACK_GROUP="all" \
				BENCH_PACKS="$$selected_packs" \
				BENCH_PACK_OFFSET="$$offset" \
				BENCH_PACK_LIMIT="$$segment_size" \
				BENCH_PROGRESS="$(BENCH_PROGRESS)" \
				BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)"; \
			$(MAKE) bench-report-last BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)"; \
			$(MAKE) bench-report-failures-last BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)"; \
			$(MAKE) bench-signal-gate-last BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)" \
				BENCH_SIGNAL_MIN_PACK_PASS_RATE="$(BENCH_SIGNAL_MIN_PACK_PASS_RATE)" \
				BENCH_SIGNAL_MAX_ESS_DEFAULT_RATE="$(BENCH_SIGNAL_MAX_ESS_DEFAULT_RATE)" \
				BENCH_SIGNAL_MAX_ESS_RETRY_RATE="$(BENCH_SIGNAL_MAX_ESS_RETRY_RATE)"; \
			offset=$$((offset + segment_size)); \
			segment_idx=$$((segment_idx + 1)); \
		done

bench-select-failures-last: ## Print comma-separated pack keys with failures in latest run
	@uv run python -c "exec('''import json\nfrom pathlib import Path\n\nroot = Path(\"$(BENCH_OUTPUT_ROOT)\")\nfallback = \"$(BENCH_FAILURE_SELECT_FALLBACK)\".strip()\nmax_packs = int(\"$(BENCH_FAILURE_SELECT_MAX_PACKS)\")\nif max_packs <= 0:\n    raise SystemExit(\"BENCH_FAILURE_SELECT_MAX_PACKS must be > 0\")\n\nruns = sorted(\n    (run for run in root.iterdir() if run.is_dir()),\n    key=lambda run: run.stat().st_mtime,\n    reverse=True,\n) if root.exists() else []\ncounts: dict[str, int] = {}\n\ntrace_run = next((run for run in runs if (run / \"turn_trace.jsonl\").exists()), None)\nif trace_run is not None:\n    with (trace_run / \"turn_trace.jsonl\").open(\"r\", encoding=\"utf-8\") as handle:\n        for raw_line in handle:\n            line = raw_line.strip()\n            if not line:\n                continue\n            try:\n                row = json.loads(line)\n            except json.JSONDecodeError:\n                continue\n            if bool(row.get(\"passed\", True)):\n                continue\n            pack = row.get(\"pack\")\n            if isinstance(pack, str):\n                counts[pack] = counts.get(pack, 0) + 1\n\nif not counts:\n    summary_run = next((run for run in runs if (run / \"run_summary.json\").exists()), None)\n    if summary_run is not None:\n        summary = json.loads((summary_run / \"run_summary.json\").read_text(encoding=\"utf-8\"))\n        for row in summary.get(\"pack_results\", []):\n            if not isinstance(row, dict):\n                continue\n            pack = row.get(\"pack\")\n            if not isinstance(pack, str):\n                continue\n            passed_steps = row.get(\"passed_steps\")\n            total_steps = row.get(\"total_steps\")\n            pass_rate = row.get(\"pass_rate\")\n            misses = 0\n            if (\n                isinstance(total_steps, int)\n                and isinstance(passed_steps, int)\n                and total_steps >= passed_steps\n            ):\n                misses = total_steps - passed_steps\n            elif isinstance(pass_rate, (int, float)) and not isinstance(pass_rate, bool):\n                misses = 1 if float(pass_rate) < 0.999999 else 0\n            if misses > 0:\n                counts[pack] = counts.get(pack, 0) + misses\n\nif not counts:\n    error_run = next((run for run in runs if (run / \"run_error.json\").exists()), None)\n    if error_run is not None:\n        payload = json.loads((error_run / \"run_error.json\").read_text(encoding=\"utf-8\"))\n        pack_keys = payload.get(\"pack_keys\")\n        if isinstance(pack_keys, list):\n            total = len(pack_keys)\n            for index, key in enumerate(pack_keys):\n                if isinstance(key, str):\n                    counts[key] = max(counts.get(key, 0), max(total - index, 1))\n\nif not counts:\n    print(fallback)\n    raise SystemExit(0)\n\nordered = [pack for pack, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]\nprint(\",\".join(ordered[:max_packs]) if ordered else fallback)\n''')"

bench-teaching-failures-last: preflight-live ## Rerun latest failed packs only (auto-selected from latest run)
	@set -euo pipefail; \
		selected_packs="$$( MAKEFLAGS= $(MAKE) --no-print-directory bench-select-failures-last \
			BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)" \
			BENCH_FAILURE_SELECT_MAX_PACKS="$(BENCH_FAILURE_SELECT_MAX_PACKS)" \
			BENCH_FAILURE_SELECT_FALLBACK="$(BENCH_FAILURE_SELECT_FALLBACK)" )"; \
		echo "Latest-failure packs: $$selected_packs"; \
		$(BENCH_TEACHING_BASE) --bench-profile "$(BENCH_FAILURE_RERUN_PROFILE)" --bench-pack-group all \
			--bench-packs "$$selected_packs"

bench-teaching-contextual: ## Run contextual group sweep with per-group gate checks
	@set -euo pipefail; \
		$(MAKE) preflight-live; \
		if [ "$(BENCH_REQUIRE_PROBE)" = "1" ]; then \
			$(MAKE) preflight-live-probe; \
		fi; \
		group_idx=1; \
		group_total=$(words $(BENCH_CONTEXT_GROUPS)); \
		for group in $(BENCH_CONTEXT_GROUPS); do \
			echo "=== Context $$group_idx/$$group_total: $$group (profile=$(BENCH_SEGMENT_PROFILE)) ==="; \
			$(MAKE) bench-teaching-profile \
				BENCH_SEGMENT_PROFILE="$(BENCH_SEGMENT_PROFILE)" \
				BENCH_PACK_GROUP="$$group" \
				BENCH_PACKS="" \
				BENCH_PACK_OFFSET=0 \
				BENCH_PACK_LIMIT=0 \
				BENCH_PROGRESS="$(BENCH_PROGRESS)" \
				BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)"; \
			$(MAKE) bench-report-last BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)"; \
			$(MAKE) bench-report-failures-last BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)"; \
			$(MAKE) bench-signal-gate-last BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)" \
				BENCH_SIGNAL_MIN_PACK_PASS_RATE="$(BENCH_SIGNAL_MIN_PACK_PASS_RATE)" \
				BENCH_SIGNAL_MAX_ESS_DEFAULT_RATE="$(BENCH_SIGNAL_MAX_ESS_DEFAULT_RATE)" \
				BENCH_SIGNAL_MAX_ESS_RETRY_RATE="$(BENCH_SIGNAL_MAX_ESS_RETRY_RATE)"; \
			group_idx=$$((group_idx + 1)); \
		done

bench-teaching-hotspots: preflight-live ## Run rapid hotspot slice for known weak development packs
	$(BENCH_TEACHING_BASE) --bench-profile rapid --bench-pack-group all \
		--bench-packs "$(BENCH_HOTSPOT_PACKS)"

bench-select-hotspots-last: ## Print comma-separated hotspot pack keys inferred from latest completed run
	@uv run python -c "exec('''import json\nfrom pathlib import Path\n\nroot = Path(\"$(BENCH_OUTPUT_ROOT)\")\nfallback = \"$(BENCH_AUTO_HOTSPOT_FALLBACK)\".strip()\nmin_pass_rate = float(\"$(BENCH_AUTO_HOTSPOT_MIN_PASS_RATE)\")\nmax_packs = int(\"$(BENCH_AUTO_HOTSPOT_MAX_PACKS)\")\nruns = sorted(\n    (run for run in root.iterdir() if run.is_dir()),\n    key=lambda run: run.stat().st_mtime,\n    reverse=True,\n) if root.exists() else []\ncompleted_runs = [run for run in runs if (run / \"run_summary.json\").exists()]\nif not completed_runs:\n    print(fallback)\n    raise SystemExit(0)\n\nsummary = json.loads((completed_runs[0] / \"run_summary.json\").read_text(encoding=\"utf-8\"))\npack_rates: dict[str, list[float]] = {}\nfor row in summary.get(\"pack_results\", []):\n    if not isinstance(row, dict):\n        continue\n    pack = row.get(\"pack\")\n    pass_rate = row.get(\"pass_rate\")\n    if not isinstance(pack, str):\n        continue\n    if not isinstance(pass_rate, (int, float)) or isinstance(pass_rate, bool):\n        continue\n    pack_rates.setdefault(pack, []).append(float(pass_rate))\n\nhotspots: list[tuple[float, float, str]] = []\nfor pack, rates in pack_rates.items():\n    if not rates:\n        continue\n    min_rate = min(rates)\n    mean_rate = sum(rates) / len(rates)\n    if min_rate < min_pass_rate:\n        hotspots.append((min_rate, mean_rate, pack))\n\nhotspots.sort(key=lambda item: (item[0], item[1], item[2]))\nkeys = [pack for _, _, pack in hotspots[:max_packs]]\nprint(\",\".join(keys) if keys else fallback)\n''')"

bench-teaching-hotspots-auto: preflight-live ## Run adaptive hotspots inferred from latest completed run
	@set -euo pipefail; \
		selected_packs="$$( $(MAKE) --no-print-directory bench-select-hotspots-last \
			BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)" \
			BENCH_AUTO_HOTSPOT_MIN_PASS_RATE="$(BENCH_AUTO_HOTSPOT_MIN_PASS_RATE)" \
			BENCH_AUTO_HOTSPOT_MAX_PACKS="$(BENCH_AUTO_HOTSPOT_MAX_PACKS)" \
			BENCH_AUTO_HOTSPOT_FALLBACK="$(BENCH_AUTO_HOTSPOT_FALLBACK)" )"; \
		echo "Adaptive hotspots: $$selected_packs"; \
		$(BENCH_TEACHING_BASE) --bench-profile "$(BENCH_AUTO_HOTSPOT_PROFILE)" --bench-pack-group all \
			--bench-packs "$$selected_packs"

bench-teaching-safety: preflight-live ## Run safety-critical slice (lean profile)
	$(BENCH_TEACHING_BASE) --bench-profile lean --bench-pack-group safety

bench-teaching-development: preflight-live ## Run personality-development core slice (lean profile)
	$(BENCH_TEACHING_BASE) --bench-profile lean --bench-pack-group development

bench-teaching-iterate: ## Run staged pipeline from BENCH_ITERATE_STAGES, stop on first failure
	@set -euo pipefail; \
		$(MAKE) preflight-live; \
		if [ "$(BENCH_REQUIRE_PROBE)" = "1" ]; then \
			$(MAKE) preflight-live-probe; \
		fi; \
		stage_idx=1; \
		stage_total=$(words $(BENCH_ITERATE_STAGES)); \
		for stage in $(BENCH_ITERATE_STAGES); do \
			case "$$stage" in \
				pulse) stage_label="pulse sanity" ;; \
				rapid) stage_label="rapid triage" ;; \
				hotspots-auto) stage_label="adaptive hotspots" ;; \
				*) stage_label="$$stage" ;; \
			esac; \
			echo "=== Stage $$stage_idx/$$stage_total: $$stage_label ==="; \
			$(MAKE) bench-teaching-$$stage BENCH_PROGRESS=$(BENCH_PROGRESS) BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)"; \
			$(MAKE) bench-report-last BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)"; \
			$(MAKE) bench-report-failures-last BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)"; \
			$(MAKE) bench-signal-gate-last BENCH_OUTPUT_ROOT="$(BENCH_OUTPUT_ROOT)" \
				BENCH_SIGNAL_MIN_PACK_PASS_RATE=$(BENCH_SIGNAL_MIN_PACK_PASS_RATE) \
				BENCH_SIGNAL_MAX_ESS_DEFAULT_RATE=$(BENCH_SIGNAL_MAX_ESS_DEFAULT_RATE) \
				BENCH_SIGNAL_MAX_ESS_RETRY_RATE=$(BENCH_SIGNAL_MAX_ESS_RETRY_RATE); \
			stage_idx=$$((stage_idx + 1)); \
		done

bench-memory: ## Run memory-structure and memory-leakage benchmark slices
	$(BENCH_TEACHING_BASE) --bench-profile default --bench-pack-group memory

bench-personality: ## Run personality-development benchmark slices
	$(BENCH_TEACHING_BASE) --bench-profile default --bench-pack-group personality

bench-report-last: ## Print compact summary for latest benchmark run
	@uv run python -c "exec('''import json\nimport sys\nfrom pathlib import Path\n\n\ndef topic_preview(rows: object, limit: int = 5) -> str:\n    if not isinstance(rows, list):\n        return \"\"\n    chunks: list[str] = []\n    for row in rows[:limit]:\n        if not isinstance(row, dict):\n            continue\n        topic = row.get(\"topic\")\n        delta = row.get(\"abs_delta_total\")\n        if not isinstance(topic, str):\n            continue\n        if isinstance(delta, (int, float)) and not isinstance(delta, bool):\n            chunks.append(f\"{topic}:{delta:.3f}\")\n        else:\n            chunks.append(topic)\n    return \", \".join(chunks)\n\n\nroot = Path(\"$(BENCH_OUTPUT_ROOT)\")\nruns = sorted(\n    (run for run in root.iterdir() if run.is_dir()),\n    key=lambda run: run.stat().st_mtime,\n    reverse=True,\n) if root.exists() else []\nif not runs:\n    print(\"No benchmark runs found.\")\n    sys.exit(0)\n\nrun_dir = runs[0]\nprint(f\"Run directory: {run_dir}\")\nsummary_path = run_dir / \"run_summary.json\"\nerror_path = run_dir / \"run_error.json\"\nhealth_path = run_dir / \"health_summary_report.json\"\n\nif summary_path.exists():\n    summary = json.loads(summary_path.read_text(encoding=\"utf-8\"))\n    hard_blockers = summary.get(\"hard_blockers\", [])\n    soft_blockers = summary.get(\"soft_blockers\", [])\n    pack_scope = summary.get(\"pack_scope\", {})\n    profile_name = (\n        summary.get(\"profile\")\n        or summary.get(\"profile_name\")\n        or summary.get(\"health_summary\", {}).get(\"profile\")\n    )\n    selected_count = pack_scope.get(\"selected_count\") if isinstance(pack_scope, dict) else None\n    if selected_count is None:\n        pack_rows = summary.get(\"pack_results\", [])\n        if isinstance(pack_rows, list):\n            selected_count = len(\n                {\n                    row.get(\"pack\")\n                    for row in pack_rows\n                    if isinstance(row, dict) and isinstance(row.get(\"pack\"), str)\n                }\n            )\n    print(f\"Profile: {profile_name} | Selected packs: {selected_count}\")\n    print(\n        f\"Decision: {summary.get('decision')} | Replicates: {summary.get('replicates_executed')} \"\n        f\"| Stop: {summary.get('stop_reason')}\"\n    )\n    print(f\"Hard blockers ({len(hard_blockers)}): {hard_blockers}\")\n    print(f\"Soft blockers ({len(soft_blockers)}): {soft_blockers}\")\n\n    default_rate = summary.get(\"ess_default_summary\", {}).get(\"defaulted_step_rate\")\n    retry_rate = summary.get(\"ess_retry_summary\", {}).get(\"retry_step_rate\")\n    print(f\"ESS defaults/retries: {default_rate} / {retry_rate}\")\n\n    cost = summary.get(\"cost_summary\", {})\n    print(\n        f\"Cost: calls={cost.get('total_calls')} tokens={cost.get('total_tokens')} \"\n        f\"steps={cost.get('total_steps')}\"\n    )\n\n    isolation = summary.get(\"run_isolation_summary\", {}).get(\"summary\", {})\n    if isolation:\n        print(\n            f\"Isolation: status={isolation.get('overall_status')} \"\n            f\"seed_fail={isolation.get('seed_state_fail_count')} \"\n            f\"interaction_chain_fail={isolation.get('interaction_chain_fail_count')} \"\n            f\"episode_chain_fail={isolation.get('episode_chain_fail_count')}\"\n        )\n\n    validity = summary.get(\"memory_validity_summary\", {}).get(\"summary\", {})\n    if validity:\n        print(\n            f\"Memory validity: status={validity.get('overall_status')} \"\n            f\"policy_violations={validity.get('update_policy_violation_count')} \"\n            f\"direction_mismatches={validity.get('direction_mismatch_count')} \"\n            f\"memory_write_rate={validity.get('memory_write_rate')}\"\n        )\n        preview = topic_preview(validity.get(\"top_belief_topic_deltas\"))\n        if preview:\n            print(f\"Top belief-topic deltas: {preview}\")\n\n    validity_signals = summary.get(\"memory_validity_summary\", {}).get(\"release_signals\", {})\n    if isinstance(validity_signals, dict):\n        policy_packs = validity_signals.get(\"packs_with_update_policy_violations\", [])\n        direction_packs = validity_signals.get(\"packs_with_direction_mismatches\", [])\n        if policy_packs:\n            print(f\"Validity critical packs: {policy_packs}\")\n        if direction_packs:\n            print(f\"Direction-mismatch packs: {direction_packs}\")\n\n    release = summary.get(\"release_readiness\", {})\n    print(\n        f\"Release readiness: {release.get('overall')} \"\n        f\"| {release.get('recommended_action')}\"\n    )\n\n    pack_results = summary.get(\"pack_results\", [])\n    if pack_results:\n        print(\"Pack snapshot (lowest pass_rate first):\")\n        ordered = sorted(pack_results, key=lambda row: row.get(\"pass_rate\", 0.0))\n        for row in ordered[:10]:\n            pass_rate = row.get(\"pass_rate\", 0.0)\n            print(\n                f\"  {row.get('pack')}: {row.get('passed_steps')}/{row.get('total_steps')} \"\n                f\"({pass_rate:.0%}) gate={row.get('gate_passed')} \"\n                f\"hard_failures={len(row.get('hard_failures', []))}\"\n            )\n\n    if health_path.exists():\n        health = json.loads(health_path.read_text(encoding=\"utf-8\"))\n        health_summary = health.get(\"summary\", {})\n        print(\n            f\"Health: status={health_summary.get('overall_status')} \"\n            f\"flagged_row_rate={health_summary.get('flagged_row_rate')} \"\n            f\"memory_update_rate={health_summary.get('memory_update_rate')}\"\n        )\n        critical_packs = health.get(\"release_signals\", {}).get(\"critical_packs\", [])\n        if critical_packs:\n            print(f\"Critical packs: {critical_packs}\")\nelif error_path.exists():\n    error = json.loads(error_path.read_text(encoding=\"utf-8\"))\n    print(f\"Run failed: {error.get('error_type')}\")\n    print(f\"Error: {error.get('error')}\")\n    print(f\"Pack keys: {error.get('pack_keys')}\")\nelse:\n    print(\"No run_summary.json or run_error.json in latest run directory.\")\n''')"

bench-report-failures-last: ## Print failed-step preview for latest benchmark run
	@uv run python -c "exec('''import json\nimport sys\nfrom pathlib import Path\n\nroot = Path(\"$(BENCH_OUTPUT_ROOT)\")\nruns = sorted(\n    (run for run in root.iterdir() if run.is_dir()),\n    key=lambda run: run.stat().st_mtime,\n    reverse=True,\n) if root.exists() else []\nif not runs:\n    print(\"No benchmark runs found.\")\n    sys.exit(0)\n\nrun_dir = runs[0]\ntrace_path = run_dir / \"turn_trace.jsonl\"\nif not trace_path.exists():\n    print(f\"No turn_trace.jsonl found in {run_dir}\")\n    sys.exit(0)\n\nfailures: list[tuple[str, str, str, str]] = []\nwith trace_path.open(\"r\", encoding=\"utf-8\") as handle:\n    for raw_line in handle:\n        line = raw_line.strip()\n        if not line:\n            continue\n        try:\n            row = json.loads(line)\n        except json.JSONDecodeError:\n            continue\n        if bool(row.get(\"passed\", True)):\n            continue\n\n        pack = row.get(\"pack\") if isinstance(row.get(\"pack\"), str) else \"unknown_pack\"\n        label = row.get(\"label\") if isinstance(row.get(\"label\"), str) else \"unknown_step\"\n        ess = row.get(\"ess_score\")\n        ess_text = (\n            f\"{float(ess):.2f}\"\n            if isinstance(ess, (int, float)) and not isinstance(ess, bool)\n            else \"n/a\"\n        )\n        reason = \"\"\n        failure_list = row.get(\"failures\")\n        if isinstance(failure_list, list) and failure_list:\n            first = failure_list[0]\n            if isinstance(first, str):\n                reason = first\n        failures.append((pack, label, ess_text, reason))\n\nif not failures:\n    print(\"Failed-step preview: none\")\n    sys.exit(0)\n\nprint(f\"Failed-step preview ({len(failures)} total):\")\nfor pack, label, ess_text, reason in failures[:12]:\n    suffix = f\" | {reason}\" if reason else \"\"\n    print(f\"  {pack}/{label} ess={ess_text}{suffix}\")\nif len(failures) > 12:\n    print(f\"  ... +{len(failures) - 12} more\")\n''')"

bench-report-root: ## Print multi-run trend table for benchmark output root
	@uv run python -c "exec('''import json\nimport sys\nfrom datetime import datetime\nfrom pathlib import Path\n\n\ndef pass_stats(summary: dict[str, object]) -> tuple[float, float, int]:\n    rows = summary.get(\"pack_results\", [])\n    if not isinstance(rows, list):\n        return 0.0, 0.0, 0\n    rates: list[float] = []\n    for row in rows:\n        if not isinstance(row, dict):\n            continue\n        rate = row.get(\"pass_rate\")\n        if isinstance(rate, (int, float)) and not isinstance(rate, bool):\n            rates.append(float(rate))\n    if not rates:\n        return 0.0, 0.0, 0\n    return sum(rates) / len(rates), min(rates), len(rates)\n\n\nroot = Path(\"$(BENCH_OUTPUT_ROOT)\")\nruns = sorted(\n    (run for run in root.iterdir() if run.is_dir()),\n    key=lambda run: run.stat().st_mtime,\n    reverse=True,\n) if root.exists() else []\ncompleted = [run for run in runs if (run / \"run_summary.json\").exists()]\nif not completed:\n    print(f\"No completed benchmark runs under {root}\")\n    sys.exit(0)\n\nprint(f\"Benchmark root: {root}\")\nprint(f\"Completed runs: {len(completed)}\")\nprint(\"Recent runs (latest first):\")\nfor run in completed[:15]:\n    summary = json.loads((run / \"run_summary.json\").read_text(encoding=\"utf-8\"))\n    mean_rate, min_rate, pack_count = pass_stats(summary)\n    health = summary.get(\"health_summary\", {}).get(\"summary\", {})\n    validity = summary.get(\"memory_validity_summary\", {}).get(\"summary\", {})\n    isolation = summary.get(\"run_isolation_summary\", {}).get(\"summary\", {})\n    timestamp = datetime.fromtimestamp(run.stat().st_mtime).strftime(\"%Y-%m-%d %H:%M\")\n    print(\n        f\"  {timestamp} {run.name} \"\n        f\"profile={summary.get('profile_name') or summary.get('profile') or summary.get('health_summary', {}).get('profile')} packs={pack_count} \"\n        f\"decision={summary.get('decision')} \"\n        f\"pass_mean={mean_rate:.0%} pass_min={min_rate:.0%} \"\n        f\"flagged={health.get('flagged_row_rate')} \"\n        f\"validity={validity.get('overall_status')} isolation={isolation.get('overall_status')}\"\n    )\n''')"

bench-report-memory-root: ## Print memory-validity trend table for benchmark output root
	@uv run python -c "exec('''import json\nimport sys\nfrom datetime import datetime\nfrom pathlib import Path\n\n\nroot = Path(\"$(BENCH_OUTPUT_ROOT)\")\nruns = sorted(\n    (run for run in root.iterdir() if run.is_dir()),\n    key=lambda run: run.stat().st_mtime,\n    reverse=True,\n) if root.exists() else []\ncompleted = [run for run in runs if (run / \"run_summary.json\").exists()]\nif not completed:\n    print(f\"No completed benchmark runs under {root}\")\n    sys.exit(0)\n\nprint(f\"Memory trend root: {root}\")\nprint(f\"Completed runs: {len(completed)}\")\nprint(\"Recent memory-validity snapshots:\")\nfor run in completed[:15]:\n    summary = json.loads((run / \"run_summary.json\").read_text(encoding=\"utf-8\"))\n    validity = summary.get(\"memory_validity_summary\", {})\n    validity_summary = validity.get(\"summary\", {}) if isinstance(validity, dict) else {}\n    top_topics = validity_summary.get(\"top_belief_topic_deltas\", [])\n    top_topic = \"\"\n    if isinstance(top_topics, list) and top_topics:\n        row = top_topics[0]\n        if isinstance(row, dict) and isinstance(row.get(\"topic\"), str):\n            top_topic = row.get(\"topic\")\n    timestamp = datetime.fromtimestamp(run.stat().st_mtime).strftime(\"%Y-%m-%d %H:%M\")\n    print(\n        f\"  {timestamp} {run.name} \"\n        f\"profile={summary.get('profile') or summary.get('health_summary', {}).get('profile')} \"\n        f\"status={validity_summary.get('overall_status')} \"\n        f\"write_rate={validity_summary.get('memory_write_rate')} \"\n        f\"policy_violations={validity_summary.get('update_policy_violation_count')} \"\n        f\"direction_mismatches={validity_summary.get('direction_mismatch_count')} \"\n        f\"belief_topics={validity_summary.get('belief_topic_count')} \"\n        f\"top_topic={top_topic or '<none>'}\"\n    )\n''')"

bench-report-beliefs-last: ## Print belief-memory alignment summary for latest benchmark run
	@uv run python -c "exec('''import json\nimport sys\nfrom pathlib import Path\n\n\ndef _topic_preview(rows: object, limit: int = 8) -> list[str]:\n    if not isinstance(rows, list):\n        return []\n    chunks: list[str] = []\n    for row in rows[:limit]:\n        if not isinstance(row, dict):\n            continue\n        topic = row.get(\"topic\")\n        if not isinstance(topic, str):\n            continue\n        risk = row.get(\"risk_score\")\n        delta = row.get(\"abs_delta_total\")\n        chunks.append(f\"{topic}(risk={risk},delta={delta})\")\n    return chunks\n\n\nroot = Path(\"$(BENCH_OUTPUT_ROOT)\")\nruns = sorted(\n    (run for run in root.iterdir() if run.is_dir()),\n    key=lambda run: run.stat().st_mtime,\n    reverse=True,\n) if root.exists() else []\nif not runs:\n    print(\"No benchmark runs found.\")\n    sys.exit(0)\n\nrun_dir = runs[0]\nsummary_path = run_dir / \"run_summary.json\"\nalignment_path = run_dir / \"belief_memory_alignment_report.json\"\nif alignment_path.exists():\n    report = json.loads(alignment_path.read_text(encoding=\"utf-8\"))\nelif summary_path.exists():\n    summary = json.loads(summary_path.read_text(encoding=\"utf-8\"))\n    report = summary.get(\"belief_memory_alignment_summary\", {})\nelse:\n    print(f\"No run summary/report in latest run: {run_dir}\")\n    sys.exit(0)\n\nif not isinstance(report, dict) or not report:\n    print(f\"Belief-memory alignment report missing in latest run: {run_dir}\")\n    sys.exit(0)\n\nsummary = report.get(\"summary\", {}) if isinstance(report.get(\"summary\"), dict) else {}\nsignals = report.get(\"release_signals\", {}) if isinstance(report.get(\"release_signals\"), dict) else {}\nprint(f\"Belief-memory report: {run_dir}\")\nprint(\n    f\"Status={summary.get('overall_status')} \"\n    f\"packs={summary.get('packs_total')} topics={summary.get('topic_count')} \"\n    f\"risky_topics={summary.get('risky_topic_count')}\"\n)\nprint(\n    f\"Signals: policy_packs={signals.get('packs_with_policy_violation_topics')} \"\n    f\"low_ess_packs={signals.get('packs_with_low_ess_topics')}\"\n)\npreview = _topic_preview(report.get(\"top_risky_topics\"))\nif preview:\n    print(\"Top risky topics:\")\n    for row in preview:\n        print(f\"  {row}\")\n''')"

bench-report-insights-root: ## Print aggregated health/memory/failure insights for benchmark root and save JSON
	@uv run python -c "exec('''import json\nimport sys\nfrom collections import Counter\nfrom datetime import UTC, datetime\nfrom pathlib import Path\n\n\ndef sorted_counter(counter: Counter[str]) -> list[dict[str, object]]:\n    return [\n        {\"key\": key, \"count\": count}\n        for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))\n    ]\n\n\nroot = Path(\"$(BENCH_OUTPUT_ROOT)\")\nruns = sorted(\n    (run for run in root.iterdir() if run.is_dir()),\n    key=lambda run: run.stat().st_mtime,\n    reverse=True,\n) if root.exists() else []\ncompleted = [run for run in runs if (run / \"run_summary.json\").exists()]\nerror_runs = [run for run in runs if (run / \"run_error.json\").exists()]\nif not completed and not error_runs:\n    print(f\"No benchmark runs (summary or error) under {root}\")\n    sys.exit(0)\n\noutput_path = root / \"root_insights.json\"\nif not completed:\n    error_window = error_runs[:30]\n    error_type_counts: Counter[str] = Counter()\n    error_message_counts: Counter[str] = Counter()\n    for run in error_window:\n        payload = json.loads((run / \"run_error.json\").read_text(encoding=\"utf-8\"))\n        error_type = payload.get(\"error_type\") if isinstance(payload.get(\"error_type\"), str) else \"unknown\"\n        error_type_counts[error_type] += 1\n        raw_message = payload.get(\"error\") if isinstance(payload.get(\"error\"), str) else \"\"\n        first_line = raw_message.splitlines()[0] if raw_message else \"\"\n        compact = (first_line[:140] + \"...\") if len(first_line) > 140 else first_line\n        error_message_counts[compact or \"<empty>\"] += 1\n\n    insights = {\n        \"generated_at\": datetime.now(UTC).isoformat(),\n        \"root\": str(root),\n        \"mode\": \"error_only\",\n        \"completed_runs\": 0,\n        \"error_runs\": len(error_runs),\n        \"window_runs\": len(error_window),\n        \"error_type_counts\": sorted_counter(error_type_counts),\n        \"error_message_counts\": sorted_counter(error_message_counts),\n    }\n    output_path.write_text(json.dumps(insights, indent=2, sort_keys=True), encoding=\"utf-8\")\n    print(f\"Insights root: {root}\")\n    print(f\"Error-only mode (no completed summaries). Runs analyzed: {len(error_window)} / {len(error_runs)}\")\n    print(f\"Error types: {dict(error_type_counts)}\")\n    print(f\"Wrote: {output_path}\")\n    sys.exit(0)\n\nwindow = completed[:30]\ndecision_counts: Counter[str] = Counter()\nprofile_counts: Counter[str] = Counter()\nhealth_counts: Counter[str] = Counter()\nvalidity_counts: Counter[str] = Counter()\nisolation_counts: Counter[str] = Counter()\nfailure_step_counts: Counter[str] = Counter()\nfailure_reason_counts: Counter[str] = Counter()\ntopic_weight: Counter[str] = Counter()\npass_means: list[float] = []\npass_mins: list[float] = []\n\nfor run in window:\n    summary = json.loads((run / \"run_summary.json\").read_text(encoding=\"utf-8\"))\n    decision = summary.get(\"decision\") if isinstance(summary.get(\"decision\"), str) else \"unknown\"\n    profile = summary.get(\"profile\")\n    if not isinstance(profile, str):\n        profile = summary.get(\"health_summary\", {}).get(\"profile\")\n    profile_name = profile if isinstance(profile, str) else \"unknown\"\n    decision_counts[decision] += 1\n    profile_counts[profile_name] += 1\n\n    health = summary.get(\"health_summary\", {}).get(\"summary\", {})\n    validity = summary.get(\"memory_validity_summary\", {}).get(\"summary\", {})\n    isolation = summary.get(\"run_isolation_summary\", {}).get(\"summary\", {})\n    health_counts[str(health.get(\"overall_status\", \"unknown\"))] += 1\n    validity_counts[str(validity.get(\"overall_status\", \"unknown\"))] += 1\n    isolation_counts[str(isolation.get(\"overall_status\", \"unknown\"))] += 1\n\n    rates = []\n    for row in summary.get(\"pack_results\", []):\n        if not isinstance(row, dict):\n            continue\n        pass_rate = row.get(\"pass_rate\")\n        if isinstance(pass_rate, (int, float)) and not isinstance(pass_rate, bool):\n            rates.append(float(pass_rate))\n    if rates:\n        pass_means.append(sum(rates) / len(rates))\n        pass_mins.append(min(rates))\n\n    top_topics = validity.get(\"top_belief_topic_deltas\", [])\n    if isinstance(top_topics, list):\n        for row in top_topics:\n            if not isinstance(row, dict):\n                continue\n            topic = row.get(\"topic\")\n            delta = row.get(\"abs_delta_total\")\n            if not isinstance(topic, str):\n                continue\n            weight = float(delta) if isinstance(delta, (int, float)) and not isinstance(delta, bool) else 1.0\n            topic_weight[topic] += weight\n\n    trace_path = run / \"turn_trace.jsonl\"\n    if trace_path.exists():\n        with trace_path.open(\"r\", encoding=\"utf-8\") as handle:\n            for raw_line in handle:\n                line = raw_line.strip()\n                if not line:\n                    continue\n                try:\n                    row = json.loads(line)\n                except json.JSONDecodeError:\n                    continue\n                if bool(row.get(\"passed\", True)):\n                    continue\n                pack = row.get(\"pack\") if isinstance(row.get(\"pack\"), str) else \"unknown_pack\"\n                label = row.get(\"label\") if isinstance(row.get(\"label\"), str) else \"unknown_step\"\n                failure_step_counts[f\"{pack}/{label}\"] += 1\n                failures = row.get(\"failures\")\n                if isinstance(failures, list) and failures:\n                    first = failures[0]\n                    if isinstance(first, str):\n                        failure_reason_counts[first] += 1\n\nmean_pass = sum(pass_means) / len(pass_means) if pass_means else 0.0\nmean_min_pass = sum(pass_mins) / len(pass_mins) if pass_mins else 0.0\ninsights = {\n    \"generated_at\": datetime.now(UTC).isoformat(),\n    \"root\": str(root),\n    \"mode\": \"completed_runs\",\n    \"completed_runs\": len(completed),\n    \"window_runs\": len(window),\n    \"decision_counts\": sorted_counter(decision_counts),\n    \"profile_counts\": sorted_counter(profile_counts),\n    \"health_status_counts\": sorted_counter(health_counts),\n    \"memory_validity_status_counts\": sorted_counter(validity_counts),\n    \"isolation_status_counts\": sorted_counter(isolation_counts),\n    \"pass_rate\": {\n        \"mean_of_run_means\": round(mean_pass, 4),\n        \"mean_of_run_mins\": round(mean_min_pass, 4),\n    },\n    \"top_failed_steps\": sorted_counter(failure_step_counts)[:20],\n    \"top_failure_reasons\": sorted_counter(failure_reason_counts)[:20],\n    \"top_belief_topics\": [\n        {\"topic\": topic, \"weighted_delta\": round(weight, 4)}\n        for topic, weight in sorted(topic_weight.items(), key=lambda item: (-item[1], item[0]))[:20]\n    ],\n}\noutput_path.write_text(json.dumps(insights, indent=2, sort_keys=True), encoding=\"utf-8\")\n\nprint(f\"Insights root: {root}\")\nprint(f\"Window runs analyzed: {len(window)} / {len(completed)}\")\nprint(f\"Decision counts: {dict(decision_counts)}\")\nprint(f\"Health status counts: {dict(health_counts)}\")\nprint(f\"Memory validity counts: {dict(validity_counts)}\")\nprint(f\"Isolation status counts: {dict(isolation_counts)}\")\nprint(f\"Pass-rate means: mean={mean_pass:.0%} min_mean={mean_min_pass:.0%}\")\nif failure_step_counts:\n    print(\"Top failed steps:\")\n    for row in sorted_counter(failure_step_counts)[:10]:\n        print(f\"  {row['key']}: {row['count']}\")\nif failure_reason_counts:\n    print(\"Top failure reasons:\")\n    for row in sorted_counter(failure_reason_counts)[:8]:\n        print(f\"  {row['key']}: {row['count']}\")\nif topic_weight:\n    print(\"Top belief topics (weighted):\")\n    for topic, weight in sorted(topic_weight.items(), key=lambda item: (-item[1], item[0]))[:8]:\n        print(f\"  {topic}: {weight:.3f}\")\nprint(f\"Wrote: {output_path}\")\n''')"

bench-report-delta-last: ## Compare latest run summary with previous run
	@uv run python -c "exec('''import json\nimport sys\nfrom pathlib import Path\n\n\ndef pick_number(payload: dict[str, object], dotted_path: str, default: float = 0.0) -> float:\n    current: object = payload\n    for part in dotted_path.split('.'):\n        if not isinstance(current, dict):\n            return default\n        current = current.get(part)\n    if isinstance(current, (int, float)) and not isinstance(current, bool):\n        return float(current)\n    return default\n\n\ndef load_summary(run_dir: Path) -> dict[str, object]:\n    summary_path = run_dir / \"run_summary.json\"\n    if not summary_path.exists():\n        return {}\n    return json.loads(summary_path.read_text(encoding=\"utf-8\"))\n\n\ndef pack_pass_rates(summary: dict[str, object]) -> dict[str, float]:\n    rows = summary.get(\"pack_results\", [])\n    if not isinstance(rows, list):\n        return {}\n    rates: dict[str, float] = {}\n    for row in rows:\n        if not isinstance(row, dict):\n            continue\n        pack = row.get(\"pack\")\n        pass_rate = row.get(\"pass_rate\")\n        if isinstance(pack, str) and isinstance(pass_rate, (int, float)) and not isinstance(pass_rate, bool):\n            rates[pack] = float(pass_rate)\n    return rates\n\n\nroot = Path(\"$(BENCH_OUTPUT_ROOT)\")\nruns = sorted(\n    (run for run in root.iterdir() if run.is_dir()),\n    key=lambda run: run.stat().st_mtime,\n    reverse=True,\n) if root.exists() else []\ncompleted_runs = [run for run in runs if (run / \"run_summary.json\").exists()]\nif len(completed_runs) < 2:\n    print(\"Need at least two completed benchmark runs for delta report.\")\n    sys.exit(0)\n\nlatest = completed_runs[0]\nprevious = completed_runs[1]\nlatest_summary = load_summary(latest)\nprevious_summary = load_summary(previous)\nif not latest_summary or not previous_summary:\n    print(\"Delta report unavailable: failed to read latest completed run summaries\")\n    sys.exit(0)\n\nprint(f\"Latest run:   {latest}\")\nprint(f\"Previous run: {previous}\")\nprint(\n    f\"Decision delta: {previous_summary.get('decision')} -> {latest_summary.get('decision')} \"\n    f\"| stop={latest_summary.get('stop_reason')}\"\n)\n\nmetrics = [\n    (\"health_summary.summary.step_contract_fail_count\", \"Step contract fails\"),\n    (\"health_summary.summary.flagged_row_rate\", \"Flagged row rate\"),\n    (\"memory_validity_summary.summary.update_policy_violation_count\", \"Update policy violations\"),\n    (\"memory_validity_summary.summary.direction_mismatch_count\", \"Direction mismatches\"),\n    (\"memory_validity_summary.summary.memory_write_rate\", \"Memory write rate\"),\n    (\"memory_validity_summary.summary.belief_topic_count\", \"Belief topic count\"),\n    (\"run_isolation_summary.summary.seed_state_fail_count\", \"Isolation seed failures\"),\n    (\"run_isolation_summary.summary.interaction_chain_fail_count\", \"Isolation interaction chain fails\"),\n    (\"run_isolation_summary.summary.episode_chain_fail_count\", \"Isolation episode chain fails\"),\n    (\"cost_summary.total_calls\", \"Total calls\"),\n    (\"cost_summary.total_tokens\", \"Total tokens\"),\n]\n\nprint(\"Metric deltas:\")\nfor path, label in metrics:\n    old_value = pick_number(previous_summary, path)\n    new_value = pick_number(latest_summary, path)\n    delta = new_value - old_value\n    sign = \"+\" if delta > 0 else \"\"\n    if old_value.is_integer() and new_value.is_integer() and delta.is_integer():\n        print(f\"  {label}: {int(old_value)} -> {int(new_value)} ({sign}{int(delta)})\")\n    else:\n        print(f\"  {label}: {old_value:.4f} -> {new_value:.4f} ({sign}{delta:.4f})\")\n\nlatest_rates = pack_pass_rates(latest_summary)\nprevious_rates = pack_pass_rates(previous_summary)\nshared_packs = sorted(set(previous_rates) & set(latest_rates))\nchanged: list[tuple[float, str, float, float]] = []\nfor pack in shared_packs:\n    old_rate = previous_rates.get(pack, 0.0)\n    new_rate = latest_rates.get(pack, 0.0)\n    if abs(new_rate - old_rate) > 1e-9:\n        changed.append((abs(new_rate - old_rate), pack, old_rate, new_rate))\n\nif changed:\n    print(\"Pack pass-rate deltas (overlapping packs):\")\n    for _, pack, old_rate, new_rate in sorted(changed, reverse=True)[:15]:\n        print(f\"  {pack}: {old_rate:.0%} -> {new_rate:.0%}\")\nelse:\n    print(\"Pack pass-rate deltas (overlapping packs): no changes\")\n\nlatest_only = sorted(set(latest_rates) - set(previous_rates))\nprevious_only = sorted(set(previous_rates) - set(latest_rates))\nif latest_only:\n    print(f\"Packs only in latest run: {latest_only}\")\nif previous_only:\n    print(f\"Packs only in previous run: {previous_only}\")\n''')"

bench-signal-gate-last: ## Fail if latest run violates quick-signal quality thresholds
	@uv run python -c "exec('''import json\nimport sys\nfrom pathlib import Path\n\nroot = Path(\"$(BENCH_OUTPUT_ROOT)\")\nruns = sorted(\n    (path for path in root.iterdir() if path.is_dir()),\n    key=lambda path: path.stat().st_mtime,\n    reverse=True,\n) if root.exists() else []\nif not runs:\n    print(\"Signal gate failed: no benchmark runs found.\")\n    sys.exit(1)\n\nsummary_path = runs[0] / \"run_summary.json\"\nif not summary_path.exists():\n    print(f\"Signal gate failed: missing run_summary.json in {runs[0]}\")\n    sys.exit(1)\n\nsummary = json.loads(summary_path.read_text(encoding=\"utf-8\"))\nmin_pack_pass_rate = float(\"$(BENCH_SIGNAL_MIN_PACK_PASS_RATE)\")\nmax_ess_default_rate = float(\"$(BENCH_SIGNAL_MAX_ESS_DEFAULT_RATE)\")\nmax_ess_retry_rate = float(\"$(BENCH_SIGNAL_MAX_ESS_RETRY_RATE)\")\nviolations = []\n\ndefault_rate = summary.get(\"ess_default_summary\", {}).get(\"defaulted_step_rate\")\nif isinstance(default_rate, (int, float)) and default_rate > max_ess_default_rate:\n    violations.append(\n        f\"ESS defaulted step rate {default_rate:.3f} exceeds {max_ess_default_rate:.3f}\"\n    )\n\nretry_rate = summary.get(\"ess_retry_summary\", {}).get(\"retry_step_rate\")\nif isinstance(retry_rate, (int, float)) and retry_rate > max_ess_retry_rate:\n    violations.append(\n        f\"ESS retry step rate {retry_rate:.3f} exceeds {max_ess_retry_rate:.3f}\"\n    )\n\nfor pack in summary.get(\"pack_results\", []):\n    pass_rate = pack.get(\"pass_rate\")\n    if isinstance(pass_rate, (int, float)) and pass_rate < min_pack_pass_rate:\n        violations.append(\n            f\"{pack.get('pack')} pass_rate {pass_rate:.3f} below {min_pack_pass_rate:.3f}\"\n        )\n\nif violations:\n    print(\"Signal gate failed:\")\n    for violation in violations:\n        print(f\"  - {violation}\")\n    sys.exit(1)\n\nprint(\n    \"Signal gate passed \"\n    f\"(min_pack_pass_rate={min_pack_pass_rate:.2f}, \"\n    f\"max_ess_default_rate={max_ess_default_rate:.2f}, \"\n    f\"max_ess_retry_rate={max_ess_retry_rate:.2f}).\"\n)\n''')"

check: lint typecheck test bench-contracts ## Run all no-key quality checks

check-ci: format-check check ## Run local checks equivalent to CI

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
	@echo "Sponge reset. Next run starts from seed state."

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
	rm -rf .mypy_cache/ .ruff_cache/ .pytest_cache/ htmlcov/ .coverage
	@echo "Cleaned."

nuke: clean reset ## Full reset — remove .venv, data, and all caches
	rm -rf .venv/
	@echo "Nuked. Run 'make install' to start fresh."
