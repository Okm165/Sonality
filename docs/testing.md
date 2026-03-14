# Testing and Evaluation

This page is an operator playbook for validating Sonality:

- what runs without API keys,
- what runs with API keys,
- which commands to run in order,
- and which artifacts matter for release decisions.

## Test Layers

| Layer | API Key | Command | Purpose |
|---|---|---|---|
| L0 Format/Lint/Type | No | `make check-ci` | Fast CI-parity quality gate |
| L1 Runtime correctness | No | `uv run pytest -q tests` | Deterministic runtime behavior and invariants |
| L2 Non-live benchmark contracts | No | `uv run pytest benches -m "bench and not live" -q` | Harness contract and release-gating logic |
| L3 Live benchmark slices | Yes | `make bench-memory` / `make bench-personality` | API-backed behavioral validation |
| L4 Full teaching benchmark | Yes | `make bench-teaching` | End-to-end release evidence pack |

## Recommended Execution Order

Run in this order from fastest to most expensive:

1. `make check-ci`
2. `uv run pytest benches -m "bench and not live" -q`
3. `make preflight-live`
4. `make bench-memory` or `make bench-personality`
5. `make bench-teaching` (release candidate evaluation)

## Live Run Preconditions

Before any live benchmark:

```bash
SONALITY_BASE_URL=http://localhost:11434/v1   # example: Ollama OpenAI-compatible endpoint
SONALITY_API_KEY=...
SONALITY_MODEL=qwen2.5:14b-instruct
SONALITY_ESS_MODEL=qwen2.5:14b-instruct
```

Validate config:

```bash
make preflight-live
```

If `SONALITY_BASE_URL` is missing, runtime should be treated as misconfigured.

## What the Non-Live Suite Must Guarantee

Non-live checks should protect these invariants:

- ESS parsing/coercion fallback remains safe.
- Memory admission/reranking behavior stays deterministic.
- Belief update/decay math preserves expected bounds.
- Scenario contract checks catch release policy regressions.
- Release-readiness aggregation still blocks unsafe candidates.

## Core Artifacts in Live Runs

Teaching benchmarks write structured artifacts under `data/teaching_bench/`.
Prioritize these when triaging:

- `summary.json` — top-level outcome and key rates.
- `release_readiness.json` — release gate view and blockers.
- `risk_tier_dashboard.json` — hard-gate evidence sufficiency by tier.
- `health_summary.json` — stability and behavioral health rollups.
- `observer_verdict_trace.jsonl` — per-step contract observer verdicts.
- `risk_event_trace.jsonl` — risk events, severity tags, and evidence context.

## Fast Failure Triage

Use this checklist:

1. Read `release_readiness.json`.
2. If blocked, inspect hard-gate failures first.
3. If not blocked but unstable, inspect `health_summary.json`.
4. If evidence is insufficient, inspect `risk_tier_dashboard.json`.
5. For step-level root cause, inspect `observer_verdict_trace.jsonl`.

## Keep Tests Lean

Prefer tests that validate behavior, contracts, or safety boundaries.
Avoid adding tests that only re-check trivial helper mechanics with no release impact.
