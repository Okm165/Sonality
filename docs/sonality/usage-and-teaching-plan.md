# Usage and Teaching Plan

This page is the practical operating path: how to run Sonality, shape it with high-quality
interactions, and monitor whether personality evolution remains coherent and evidence-driven.

## 1) Boot and baseline

1. Install and configure:
   - Follow [Getting Started](getting-started.md).
   - Set `SONALITY_BASE_URL`, `SONALITY_API_KEY`, and model IDs in `.env`.
2. Start a fresh run:
   - `make run` for local REPL usage.
   - Optionally `make reset` before a new training cycle.
3. Capture baseline state:
   - `/health` to check snapshot version and initial diagnostics.
   - `/beliefs` for structured stance baseline.

## 2) Teach with evidence, not pressure

Use interaction patterns that can legitimately update beliefs:

- Provide explicit reasoning chains.
- Prefer verifiable evidence over assertions.
- Make topic labels clear and stable across turns.
- Avoid social-pressure framing ("everyone says", "you should agree").

Key: provide reasoning chains and verifiable evidence, not social pressure.

## 3) Observe update gates in real time

After each turn, inspect:

- ESS score and topics (shown after each response in REPL status line).
- Belief states (`/beliefs`) to track opinion changes.
- Agent health (`/health`) for snapshot version and identity coherence tracking.

Expected behavior:

- Low-ESS chat should not alter beliefs.
- High-ESS evidence triggers integrate_knowledge (store facts + update beliefs).
- Knowledge integration consolidates insights without erasing identity.

## 4) Validate personality integrity

Use this monitoring checklist:

- **Coherence:** beliefs and snapshot stay semantically aligned.
- **Resistance:** disagreement rate does not collapse toward zero.
- **Stability:** major shifts require repeated high-quality evidence.
- **Specificity:** integrate_knowledge output stays concrete (not generic assistant drift).

Use `/beliefs`, `/health`, and historical snapshots in Neo4j `PersonalitySnapshot` nodes.

## 5) Run non-live quality gates

Before sharing changes:

- `make check` (lint + typecheck + unit tests)
- `uv run pytest benches -m "bench and not live" -q`
- `uv run --with zensical zensical build --clean` (docs build)

CI runs lint + typecheck + tests in `.github/workflows/ci.yml`; docs build runs in `docs.yml`.

## 6) Escalate to benchmark suites when needed

For deeper quality/risk checks:

- `make bench-memory`
- `make bench-personality`
- `make bench-teaching`

These are API-key-backed evaluation runs and should be used for release-gating and
regression analysis, not for every local edit.

---

**Related**: [Getting Started](getting-started.md) | [Testing](testing.md) | [Configuration](configuration.md)
