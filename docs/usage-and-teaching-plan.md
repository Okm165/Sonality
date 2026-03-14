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
   - `/snapshot` for narrative baseline.
   - `/beliefs` for structured stance baseline.
   - `/health` for initial diagnostics.

## 2) Teach with evidence, not pressure

Use interaction patterns that can legitimately update beliefs:

- Provide explicit reasoning chains.
- Prefer verifiable evidence over assertions.
- Make topic labels clear and stable across turns.
- Avoid social-pressure framing ("everyone says", "you should agree").

See [Training Guide](training-guide.md) for deep methodology and curricula.

## 3) Observe update gates in real time

After each turn, inspect:

- ESS score and topics (`/health`, status line output).
- Staged updates (`/staged`) rather than immediate opinion flips.
- Topic engagement growth (`/topics`).

Expected behavior:

- Low-ESS chat should not alter opinions.
- High-ESS evidence should stage topic-specific deltas.
- Reflection should consolidate insights periodically without erasing identity.

## 4) Validate personality integrity

Use this monitoring checklist:

- **Coherence:** beliefs and snapshot stay semantically aligned.
- **Resistance:** disagreement rate does not collapse toward zero.
- **Stability:** major shifts require repeated high-quality evidence.
- **Specificity:** reflection output stays concrete (not generic assistant drift).

Use `/diff`, `/shifts`, `/health`, and historical sponge versions in `data/sponge_history/`.

## 5) Run non-live quality gates

Before sharing changes:

- `make check` (lint + typecheck + unit tests)
- `uv run pytest benches -m "bench and not live" -q`
- `uv run --with zensical zensical build --clean` (docs build)

CI mirrors these no-key checks in `.github/workflows/ci.yml`.

## 6) Escalate to benchmark suites when needed

For deeper quality/risk checks:

- `make bench-memory`
- `make bench-personality`
- `make bench-teaching`

These are API-key-backed evaluation runs and should be used for release-gating and
regression analysis, not for every local edit.

---

**Related**

- [Getting Started](getting-started.md) — installation and first run
- [Training Guide](training-guide.md) — detailed teaching methodology
- [Testing & Evaluation](testing.md) — validation layers and benchmark interpretation
- [Configuration](configuration.md) — runtime knobs and expected effects
