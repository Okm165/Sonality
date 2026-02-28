# Research-Backed Upgrade Plan

This page translates the research synthesis into concrete architecture decisions for Sonality.
It focuses on high-impact changes that keep the codebase small: no graph DB migration, no heavy orchestration framework, no speculative subsystems.

---

## Executive Priority Order

1. **Reduce personality corruption risk first**: staged belief commits, reflection-only narrative rewrites, strict snapshot validation.
2. **Improve memory quality before memory complexity**: typed vector memories and quality reranking, not graph expansion.
3. **Instrument behavior, not self-reports**: per-turn health events, disagreement dynamics, belief confidence trajectories.
4. **Calibrate ESS continuously**: keep score quality tied to external data (IBM ArgQ), not intuition.

This ordering follows ENGRAM (typed memory over graph complexity), ABBEL (belief bottleneck usefulness + propagation risk), PersistBench/SYConBench (memory-induced sycophancy + self-judge bias), and Narrative Continuity Test findings.

---

## Task 1 - Architecture Improvements

## 1) Memory Architecture

### Decision

Keep **ChromaDB + JSON** as the primary storage architecture, but enforce lightweight memory typing.

### Why

- **ENGRAM (2025)**: episodic/semantic/procedural typing with lightweight routing beats full-context baselines without requiring a graph DB.
- **Mem0 vs Graphiti (2026)**: vector memory significantly outperforms graph memory on efficiency, with no statistically significant accuracy gain from graph structure in the tested setup.
- **ABBEL (2025)**: compact belief states are useful, but error propagation increases when compression is lossy and unobserved.

### What this project does now

- **Episodic memory**: Chroma episodes tagged `memory_type=episodic` (low-ESS or conversational events).
- **Semantic memory**: Chroma episodes tagged `memory_type=semantic` (higher-ESS, belief-relevant events).
- **Procedural memory**: immutable behavior contract in `CORE_IDENTITY` and instruction blocks.

This gives ENGRAM-style typing with minimal code and avoids graph overhead.

### Flat `opinion_vectors` vs relational graph

- Keep flat vectors for now.
- Add relation logic only if a measurable failure appears (for example, contradictory topic bundles that cannot be resolved by reflection).
- This matches the vector-first evidence from Mem0 vs Graphiti and avoids speculative complexity.

---

## 2) ESS Pipeline

### Decision

Keep ESS as a dedicated classifier call, but harden it operationally:

- use a **separate ESS model** in production;
- keep **third-person framing** and **user-only evaluation**;
- continuously calibrate against a ground-truth set.

### Why

- **SYConBench (EMNLP 2025)**: self-judge and attribution effects can move outcomes dramatically; third-person framing reduces sycophantic bias substantially.
- **BASIL (2025)**: post-hoc Bayesian-style calibration improves separation of rational updates vs social-compliance updates.
- **ConQRet/MArgE (2025)**: multi-judge and structured argument quality scoring are useful when you need evaluation rigor.

### Implementation stance

- Sonality keeps single-model ESS calls by default for cost/latency.
- For high-stakes deployments, set `SONALITY_ESS_MODEL` different from the main response model and run calibration tests regularly.
- MArgE-style argument trees are a good future addition for offline audits, but are intentionally not in the runtime path to keep latency bounded.

---

## 3) Snapshot Update Mechanism

### Decision

Keep **append-first, consolidate-later**:

- per turn: extract one insight + stage belief deltas;
- periodic/event-driven reflection: rewrite narrative snapshot;
- never do wholesale snapshot rewrites every message.

### Why

- **ABBEL (2025)**: compact belief states are powerful but vulnerable to propagation error.
- **Stanford Generative Agents (Park et al., 2023)**: reflection is the critical consolidation mechanism.
- **Broken-telephone effects in iterative rewriting studies (2025)**: repeated full rewrites erase minority traits.

### Structural guardrails

- snapshot retention validation (`validate_snapshot`);
- staged opinion commits with cooling period;
- versioned archive in `sponge_history`.

---

## 4) Embedding Model

### Current stance

Chroma default embedding remains acceptable for a small codebase and short summaries.

### Recommended migration path to a long-context embedding backend

Switch only when retrieval quality becomes a measured bottleneck. Keep concrete model names in `docs/model-considerations.md`.

1. baseline retrieval metrics on current embeddings (precision-at-k on held-out memory probes);
2. dual-index migration window (old + new embeddings side by side);
3. backfill embeddings;
4. compare retrieval quality and latency;
5. cut over and drop old index.

### Why not force migration immediately

- No new dependency pressure unless quality requires it.
- Keeps the system within the current dependency budget (LLM provider SDK, `chromadb`, `pydantic`, `dotenv`).

---

## 5) Anti-Sycophancy Measures

### Implemented and recommended

- immutable anti-sycophancy core identity;
- third-person ESS framing;
- structural disagreement detection;
- **staged opinion updates with cooling period** (`SONALITY_OPINION_COOLING_PERIOD`, default 3);
- health warning when disagreement rate collapses.

### Why

- **PersistBench (2025)**: memory + preference persistence can collapse into sycophancy without memory-specific controls.
- **BASIL (2025)**: distinguish evidence-driven changes from social-compliance changes.
- **MONICA (2025)**: real-time monitoring is needed because sycophancy appears during inference, not just in training.
- **Third-person framing work**: materially reduces social-attribution bias.

### Cooling period rationale

Immediate updates are easy to exploit with repeated pressure. A short delay + aggregation over a few turns reduces reactive flips while preserving evidence-driven adaptation.

---

## 6) Forgetting / Decay

### Decision

Keep power-law decay and apply it at reflection boundaries.

### Why

- **Ebbinghaus-in-LLMs work (2025)** and **FadeMem**: power-law forgetting tracks realistic retention behavior better than no decay.
- prevents zombie beliefs and keeps high-confidence beliefs from permanent lock-in.

### Interaction with trait state

Sonality intentionally does not use dynamic OCEAN updates as a primary control variable because high measurement noise and self-report instability are repeatedly reported (PERSIST, Personality Illusion). Behavioral metrics remain primary.

---

## Task 3 - Code Quality and Monitoring

## 1) Simplification policy

- no graph DB;
- no runtime argument-tree parser;
- no new orchestration framework;
- keep memory typing in metadata and routing logic only.

This is consistent with project philosophy: smallest mechanism that can carry evidence.

## 2) Logging pipeline (implemented)

`data/ess_log.jsonl` now carries a turn-by-turn evolution story:

- `context`: prompt assembly footprint, retrieval composition, context sizes.
- `ess`: classification result and topic-linked belief state.
- `opinion_staged`: staged deltas with due interaction.
- `opinion_commit`: committed aggregate deltas after cooling period.
- `health`: disagreement, belief growth, lexical diversity, warning flags.
- `reflection`: consolidation summary and retention-oriented metrics.

## 3) Monitoring data contract

Recommended downstream aggregations:

- **stability**: snapshot lexical diversity, snapshot size, reflection jaccard.
- **conviction**: high-confidence ratio, belief-count trajectory.
- **plasticity**: staged-to-committed ratio, mean commit lag.
- **sycophancy risk**: disagreement-rate trend and low-disagreement streaks.
- **memory quality**: semantic vs episodic retrieval share.

## 4) Divergence detection checks

Run after every interaction:

- `possible_sycophancy`: disagreement rate too low after warmup.
- `snapshot_bland`: low lexical diversity.
- `snapshot_too_short`: collapse risk.
- `low_belief_growth`: no meaningful belief formation over long windows.

These are intentionally simple, cheap, and directly auditable in logs.

---

## Open but Deferred Changes

These are valid ideas but intentionally deferred to keep complexity bounded:

- MArgE-style formal argument trees in runtime ESS path.
- Graph relational belief store.
- Domain-specific stubbornness schedules per topic family.
- Multi-judge ESS at inference time for every turn.

If adopted later, each should be gated by measurable failure in current metrics, not by speculative architecture preference.

---

## Operational Recommendations

For research-grade runs:

1. set a separate `SONALITY_ESS_MODEL`;
2. keep `SONALITY_OPINION_COOLING_PERIOD=3` (or 4 for high-manipulation environments);
3. run ESS calibration benchmarks weekly (`benches/test_ess_calibration_live.py`);
4. review `health` + `reflection` event trends before changing thresholds.

This keeps the system aligned with current evidence: quality-gated, typed, observable, and minimal.
