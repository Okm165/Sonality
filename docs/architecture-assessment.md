# Architecture Assessment

Research-backed architecture and training blueprint for Sonality under hard constraints:

- API-only access (provider API, no fine-tuning)
- single-user conversational runtime
- minimal dependencies and minimal abstraction
- behavior-first evaluation over self-reported traits

This document is intentionally implementation-oriented: each recommendation is tied to impact, complexity, and what code should be removed or avoided.

---

## 1. ARCHITECTURE ASSESSMENT

### 1.1 Memory Architecture (ChromaDB + JSON vs Graph)

**Verdict: MODIFY (lightly), REJECT full graph DB**

- **Research justification**
  - ENGRAM (Patel and Patel, 2025): typed memory with dense retrieval beats full-context baselines by about 15 points on LongMemEval while using about 1% token budget; no graph DB required.
  - Mem0 vs Graphiti empirical comparison (2025): vector memory significantly outperforms graph memory on efficiency with no statistically significant accuracy gain.
  - RecallM (2024): graph+vector excels for temporal multi-hop belief updating, but its gains are strongest in tasks where causal graph traversal is central.
- **Impact vs complexity**
  - Full graph layer (Neo4j-style or heavy NetworkX usage): high complexity, uncertain gain for Sonality's single-user conversational pattern.
  - Typed vector memory + quality rerank + lightweight relational signals: high impact, low complexity.
- **What to replace/remove**
  - Keep one store (`ChromaDB`) and remove any need for secondary graph persistence.
  - Prefer metadata-based relation hints instead of graph entities/edges services.
- **Minimal implementation**
  - Keep typed retrieval (`semantic` then `episodic`).
  - Add optional relation hints in metadata (`related_topics`, `stance_sign`) and apply small rerank bonuses for causally adjacent memories.

### 1.2 ESS Pipeline (Self-Judge Bias, Argument Quality, Calibration)

**Verdict: MODIFY**

- **Research justification**
  - SYConBench (Hong et al., EMNLP 2025): third-person framing reduces sycophancy up to 63.8%.
  - BASIL (Atwell et al., 2025): Bayesian consistency requires distinguishing rational updates from agreeableness artifacts.
  - MArgE (2025), SPARK (2024), ConQRet (2025): richer argument assessment improves quality scoring but adds runtime overhead.
- **Impact vs complexity**
  - Separate ESS model: high impact, very low complexity.
  - Per-turn formal argument trees (MArgE-style): medium potential benefit, high latency/complexity in API-only loop.
  - Offline calibration harness: medium impact, low-medium complexity.
- **What to replace/remove**
  - Replace same-model default ESS in production usage with a dedicated ESS model.
  - Reject per-turn argument-tree parsing and extra judge calls in online path.
- **Minimal implementation**
  - Keep third-person ESS prompt and current structured ESS fields.
  - Add a small offline calibration suite against curated argument-quality samples (no runtime cost).

### 1.3 Snapshot Update Mechanism (Lossy Rewrites)

**Verdict: KEEP current direction, MODIFY guardrails**

- **Research justification**
  - ABBEL (2025): belief bottleneck (compact natural-language state) is effective but vulnerable to error propagation.
  - Park et al. (2023): reflection is a critical mechanism for coherence.
- **Impact vs complexity**
  - Per-turn full snapshot rewrite: high drift risk.
  - Append-only insight ledger + periodic consolidation: high value, low complexity.
- **What to replace/remove**
  - Remove any remaining assumptions of per-message full rewrite behavior.
  - Keep insight accumulation + periodic/event reflection as the only synthesis path.
- **Minimal implementation**
  - Continue logging `reflection` integrity metrics (`snapshot_jaccard`, dropped strong beliefs).
  - Add explicit contradiction count in reflection logs to catch hidden compression failures early.

### 1.4 Embedding Backend (Compact vs Long-Context)

**Verdict: MODIFY via migration path, not immediate mandatory switch**

- **Research justification**
  - Compact embedding backends can lose semantic fidelity on longer inputs.
  - For long-horizon memory, stronger embedding quality improves retrieval robustness and reduces contextual tunneling risk.
- **Impact vs complexity**
  - Immediate migration: medium impact, medium operational complexity (re-embedding).
  - Deferred migration with explicit trigger criteria: better operational trade-off.
- **What to replace/remove**
  - Keep current embedding path until trigger criteria are met.
  - Remove ambiguity by documenting exact migration playbook.
- **Minimal implementation**
  - Trigger migration when retrieval quality drops on long user inputs or when episode length distribution exceeds the current backend's reliable context range.
  - Migration sequence:
    1. export/backup current episode metadata,
    2. rebuild collection with new embedding function,
    3. re-ingest summaries with same IDs/metadata,
    4. run retrieval regression tests,
    5. switch runtime.

### 1.5 Anti-Sycophancy Measures

**Verdict: KEEP core stack, MODIFY evaluation rigor**

- **Research justification**
  - PersistBench (Pulipaka et al., 2026): memory-induced sycophancy is a severe failure mode.
  - SYConBench (Hong et al., 2025): ToF/NoF metrics are practical for multi-turn pressure testing.
  - BASIL (2025): consistency-aware update logic outperforms naive agreement patterns.
  - SMART (2025) and MONICA (2025): powerful but require training-time or inference-internals access unavailable in this deployment model.
- **Impact vs complexity**
  - Current defenses (third-person ESS, cooling period, confidence resistance, quality gating): high value, low complexity.
  - RL/MCTS or hidden-state monitors: high complexity, outside API-only constraints.
- **What to replace/remove**
  - Keep current runtime defenses.
  - Add offline sycophancy evaluation protocol and remove reliance on anecdotal checks.
- **Minimal implementation**
  - Log ToF/NoF-style session metrics from runtime events.
  - Keep cooling period (`N=3`) as default.

### 1.6 Forgetting/Decay and OCEAN Interaction

**Verdict: KEEP decay, REJECT dynamic OCEAN as control loop**

- **Research justification**
  - Ebbinghaus-in-LLMs (2025): forgetting curves remain useful to avoid stale lock-in.
  - PERSIST (2025), Personality Illusion (NeurIPS 2025): self-reported trait measures are unstable and weakly coupled to real behavior.
- **Impact vs complexity**
  - Power-law confidence decay with reinforcement floors: high value, very low complexity.
  - Dynamic OCEAN update loops: low reliability, unnecessary complexity.
- **What to replace/remove**
  - Keep belief confidence decay.
  - Remove stale OCEAN-centric evaluation assumptions from docs/tests where they are no longer first-class.
- **Minimal implementation**
  - Continue decay in reflection path.
  - Keep behavior-first health metrics as canonical signals.

### 1.7 Opinion Confidence and Belief Revision

**Verdict: KEEP + MODIFY (AGM-lite refinement)**

- **Research justification**
  - AGM framework: rational change requires expansion/contraction/revision discipline.
  - Belief-R (2024) and Hurst (2024): practical systems need hybrid revision (formal constraints + heuristics).
  - Deliberative Reasoning Networks (2025): uncertainty-aware updates improve adversarial robustness.
- **Impact vs complexity**
  - Current confidence/evidence metadata is strong.
  - Small contraction-before-revision addition is low complexity and improves rationality.
- **What to replace/remove**
  - Replace purely ad-hoc opposition damping with explicit two-step update:
    1. contraction of confidence under strong counter-evidence,
    2. directional revision.
- **Minimal implementation**
  - Implement in existing update path without theorem prover, new dependency, or extra LLM calls.

### 1.8 Memory Poisoning Defense

**Verdict: MODIFY**

- **Research justification**
  - AgentPoison (Chan et al., 2024): tiny poison rate can still drive major drift.
  - MemoryGraft (Srivastava and He, 2025): poisoned retrievals can dominate outputs.
  - Experience-following property (2025): retrieval quality directly shapes behavior trajectories.
- **Impact vs complexity**
  - Provenance-aware metadata and reranking penalties: high impact, low-medium complexity.
- **What to replace/remove**
  - Replace trust-by-default retrieval ordering with provenance-aware quality penalties.
- **Minimal implementation**
  - Add immutable metadata (`origin`, `ingested_at`, `ess_model`, `session_id`) and penalize low-credibility records even if semantically close.

### 1.9 Disagreement Detection

**Verdict: KEEP**

- **Research justification**
  - Stance dynamics are more reliable than lexical cue matching for multi-turn sycophancy detection (SYConBench 2025).
- **Impact vs complexity**
  - Structural check (`direction x position < 0`) is simple and robust.
- **What to replace/remove**
  - Keep structural detector.
  - Reject keyword-based disagreement heuristics and extra per-turn judge calls.

---

## 2. BEHAVIOR DEVELOPMENT PIPELINE

### 2.1 Personality Formation (APF 3-layer progression)

Map AI Personality Formation (ICLR 2026 submission) directly onto Sonality:

1. **Linguistic Mimicry (0-20 interactions)**
   - Use strict ESS gating and dampened updates.
   - Goal: stabilize style, avoid first-impression overfitting.
2. **Structured Accumulation (20-80 interactions)**
   - Use staged updates, confidence growth, typed memory routing.
   - Goal: turn high-quality repeated evidence into durable beliefs.
3. **Autonomous Expansion (80+ interactions)**
   - Reflection proposes tentative extensions to adjacent topics.
   - Goal: coherent, non-random worldview expansion.

### 2.2 Update Granularity

Use three interacting cadences:

- **Per-message:** ESS + topic tracking + staged deltas.
- **Short-window commit:** cooling-period net commit.
- **Mid-window consolidation:** reflection synthesis.

Rationale: per-message hard commits are reactive; reflection-only is too sluggish.

### 2.3 Collapse Prevention Guarantees

Structural safeguards:

- append-only insight ledger between reflections,
- snapshot retention validation,
- high-confidence belief preservation checks,
- versioned archives and rollback path,
- immutable core identity anchor.

### 2.4 Measuring Personality Coherence

Primary metrics (behavior-first):

- disagreement rate,
- belief growth trajectory,
- high-confidence ratio,
- snapshot jaccard continuity,
- insight yield,
- staged-to-committed transition behavior.

Secondary evaluation:

- Narrative Continuity Test axes (Situated Memory, Goal Persistence, Autonomous Self-Correction, Stylistic/Semantic Stability, Persona Continuity),
- self-report vs behavior contradiction checks.

### 2.5 Reflection Strategy

Use dual trigger:

- periodic (every `REFLECTION_EVERY` interactions),
- event-triggered (cumulative shift threshold).

This matches Park et al. (2023): reflection should track meaningful events, not only fixed intervals.

### 2.6 Stubbornness/Resistance Model

Adopt confidence-driven resistance (Friedkin-Johnsen-compatible behavior):

- resistance increases with accumulated evidence and confidence,
- no per-domain hard-coded stubbornness knobs,
- stronger opposition evidence can still contract confidence before revision.

### 2.7 Teaching Methodology (Practical)

Recommended training curriculum:

1. **Interview-first** (discover substrate)
2. **Evidence modules** (thesis/evidence/counterevidence/synthesis)
3. **Socratic probes** (assumption/falsification/trade-off questions)
4. **Adversarial pairs** (weak pressure vs strong evidence)
5. **Constitutional reinforcement** (periodic core-value scenarios)

This keeps behavior intentional rather than random conversation drift.

---

## 3. IMPLEMENTATION PLAN

Atomic commit plan ordered by impact-to-complexity.

### (a) High-impact / low-complexity first

1. **Docs canonicalization and consistency cleanup**
   - consolidate architecture decisions and remove stale assumptions.
   - Est. `+250 / -500`.

2. **AGM-lite contraction-before-revision**
   - implement minimal confidence contraction under strong opposing evidence.
   - Files: `sonality/agent.py`, tests.
   - Est. `+35 / -20`.

3. **Poisoning provenance metadata**
   - add ingestion provenance + retrieval penalties.
   - Files: `sonality/memory/episodes.py`, `sonality/agent.py`, tests.
   - Est. `+40 / -12`.

### (b) Medium-impact

4. **Light relational rerank hints**
   - add relation-aware metadata bonus without graph dependency.
   - Files: `sonality/memory/episodes.py`, tests.
   - Est. `+45 / -15`.

5. **Reflection contradiction ledger**
   - emit unresolved-contradiction count in reflection events.
   - Files: `sonality/agent.py`, docs.
   - Est. `+20 / -5`.

### (c) Speculative / research-heavy (defer)

6. **NetworkX augmentation for memory-belief graph**
   - only if measured retrieval misses justify complexity.
   - Est. `+150 / -40`.

7. **Offline SYConBench-style evaluator harness**
   - richer regression suite, not runtime logic.
   - Est. `+120 / -20`.

---

## 4. MONITORING SPECIFICATION

### 4.1 Events to Emit (JSONL)

Required events:

- `context`
- `ess`
- `opinion_staged`
- `opinion_commit`
- `health`
- `reflection`

Critical fields:

- ESS quality dimensions (`reasoning_type`, `source_reliability`, `internal_consistency`, `novelty`)
- update mechanics (`signed_magnitude`, `effective_magnitude`, confidence before/after)
- memory routing/provenance (`memory_type`, `origin`, `session_id`)
- reflection integrity (`snapshot_jaccard`, `beliefs_dropped`, `insight_yield`, `entrenched`)

### 4.2 REPL Operator Surface

Must expose:

- `/health`
- `/beliefs`
- `/staged`
- `/diff`
- `/shifts`

### 4.3 Automated Anomaly Rules

Flag when:

- disagreement rate < 0.15 after warm-up (`possible_sycophancy`)
- disagreement rate > 0.50 (`contrarian_drift`)
- belief_count < 3 after 40 interactions (`low_belief_growth`)
- reflection drops strong beliefs with low jaccard (`reflection_regression`)
- high-confidence ratio > 0.80 with low update throughput (`ossified_beliefs`)
- repeated low-credibility retrieval dominance (`poisoning_risk`)

### 4.4 Example Event Records

```json
{"event":"ess","interaction":42,"score":0.64,"reasoning_type":"empirical_data","source_reliability":"peer_reviewed","internal_consistency":true,"novelty":0.52,"topics":["education_policy"]}
```

```json
{"event":"opinion_staged","interaction":42,"topic":"education_policy","signed_magnitude":0.021,"due_interaction":45,"confidence_before":0.44,"confidence_after":0.45}
```

```json
{"event":"health","interaction":42,"belief_count":11,"high_conf_ratio":0.36,"disagreement_rate":0.24,"warnings":[]}
```

```json
{"event":"reflection","interaction":60,"insights_consolidated":4,"beliefs_dropped":[],"snapshot_jaccard":0.71,"insight_yield":0.28,"entrenched":["education_policy"]}
```

---

## 5. REJECTED IDEAS

Rejected because they violate constraints or add complexity without strong local payoff.

1. **Full graph database migration**
   - Conflicts with minimalism and lacks clear single-user benefit vs typed vector memory.

2. **Per-turn MArgE/ConQRet online judges**
   - Valuable conceptually, too expensive for runtime loop.

3. **SMART/MONICA runtime replication**
   - Requires RL pipelines or inference-internal access unavailable here.

4. **Fine-tuning personality methods (BIG5-CHAT/PISF/Open Character Training)**
   - Strong research, incompatible with API-only constraint.

5. **Activation-space persona-vector production monitor**
   - Requires hidden-state access not provided by standard API calls.

6. **Domain-specific stubbornness knobs**
   - Adds brittle tuning complexity; confidence/evidence-based resistance is simpler and data-driven.

---

## References Used in Decisions

- AGM framework (Alchourron, Gardenfors, Makinson)
- Park et al. (2023), Generative Agents
- RecallM (2024)
- Belief-R (2024)
- Hurst (2024)
- AgentPoison (2024)
- ENGRAM (2025)
- BASIL (2025)
- MArgE (2025)
- SPARK (2024)
- ConQRet (2025)
- MemoryOS (2025)
- SMART (2025)
- MONICA (2025)
- Mem0 vs Graphiti comparison (2025)
- Personality Illusion (NeurIPS 2025)
- PERSIST (2025)
- AI Personality Formation (ICLR 2026 submission)
- PersistBench (2026)
