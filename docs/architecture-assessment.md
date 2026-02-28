# Architecture Assessment

A research-driven evaluation of Sonality's architecture against the academic literature on LLM agent memory, belief revision, personality dynamics, and behavioral alignment. Each area receives a verdict (KEEP, MODIFY, ADD, REJECT) with specific citations.

---

## 1. Memory Architecture

**Verdict: KEEP**

### What the research says

RecallM (2023) reports 4x better belief updating with graph DB vs vector alone. However, this comparison measures *retrieval-driven* belief updates — the agent queries memory and updates beliefs based on what it retrieves.

Sonality does not work this way. Beliefs are tracked in `opinion_vectors` + `belief_meta` — a structured Bayesian system updated via ESS classification, not via memory retrieval. Memory retrieval serves context assembly, not belief revision. The 4x claim does not apply to this architecture.

Mem0 vs Graphiti (arXiv:2601.07978) provides the decisive empirical evidence: vector databases "significantly outperform graph databases in efficiency" with "NO statistically significant accuracy difference." Graphiti generated 1.17M tokens per test case at $152 before abort.

ENGRAM (2025) achieves SOTA on LoCoMo using typed memory (episodic/semantic/procedural) *without* graph infrastructure. Sonality already implements this pattern via `memory_type` metadata tagging in ChromaDB and `retrieve_typed()` routing.

### Current implementation

```
episodes.py: store() tags memory_type ("semantic" if ESS > threshold, "episodic" otherwise)
episodes.py: retrieve_typed() queries semantic first, then episodic
episodes.py: ESS-weighted reranking: similarity × (1 + ess_score)
```

### What would need to change for a graph layer

Adding NetworkX for relational edges between topics would require:
- ~150 lines of graph construction/maintenance
- A new dependency
- Topic co-occurrence extraction logic
- Graph traversal during retrieval

The benefit — "contextual tunneling" resolution (SYNAPSE 2025) — addresses multi-hop reasoning across semantically distant memories. For a personality agent with < 1000 episodes, cosine similarity + ESS reranking is sufficient. Graph becomes valuable at scale (200+ episodes on overlapping topics).

**Upgrade path documented in [Future Opportunities](design-decisions.md#future-opportunities).**

---

## 2. Temporal Decay

**Verdict: KEEP (already implemented)**

### What the research says

FadeMem (Jan 2026): biologically-inspired power-law forgetting achieves 45% storage reduction while improving reasoning. Ebbinghaus (1885): power-law (not exponential) matches human forgetting curves. "Ebbinghaus in LLMs" (2025): neural networks exhibit human-like forgetting patterns.

### Current implementation

```python
# sponge.py: decay_beliefs()
retention = (1 + gap) ** (-decay_rate)        # power-law, β=0.15
floor = min(0.6, meta.evidence_count * 0.06)  # evidence-based reinforcement floor
new_conf = max(floor, meta.confidence * retention)
```

This is exactly the FadeMem-inspired approach. Power-law decay (not exponential), with a reinforcement floor that prevents well-evidenced beliefs from vanishing. Beliefs below `min_confidence=0.05` are dropped entirely.

The decay rate β=0.15 means:
- A belief unreinforced for 10 interactions retains ~70% confidence
- Unreinforced for 30 interactions: ~52%
- Unreinforced for 100 interactions: ~32%
- But a belief with 10 evidence points has floor=0.6, so it persists regardless

This matches the Friedkin-Johnsen stubbornness model: agents balance initial beliefs vs. new information, with diminishing stubbornness leading to consensus but at rates that vary with decay.

**No changes needed.** The implementation already follows the research recommendations.

---

## 3. Opinion Confidence

**Verdict: KEEP (already implemented)**

### What the research says

Deliberative Reasoning Networks (2025): tracking epistemic uncertainty per hypothesis improves adversarial reasoning by 15.2%. Graph-theoretic belief model (2025): distinguish credibility (source) from confidence (structural support).

### Current implementation

```python
# sponge.py: BeliefMeta
class BeliefMeta(BaseModel):
    confidence: float = 0.0          # epistemic status
    evidence_count: int = 1          # structural support
    last_reinforced: int = 0         # temporal tracking
    provenance: str = ""             # source tracking

# confidence grows logarithmically:
confidence = min(1.0, log2(evidence_count + 1) / log2(20))
```

This captures:
- **Epistemic status** (confidence) — maps to Deliberative Reasoning Networks' uncertainty tracking
- **Structural support** (evidence_count) — maps to graph-theoretic model's "structural confidence"
- **Source tracking** (provenance) — maps to credibility distinction
- **Temporal context** (last_reinforced) — enables decay

The Bayesian resistance mechanism in `agent.py:_update_opinions()` scales magnitude by `1 / (confidence + 1)`, making established beliefs harder to shift. Counter-arguments face additional resistance via `conf += abs(old_pos)`.

**The minimal representation that captures epistemic status is already implemented.**

---

## 4. Belief Revision

**Verdict: KEEP**

### What the research says

AGM framework (Alchourrón, Gärdenfors, Makinson): formal postulates for rational belief change — contraction, revision, expansion. Belief-R (2024): LLMs struggle with the trade-off between updating and stability. Hurst (2024): hybrid models combining Bayesian and heuristic approaches capture realistic updating best.

### Current implementation

Sonality approximates AGM-rational revision through a multi-mechanism pipeline:

1. **Expansion** — new topics enter via ESS-gated `stage_opinion_update()`
2. **Revision** — counter-evidence against existing beliefs faces Bayesian resistance (confidence-scaled magnitude reduction + extra resistance for direction reversal)
3. **Contraction** — unreinforced beliefs decay via power-law in `decay_beliefs()`
4. **Cooling period** — staged updates wait 3 interactions before committing, allowing contradictory evidence to cancel out (BASIL 2025: separating sycophantic from rational shifts)

This is exactly Hurst's "hybrid model" — Bayesian updating (confidence scaling) combined with heuristics (ESS gating, cooling period, decay). A formal theorem prover would add hundreds of lines for marginal benefit in a personality agent.

**No changes needed.** The heuristic approximation matches what the research recommends for practical systems.

---

## 5. Memory Tiers

**Verdict: KEEP (already implemented)**

### What the research says

MemoryOS (EMNLP 2025): OS-inspired hierarchical storage improves LoCoMo F1 by 49.11%. ENGRAM (2025): memory typing without graph infrastructure achieves SOTA.

### Current implementation

Sonality implements a three-tier hierarchy:

| Tier | Storage | Contents | Access |
|------|---------|----------|--------|
| Working memory | `self.conversation` (in-process) | Current session turns | Direct |
| Episodic memory | ChromaDB (type="episodic") | Low-ESS interactions | Cosine similarity |
| Semantic memory | ChromaDB (type="semantic") | High-ESS interactions | Cosine similarity + ESS reranking |
| Personality state | JSON (sponge.json) | Snapshot + beliefs + traits | Direct |

The `retrieve_typed()` method routes queries through semantic memory first (high-quality), then episodic (contextual), matching ENGRAM's typed retrieval pattern.

MemoryOS's full short/mid/long-term hierarchy adds value for multi-user agents with thousands of interactions. For a single-user agent, the current two-tier episodic/semantic split plus JSON personality state is sufficient.

**No changes needed.**

---

## 6. Poisoning Defense

**Verdict: KEEP (already implemented)**

### What the research says

MemoryGraft (Dec 2025): poisoned experience retrieval causes persistent behavioral drift; 47.9% poisoned retrievals on benign workloads. AgentPoison (NeurIPS 2024): > 80% attack success with < 0.1% poison rate. Defenses: provenance attestations, safety-aware reranking, memory quality regulation.

### Current implementation

Sonality has four defense layers:

1. **ESS gating** — low-quality inputs score below threshold and don't trigger opinion updates
2. **ESS-weighted reranking** — retrieved memories are ranked by `similarity × (1 + ess_score)`, preferring high-quality memories (episodes.py line 99)
3. **Memory typing** — high-ESS episodes are stored as "semantic" (higher retrieval priority), low-ESS as "episodic"
4. **Anti-sycophancy memory framing** — the system prompt instructs "evaluate on merit, not familiarity" (prompts.py line 31)
5. **Cooling period** — staged updates don't commit immediately, allowing contradictory evidence to cancel
6. **Bayesian resistance** — existing high-confidence beliefs resist change proportionally

PersistBench (2025) shows 97% sycophancy failure when memories contain user preferences. Sonality's anti-sycophancy framing directly addresses this by decoupling memory from endorsement.

The remaining risk: a well-structured poisoned argument (high ESS score with fabricated evidence) will pass all gates. This is an accepted limitation — ESS evaluates argument *structure*, not factual truth.

**No changes needed.** The multi-layer defense already implements the recommended approach.

---

## 7. Disagreement Detection

**Verdict: KEEP (already implemented)**

### What the research says

SYConBench (EMNLP 2025): self-judge bias causes up to 50 percentage point shifts. Third-person prompting reduces sycophancy by 63.8%. Keyword matching is brittle and easily circumvented.

### Current implementation

```python
# agent.py: _detect_disagreement()
def _detect_disagreement(self, ess: ESSResult) -> bool:
    sign = ess.opinion_direction.sign  # from ESS structured output
    if sign == 0.0:
        return False
    for topic in ess.topics:
        pos = self.sponge.opinion_vectors.get(topic, 0.0)
        if abs(pos) > 0.1 and pos * sign < 0:
            return True
    return False
```

This is **structural disagreement detection** — the ESS classifier (third-person framing, separate from response generation) determines opinion direction, and the agent checks whether this direction opposes its existing stance. No keyword matching involved.

This is more reliable than:
- Keyword matching ("I disagree", "that's not") — trivially circumvented
- LLM self-judgment — self-judge bias (SYConBench)
- Sentiment analysis — doesn't distinguish argument direction from emotional tone

**No changes needed.** The structural approach is already the right one.

---

## Summary Table

| Area | Verdict | Justification |
|------|---------|---------------|
| Graph memory layer | KEEP (no graph) | Mem0 vs Graphiti: no accuracy gain; ENGRAM: typed memory without graphs is sufficient |
| Temporal decay | KEEP (implemented) | Power-law decay already follows FadeMem/Ebbinghaus research |
| Opinion confidence | KEEP (implemented) | BeliefMeta already tracks confidence, evidence count, provenance, temporal context |
| Belief revision | KEEP (implemented) | ESS + Bayesian resistance + cooling = Hurst's hybrid model |
| Memory tiers | KEEP (implemented) | Episodic/semantic typing follows ENGRAM pattern |
| Poisoning defense | KEEP (implemented) | ESS gating + reranking + typing + anti-sycophancy framing |
| Disagreement detection | KEEP (implemented) | Structural via ESS direction vs existing stance; not keyword-based |

---

## Behavior Development Pipeline

### How Personality Emerges (APF Three-Layer Theory)

AI Personality Formation (ICLR 2026 submission) identifies three progressive layers. Sonality maps them to interaction phases:

**Layer 1 — Linguistic Mimicry (0–20 interactions)**

The model mirrors `CORE_IDENTITY` and `SEED_SNAPSHOT` style. ESS scores are low for casual chat. Bootstrap dampening (0.5× magnitude) prevents early over-commitment.

Mechanism: `compute_magnitude()` applies `dampening=0.5` when `interaction_count < BOOTSTRAP_DAMPENING_UNTIL`.

**Layer 2 — Structured Accumulation (20–50+ interactions)**

Opinions form through repeated high-ESS exposure. The `belief_meta` tracks evidence count and confidence. Reflection cycles consolidate insights into the snapshot narrative.

Mechanism: `stage_opinion_update()` → cooling period → `apply_due_staged_updates()` → `update_opinion()` with logarithmic confidence growth.

**Layer 3 — Autonomous Expansion (50+ interactions, 10+ beliefs)**

The reflection prompt's maturity instruction changes: "Your worldview is developing coherence. If a pattern suggests a new position, articulate it tentatively."

Mechanism: `_maybe_reflect()` checks `ic >= 50 and nb >= 10` to switch maturity instruction.

### Teaching the Agent

Teaching means presenting structured arguments that pass ESS thresholds. See the [Training Guide](training-guide.md) for detailed curricula, methodologies, and monitoring approaches.

Key mechanisms:
- Arguments with evidence and reasoning score ESS > 0.5 (effective teaching)
- Social pressure and bare assertions score ESS < 0.15 (filtered out)
- Logical fallacies score ESS < 0.25 (filtered with calibration examples)
- Established beliefs resist change via Bayesian confidence scaling
- Unreinforced beliefs decay, creating room for new positions

### Measuring Personality Coherence

Based on PERSIST (2025) and Narrative Continuity Test (2025):

1. **Snapshot Jaccard** — lexical overlap between successive snapshots (logged per reflection)
2. **Disagreement rate** — running mean of structural disagreements (20–35% = healthy)
3. **Belief count trajectory** — should grow during Layer 2, stabilize during Layer 3
4. **High-confidence ratio** — fraction of beliefs with confidence > 0.5 (20–50% = healthy)
5. **Insight yield** — fraction of interactions producing identity insights (0.1–0.5 = healthy)

All five metrics are logged to `data/ess_log.jsonl` in `health` and `reflection` events.

### Real-Time Monitoring

The JSONL audit trail emits six event types:

| Event | Frequency | Key Fields |
|-------|-----------|------------|
| `context` | Every interaction | prompt_chars, relevant_count, snapshot_chars |
| `ess` | Every interaction | score, type, direction, topics, beliefs |
| `opinion_staged` | When ESS > threshold | topic, signed_magnitude, due_interaction |
| `opinion_commit` | When cooling period expires | committed topics, remaining staged |
| `health` | Every interaction | belief_count, disagree_rate, warnings |
| `reflection` | Every 20 interactions | jaccard, insight_yield, dropped beliefs |

Health warnings auto-detect:
- `possible_sycophancy` — disagreement rate < 15% after 20+ interactions
- `snapshot_too_short` — snapshot < 15 words
- `snapshot_bland` — vocabulary diversity < 0.4
- `low_belief_growth` — < 3 beliefs after 40+ interactions

### Reflection Quality

Park et al. (2023) ablation shows reflection is the most critical component. Sonality's reflection uses four structured phases:

1. **EVALUATE** — compare snapshot to accumulated evidence
2. **RECONCILE** — check beliefs for tensions/contradictions
3. **SYNTHESIZE** — find meta-patterns across beliefs and insights
4. **GUARD** — preserve core personality that should not change

The maturity-aware instruction adapts the reflection depth to the agent's development stage, following SAMULE's (EMNLP 2025) insight that reflection should scale with experience.

Belief preservation checking warns when reflection drops high-confidence beliefs, following Constitutional AI Character Training (Nov 2025): losing a trait from the narrative = losing it from behavior.

---

## Rejected Ideas

### Knowledge Graph (Neo4j/NetworkX)

**Why rejected:** Mem0 vs Graphiti (arXiv:2601.07978): no statistically significant accuracy difference; Graphiti costs $152/test case. ENGRAM achieves SOTA without graphs. Sonality's belief tracking is structural (opinion_vectors), not retrieval-driven — the 4x RecallM claim doesn't apply to this architecture.

### Full AGM Belief Revision

**Why rejected:** Requires a theorem prover or constraint solver. The ESS-gated heuristic with Bayesian resistance already approximates AGM's expansion/revision/contraction. Hurst (2024) confirms hybrid models outperform pure formal approaches for practical systems.

### Fine-Tuning for Personality (BIG5-CHAT, PISF, Constitutional AI)

**Why rejected:** Requires model weights access. Sonality is API-only. RAG-based personalization achieves 14.92% improvement vs 1.07% for parameter-efficient fine-tuning (arXiv:2409.09510).

### OCEAN as Personality Driver

**Why rejected:** PERSIST (2025): σ > 0.3 measurement noise even in 400B+ models. Personality Illusion (NeurIPS 2025): self-reported traits don't predict behavior. Sonality uses behavioral metrics (disagreement rate, opinion vectors, topic engagement) instead.

### Activation Steering (PERSONA, Persona Vectors)

**Why rejected:** Requires model internals (hidden states) not exposed by API. Could complement Sonality if using open-weight models, but adds complexity for marginal benefit when the prompt-based approach already works.

### Multi-Agent Reflection (MAR)

**Why rejected:** Adds multiple LLM calls per reflection cycle. The single-agent four-phase reflection already covers self-critique (Phase 1), contradiction detection (Phase 2), synthesis (Phase 3), and identity preservation (Phase 4).

### Probabilistic Memory Gating (FluxMem)

**Why rejected:** FluxMem's distribution-aware fusion requires training a gating network. ESS-weighted reranking is a simpler analog. Worth revisiting if retrieval quality degrades at scale.

### Prospective Reflection (PreFlect)

**Why rejected:** PreFlect shifts from post-hoc correction to pre-execution foresight. Valuable for planning agents, but Sonality is a conversational agent where retrospective consolidation is the appropriate pattern. Adding pre-execution reflection would double LLM calls per interaction.

---

## What Actually Needs Improvement

The architecture is research-aligned. The genuine remaining opportunities are:

1. **Embedding model upgrade** — MiniLM-L6-v2 has a 256-token window; longer summaries are truncated. Migration to a larger-context model would improve retrieval quality. (Documented in [Future Opportunities](design-decisions.md#embedding-model-upgrade))

2. **Separate ESS model** — Using a different model for ESS classification reduces Neural Howlround (arXiv:2504.07992: same model at every stage creates self-reinforcing bias in 67% of conversations). The config already supports `SONALITY_ESS_MODEL` but defaults to the same model.

3. **Martingale entrenchment detection** — NeurIPS 2025 (arXiv:2512.02914): all LLMs exhibit belief entrenchment violating Bayesian rationality. A periodic check during reflection could detect when opinion trajectories become predictable. Low complexity, medium impact.

These are documented as future opportunities, not architectural gaps. The current system works as designed.
