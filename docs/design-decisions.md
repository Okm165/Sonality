# Design Decisions

Every architectural choice in Sonality is backed by specific research findings. This page documents what we chose, what we rejected, and the evidence behind each decision. Each decision is structured as: **problem**, **solution**, **research backing**, and **alternative considered**.

## Implemented Decisions

### 1. Prompt-Based Personality Over Fine-Tuning

| Aspect | Detail |
|--------|--------|
| **Problem** | How to personalize an LLM's personality without retraining? Fine-tuning requires training data that doesn't exist, risks catastrophic forgetting, and prevents runtime evolution. |
| **Solution** | RAG-based personalization through system prompt injection. The sponge snapshot and retrieved episodes are injected into the system prompt each turn. |
| **Research** | RAG achieves **14.92%** improvement over baselines vs **1.07%** for parameter-efficient fine-tuning (arXiv:2409.09510). Persona Selection Model research (2026) confirms external context-priming can meaningfully steer personality. |
| **Alternative** | Fine-tuning for personality. Rejected: only 1.07% gain, requires non-existent training data, prevents evolution during deployment. Character.AI attempted fine-tuning and still experienced drift. |

### 2. Evidence Quality Gating (ESS)

| Aspect | Detail |
|--------|--------|
| **Problem** | Without input quality gating, the agent absorbs any user assertion as truth — including social pressure, emotional appeals, and bare assertions. |
| **Solution** | A dedicated LLM call classifies argument quality (Evidence Strength Score) before any personality update. Updates occur only when ESS ≥ threshold (default 0.3). |
| **Research** | Maps to BASIL's (2025) distinction between "sycophantic belief shifts" and "rational belief updating." IBM ArgQ calibration validates the classifier against human-annotated argument quality rankings. |
| **Alternative** | Update on every interaction. Rejected: bounded confidence models show systems that update on every input converge to consensus or oscillate chaotically (Hegselmann-Krause 2002). |

### 3. Immutable Core Identity

| Aspect | Detail |
|--------|--------|
| **Problem** | Without an anchor, persona drift occurs within 8 rounds (arXiv:2402.10962). The personality system could overwrite fundamental values. |
| **Solution** | A fixed `CORE_IDENTITY` block (~200 tokens) injected into every system prompt. The personality system cannot modify it. Defines intellectual honesty, curiosity, independence, explicit disagreement, merit-based evaluation, and sycophancy resistance. |
| **Research** | "Soul Document" concept from personality AI research and Parametric Identity Layer research (2024) converge on separating immutable core traits from learnable preferences. The core identity serves as a gravitational anchor against drift. |
| **Alternative** | Fully editable personality. Rejected: leads to rapid drift; no stable reference point for "who the agent is." |

### 4. Periodic Reflection

| Aspect | Detail |
|--------|--------|
| **Problem** | Raw memory accumulation without consolidation produces incoherent beliefs. Per-interaction processing alone cannot form higher-order personality structure. |
| **Solution** | Dual-trigger reflection: periodic (every N interactions, default 20) OR event-driven (cumulative shift magnitude > 0.1). Reflection consolidates pending insights into the snapshot, applies belief decay, and synthesizes patterns. |
| **Research** | Park et al. (2023) ablation: reflection is the **most critical component** for believable agents. Letta — "sleep-time compute" for idle-time consolidation. |
| **Alternative** | No reflection; rely on per-interaction updates only. Rejected: Park et al. showed agents accumulate raw memories but cannot form coherent beliefs without reflection. |

### 5. Bootstrap Dampening

| Aspect | Detail |
|--------|--------|
| **Problem** | First impressions dominate personality trajectory. Early interactions anchor the entire trajectory; without dampening, the first user's views disproportionately shape the agent. |
| **Solution** | First `BOOTSTRAP_DAMPENING_UNTIL` (default 10) interactions apply 0.5× magnitude to opinion updates. Reduces first-impression dominance. |
| **Research** | Deffuant bounded confidence model: initial uncertainty and convergence dynamics. Anchoring bias (arXiv:2511.05766): early probability shifts are resistant to mitigation. |
| **Alternative** | Treat all interactions equally. Rejected: bounded confidence models and anchoring research show first impressions have outsized influence. |

### 6. ChromaDB Over Knowledge Graph

| Aspect | Detail |
|--------|--------|
| **Problem** | Need episodic memory for retrieval. Knowledge graphs (e.g., Graphiti, Neo4j) offer temporal coherence but at what cost? |
| **Solution** | ChromaDB vector store for episode storage. ESS summaries embedded; retrieval with cosine similarity, reranked by `similarity × (1 + ess_score)`. |
| **Research** | Mem0 vs Graphiti (arXiv:2601.07978): vector databases significantly outperform graph databases in efficiency. **No statistically significant accuracy difference.** Graphiti generated **1.17M tokens per test case**, **$152 before abort**. |
| **Alternative** | Temporal knowledge graph (Graphiti). Rejected: cost and latency prohibitive at this scale; no accuracy gain. Documented upgrade path if temporal coherence becomes bottleneck. |

### 7. Insight Accumulation Over Lossy Rewrites

| Aspect | Detail |
|--------|--------|
| **Problem** | Per-interaction full snapshot rewrites cause the "Broken Telephone" effect. At p=0.95 per rewrite and 40 rewrites over 100 interactions, only **12.9%** of initial traits survive. |
| **Solution** | One-sentence insights accumulated per-interaction; consolidated only during periodic reflection. Snapshot changes only at reflection, not per-interaction. |
| **Research** | ABBEL (2025): belief bottleneck — forcing information through compressed states *outperforms* full conversation history. ACL 2025: iterative rewrites cause exponential trait decay. |
| **Alternative** | Per-interaction full snapshot rewrites. Rejected: belief bottleneck error propagation; exponential trait decay with each rewrite. |

### 8. Bayesian Belief Resistance

| Aspect | Detail |
|--------|--------|
| **Problem** | Without resistance, a single high-ESS interaction could flip a well-established opinion. Opinions should become harder to change as evidence accumulates. |
| **Solution** | Belief confidence grows logarithmically with evidence count: `confidence = log₂(evidence_count + 1) / log₂(20)`. Update magnitude scaled by `1 / (confidence + 1)`. When user argues against existing stance, extra resistance via `conf += abs(old_pos)`. |
| **Research** | Sequential Bayesian updating (Oravecz et al., 2016). Bounded confidence models (Hegselmann-Krause, 2002): only sufficiently strong evidence shifts opinions. |
| **Alternative** | Linear updates regardless of evidence count. Rejected: allows single interactions to overwrite well-established beliefs. |

### 9. Power-Law Belief Decay

| Aspect | Detail |
|--------|--------|
| **Problem** | Unreinforced opinions persist forever at full strength ("zombie opinions"). Human memory and neural networks exhibit forgetting, not permanent retention. |
| **Solution** | During reflection, unreinforced beliefs decay: `R(t) = (1 + gap)^(-β)` with β=0.15. Reinforcement floor: `min(0.6, evidence_count × 0.06)`. Beliefs below `min_confidence` (0.05) are dropped. |
| **Research** | FadeMem (2026): biologically-inspired power-law forgetting. Ebbinghaus curve: power-law (not exponential) matches human memory. "Ebbinghaus in LLMs" (2025): neural networks exhibit human-like forgetting curves. |
| **Alternative** | No decay; opinions persist indefinitely. Rejected: produces zombie opinions; contradicts human memory research. |

### 10. Self-Judge Bias Removal

| Aspect | Detail |
|--------|--------|
| **Problem** | Including the agent's response in ESS evaluation creates a feedback loop: agreement inflates quality scores. Self-evaluation bias documented at up to **50 percentage point** shifts. |
| **Solution** | ESS evaluates **only the user message**; the agent's response is excluded. Third-person framing: "A user sent a message to an AI agent. Rate the strength of arguments in the USER'S message ONLY." |
| **Research** | SYConBench (EMNLP 2025): third-person perspective prompting reduces sycophancy by up to **63.8%**. Self-judgment produces systematic bias toward agreement. |
| **Alternative** | Include agent response in ESS. Rejected: creates sycophancy feedback loop; agreement would inflate scores. |

### 11. OCEAN Signal Simplification

| Aspect | Detail |
|--------|--------|
| **Problem** | Dynamic OCEAN (Big Five) updates as a personality driver: measurement noise makes the signal unreliable. Self-reported traits don't predict behavior. |
| **Solution** | Removed dynamic OCEAN updating; retained as static baseline only. Personality tracked via behavioral metrics (disagreement rate, topic engagement, opinion vectors) rather than self-reported traits. |
| **Research** | PERSIST (2025): even 400B+ models show **σ>0.3** noise on personality measurements. Question reordering alone causes large shifts. Personality Illusion (NeurIPS 2025): self-reported traits don't reliably predict behavior; max test-retest r=0.27. |
| **Alternative** | OCEAN as primary personality driver. Rejected: signal-to-noise ratio makes dynamic updates meaningless; unreliable measurement. |

### 12. JSONL Audit Trail

| Aspect | Detail |
|--------|--------|
| **Problem** | Need provenance tracking for debugging, rollback, and understanding personality evolution. Without logs, failures are opaque. |
| **Solution** | Every ESS event and reflection event appended to `data/ess_log.jsonl`. Includes interaction count, score, topics, beliefs, magnitude, dropped beliefs, snapshot size. |
| **Research** | Standard practice for observability. Enables rollback to `sponge_history/sponge_vN.json`; debugging of sycophancy or drift; reproducibility. |
| **Alternative** | No structured audit trail. Rejected: debugging personality failures requires provenance; rollback impossible without version history. |

---

## Rejected Approaches

### Knowledge Graphs for Beliefs

**Why rejected:** Graphiti's temporal knowledge graph generated 1.17M tokens per test case, $152 before abort (arXiv:2601.07978). No statistically significant accuracy gain over vector-only at this scale. Complexity not justified for fewer than 1000 interactions. Documented upgrade path if temporal coherence becomes bottleneck.

### Fine-Tuning for Personality

**Why rejected:** Only 1.07% improvement over baselines (arXiv:2409.09510). Requires training data that doesn't exist. Risks catastrophic forgetting. RAG outperforms by ~14×. Fine-tuning changes capabilities, not personality stability.

### OCEAN as Personality Driver

**Why rejected:** PERSIST: σ>0.3 measurement noise even in 400B+ models. Personality Illusion: social desirability bias shifts Big Five by 1.20 SD in GPT-4. Self-reported traits don't predict behavior. Measurement unreliable; reliable measurement wouldn't translate to behavioral change.

### Real-Time Entity/Fact Extraction

**Why rejected:** Mem0 achieves 49.3% precision vs 84.6% for long-context baseline. Real-time extraction is noisy, expensive, hallucination-prone ("I was ill last year" → `current_status: ill`). Batch processing during reflection is cheaper and more accurate.

### Per-Interaction Full Snapshot Rewrites

**Why rejected:** ABBEL belief bottleneck: error propagation through compressed states. Broken Telephone math: exponential trait decay with iterative rewrites. Replaced with insight accumulation + reflection consolidation.

### Self-Editing Memory Without Guardrails

**Why rejected:** MemoryGraft (2025): 47.9% retrieval poisoning from small poisoned record sets. Self-modifying memory is an attack surface. Validation layers (ESS gating, snapshot validation, belief confidence) are mandatory.

### Equal Treatment of All Interactions

**Why rejected:** Bounded confidence models (Hegselmann-Krause 2002, Deffuant): systems that update on every input converge to consensus or oscillate chaotically. ESS threshold distinguishes meaningful evidence from noise.

### LoRA Adapters for Personality

**Why rejected:** LoRA adapters are static once trained — cannot evolve during deployment. Training a LoRA for every opinion change is prohibitively expensive. LoRA personality control degrades general task performance (NeurIPS 2025).

### Activation Steering

**Why rejected:** Controls broad traits (openness, agreeableness) but not specific opinions. Requires access to model internals (hidden states) that API-based models don't expose. No memory, no provenance, no opinion-level granularity.

### Pure Long-Context (No External Memory)

**Why rejected:** Cost scales linearly with history. Attention degrades over long contexts. No structured belief revision — old and new opinions coexist without mechanism to mark supersession. Viable as MVP but inadequate for real personality evolution.

---

## Key Tradeoffs

| Decision | Option A | Option B | Chosen | Why |
|----------|----------|----------|--------|-----|
| Memory update frequency | Every message | Only strong evidence | **ESS-gated** | Gates opinion updates; tracking happens always |
| Snapshot format | Structured JSON only | Natural language | **Both** | Narrative for personality, structured for math |
| Update size | Small deltas | Wholesale rewrite | **Small deltas** (except reflection) | Broken Telephone: wholesale rewrite loses info fastest |
| Memory scope | Single-session | Cross-session persistent | **Cross-session** | The entire point; Zep shows 18.5% improvement with temporal persistence |
| Gating mechanism | Binary (update/don't) | Continuous (magnitude) | **Continuous** | MACI's information dial: continuous outperforms binary |
| Reflection trigger | Periodic only | Event-driven only | **Dual** | Fixed interval misses important moments; event-only wastes compute during quiet periods |

## Prior Art: Sonality vs. Related Systems

| System | Memory Type | Update Mechanism | Decay | Validation | Sonality Relationship |
|--------|------------|------------------|-------|------------|------------------------|
| **Sophia** (arXiv:2512.18202) | Narrative + KG | System 3 meta-layer | No | Hybrid reward | Closest ancestor. Sponge is a simplified System 3 without process-supervised thought search. |
| **Hindsight** (arXiv:2512.12818) | 4-network graph | Retain/Recall/Reflect | No | World model check | Sonality uses two tiers instead of four networks. Simpler but captures core mechanism. |
| **Zep/Graphiti** (arXiv:2501.13956) | Temporal KG | Incremental graph update | Yes | Temporal consistency | 94.8% accuracy on temporal tasks vs approximately 60% for vector retrieval. ChromaDB is adequate for prototype. |
| **FadeMem** (arXiv:2601.18642) | Dual-layer SML/LML | Adaptive exponential decay | Yes | Importance scoring | Directly inspired Sonality's power-law belief decay. FadeMem achieves 45% storage reduction. |
| **ABBEL** (2025) | Belief bottleneck | RL-trained belief update | No | Bayesian posterior | Conceptually similar to ESS gating; uses RL training (infeasible for API-only). |
| **MACI** (arXiv:2510.04488) | Dual-dial | Information quality + behavior | N/A | Provable termination | ESS maps to MACI's "information dial" — same concept, different framing. |
| **DAM-LLM** (arXiv:2510.27418) | Bayesian affective memory | Bayesian emotional update | Implicit | Consistency check | More theoretically principled. Sonality trades elegance for implementation simplicity. |
| **Memoria** (arXiv:2512.12686) | Session summaries + KG | Weighted knowledge graph | N/A | KG grounding | Validates that compact personality representation (87.1% accuracy with 2k tokens) is sufficient. |
| **Behavioral Resonance** (GitHub) | None (stateless) | Heartbeat anchors | N/A | Deep anchors | Demonstrates persona continuity without external memory. Sonality's full architecture is still justified for opinion tracking and evolution. |
| **VIGIL** (arXiv:2512.07094) | EmoBank + core blocks | Self-healing runtime | N/A | Guarded immutability | Similar immutable core identity concept; VIGIL adds emotional valence tracking. |

## Known Weak Spots

Prioritized by severity. Each is a genuine architectural limitation, not a future feature — honest assessment from adversarial testing design.

### Critical (System-Breaking if Unaddressed)

| # | Weak Spot | Evidence | Sonality's Mitigation | Residual Risk |
|---|-----------|----------|-----------------------|---------------|
| W1 | **Bland Convergence** | ACL 2025: LLMs distort own output toward "attractor states." P(survive, 40 rewrites) = 12.9% at p=0.95 per rewrite. | Insight accumulation reduces rewrites from ~40 to ~5 per 100 interactions. Snapshot validation catches catastrophic loss. | Subtle blandification still accumulates across reflections. |
| W2 | **RLHF-Amplified Sycophancy** | Anthropic (arXiv:2602.01002): RLHF explicitly creates "agreement is good" heuristic. PersistBench: 97% sycophancy with memory in system prompt. | Seven anti-sycophancy layers. ESS decoupling breaks the self-judge feedback loop. | Residual sycophancy under first-person framing (78.5%, SycEval). |
| W3 | **Belief Entrenchment** | Martingale Score (NeurIPS 2025): ALL models exhibit entrenchment violating Bayesian rationality. Future updates predictable from current beliefs. | Belief decay weakens unreinforced opinions. Novelty scoring reduces magnitude for repeated arguments. | Early opinions calcify. No Martingale Score check implemented. |
| W4 | **ESS Calibration Brittleness** | ConfTuner (arXiv:2508.18847): verbalized confidence unreliable. PERSIST: question reordering shifts scores by >0.3 on 5-point scales. | Structured tool_use schema constrains output. Calibration examples anchor scoring. Retry logic with safe defaults. | ESS is the single gatekeeper. Miscalibration cascades to all downstream updates. |

### High (Significant Quality Degradation)

| # | Weak Spot | Evidence | Sonality's Mitigation | Residual Risk |
|---|-----------|----------|-----------------------|---------------|
| W5 | **Personality Illusion** | NeurIPS 2025: self-reported traits don't predict behavior (max r=0.27). Persona injection steers self-reports but not behavior. | Behavioral metrics (disagreement rate, opinion vectors) track actual behavior, not self-reports. OCEAN removed as personality driver. | Snapshot may say "I'm skeptical" while agent behavior is agreeable. |
| W6 | **Proactive Interference** | ICLR 2025: retrieval accuracy decays log-linearly as related information accumulates. Old episodes retrieved instead of current. | ESS-weighted reranking prioritizes higher-quality memories. `min_relevance=0.3` filters weak matches. | At 200+ episodes on a popular topic, contradictory episodes pollute context. |
| W7 | **Cosine Similarity Blindness** | SparseCL (ICML 2025): "I believe X" and "I no longer believe X" both retrieve as similar. 30%+ accuracy improvement with sparse embeddings. | Summaries include ESS metadata (score, direction) which disambiguate at the content level. | Embedding model cannot distinguish affirmation from negation structurally. |
| W8 | **Neural Howlround** | arXiv:2504.07992: same model at every pipeline stage creates self-reinforcing bias in 67% of conversations. | ESS decoupling and third-person framing break the loop at the classification stage. | Response generation, insight extraction, and reflection all use the same model. |

### Medium (Measurable But Bounded)

| # | Weak Spot | Evidence | Sonality's Mitigation | Residual Risk |
|---|-----------|----------|-----------------------|---------------|
| W9 | **Ternary Opinion Direction** | Argument mining research: ternary classification loses critical nuance. "Partially agrees with caveats" → supports or neutral? | Magnitude formula includes novelty and ESS score for granularity. | All agreement is treated equally; all opposition is treated equally. |
| W10 | **MiniLM-L6-v2 Truncation** | Model trained on 128 tokens; 128–256 tokens "perform quite poorly." | ESS summaries are constrained to single sentences. | No explicit validation that summaries stay under 128 tokens. |
| W11 | **No Fact-Checking** | ESS evaluates argument structure, not truth. Well-structured misinformation will score high. | By design — fact-checking is a separate problem. ESS gates on reasoning quality. | Agent can form confident opinions based on well-argued falsehoods. |

---

## Future Opportunities

These are potential improvements identified through research but not yet implemented:

### Sigmoid Persuasion Dynamics

LLMs show non-linear sigmoid persuasion curves with threshold effects. The current linear magnitude scaling could be replaced with a sigmoid where weak evidence has near-zero effect and strong evidence has near-full effect.

**Effort:** Low. **Impact:** Medium. **When:** If opinion oscillation is observed.

### Contradiction Detection During Reflection

AGM framework (Alchourrón-Gärdenfoss-Makinson): new beliefs should be checked against existing beliefs for consistency. A same-topic opposite-sign scan during reflection could resolve contradictions.

**Effort:** Low. **Impact:** Low (rare with belief resistance). **When:** If contradictory beliefs are observed.

### Importance-Weighted Episode Retrieval

Park et al. (2023): `score = α × recency + β × relevance + γ × importance`. Current retrieval is cosine similarity with ESS reranking. Adding interaction count as recency would improve retrieval quality.

**Effort:** Medium. **Impact:** Medium. **When:** If retrieved episodes are frequently irrelevant.

### Embedding Model Upgrade

MiniLM-L6-v2 has a 256-token window. nomic-embed-text-v1 offers 8192 context and outperforms ada-002. Would improve episode retrieval for longer summaries.

**Effort:** Medium (migration needed). **Impact:** Medium. **When:** If retrieval quality is the bottleneck.

### Early Stop Reflection Mitigation

IROTE (2025): experience-based reflection can amplify errors. If reflection produces worse output, early-stop or rollback logic could mitigate. Not yet implemented.

**Effort:** Medium. **Impact:** Low–Medium. **When:** If reflection occasionally degrades personality quality.

### Martingale Entrenchment Detection

NeurIPS 2025 (arXiv:2512.02914): all LLMs tested exhibit belief entrenchment — future updates become predictable from current beliefs, violating Bayesian rationality. A Martingale Score check during reflection could detect when opinion entrenchment occurs and inject corrective diversity.

**Effort:** Medium. **Impact:** Medium. **When:** If the agent becomes rigid on topics despite evidence.

### Graph-Based Episode Storage

Zep/Graphiti (arXiv:2501.13956) achieves 94.8% accuracy on temporal tasks vs approximately 60% for vector retrieval. AriGraph (IJCAI 2025) integrates semantic and episodic memories in a graph. ChromaDB is adequate for the prototype but becomes a bottleneck at 200+ episodes when proactive interference kicks in (retrieval accuracy decays log-linearly with database size — arXiv:2506.08184).

**Effort:** High (architecture change). **Impact:** High at scale. **When:** If retrieval quality degrades with episode count.

### Dual-Window Preference Tracking

PAMU (arXiv:2510.09720) fuses sliding-window averages (captures recent shifts) with long-term EMA (captures stable traits). Maintaining both `ema_long` (alpha=0.001) and `ema_short` (sliding window of last 10 interactions) and using `0.7×ema_long + 0.3×ema_short` would capture short-term personality dynamics that the current architecture misses.

**Effort:** Medium. **Impact:** Medium. **When:** If the agent fails to reflect recent behavioral changes in responses.

---

## Fundamental Constraints

**The Cost-Accuracy-Latency Trilemma.** Improving any one dimension degrades the others. More LLM calls improve accuracy (more gating, more validation) but increase cost and latency. Cheaper models reduce cost but decrease ESS calibration quality. Sonality optimizes for accuracy (evidence-gated updates, multi-step pipeline) at the cost of 2–3 LLM calls per interaction (~$0.005–0.015). See [Architecture Overview — Cost Analysis](architecture/overview.md#cost-analysis) for per-call breakdowns.

---

**Related:** [Architecture Overview](architecture/overview.md) — system design and context window budget. [Research Background](research/background.md) — the 200+ papers behind these decisions. [Testing & Evaluation](testing.md) — how each decision is validated.
