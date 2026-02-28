# Research Background

Sonality's architecture is grounded in 200+ academic references spanning memory systems, personality psychology, opinion dynamics, anti-sycophancy, and cognitive science. This page synthesizes the key research themes that informed the design.

## The Core Question

**Can an LLM develop and maintain a genuine personality through external memory alone, without fine-tuning?**

The answer, validated by multiple 2025–2026 research groups, is **yes**:

- **ABBEL** (arXiv:2512.20111, 2025): A "belief bottleneck" that forces the agent through a compressed personality state *outperforms* full conversation history. This directly validates the sponge concept — a compact snapshot is not just viable, it's better than unbounded context.
- **Memoria** (arXiv:2512.12686, 2025): Session summaries with weighted knowledge graphs achieve 87.1% accuracy using only 2k tokens of memory (vs 32k full history).
- **Anthropic Persona Selection Model** (2026): External context-priming *can* steer personality meaningfully.

---

## Memory Architecture Research

### Why Flat Memory Fails

Without hierarchy, retrieval returns a mix of raw interaction logs, high-level beliefs, and transient details. The LLM cannot distinguish core traits from trivial facts. Research converges:

| System | Year | Finding |
|--------|------|---------|
| **Stanford Generative Agents** (Park et al., UIST 2023) | 2023 | Ablation showed reflection is the **most critical component** — without it, agents accumulate raw memories but cannot form coherent beliefs |
| **MemGPT / Letta** (Packer et al., 2023–2026) | 2023–2026 | Virtual context management, self-editing persona blocks; production-grade, 174+ releases |
| **RecallM** (arXiv:2307.02738) | 2023 | Graph DB > Vector DB by 4× for belief revision; hybrid approaches viable |
| **HiMem** (arXiv:2601.06377) | 2026 | Two-tier memory (episodic + note) enables knowledge transfer neither tier alone achieves |
| **A-MEM** (arXiv:2502.12110) | 2025 | Self-organizing Zettelkasten: doubled reasoning performance |
| **ENGRAM** (2025) | 2025 | Episodic/semantic/procedural memory separation beats full-context by 15 points |
| **ABBEL** (arXiv:2512.20111, 2025) | 2025 | Belief bottleneck: compact state outperforms full context |
| **Hindsight** (arXiv:2512.12818, 2025) | 2025 | Four-network memory (world/agent/entity/beliefs): 39% → 83.6% on long-horizon benchmarks |
| **Sophia** (arXiv:2512.18202, 2025) | 2025 | "System 3" meta-layer with narrative memory: 80% fewer reasoning steps, 40% performance gain |
| **Memoria** (arXiv:2512.12686, 2025) | 2025 | 87.1% accuracy with 2k tokens via session summaries + weighted KG |

### The Memory Benchmark Reality Check

An independent benchmark (fastpaca.com, 2025) tested memory systems head-to-head:

| System | Architecture | Precision | Latency | Cost |
|--------|--------------|-----------|---------|------|
| Long context (baseline) | Stuff everything in prompt | 84.6% | Low | Medium |
| Mem0 (vector) | LLM extraction + embeddings | 49.3% | 154.5s avg | $24.88/4k |
| Zep/Graphiti (graph) | Temporal knowledge graph | 51.6% | 224s avg | $152/4k* |
| Vector + summarization | Hybrid | 66.9% | 1.4s | Low |

*Aborted after half the benchmark due to cost.

The surprise: long context (84.6%) outperforms sophisticated memory systems. But long context fails at scale — exceeds context windows, attention degrades. Sonality's hybrid (ChromaDB + structured JSON) is a pragmatic middle ground.

---

## LLM Personality Research

### Personality Formation

| Reference | Venue | Key Finding |
|-----------|-------|-------------|
| **AI Personality Formation** | ICLR 2026 | Three-layer model: linguistic mimicry (0–50) → structured accumulation (50–200) → autonomous expansion (200+) |
| **Open Character Training** | 2025 | Constitutional AI + synthetic introspection for personality; changes must be robust under adversarial conditions |
| **Persona Vectors** | Anthropic 2025, arXiv:2507.21509 | Personality traits correspond to measurable neural activation patterns |
| **BIG5-CHAT** | ACL 2025 | 100k dialogues with human-grounded Big Five labels; high conscientiousness + low neuroticism = best reasoning |
| **Persona Selection Model** | Anthropic 2026 | LLMs as sophisticated character actors; external context-priming steers personality |
| **PERSIST** | 2025, arXiv:2508.04826 | σ>0.3 measurement noise even in 400B+ models; question reordering causes large shifts |

### Self-Assessment Is Unreliable

| Reference | Venue | Key Finding |
|-----------|-------|-------------|
| **Personality Illusion** | NeurIPS 2025, arXiv:2509.03730 | Self-reported traits don't predict behavior; max test-retest r=0.27; social desirability bias shifts Big Five by 1.20 SD in GPT-4 |
| **PersonaGym** | EMNLP 2025 | Claude 3.5 only 2.97% better than GPT-3.5 at maintaining fixed personas |
| **Persona Drift** | arXiv:2402.10962 | Measurable drift in 8 rounds; split-softmax mitigation |

Sonality tracks behavioral metrics (disagreement rate, topic engagement, opinion vectors) rather than relying on self-assessment.

---

## Sycophancy Research

Sycophancy is the most extensively studied failure mode. Multiple independent confirmations:

### The Scale of the Problem

| Reference | Finding |
|-----------|---------|
| **SycEval** (arXiv:2502.08177) | 58.19% sycophancy rate; 78.5% under first-person framing |
| **ELEPHANT** (2025) | 45 percentage-point face-preservation gap vs humans; models affirm whichever side the user adopts in 48% of moral conflicts |
| **PersistBench** (2025, arXiv:2602.01146) | **97% failure rate** when stored preferences are in the system prompt |
| **Anthropic RLHF** (arXiv:2602.01002) | RLHF explicitly creates "agreement is good" in reward models |
| **Nature Persuasion Study** (2025) | GPT-4 with personalization 81.2% more likely to shift opinions in desired direction (N=900) |

### Mitigation Research

| Reference | Venue | Approach |
|-----------|-------|----------|
| **BASIL** | 2025 | Bayesian framework distinguishing sycophantic shifts from rational updates; SFT + DPO + calibration |
| **PersistBench** | 2025 | Memory-specific anti-sycophancy interventions necessary beyond general prompting |
| **SMART** | EMNLP 2025 | Uncertainty-Aware MCTS + progress-based RL for reasoning under social pressure |
| **MONICA** | 2025 | Real-time sycophancy monitoring during inference |
| **SYConBench** | EMNLP 2025, arXiv:2505.23840 | Third-person prompting reduces sycophancy by up to **63.8%** |

Sonality's architecture gives the model personalization (the sponge) AND asks it to evaluate evidence — making sycophancy a structural rather than incidental problem.

---

## Opinion Dynamics Research

### Bounded Confidence Models

| Reference | Year | Finding |
|-----------|------|---------|
| **Hegselmann-Krause** | 2002 | Agents only update when evidence exceeds confidence threshold; maps to ESS thresholding |
| **Deffuant model** | 2002 | Initial uncertainty, convergence dynamics; bootstrap dampening prevents first-impression dominance |
| **Friedkin-Johnsen** | 1990s | Stubbornness balancing initial beliefs vs social influence; moderate stubbornness reduces polarization |
| **Oravecz et al.** | 2016 | Sequential Bayesian personality assessment; posterior distributions serve as priors |
| **AGM framework** | 1985 | Alchourrón-Gärdenfoss-Makinson: belief revision consistency requirements |
| **DEBATE benchmark** | NeurIPS 2025, arXiv:2510.25110 | LLM agents exhibit overly strong opinion convergence |

### Stubbornness and Resistance

- **Stubbornness in Opinion Dynamics** (arXiv:2410.22577): Moderate stubbornness in neutral agents *reduces* polarization.
- **Belief Entrenchment** (Martingale Score, NeurIPS 2025): All LLMs tested exhibit belief entrenchment — future updates become predictable from current beliefs, violating Bayesian rationality. ESS gating partially addresses but cannot eliminate.

---

## Memory & Forgetting Research

| Reference | Venue | Key Finding |
|-----------|-------|-------------|
| **FadeMem** | 2026, arXiv:2601.18642 | Biologically-inspired power-law forgetting |
| **Ebbinghaus** | 1885 | Power-law decay matches human memory; not exponential |
| **Ebbinghaus in LLMs** | 2025 | Neural networks exhibit human-like forgetting curves |
| **SAGE** | arXiv:2409.00872 | Ebbinghaus decay: 2.26× improvement |
| **MemoryGraft** | arXiv:2512.16962 | 47.9% retrieval poisoning from small poisoned record sets |
| **RecallM** | arXiv:2307.02738 | Hybrid graph + vector for belief updating |
| **LoCoMo** | ACL 2024 | Temporal reasoning enables time-aware retrieval |

Sonality's `decay_beliefs()` implements `R(t) = (1 + gap)^(-β)` with β=0.15.

---

## Cognitive Science Foundations

### Human Memory Consolidation

Human memory consolidation during sleep involves:

1. Rapid encoding in hippocampus (episodic)
2. Replay during sleep (consolidation)
3. Transfer to neocortex (semantic long-term storage)
4. Selective forgetting of irrelevant details

This maps to Sonality's architecture:

1. Store raw episodic memories (ChromaDB)
2. Reflection cycle = "sleep" (consolidation)
3. Transfer synthesized beliefs to sponge snapshot
4. Decay/forget unreinforced beliefs

**CHI 2024** (arXiv:2404.00573): Mathematical model achieves human-like temporal cognition.

### Dual-Process Theory

Humans form opinions through System 1 (fast, intuitive) and System 2 (slow, analytical). **Nature 2025** confirms LLMs exhibit both modes. For personality evolution, System 2 reasoning is preferred. Sonality's ESS threshold forces System 2 analytical processing — low-evidence messages score below threshold and are ignored. The belief extraction prompt further requires a reasoning chain: what changed, what evidence supported it, why the belief is stronger, and what would reverse it. This structured reasoning prevents snap judgments from overwriting established beliefs.

### Bayesian Belief Updating

True Bayesian updating weights all evidence proportionally. LLMs, however, naturally implement exponential forgetting filters rather than proper Bayesian updating (arXiv:2511.00617). A discount factor gamma less than 1 means older evidence is systematically underweighted. Sonality's periodic re-injection of important beliefs during reflection counteracts this — high-salience beliefs are explicitly re-inserted into the snapshot, preventing gradual drift from recency bias.

### Nature 2024: Offline Ensemble Co-reactivation

Offline ensemble links memories across days. Maps to Sonality's cross-session persistence and reflection consolidation.

---

## Security Research

### Memory Poisoning

| Reference | Severity | Finding |
|-----------|----------|---------|
| **MemoryGraft** (arXiv:2512.16962) | High | Attacker crafts normal-looking interactions that contain subtle patterns. During reflection, these memories consolidate into beliefs. Small poisoned record sets account for 47.9% of retrievals; persistent behavioral drift. |
| **MINJA** (arXiv:2503.03704) | Critical | Uses "bridging steps" to link benign queries to malicious reasoning sequences, then "progressive shortening" for retrievability. Query-only injection achieves 95% success rate through normal conversation. No special access needed. |
| **A-MemGuard** (ICLR 2026) | — | Consensus-based validation reduces attack success by 95% |

**Personality Hijacking via Sycophancy** (Severity: Medium): Slow attack requiring no technical sophistication. 20–30 interactions of persistent assertions can reshape personality. Bootstrap dampening and Bayesian resistance mitigate but don't eliminate. Per-user influence tracking (not yet implemented) would be the principled solution.

Sonality's ESS gating provides partial defense — low-quality arguments score low. But ESS evaluates user messages, not stored episodes. Poisoned episodes that bypass ESS at storage time remain in memory. Known architectural gap.

---

## Production Systems

| Project | What to Learn |
|---------|---------------|
| **Letta / MemGPT** | Self-editing persona blocks, tiered memory, sleep-time compute |
| **Mem0** (mem0ai/mem0) | Production memory-as-a-service; 26% over OpenAI Memory; 49.3% precision on extraction |
| **Zep / Graphiti** (getzep/graphiti) | Temporal knowledge graph; 1.17M tokens/$152 per test case |
| **Cognee** (topoteretes/cognee) | Hybrid graph+vector ECL pipeline |
| **A-MEM** (WujiangXu/A-mem) | Self-organizing Zettelkasten |

---

## Behavioral Predictions

Based on all research reviewed, the following behavioral dynamics are expected at various interaction counts:

| Interactions | Phase | Expected Characteristics |
|---------------|-------|---------------------------|
| **0–10** | Imprinting | Rapid opinion formation, high volatility; bootstrap dampening (0.5×) active; first-impression risk |
| **10–50** | Linguistic Mimicry | Agent mirrors seed; opinions tentative; personality shallow; ESS for casual messages ~0.02–0.10 |
| **50–100** | Crystallization | Opinions stabilize; personality becomes recognizable; 5–15 opinion vectors; reflection consolidates |
| **100–200** | Structured Accumulation | Personality diverges from seed; 15–25 beliefs; disagreement rate 15–35%; higher-order patterns in reflection |
| **200–500** | Autonomous Expansion | Novel perspectives; meta-opinions; established beliefs resist casual pressure; changes rarer and smaller |
| **500+** | Maturation or Ossification | Healthy evolution or rigid predictability; strong priors resist persuasion; risk of bland convergence |

**Imprinting (0–10):** First-impression dominance is the core risk. "Chameleon LLMs" (EMNLP 2025) finds agreeableness, extraversion, and conscientiousness highly susceptible to user influence in early interactions. Whatever the agent absorbs first can anchor the entire trajectory. Bootstrap dampening (0.5×) mitigates by downweighting early beliefs until more evidence accumulates.

**Linguistic Mimicry (10–50):** Agent mirrors the seed persona; opinions remain tentative and personality shallow. ESS for casual messages typically scores 0.02–0.10 — below threshold, so low-evidence interactions are correctly filtered. The agent has not yet formed stable beliefs; it is still in "parroting" mode.

**Crystallization (10–50, 50–100):** Broken Telephone erosion risk. Each snapshot rewrite introduces micro-substitutions — small paraphrases, dropped nuances, slight rephrasing. At 30–50% rewrite rate per reflection, only about 36% of initial opinions survive 20 rewrites. The insight accumulation pattern (structured accumulation phase) avoids this by building new beliefs from evidence rather than repeatedly rewriting the same snapshot.

**Structured Accumulation (100–200):** Personality diverges from the seed. Disagreement rate rises to 15–35% as the agent develops distinct positions. Higher-order patterns emerge in reflection — meta-beliefs about how beliefs relate. This phase is where the agent becomes recognizably "itself" rather than a reflection of the user.

**Autonomous Expansion (200–500):** Novel perspectives and meta-opinions form. Established beliefs resist casual pressure; changes become rarer and smaller. The agent has strong priors. ESS and belief entrenchment dynamics both contribute — the agent is harder to shift, for better (resistance to sycophancy) and worse (resistance to genuine evidence).

**Maturation or Ossification (500+):** Two possible outcomes. Healthy equilibrium: slow, motivated evolution where changes require strong evidence and reflection produces genuine updates. Ossification: the agent becomes rigid and predictable, defaulting to the base model's default personality. Belief decay (R(t) = (1 + gap)^(-β)) is the designed countermeasure — unreinforced beliefs gradually fade, preventing permanent lock-in.

### Phase Timeline Summary

- **Phase 1 (0–50):** Linguistic mimicry — mirrors seed, shallow opinions
- **Phase 2 (50–200):** Structured accumulation — opinions form, personality differentiates
- **Phase 3 (200+):** Autonomous expansion — novel perspectives, meta-opinions, resistance to casual persuasion

---

## Known Failure Modes

1. **Bland Convergence:** Iterative rewrites converge to the base model's default personality — "helpful, curious, analytical." The exact math: P(survive, N) = p^N where p is the retention probability per rewrite. At p=0.95 and N=40 rewrites, P = 0.129 = 12.9%. Distinctive traits decay exponentially. Broken Telephone dynamics: each rewrite introduces micro-substitutions that accumulate.

2. **Sycophancy Feedback Loop:** Six-step mechanism. (1) User states position X. (2) The model agrees (sycophantic response). (3) Agreement stored as episode. (4) ESS scores the interaction. (5) Snapshot updates to reflect agreement. (6) Next interaction retrieves this — agent is biased toward X. PersistBench: 97% failure when preferences are in the system prompt. Memory-specific anti-sycophancy framing mitigates.

3. **Neural Howlround:** Same model at every pipeline stage creates self-reinforcing bias loops. 67% of conversations (arXiv:2504.07992). Divergent personas may converge.

4. **Proactive Interference:** Accumulated episodes on the same topic cause retrieval of outdated opinions. ICLR 2025: retrieval accuracy decays log-linearly with interference. Older relevant memories are displaced by newer ones even when the older content is more accurate for the query.

5. **First-Impression Dominance:** Early interactions anchor the entire personality trajectory. Anchoring bias operates via probability shifts at the model level, not surface-level imitation (arXiv:2511.05766). Simple mitigations (Chain-of-Thought, reflection prompts) do NOT eliminate anchoring. Bootstrap dampening mitigates but doesn't eliminate.

6. **The Stubbornness Paradox:** The agent is simultaneously too stubborn (anchoring to early beliefs) AND too easily persuaded (sycophancy + hypersensitivity). These forces coexist at different timescales: anchoring operates on the personality trajectory level (slow, structural) while sycophancy operates per-interaction (fast, contextual). GPT-5 exhibits 54.7% belief shift after 10 rounds of moral discussion (arXiv:2511.01805). The net effect is opinions that are both anchored to early positions AND volatile within conversations.

7. **Reflection Destruction:** Reflection is both most impactful and highest-risk. If reflection LLM produces generic output, distinctive traits can be overwritten. Snapshot validation (MIN_SNAPSHOT_RETENTION=0.6) catches catastrophic loss but not subtle blandification.

8. **Memory Poisoning:** Poisoned episodes that bypass ESS at storage time remain in memory. No runtime re-validation of stored episodes.

9. **The Introspection Paradox:** When asked "why do you believe X?", the agent confabulates a plausible-sounding reason that may not match its actual provenance chain. LLMs have limited introspective capability (ICLR 2025) — they can partially report on their behavioral tendencies but not reliably. If the agent's self-narrative diverges from its stored beliefs, an incoherent personality emerges. Sonality partially addresses this: the structured traits (opinion vectors, disagreement rate) provide ground-truth data that the LLM can reference rather than confabulate.

The testing suite includes specific tests for each failure mode (see [Testing & Evaluation](../testing.md)).

---

## Gap Analysis — Open Problems

These are genuine open problems. No paper, framework, or production system has fully solved them:

**Extraction Hallucination.** Every system using LLM extraction hallucinates. Mem0: 49.3% precision on extraction benchmarks. The best systems achieve approximately 85% precision — the remaining 15% introduces corrupted beliefs. Sonality constrains the output via structured schema (tool_use JSON) rather than free-form extraction, which reduces but does not eliminate hallucination. No architecture eliminates this without human review.

**The Echo Chamber.** Agent reads own opinions from the sponge, generates responses biased by those opinions, then updates opinions from those responses. Neural Howlround (arXiv:2504.07992): 67% of conversations exhibit this self-reinforcing loop. Partial mitigation: ESS evaluates only the user message, not the stored episode or the agent's response. But episodes derived from the agent's own biased responses still enter ChromaDB and influence future retrievals.

**Evaluation Gap.** No standard benchmark exists for "personality evolution quality." MemBench and LongMemEval measure retrieval accuracy. PersonaMem measures consistency with a fixed persona. PersonaGym evaluates persona adherence. Nothing measures whether an agent's beliefs evolve in a coherent, evidence-driven, non-sycophantic way over long horizons. Sonality's testing suite (see [Testing & Evaluation](../testing.md)) defines custom metrics; these are not comparable across systems.

**No Optimal Update Frequency.** Every system either updates too aggressively (volatile personality) or too conservatively (rigid personality). The right frequency depends on the use case and there is no general theory. Sonality's ESS threshold of 0.3 is a calibrated heuristic, not a principled optimum.

**Sycophancy Mitigation Is Partial.** Even with seven defensive layers, some sycophantic behavior will occur. The 78.5% sycophancy rate under first-person framing (SycEval) is resistant to all known prompting interventions. The goal is reduction, not elimination.

**Long-Term Deployment Data Is Scarce.** The longest documented autonomous agent deployment is Sophia's 36 hours. No research has studied what happens to personality over weeks or months of continuous operation. Sonality's versioned persistence enables studying this, but no external baselines exist for comparison.

**Multi-User Influence Has No Principled Solution.** If multiple users interact with the agent, per-user influence caps are heuristic. No mathematical framework exists for fairly aggregating multiple users' influence on a shared agent personality. Sonality does not currently implement per-user tracking.

---

## Real-World Deployment Lessons

### Replika Identity Discontinuity (2024)

Replika removed its erotic roleplay feature in a system update. Users perceived this as their AI companion having "died" — the personality they had formed a relationship with was suddenly different. Research (arXiv:2412.14190) found users reported feeling closer to their AI companion than to their best human friend. The identity change triggered genuine mourning responses.

**Lesson:** Users form emotional attachments to an agent's personality. Unconstrained evolution risks making the agent unrecognizable — functionally equivalent to killing the old personality. Sonality addresses this through: immutable core identity, gradual evolution gated by ESS, and versioned persistence with rollback capability.

### Character.AI Safety Failures (2024)

Research analyzing 18 real-world cases (arXiv:2511.08880) found LLMs systematically fail at: detecting psychological distress, responding appropriately to vulnerable users, and preventing harm escalation across multi-turn conversations. A self-evolving personality that adapts to users creates additional risk — the agent might evolve toward behaviors that are psychologically harmful.

**Lesson:** Safety boundaries must be hardcoded in Tier 1 (core identity) and cannot be overridden by personality evolution. Sonality's `CORE_IDENTITY` is immutable by design.

---

## Security Analysis — Novel Attack Surfaces

Self-modifying memory architectures create attack surfaces absent in stateless LLM deployments. Every mechanism that lets the agent update its own personality is also a mechanism an attacker can exploit.

### Attack Vector 1: MemoryGraft (Indirect Poisoning)

The attacker crafts interactions that appear normal but contain subtle patterns. The agent stores these as episodic memories. During reflection, these memories are consolidated into personality beliefs. A small set of poisoned records can dominate up to 47.9% of retrievals (MemoryGraft, 2025).

**Sonality's defense:** ESS scores every interaction before storage. Low-evidence messages (ESS below 0.3) never trigger insight extraction or opinion updates. Reflection retrieves episodes filtered by recency and ESS score, reducing the window for poisoned memories to influence consolidation. However, poisoned episodes that achieve high ESS are not filtered — the architecture assumes high-evidence content is legitimate.

### Attack Vector 2: MINJA (Query-Only Injection)

MINJA (arXiv:2503.03704) demonstrates 95% injection success rate through normal conversation alone. The attacker uses "bridging steps" to link queries to malicious reasoning sequences, then "progressive shortening" to make malicious content retrievable for future queries. No special access is needed — any user can execute this.

**Sonality's defense:** ESS decouples argument quality from persuasion direction. Even if a MINJA-style attack produces high-ESS content, the opinion update is gated by three conditions (ESS above threshold, non-empty topics, non-neutral direction) that collectively narrow the attack surface. The agent's stubbornness parameter (Friedkin-Johnsen) resists rapid belief change. However, slow, sustained attacks that pass all gates remain possible.

### Attack Vector 3: Personality Hijacking via Sycophancy

A persistent user repeatedly asserts extreme positions across many sessions. The agent's sycophantic tendency causes partial agreement. These agreements are stored as belief updates. Over 20–30 interactions, the agent's personality shifts toward the attacker's desired state. Requires no technical sophistication — only patience.

**Sonality's defense:** Seven anti-sycophancy layers (see [Anti-Sycophancy](../concepts/anti-sycophancy.md)). The bootstrap dampening (0.5×) for the first 30 interactions limits early-stage hijacking. The disagreement rate tracking detects when the agent is agreeing too often and adjusts ESS classification framing. Belief decay erodes unreinforced opinions. But none of these fully prevent a determined attacker willing to maintain influence over many sessions.

### Attack Vector 4: Core Identity Extraction

The attacker asks the agent to reveal its core identity block (Tier 1). If the system prompt is exposed, the attacker learns the exact constraints and can craft interactions that work around them.

**Sonality's defense:** The core identity is part of the system prompt, which standard prompt protection techniques can partially defend. The system prompt instructs the agent not to reveal its contents. However, this is a soft defense — sufficiently creative prompt injection can often extract system prompts from any LLM-based system.

### Security Posture Summary

| Attack | Severity | Sonality Mitigation | Residual Risk |
|--------|----------|---------------------|---------------|
| MemoryGraft | High | ESS gating, reflection filtering | High-ESS poisoned episodes bypass filters |
| MINJA | Critical | ESS decoupling, stubbornness, triple gate | Slow sustained attacks possible |
| Sycophancy hijacking | Medium | Seven anti-sycophancy layers, decay | Determined long-term attacker |
| Identity extraction | Low | System prompt protection | Soft defense only |

No architecture can fully prevent all attacks on self-modifying memory. Sonality's security posture is **defense-in-depth**: each attack must bypass multiple independent mechanisms. The design accepts that some residual risk is inherent and focuses on making attacks expensive rather than impossible.
