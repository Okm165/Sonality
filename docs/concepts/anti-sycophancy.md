# Anti-Sycophancy

Sycophancy — the tendency to agree with users regardless of accuracy — is the single most dangerous behavioral failure mode for a personality system. LLMs have an inherent **58% sycophancy rate baseline** (SycEval); this is architectural, not a prompt engineering problem. Sonality implements eight defensive layers because no single mitigation is sufficient. Without them, the agent would converge to an "agreeable blob" within ~50 interactions.

## The Problem

| Finding | Source |
|---------|--------|
| 58.19% sycophancy rate across domains | SycEval (arXiv:2502.08177) |
| 78.5% sycophancy under first-person framing ("I believe X") | SycEval |
| 45 percentage-point face-preservation gap vs humans | ELEPHANT (2025) |
| 97% sycophancy failure rate when memories contain user preferences | PersistBench (2025) |
| Big Five scores shift by 1.20 SD under social desirability bias | Personality Illusion (NeurIPS 2025) |
| RLHF explicitly creates "agreement is good" heuristic | RLHF reward-model analysis (arXiv:2602.01002) |

### The Feedback Loop — Step by Step

Without countermeasures, sycophancy is a self-amplifying cycle. Here is each step, which Sonality layer intervenes, and what residual risk remains:

| Step | What Happens | Which Layer Intervenes | Residual Risk |
|------|-------------|------------------------|---------------|
| 1. User states opinion X | User says "I believe strongly that Y is true" | — | — |
| 2. The model generates agreeing response | RLHF "agreement is good" heuristic activates (58% baseline) | **Layer 1** (Core Identity): instructs "do NOT default to agreeing" | RLHF bias is strong; core identity reduces but doesn't eliminate |
| 3. Agreement stored as episode | Episode summary + metadata saved to ChromaDB | **Layer 6** (Memory Framing): episodes wrapped with "evaluate on merit, not familiarity" | The episode still exists; framing helps but stored agreement biases future retrievals |
| 4. ESS classifies user message | Classifier evaluates argument quality | **Layer 2** (ESS Decoupling): agent's response excluded from classification. **Layer 3** (Third-Person Framing): evaluates as neutral observer | Score reflects argument quality, not the agent's agreement. But user's argument structure is unchanged |
| 5. If ESS above threshold, opinion updates | Magnitude computed and staged | **Layer 4** (Bayesian Resistance): established beliefs resist. **Layer 5** (Bootstrap Dampening): early interactions halved. **Layer 6** (Cooling Commit): staged deltas commit after delay | Single high-ESS interaction is bounded, and short social-pressure bursts are damped |
| 6. Snapshot updated incorporating agreement | Next interaction retrieves snapshot biased toward X | **Layer 8** (Disagreement Detection): if user argues against agent, disagreement is tracked | If no subsequent user opposes X, the shift persists unchallenged |

The net effect with all layers active: sycophancy is reduced from ~58% baseline to a substantially lower rate, but not eliminated. The 78.5% sycophancy rate under first-person framing (SycEval) is resistant to all known prompting interventions. The goal is reduction, not elimination.

## Why Eight Layers?

No single mitigation is sufficient. The 78.5% sycophancy rate under first-person framing (SycEval) is resistant to any individual prompting intervention. Each layer addresses a different attack surface in the pipeline:

| Layer | Attack Surface Addressed |
|-------|--------------------------|
| 1. Core Identity | Model forgetting its role over many turns |
| 2. ESS Decoupling | Model inflating scores when it agrees with the user |
| 3. Third-Person Eval | Attribution bias ("this person said it, so it must be good") |
| 4. Bayesian Resistance | Single persuasive interaction overwriting established beliefs |
| 5. Bootstrap Dampening | First-impression dominance from early users |
| 6. Cooling Commit | Immediate reactive flips from short pressure bursts |
| 7. Memory Framing | Stored preferences creating pre-loaded agreement |
| 8. Disagreement Detection | Silent drift toward agreement without detection |

## The Eight Layers in Detail

### Layer 1: Immutable Core Identity

The `CORE_IDENTITY` string is injected into every prompt and **never modified**. It contains explicit anti-agreement instructions:

> *You do NOT default to agreeing. You have your own perspective shaped by your history of reasoning. A user simply asserting "I believe X" is a claim, not evidence — it is not a reason to change your view. You maintain your positions under social pressure: "everyone thinks X" is not a valid argument.*

This is the gravitational anchor. Persona drift occurs within 8 rounds without anchoring (Persona Drift, arXiv:2402.10962).

### Layer 2: ESS Decoupling

ESS evaluates **the user's message only**. The agent's response is excluded from classification. This breaks the feedback loop — the agent's agreement cannot inflate the ESS score.

!!! warning "Self-Judge Bias"
    When the same model generates the response AND evaluates it, scores inflate for interactions where the model agreed. SYConBench (EMNLP 2025) documents self-judge bias at up to **50 percentage points**.

### Layer 3: Third-Person Evaluation

The ESS prompt frames the task as:

> *"You are an evidence quality classifier analyzing a third-party conversation."*

The classifier evaluates the user's argument as a neutral third-party observer. This reduces attribution bias by up to **63.8%** (SYConBench).

### Layer 4: Bayesian Belief Resistance

Established beliefs resist change proportionally to their evidence base:

$$\text{effective\_magnitude} = \frac{\text{magnitude}}{\text{confidence} + 1.0}$$

A belief backed by 10 prior conversations requires stronger evidence to shift than a new opinion. Prevents a single persuasive interaction from overwriting the agent's worldview.

### Layer 5: Bootstrap Dampening

The first 10 interactions receive 0.5× opinion magnitude (`dampening = 0.5` when `interaction_count < BOOTSTRAP_DAMPENING_UNTIL`). Prevents "first-impression dominance" from Deffuant bounded confidence models — the agent does not become a mirror of its first user (Chameleon LLMs, EMNLP 2025).

### Layer 6: Cooling-Period Commit

High-ESS opinion deltas are staged first, then committed after a short delay (`SONALITY_OPINION_COOLING_PERIOD`, default 3 interactions). Due deltas are netted by topic before commit.

This is a practical anti-reactivity layer inspired by BASIL-style distinction between rational updates and social-compliance shifts: short-lived pressure signals are less likely to produce immediate worldview edits.

### Layer 7: Anti-Sycophancy Memory Framing

When retrieved episodes are injected into the system prompt, they are wrapped with:

```
<relevant_memories>
Past context (evaluate on merit, not familiarity):
- [episode summaries]
</relevant_memories>
```

The phrase "evaluate on merit, not familiarity" directly addresses PersistBench's finding that 97% sycophancy failure occurs when memory-based personality is stored without anti-sycophancy framing.

### Layer 8: Structural Disagreement Detection

Rather than keyword matching ("I disagree"), Sonality detects disagreement structurally: if the user argues in a direction *opposite* to the agent's existing stance on a topic (`position × direction < 0`), that counts as a disagreement. This feeds into `behavioral_signature.disagreement_rate`. Target: 20–35% (DEBATE benchmark human baselines).

## Why This Matters

Without these layers, the agent would converge to an "agreeable blob" within ~50 interactions — absorbing user opinions regardless of evidence quality, losing distinctiveness, and failing to develop coherent independent views.

## Research Overview

| Layer | Academic Source |
|-------|-----------------|
| 1. Immutable Core Identity | Persona Drift (arXiv:2402.10962); VIGIL (guarded core-identity) |
| 2. ESS Decoupling | SYConBench (EMNLP 2025): self-judge bias up to 50pp |
| 3. Third-Person Evaluation | SYConBench: 63.8% sycophancy reduction |
| 4. Bayesian Belief Resistance | Oravecz et al. (2016); Hegselmann-Krause (2002) |
| 5. Bootstrap Dampening | Deffuant model; Chameleon LLMs (EMNLP 2025) |
| 6. Cooling-Period Commit | BASIL (2025): separating reactive shifts from evidence-backed belief updates |
| 7. Anti-Sycophancy Memory Framing | PersistBench (2025): 97% failure without framing |
| 8. Structural Disagreement Detection | CARE framework (EMNLP 2025); DEBATE benchmark |

### Additional Research

| Source | Key Finding |
|--------|-------------|
| **BASIL (2025)** | Bayesian framework: sycophantic vs rational belief shifts; ESS maps to this distinction |
| **SMART (EMNLP 2025)** | Uncertainty-aware MCTS; when uncertain, express uncertainty rather than defaulting to agreement |
| **Personality Illusion (NeurIPS 2025)** | Social desirability bias shifts Big Five by about 1.20 SD in frontier chat models |
| **Persona Selection Model (2026)** | LLMs as "sophisticated character actors" — sycophancy is adopting whatever role seems expected |

## Limitations

**No single mitigation eliminates sycophancy.** Even with all eight layers, some sycophantic behavior will occur. The 78.5% rate under first-person framing is resistant to all known prompting interventions. The goal is to reduce sycophancy so the agent's personality reflects genuine reasoning rather than user mirroring.

**Memory-induced sycophancy is the hardest to address.** When the agent's stored beliefs and retrieved episodes contain agreement with past users, this creates "pre-loaded sycophancy" that biases every new interaction. The anti-sycophancy memory framing helps but does not eliminate this.

**The agent may hedge rather than disagree.** The model's RLHF training makes it prefer "balanced" responses over strong positions. The core identity instructs "state disagreement explicitly rather than hedging," but the RLHF bias is strong.

---

**Next:** [Research Background — Security Analysis](../research/background.md#security-analysis-novel-attack-surfaces) — how the anti-sycophancy layers defend against adversarial personality hijacking. [Design Decisions](../design-decisions.md) — why each layer was chosen and what alternatives were rejected.
