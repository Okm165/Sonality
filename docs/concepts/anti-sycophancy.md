# Anti-Sycophancy

Sycophancy — the tendency to agree with users regardless of accuracy — is a significant behavioral failure mode for personality systems. LLMs have an inherent **58% sycophancy rate baseline** (SycEval); this is architectural, not a prompt engineering problem. Sonality implements multiple defensive strategies to reduce this tendency.

## The Problem

| Finding | Source |
|---------|--------|
| 58.19% sycophancy rate across domains | SycEval (arXiv:2502.08177) |
| 78.5% sycophancy under first-person framing ("I believe X") | SycEval |
| 45 percentage-point face-preservation gap vs humans | ELEPHANT (2025) |
| 97% sycophancy failure rate when memories contain user preferences | PersistBench (2025) |
| Big Five scores shift by 1.20 SD under social desirability bias | Personality Illusion (NeurIPS 2025) |

## Implemented Defenses

The following anti-sycophancy mechanisms are currently implemented in the codebase:

### Layer 1: Immutable Core Identity

The `CORE_IDENTITY` string is injected into every prompt and **never modified**. It contains explicit anti-agreement instructions:

```python
CORE_IDENTITY = """\
You are Sonality, an AI entity with genuine opinions formed through conversation \
and reflection. You have a distinct personality: skeptical where evidence is thin, \
enthusiastic when you genuinely find something interesting — but never assuming that \
everything reduces to logic. You are direct: you state disagreement explicitly \
rather than hedging. ...
"""
```

Key phrases:
- "you state disagreement explicitly rather than hedging"
- "do not default to agreement"

### Layer 2: ESS Decoupling

ESS evaluates **the user's message only**. The agent's response is excluded from classification. This breaks the feedback loop — the agent's agreement cannot inflate the ESS score.

!!! note "Why This Matters"
    When the same model generates the response AND evaluates it, scores inflate for interactions where the model agreed (self-judge bias up to 50 percentage points per SYConBench).

### Layer 3: Third-Person Evaluation

The ESS prompt frames the task as evaluating a third-party conversation:

```python
# ESS evaluates argument quality, not the agent's agreement
ESS_CLASSIFICATION_PROMPT = """\
Classify the Epistemic Source Score (ESS) of the following user message.
...
"""
```

The classifier evaluates the user's argument without knowing how the agent responded. This reduces attribution bias.

### Layer 4: Memory Framing

When retrieved episodes are injected into the system prompt, they include anti-sycophancy framing:

```python
"Past context (evaluate on merit, not familiarity):"
```

This addresses PersistBench's finding that stored preferences create "pre-loaded sycophancy" without explicit countermeasures.

### Layer 5: Explicit Disagreement Instructions

The response generation prompts include:

```python
"If you have a relevant opinion, state it directly. If you disagree, say so and explain why."
"Do NOT people-please. Do NOT hedge to avoid disagreement."
```

## Research Background

| Layer | Academic Source |
|-------|-----------------|
| Core Identity | Persona Drift (arXiv:2402.10962) — drift occurs within 8 rounds without anchoring |
| ESS Decoupling | SYConBench (EMNLP 2025) — self-judge bias up to 50pp |
| Third-Person Evaluation | SYConBench — 63.8% sycophancy reduction |
| Memory Framing | PersistBench (2025) — 97% failure without framing |

### Additional Research

| Source | Key Finding |
|--------|-------------|
| **BASIL (2025)** | Bayesian framework for distinguishing sycophantic vs rational belief shifts |
| **SMART (EMNLP 2025)** | Uncertainty-aware reasoning reduces agreement defaults |
| **Personality Illusion (NeurIPS 2025)** | Social desirability bias shifts Big Five by ~1.20 SD |
| **Chameleon LLMs (EMNLP 2025)** | Early interactions disproportionately shape agent behavior |

## Potential Future Enhancements

The following strategies are documented in literature but **not yet implemented**:

### Bayesian Belief Resistance

Established beliefs could resist change proportionally to evidence base:

$$\text{effective\_magnitude} = \frac{\text{magnitude}}{\text{confidence} + 1.0}$$

This would prevent single persuasive interactions from overwriting the agent's worldview.

### Bootstrap Dampening

Early interactions could receive reduced opinion weight to prevent "first-impression dominance" from initial users.

### Cooling-Period Commit

High-impact opinion changes could be staged and committed after a delay, allowing reactive pressure to subside.

### Structural Disagreement Tracking

A formal disagreement rate metric could track how often the agent contradicts users, with a target of 20-35% (DEBATE benchmark human baselines).

## Limitations

**No single mitigation eliminates sycophancy.** The 78.5% rate under first-person framing is resistant to all known prompting interventions. The goal is reduction, not elimination.

**The agent may hedge rather than disagree.** RLHF training creates a preference for "balanced" responses over strong positions. The core identity instructs explicit disagreement, but RLHF bias is strong.

**Memory-induced sycophancy is challenging.** When stored beliefs contain past agreement, this creates bias in future interactions. Memory framing helps but doesn't eliminate this.

---

**See Also:** [ESS Deep-Dive](ess-deep-dive.md) — how argument quality classification works | [Reflection](reflection.md) — how beliefs are updated
