# Anti-Sycophancy

LLMs have ~58% sycophancy baseline (SycEval). This is architectural, not fixable by prompting alone.

## Implemented Defenses

| Layer | Mechanism | Research |
|-------|-----------|----------|
| **Core Identity** | Immutable `CORE_IDENTITY` with explicit anti-agreement instructions | Persona Drift: drift in 8 rounds without anchoring |
| **ESS Decoupling** | ESS evaluates user message only (agent response excluded) | SYConBench: self-judge bias up to 50pp |
| **Third-Person Framing** | ESS evaluates as neutral observer | SYConBench: 63.8% sycophancy reduction |
| **Memory Framing** | "evaluate on merit, not familiarity" prefix | PersistBench: 97% failure without framing |
| **Disagreement Instructions** | "state disagreement explicitly rather than hedging" | — |

## Key Research Findings

| Finding | Source |
|---------|--------|
| 58% baseline sycophancy | SycEval |
| 78.5% under "I believe X" framing | SycEval |
| 97% failure with stored preferences | PersistBench |

## Not Yet Implemented

- **Bayesian Resistance** — beliefs resist change proportional to evidence count
- **Bootstrap Dampening** — early interactions weighted less
- **Cooling-Period Commit** — delay high-impact changes
- **Disagreement Tracking** — target 20-35% disagreement rate

## Limitations

The 78.5% rate under first-person framing resists all prompting interventions. Goal is reduction, not elimination.
