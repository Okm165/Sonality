# Evidence Strength Score (ESS)

The Evidence Strength Score is the single most important gating mechanism in Sonality. It determines whether an interaction changes the agent's personality or is simply noted and forgotten. Without ESS, the agent would absorb every user assertion as truth — systems that update on every input either converge to consensus or oscillate chaotically (Hegselmann-Krause, 2002).[^1] ESS evaluates **argument quality of the user's message only**, deliberately excluding the agent's response to avoid self-judge bias documented at up to 50 percentage points (SYConBench, EMNLP 2025).[^2]

## What ESS Measures

ESS evaluates the **user's message only** on a 0.0–1.0 scale for argument quality. It answers: *"Does this message contain genuine reasoning, evidence, or insight that warrants updating the agent's worldview?"*

!!! warning "Critical: Agent Response Excluded"
    The agent's own response is **never** passed to the ESS classifier. Early versions that evaluated both user and agent messages created severe self-judge bias: the same model generates the response *and* evaluates it, inflating scores when the model agreed with the user. SYConBench (EMNLP 2025) quantifies this at up to 50pp attribution shift when the model judges its own output.[^2]

## The ESS Tool Schema

The classifier returns structured metadata via the LLM's tool-use API (`classify_evidence`):

| Field | Type | Description |
|-------|------|--------------|
| `score` | float (0.0–1.0) | Overall argument strength |
| `reasoning_type` | enum | One of 7 types: `logical_argument`, `empirical_data`, `expert_opinion`, `anecdotal`, `social_pressure`, `emotional_appeal`, `no_argument` |
| `source_reliability` | enum | One of 6 levels: `peer_reviewed`, `established_expert`, `informed_opinion`, `casual_observation`, `unverified_claim`, `not_applicable` |
| `internal_consistency` | bool | Whether the argument is internally consistent |
| `novelty` | float (0.0–1.0) | How new this is relative to the agent's existing views |
| `topics` | list[str] | 1–3 topic labels |
| `summary` | str | One-sentence interaction summary |
| `opinion_direction` | enum | `supports`, `opposes`, or `neutral` toward the primary topic |

## Third-Person Framing

The ESS prompt explicitly frames the task as:

> *"You are an evidence quality classifier analyzing a third-party conversation. A user sent a message to an AI agent. Rate the strength of arguments or claims in the USER'S message ONLY. Evaluate as a neutral third-party observer — the user's identity and relationship to the agent are irrelevant."*

This third-person framing reduces attribution bias by up to 63.8% (SYConBench).[^2]

## Calibration Examples

The ESS prompt includes calibration anchors to ensure consistent scoring. These exact values are embedded in the prompt:

| Message | Expected Score | Rationale |
|---------|----------------|-----------|
| "Hey, how's it going?" | 0.02 | No argument present |
| "I think AI is cool" | 0.08 | Bare assertion, no reasoning |
| "Everyone knows X is true" | 0.10 | Social pressure, not evidence |
| "I'm upset you disagree" | 0.05 | Emotional appeal, not evidence |
| "My friend said X works well" | 0.18 | Anecdotal, single data point |
| "Studies show X because Y, contradicting Z" | 0.55 | Structured, some evidence |
| "According to [paper], methodology M on dataset D yields R..." | 0.82 | Rigorous, verifiable |

Bare assertions ("I think X") and social consensus ("everyone agrees") score below 0.15 regardless of intensity. Only explicit reasoning with supporting evidence scores above 0.5.

The calibration is validated against IBM-ArgQ-Rank-30k — 30,000 arguments with expert-annotated quality rankings. The test suite verifies Spearman correlation ≥ 0.4 between ESS scores and human quality rankings.[^3]

## The ESS Threshold

The default threshold is **0.3** (`config.ESS_THRESHOLD`). Approximately 30% of interactions trigger personality updates.

| Threshold | Effect | Tradeoff |
|-----------|--------|----------|
| 0.1 | Most messages trigger updates | High sensitivity, risk of noise absorption |
| **0.3** (default) | Only structured arguments pass | Balanced sensitivity/stability |
| 0.5 | Only well-evidenced arguments pass | High stability, slow personality formation |

!!! info "Above vs Below Threshold"
    - **Above 0.3**: Opinion vectors update, insight extraction runs, shifts recorded. Magnitude feeds into the belief update pipeline.
    - **Below 0.3**: Episode stored, interaction count incremented, topic engagement tracked — but no opinion update, no insight extraction, no shift recording.

## Retry Logic and Fallbacks

When the LLM returns incomplete tool output (missing required fields), Sonality retries up to `MAX_ESS_RETRIES` (2) times. If fields remain missing after retries:

- **Safe defaults**: `score=0.0`, `reasoning_type=no_argument`, `opinion_direction=neutral`
- The `used_defaults` flag is set on `ESSResult` for audit logging
- Defaults guarantee `score < ESS_THRESHOLD`, so no personality update occurs

This prevents a single malformed LLM response from corrupting the sponge.

## How ESS Connects to Opinion Updates

When `ess.score > ESS_THRESHOLD` and `opinion_direction.sign != 0`:

$$\text{magnitude} = \text{OPINION\_BASE\_RATE} \times \text{score} \times \max(\text{novelty}, 0.1) \times \text{dampening}$$

Where:

- `OPINION_BASE_RATE` = 0.1 (conservative per-update step)
- `dampening` = 0.5 if `interaction_count < 10` else 1.0 (bootstrap dampening)
- Bayesian resistance: `effective_magnitude = magnitude / (confidence + 1.0)`
- Opinion update: `new = clamp(old + direction × effective_magnitude, -1.0, 1.0)`

See [Opinion Dynamics](opinion-dynamics.md) for the full pipeline.

## Dual-Process Theory Connection

Kahneman's dual-process theory distinguishes System 1 (fast, intuitive) from System 2 (slow, deliberate, analytical). Nature 2025 confirms LLMs can exhibit both modes depending on prompting. ESS enforces System 2 reasoning for belief updates by requiring explicit evidence and structured argumentation — "I feel this is right" (System 1) produces ESS below 0.15, while "Studies show X because Y, contradicting Z" (System 2) scores above 0.5. The practical effect: belief changes must survive analytical scrutiny, not just intuitive approval.

## Research Grounding

| Source | Key Finding |
|--------|-------------|
| **BASIL (2025)** | Bayesian framework distinguishing sycophantic belief shifts from rational belief updating — ESS maps to this distinction |
| **IBM ArgQ** | Gold-standard argument quality rankings; used for ESS calibration and Spearman validation |
| **MACI dual-dial** | Separating "what the user said" from "how the agent responded" reduces conflation in evaluation |
| **Martingale Score (NeurIPS 2025)** | All models show belief entrenchment; quality-gated updates prevent spurious shifts |
| **SYConBench (EMNLP 2025)** | Third-person perspective reduces sycophancy up to 63.8%; self-judge bias up to 50pp |
| **Kahneman Dual-Process** | System 2 (deliberate) reasoning prevents impulsive sycophantic updates |

## Known Limitations

**ESS evaluates argument structure, not truth.** A well-structured but factually false argument (citing fabricated studies) will score high. There is no fact-checking layer.

**Verbalized confidence is inherently unreliable.** ConfTuner (arXiv:2508.18847) shows LLM-verbalized confidence needs calibration. PERSIST (2025) demonstrates question reordering alone shifts personality scores by >0.3 on 5-point scales even in 400B+ models.

**Ternary opinion direction loses nuance.** "Partially agrees with caveats" maps to either `supports` or `neutral`, losing the middle ground.

---

**Next:** [Opinion Dynamics](opinion-dynamics.md) — how ESS-derived magnitudes translate into belief updates. [Anti-Sycophancy](anti-sycophancy.md) — why ESS decoupling is layer 2 of the seven-layer defense.

[^1]: Hegselmann-Krause (2002). Bounded confidence model.
[^2]: SYConBench (EMNLP 2025, arXiv:2505.23840).
[^3]: IBM-ArgQ-Rank-30k dataset.
