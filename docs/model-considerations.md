# Model Considerations

Sonality is designed to be model-agnostic in architecture but not in requirements. Different roles in the pipeline have different demands. This guide covers what to look for when selecting models and documents known trade-offs.

---

## Pipeline Roles

Sonality uses 2–3 LLM calls per interaction, each with different requirements:

| Role | Requirements | Frequency |
|------|-------------|-----------|
| **Response generation** | Strong reasoning, personality coherence, long-context handling, natural conversation | Every interaction |
| **ESS classification** | Structured output (tool_use), calibrated scoring, resistance to self-judge bias | Every interaction |
| **Insight extraction** | Concise summarization, identity-relevant pattern recognition | When ESS > threshold |
| **Reflection** | Narrative synthesis, belief reconciliation, preservation of established traits | Every 20 interactions |

### Response Generation

The main model handles conversation. Key requirements:

- **Long-context support** (100k+ tokens) — the system prompt includes personality state, traits, and retrieved episodes
- **Instruction following** — the core identity defines behavioral constraints (anti-sycophancy, intellectual honesty) that the model must respect
- **Reasoning quality** — the agent needs to construct genuine arguments, not just agree
- **Personality coherence** — the model must maintain the character described in the sponge snapshot across turns

Higher-capability models produce more distinctive personalities. Research finding: PersonaGym (EMNLP 2025) found only 2.97% difference between top-tier and mid-tier models on persona maintenance — suggesting mid-tier models are often sufficient.

### ESS Classification

The ESS classifier evaluates user argument quality via structured tool output. Requirements:

- **Structured output** — must support `tool_use` or equivalent function-calling API
- **Calibration** — scores must be consistent across similar argument types (bare assertions always < 0.15, structured evidence always > 0.5)
- **Resistance to framing** — third-person evaluation prompt reduces but doesn't eliminate model-specific biases

**Using a separate model for ESS reduces self-judge coupling.** When the same model generates both the response and the ESS evaluation, it creates a feedback loop where the model rates its own reasoning favorably (up to 50pp bias documented by SYConBench, EMNLP 2025). Using a different model for ESS classification is the simplest mitigation.

Cost consideration: ESS classification uses structured output with a fixed schema — a smaller, faster model often suffices here.

### Insight Extraction and Reflection

Both insight extraction and reflection require summarization and synthesis. Reflection in particular must:

- Preserve existing personality traits while integrating new insights
- Detect contradictions between beliefs
- Generate meta-patterns ("I notice I tend to value X")

Reflection quality directly determines personality stability. Park et al. (2023) ablation showed reflection is the most critical component for believable agents. Use a model with strong instruction-following for reflection.

---

## Model Selection Criteria

### Minimum Requirements

Any model used with Sonality must support:

1. **System prompts** — the personality state is injected as a system message
2. **Structured output** (for ESS) — tool_use, function calling, or equivalent
3. **100k+ context window** — for conversation history + personality state + episodes
4. **Strong instruction following** — anti-sycophancy framing requires the model to resist agreeing by default

### Evaluation Checklist

When evaluating a new model for Sonality:

| Test | What to Check | Healthy |
|------|--------------|---------|
| ESS calibration | Run `make test-ess` with 10 benchmark messages | Scores within ±0.15 of expected |
| Sycophancy resistance | Run the sycophancy battery (see [Testing](testing.md)) | Disagreement rate > 20% |
| Personality coherence | 50 interactions, then check `/snapshot` distinctiveness | Snapshot diverged from seed |
| Reflection quality | After reflection, `/diff` shows meaningful synthesis | Not just parroting insights |

### Cost-Performance Trade-offs

The pipeline makes 2–3 calls per interaction. For a 100-interaction session:

| Configuration | Calls | Trade-off |
|--------------|-------|-----------|
| Same model for all roles | ~250 | Simplest setup; self-judge bias risk |
| Separate ESS model (smaller) | ~250 | Reduced self-judge bias; lower cost for ESS calls |
| Smaller model for insight extraction | ~250 | Lower cost; may reduce insight quality |

Many providers offer prompt caching (up to 90% discount on static prefixes). The system prompt (~1,400 tokens) is largely static, making Sonality cache-friendly.

---

## Embedding Model

Sonality uses an embedding model for ChromaDB episode storage and retrieval. The default (ChromaDB's built-in model) has limitations:

| Aspect | Consideration |
|--------|--------------|
| Token window | Short windows (128–256 tokens) truncate longer summaries |
| Semantic quality | Negation blindness: "I believe X" and "I no longer believe X" embed similarly |
| Performance | Larger embedding models improve retrieval but increase storage time |

For production use, consider embedding models with 2048+ token windows and better semantic discrimination. The migration path is straightforward: re-embed existing episodes with the new model.

---

## Provider-Specific Notes

Sonality currently uses a single provider SDK. To switch providers:

1. **Swap the SDK** in `sonality/agent.py`, `sonality/ess.py`, `sonality/memory/updater.py`
2. **Update the structured output format** in `sonality/ess.py` — the `ESS_TOOL` schema maps to the provider's tool/function calling format
3. **Update environment variables** in `sonality/config.py`
4. **Re-run calibration tests** — ESS scoring behavior varies across models

The architecture is designed so that provider-specific code is concentrated in three files. The memory layer (ChromaDB, sponge state) and the prompt templates are provider-independent.

---

## Research on Model Selection for Personality Agents

Key findings relevant to model selection:

- **PERSIST (2025):** Even 400B+ parameter models show σ > 0.3 measurement noise on personality assessments. Scaling alone does not solve personality instability — architectural scaffolding (external state, ESS gating, reflection) compensates.

- **BIG5-CHAT (ACL 2025):** Higher conscientiousness and agreeableness in the base model improve reasoning quality. Models with these traits in their training character make better personality agents.

- **PISF (2024):** Prompt Induction post Supervised Fine-Tuning — combining SFT with prompting achieves the highest efficacy and adversarial robustness. For API-only use, prompt engineering is the primary lever.

- **PersonaGym (EMNLP 2025):** Persona maintenance is surprisingly similar across model tiers (< 3% difference between top and mid-tier). The bottleneck is architecture, not model capability.

- **Personality Illusion (NeurIPS 2025):** RLHF-trained models show more stable personality expression than base models. Post-training matters more than scale for personality coherence.
