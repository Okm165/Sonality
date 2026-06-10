# Rejected Approaches

Every significant design decision in Sonality was preceded by evaluation and rejection of alternatives. This page documents what was considered and why it was dismissed — providing context for the current architecture.

---

## Knowledge Graphs for Memory

**Evaluated:** Graphiti, Neo4j-backed entity extraction (triple stores for episodic memory).

**Rejected because:**

- Empirical benchmarks (Lam et al., arXiv:2601.07978) showed vector databases significantly outperform graph databases in retrieval efficiency with no statistically significant accuracy difference
- Graphiti generated 1.17M tokens per test case, costing $152 before the benchmark was aborted
- Entity extraction at scale introduces hallucination (Mem0 achieves only 49.3% precision vs 84.6% for long-context baselines)
- Triple stores require ontology design that becomes brittle as topics expand

**What Sonality does instead:** Neo4j stores structural relationships (temporal chains, belief provenance, topic connections) while Qdrant handles semantic retrieval. The graph captures *relationships between memories* rather than trying to decompose every memory into entities.

---

## Fine-Tuning for Personality

**Evaluated:** LoRA, QLoRA, and full fine-tuning to embed personality directly in model weights.

**Rejected because:**

- RAG-based personalization achieves ~14.92% improvement over baselines vs only 1.07% for parameter-efficient fine-tuning ([arXiv:2409.09510](https://arxiv.org/abs/2409.09510))
- Fine-tuning is irreversible — personality changes require retraining
- Catastrophic forgetting erases prior personality when new data is added
- No interpretability — cannot inspect *why* the model holds a belief
- Requires labeled training data that doesn't exist for organic personality

**What Sonality does instead:** Personality lives in external memory (natural-language narrative + structured belief vectors). The model is stateless; personality is injected via the system prompt and retrieved context on every request. This enables inspection, rollback, and debugging of belief formation.

---

## OCEAN Personality Vectors

**Evaluated:** Big Five (OCEAN) trait scores as the primary personality representation.

**Rejected because:**

- PERSIST (AAAI 2026): even 400B+ parameter models show standard deviation >0.3 on 5-point personality scales — measurement noise exceeds meaningful signal
- Personality Illusion (NeurIPS 2025): self-reported traits don't reliably predict behavior; social desirability bias shifts Big Five scores by 1.20 SD in GPT-4
- Question reordering alone causes large shifts in measured traits
- Numeric scores lose nuance: "skeptical of blockchain claims except when backed by on-chain data" cannot be expressed as a single Openness score

**What Sonality does instead:** Natural-language personality narrative (~500 tokens) that the LLM reads directly. Belief vectors track *per-topic* opinions rather than abstract trait dimensions. OCEAN scores are retained only as a static baseline context, never updated.

---

## Per-Interaction Full Snapshot Rewrites

**Evaluated:** Rewriting the personality narrative after every qualifying interaction.

**Rejected because:**

- ABBEL (2025) demonstrated the "belief bottleneck" — iterative LLM rewrites propagate errors and silently drop minority opinions over hundreds of updates
- The "Broken Telephone" effect: each rewrite is a lossy compression that converges toward generic text
- Compute-expensive: full reflection per interaction is wasteful when most interactions don't warrant personality change

**What Sonality does instead:** Insight accumulation — each qualifying interaction contributes a one-sentence observation to a buffer. The personality snapshot is only rewritten during periodic reflection consolidation (every ~20 interactions or when cumulative belief shift exceeds threshold). Research from Park et al. (2023) confirms: reflection is where personality formation happens, not per-interaction writes.

---

## Self-Editing Memory Without Guardrails

**Evaluated:** Allowing the agent to freely modify its own memory (MemGPT-style).

**Rejected because:**

- MemoryGraft (2025): 47.9% retrieval poisoning rate from small poisoned record sets
- Unrestricted self-editing enables adversarial memory injection through conversational manipulation
- Without gating, the system cannot distinguish valuable memories from noise

**What Sonality does instead:** All memory writes pass through the ESS classifier. Manipulative inputs are captured as facts (the system records what was said) but cannot modify beliefs or personality state. Memory writes are gated, not free-form.

---

## Treating Every Interaction Equally

**Evaluated:** Updating beliefs on every user message regardless of content quality.

**Rejected because:**

- Bounded confidence models (Hegselmann-Krause 2002, Deffuant): systems that update on every input converge to consensus or oscillate chaotically
- Without quality differentiation, casual remarks ("I heard X might be true") carry the same weight as cited empirical evidence
- Produces "noise absorption" — personality becomes a mirror of conversation frequency rather than evidence quality

**What Sonality does instead:** Evidence Strength Score (ESS) gates all belief updates. Only messages with score ≥ 0.25 and non-manipulative reasoning type can trigger personality changes. This implements the principle that not all inputs deserve equal weight.

---

## Deterministic Top-K URL Selection

**Evaluated:** Always selecting the highest-ranked URLs for research (greedy selection).

**Rejected because:**

- Greedy selection causes "tunnel vision" — the system only visits known-good domains, missing serendipitous discoveries
- Deterministic selection means identical queries always produce identical research paths, limiting coverage
- High-authority domains are not universally good — a medical journal is irrelevant for a coding question

**What Sonality does instead:** Fathom applies softmax temperature sampling over RRF-fused ranking scores. Low-scoring URLs still have nonzero probability of selection. Each research session takes a different path while concentrating on quality sources. The temperature parameter controls the exploration/exploitation tradeoff.

---

## Heuristic Scoring Functions

**Evaluated:** Hand-tuned scoring weights, quality thresholds, decay factors, and Thompson Sampling for web research.

**Rejected because:**

| Heuristic | Problem |
|-----------|---------|
| Magic weights (0.3 / 0.7) | Arbitrary; tuned on unknown data |
| Quality threshold (> 0.4) | Why 0.4 and not 0.35? No principled basis |
| Decay factor (× 0.99) | Pretends to handle non-stationarity with a random constant |
| Thompson Sampling | Domain quality is topic-dependent; persistent reputation is wrong when sub-topics shift |
| Every-N-pages check | Why N? Arbitrary batch boundaries |

**What Sonality does instead:** Fathom's core principle is **zero heuristics** — every judgment is an LLM call. The LLM understands context, relevance, and nuance better than any formula. Decisions about "is this page useful?" or "what should we read next?" are made by the model, not by hand-crafted rules.

---

## Summary

| Rejected Approach | Core Problem | Sonality's Alternative |
|-------------------|-------------|----------------------|
| Knowledge graphs for memory | Cost-prohibitive, low precision | Hybrid graph (structure) + vector (semantics) |
| Fine-tuning | Irreversible, opaque, worse performance | External memory + RAG |
| OCEAN vectors | Noisy measurement, low expressiveness | Natural-language narrative |
| Per-interaction rewrites | Lossy compression, Broken Telephone | Insight accumulation + batch reflection |
| Free self-editing memory | Poisoning vulnerability | ESS-gated writes |
| Equal-weight updates | Noise absorption, chaotic convergence | Evidence quality scoring |
| Greedy URL selection | Tunnel vision, no exploration | Softmax temperature sampling |
| Heuristic scoring | Arbitrary constants, brittle | Zero-heuristic LLM judgment |

---

## References

- Lam et al. (2026). SSGM: Stability and Safety-Governed Memory governance framework.
- [Mem0 vs Graphiti benchmark](https://arxiv.org/abs/2601.07978) — Vector DB wins on efficiency.
- [RAG vs fine-tuning for personality](https://arxiv.org/abs/2409.09510) (2024) — 14x effectiveness gap.
- PERSIST (AAAI 2026) — Personality measurement noise in large models.
- Personality Illusion (NeurIPS 2025) — Self-reported traits vs actual behavior.
- ABBEL (2025) — Belief bottleneck and error propagation in iterative rewrites.
- MemoryGraft (2025) — Memory poisoning attack surface.
- Park et al. (2023). "Generative Agents: Interactive Simulacra of Human Behavior." *UIST*.
- [Hegselmann-Krause (2002)](https://jasss.soc.surrey.ac.uk/5/3/2.html). Bounded confidence opinion dynamics.

See also: [Sponge Architecture](../concepts/sponge.md), [Evidence Strength Score](../concepts/ess.md), [Fathom Research](../architecture/fathom.md), [Agentic Loop](agentic-loop.md).
