## System Overview

Sonality is a self-evolving LLM personality agent built on a "Sponge architecture" — a compact natural-language narrative (~500 tokens) that absorbs conversations, modulated by an Evidence Strength Score (ESS) that gates which interactions actually change the agent's worldview. The memory subsystem uses a three-layer architecture: Short-Term Memory (bounded buffer with LLM summarization), Long-Term Episodic Memory (Neo4j graph + pgvector derivatives), and Semantic Features (PostgreSQL). Every architectural decision is grounded in academic research. This document describes the core interaction loop, each pipeline stage in depth, the technology stack, and the rationale behind key design choices.

## The Core Interaction Loop

Every interaction flows through `SonalityAgent.respond()` — a single entry point that orchestrates context assembly, response generation, and post-processing. The agent uses a sync→async bridge (background event loop + `asyncio.run_coroutine_threadsafe()`) to call async database operations from the synchronous `respond()` method. When Neo4j/PostgreSQL are unavailable, the system gracefully falls back to ChromaDB-only mode.

```mermaid
flowchart TB
    subgraph Input
        UM[User Message]
    end

    subgraph Stage0["0. STM Buffering"]
        STM_ADD[ShortTermMemory.add_message]
    end

    subgraph Stage1["1. Context Assembly"]
        ROUTE[QueryRouter.route]
        RETRIEVE[ChainOfQuery / SplitQuery / Recency]
        RERANK[LLM Listwise Reranking]
        TRAITS[_build_structured_traits]
        BUILD[build_system_prompt + STM summary]
    end

    subgraph Stage2["2. Response Generation"]
        LLM["LLM API · messages.create"]
    end

    subgraph Stage3["3. ESS Classification"]
        CLASSIFY[classify in ess.py]
    end

    subgraph Stage4["4. Post-Processing"]
        BOUNDARY[EventBoundaryDetector]
        DUAL_STORE[DualEpisodeStore: Neo4j + pgvector]
        SEMANTIC[SemanticIngestionWorker]
        STORE_LEGACY[ChromaDB fallback]
        TOPICS[_update_topics]
        OPINIONS[_update_opinions]
        DISAGREE[_detect_disagreement + note_disagreement/agreement]
        INSIGHT[_extract_insight]
    end

    subgraph Stage5["5. Reflection"]
        CONSOLIDATE[ConsolidationEngine]
        FORGET[ForgettingEngine: LLM importance assessment]
        DECAY[decay_beliefs]
        REFLECT[REFLECTION_PROMPT + LLM]
        VALIDATE[validate_snapshot]
        HEALTH[assess_health: LLM health check]
    end

    subgraph Stage6["6. Persistence"]
        SAVE[SpongeState.save + STM.persist]
    end

    UM --> STM_ADD
    STM_ADD --> ROUTE
    ROUTE --> RETRIEVE
    RETRIEVE --> RERANK
    RERANK --> TRAITS
    TRAITS --> BUILD
    BUILD --> LLM
    LLM --> CLASSIFY
    CLASSIFY --> BOUNDARY
    BOUNDARY --> DUAL_STORE
    DUAL_STORE --> SEMANTIC
    CLASSIFY --> TOPICS
    CLASSIFY --> OPINIONS
    CLASSIFY --> DISAGREE
    CLASSIFY --> |"ESS above 0.3"| INSIGHT
    INSIGHT --> CONSOLIDATE
    DUAL_STORE --> CONSOLIDATE
    CONSOLIDATE --> FORGET
    FORGET --> DECAY
    DECAY --> |"periodic or event-driven"| REFLECT
    REFLECT --> VALIDATE
    VALIDATE --> HEALTH
    HEALTH --> SAVE
    OPINIONS --> SAVE
```

!!! info "Entry Point"
    The main loop is `SonalityAgent.respond(user_message: str) -> str` in `agent.py`. It buffers the user message in STM, retrieves memories via the routing pipeline, builds the system prompt with STM summary, generates a response, then invokes `_post_process()` for boundary detection, dual-store storage, ESS classification, and conditional updates. A background event loop bridges sync→async for all database operations.

## Pipeline Stages in Depth

### 1. Context Assembly

Before every LLM call, the system assembles a system prompt from five components:

| Component | Size | Source | Always Present? |
|-----------|------|--------|-----------------|
| Core Identity | ~200 tokens | `CORE_IDENTITY` in `prompts.py` | Yes |
| Personality Snapshot | ~500 tokens | `sponge.snapshot` | Yes |
| STM Running Summary | Variable | `ShortTermMemory.running_summary` | If available |
| Structured Traits | ~100 tokens | `_build_structured_traits()` | Yes |
| Retrieved Memories | ~400 tokens | Retrieval pipeline (QueryRouter → agent search → LLM rerank) | If available |

**Retrieval pipeline** (new architecture): `QueryRouter` classifies the query into categories (factual, personal, recent, meta). Factual queries dispatch to `ChainOfQueryAgent` (iterative refinement with confidence threshold). Personal/complex queries dispatch to `SplitQueryAgent` (parallel sub-query decomposition). Results are deduplicated, temporally expanded via Neo4j graph edges, and reranked using LLM listwise comparison. Falls back to ChromaDB `retrieve_typed()` when the new architecture is unavailable.

`build_system_prompt(sponge_snapshot, relevant_episodes, structured_traits)` wraps each in XML-style tags: `<core_identity>`, `<personality_state>`, `<personality_traits>`, `<relevant_memories>`, and `<instructions>`. The STM running summary is injected into the personality state section. The core identity is immutable and anchors the agent's fundamental values regardless of how opinions evolve — research shows persona drift occurs within 8 rounds without such an anchor[^1].

[^1]: arXiv:2402.10962 — measurable persona drift in 8 conversation rounds.

### 2. Response Generation

A standard LLM API call with `model=config.MODEL`, `max_tokens=2048`, the assembled system prompt, and `self.conversation` as messages. The agent responds naturally, drawing on its personality state and retrieved memories. No tool use or structured output is required for the main response.

### 3. ESS Classification

A **separate** LLM call evaluates the user's message for argument strength. Critically, **only the user message** is passed — the agent's response is deliberately excluded to avoid self-judge bias (documented at up to 50 percentage points in SYConBench, EMNLP 2025)[^2].

The `classify()` function in `ess.py` uses the LLM's `tool_use` with a `classify_evidence` tool to extract structured output:

| Field | Type | Description |
|-------|------|--------------|
| `score` | float (0.0–1.0) | Overall argument strength |
| `reasoning_type` | enum | `logical_argument`, `empirical_data`, `expert_opinion`, `anecdotal`, `social_pressure`, `emotional_appeal`, `no_argument` |
| `source_reliability` | enum | `peer_reviewed` through `not_applicable` |
| `internal_consistency` | bool | Whether the argument is internally consistent |
| `novelty` | float (0.0–1.0) | How new this is relative to the agent's existing views |
| `topics` | list[str] | 1–3 topic labels |
| `summary` | str | One-sentence interaction summary |
| `opinion_direction` | enum | `supports`, `opposes`, `neutral` |

[^2]: SYConBench (EMNLP 2025): third-person prompting reduces sycophancy by up to 63.8%.

### 4. Conditional Processing

Post-processing runs unconditionally for: episode storage, interaction counting, staged-commit checks, topic tracking, and disagreement detection. The ESS score gates opinion staging and insight extraction:

| ESS | Episode Storage | Topic Tracking | Opinion Update | Insight Extraction |
|-----|-----------------|----------------|----------------|--------------------|
| ≤ 0.3 | Yes | Yes | No | No |
| > 0.3 | Yes | Yes | Conditional | Conditional |

**Opinion updates** require all three conditions: (1) `ess.score > ESS_THRESHOLD`, (2) `ess.topics` is non-empty, and (3) `opinion_direction.sign != 0` (user takes a stance). When all three hold:

1. **Magnitude computation**: `compute_magnitude()` yields `base_rate × score × max(novelty, 0.1) × dampening`; Bayesian resistance applies `effective_mag = magnitude / (confidence + 1)`. When the user argues against an existing stance, additional resistance: `conf += abs(old_position)`.
2. **Insight extraction**: A third LLM call extracts one personality-relevant sentence via `INSIGHT_PROMPT`; the result is appended to `pending_insights`. Only requires `ess.score > ESS_THRESHOLD`.
3. **Shift recording**: `record_shift()` logs magnitude and description for reflection triggers.

!!! tip "Bootstrap Dampening"
    For the first `BOOTSTRAP_DAMPENING_UNTIL` (default 10) interactions, `dampening = 0.5`. This prevents "first-impression dominance" documented in bounded confidence models (Deffuant).

### 5. Reflection

Reflection is triggered by:

- **Periodic**: every `REFLECTION_EVERY` (default 20) interactions since `last_reflection_at`
- **Event-driven**: cumulative shift magnitude since last reflection exceeds `REFLECTION_SHIFT_THRESHOLD` (0.1)

When triggered:

1. **Consolidation**: `ConsolidationEngine.maybe_consolidate_segment()` merges related episodes in the current segment via LLM assessment.
2. **Forgetting**: `ForgettingEngine.assess_and_forget()` queries Neo4j for low-utility episodes, runs batch LLM importance assessment, and soft-archives dispensable episodes (removed from pgvector, graph node retained for provenance).
3. **Decay**: `decay_beliefs(decay_rate=0.15)` applies power-law forgetting: \( R(t) = (1 + \text{gap})^{-0.15} \), with floor \( \min(0.6, \max(0.0, (\text{evidence\_count} - 1) \times 0.04)) \). Beliefs below 0.05 confidence are dropped.
4. **Retrieve**: Recent episodes are fetched with `where={"interaction": {"$gte": last_reflection_at}}`.
5. **Consolidate**: `REFLECTION_PROMPT` is sent to the LLM with current snapshot, traits, beliefs, pending insights, episode summaries, and recent shifts. The LLM outputs a revised narrative.
6. **Validate**: `validate_snapshot()` rejects if `len(new) / len(old) < 0.6` (minimum retention ratio).
7. **Health**: `assess_health()` runs an LLM health assessment on the memory system state.
8. **Clear**: `pending_insights` is cleared; `last_reflection_at` is updated.

!!! warning "Reflection Ablation"
    Park et al. (2023) ablation showed reflection is the **most critical component** for believable agents. Sonality's accumulate-then-consolidate approach avoids the "Broken Telephone" effect where iterative per-interaction rewrites converge to generic text (ACL 2025).

### 6. Persistence

`SpongeState.save(config.SPONGE_FILE, config.SPONGE_HISTORY_DIR)` runs after every interaction. Before writing, the current file is copied to `sponge_history/sponge_v{N}.json`. The full `SpongeState` is serialized as JSON via Pydantic's `model_dump_json()`. When the new architecture is active, `ShortTermMemory.persist()` also writes the STM buffer and running summary to PostgreSQL for crash recovery.

## Technology Stack

| Component | Technology | Justification |
|-----------|------------|---------------|
| LLM | Configurable (see [Configuration — Model Selection](../configuration.md#model-selection)) | Best reasoning for structured output and belief extraction without fine-tuning; RAG-based personalization achieves ~14.92% improvement vs 1.07% for PEFT[^3] |
| Episode Graph | Neo4j (async driver) | Temporal edges, provenance tracking, graph traversal for episode expansion; hybrid graph+vector approach for belief revision (RecallM: 4× improvement) |
| Derivative Vectors | PostgreSQL + pgvector | Sentence-level derivative embeddings for granular retrieval; co-located with STM and semantic features |
| Legacy Vector Store | ChromaDB | Fallback when Neo4j/PostgreSQL unavailable; cosine similarity[^4] |
| Embeddings | OpenAI text-embedding-3-large (configurable) | High-dimensional (4096) embeddings for derivative chunks |
| Personality State | Pydantic models → JSON on disk | Simple, inspectable, versionable |
| Short-Term Memory | In-memory deque + PostgreSQL persistence | Bounded buffer with LLM summarization; crash recovery via PostgreSQL |
| Orchestration | Plain Python + asyncio bridge | Sync `respond()` bridges to async database operations via background event loop |
| Package Manager | uv | Fast, lockfile-based, modern Python tooling |

[^3]: arXiv:2409.09510 — RAG vs fine-tuning comparison.
[^4]: arXiv:2601.07978 — Mem0 vs Graphiti benchmark.

### Why No Heavy Agent Framework?

Sonality's pipeline requires branching logic (conditional processing based on ESS), stateful persistence (personality survives sessions), and cyclic operations (reflection feeds back into the personality). General-purpose orchestration frameworks can provide these features, but they also add abstraction overhead that is unnecessary at this scale. Building this with plain Python functions keeps the flow compact in a single `respond()` method, reduces moving parts, and makes debugging faster.

### Why Neo4j + pgvector (Hybrid Graph+Vector)?

RecallM reports 4× improvement with hybrid graph+vector over vector-only for belief revision. The new memory architecture uses Neo4j for the episode graph (temporal edges, provenance tracking, graph traversal) and PostgreSQL+pgvector for derivative embeddings (granular sentence-level retrieval). This hybrid approach provides temporal coherence and provenance tracking that pure vector search cannot offer, while keeping vector similarity search efficient via pgvector. ChromaDB is retained as a graceful fallback when databases are unavailable, ensuring the agent always functions.

### Why Not Fine-Tuning?

RAG-based personalization achieves ~14.92% improvement over baselines vs 1.07% for parameter-efficient fine-tuning. Fine-tuning also requires training data that doesn't exist yet, risks catastrophic forgetting, and prevents the personality from evolving at runtime. Sonality's personality is **external state** — it evolves through interaction, not retraining.

## Context Window Budget

Research shows the bottleneck is **reasoning quality, not context space** — PersonaMem-v2 found frontier LLMs achieve only 37–48% accuracy on personalization tasks despite long context. Sonality allocates aggressively to compressed, structured memory rather than raw conversation history.

| Component | Token Budget | Notes |
|-----------|-------------|-------|
| System prompt (instructions) | 500–1,000 | Static prefix; prompt caching eligible |
| Core identity (Tier 1) | ~500 | Always present; immutable; highly cacheable |
| Personality snapshot (sponge) | ~500 | Current narrative; updated after reflection |
| Structured traits (Tier 2) | 200–600 | Opinion vectors, topics, meta-beliefs |
| Retrieved episodes (Tier 3) | 500–1,500 | 5 most relevant past interactions |
| Conversation history | 2,000–8,000 | Current session, truncated at 100k chars |
| User's current message | Variable | |
| **Total input** | **~4,500–12,000** | Well under any modern model's 128k–200k capacity |
| Reserved for output | 2,000–4,000 | Agent response |

The entire personality state uses less than 10% of a 128k context window. Context space is not the bottleneck — retrieval quality and reasoning fidelity are.

**Prompt caching:** The system prompt and core identity are placed at the start of the message (static prefix). Many LLM providers offer prompt caching discounts (up to 90%) for static prefixes. For ~1,500 tokens of static personality, this can reduce costs significantly.

## Cost Analysis

Per-interaction cost: **~$0.005–0.015** depending on the configured model.

| Call | Model | Purpose | Approx. Tokens |
|------|-------|---------|----------------|
| Response generation | `config.MODEL` | Main conversational response | ~2,000 in / ~500 out |
| ESS classification | `config.ESS_MODEL` | Evidence strength scoring (tool_use) | ~800 in / ~200 out |
| Insight extraction | `config.ESS_MODEL` | Personality insight (conditional, ESS above 0.3) | ~400 in / ~50 out |
| Reflection | `config.ESS_MODEL` | Snapshot consolidation (periodic/event-driven) | ~1,500 in / ~500 out |

Each interaction makes **2–3** API calls: always response + ESS; conditionally insight; periodically reflection. At 20 interactions per reflection cycle, reflection adds ~0.05/20 ≈ $0.0025 per interaction amortized.

**Daily cost estimate (100 messages/day):**

- Episodic storage: 100 × $0.001 = $0.10
- ESS classification: 100 × $0.003 = $0.30
- Insight extraction (~30% trigger): 30 × $0.002 = $0.06
- Reflection cycles (~5×): 5 × $0.01 = $0.05
- **Total: ~$4–5/day** (vs $38+/day with per-message knowledge graph updates like Zep)

## Retrieval Strategy: Agent-Based Routing

Sonality uses **always-retrieve with agent-based routing**: every interaction triggers the retrieval pipeline before building the system prompt. The `QueryRouter` classifies each query into a category (factual, personal, recent, meta) and dispatches to the appropriate retrieval agent:

| Category | Agent | Strategy |
|----------|-------|----------|
| Factual | `ChainOfQueryAgent` | Iterative refinement: search → assess confidence → refine query (up to 3 iterations) |
| Personal / Complex | `SplitQueryAgent` | Decompose into parallel sub-queries, merge and deduplicate results |
| Recent | Recency search | Direct temporal query on Neo4j graph |
| Meta | Direct | Minimal retrieval needed |

Results are temporally expanded via Neo4j graph edges (neighboring episodes in the same conversation segment), then reranked using LLM listwise comparison that considers relevance, recency, and provenance quality. Falls back to ChromaDB `retrieve_typed()` when the new architecture is unavailable.

| Strategy | Pros | Cons |
|----------|------|------|
| **Agent-routed** (Sonality) | Category-optimal retrieval, iterative refinement, LLM reranking | Additional LLM calls for routing and reranking |
| **Always-retrieve flat** (legacy) | Deterministic, fast | Fixed strategy; may retrieve irrelevant episodes for novel topics |
| **Tool-based** (memory-OS style) | Agent decides when to search; flexible | MemTool (arXiv:2507.21428): 0–60% efficiency on medium models; under-retrieval risk |

---

**Next:** [Memory Model](memory-model.md) — the five-tier memory hierarchy in detail. [Data Flow](data-flow.md) — how data moves through the pipeline for a single interaction.
