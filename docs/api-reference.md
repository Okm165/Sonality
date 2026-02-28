# API Reference

Module-level documentation for the Sonality package. Function signatures, types, and values match the actual codebase.

---

## `sonality.agent`

The core agent loop: context assembly → LLM → post-processing → persistence.

### `SonalityAgent`

Main agent class. Manages the conversation loop, ESS classification, personality updates, and reflection.

```python
from sonality.agent import SonalityAgent

agent = SonalityAgent()
response = agent.respond("Your message here")
```

#### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `client` | `Anthropic` | LLM API client (current provider SDK) |
| `sponge` | `SpongeState` | Current personality state |
| `episodes` | `EpisodeStore` | ChromaDB episode storage |
| `conversation` | `list[dict[str, str]]` | Current session conversation history |
| `last_ess` | `ESSResult | None` | Most recent ESS classification result |
| `previous_snapshot` | `str | None` | Snapshot before last change (for diffing) |

#### Methods

##### `respond(user_message: str) -> str`

Process a user message and return the agent's response. Main entry point. Handles: context assembly, response generation, ESS classification, opinion updates, insight extraction, reflection.

1. Retrieve relevant episodes from ChromaDB
2. Build system prompt with `build_system_prompt()`
3. Append user message to conversation, truncate if needed
4. Call LLM API (`max_tokens=2048`)
5. Call `_post_process()` for ESS, updates, reflection
6. Save sponge state

##### `_post_process(user_message: str, agent_response: str) -> None`

Post-processing pipeline after each response:

1. `_classify_ess(user_message)` — ESS classification (user message only)
2. `_log_ess()` — append to `ess_log.jsonl`
3. `_store_episode()` — store in ChromaDB
4. Increment `sponge.interaction_count`
5. `_update_topics()`, `_update_opinions()` — opinion vectors
6. `track_disagreement()` — structural disagreement detection
7. `_extract_insight()` — one-sentence insight if ESS ≥ threshold
8. `_maybe_reflect()` — periodic or event-driven reflection
9. `sponge.save()` — persist state

##### `_maybe_reflect() -> None`

Dual-trigger reflection:

- **Periodic:** `interaction_count - last_reflection_at >= REFLECTION_EVERY` (20)
- **Event-driven:** Cumulative shift magnitude in `recent_shifts` since last reflection &gt; `REFLECTION_SHIFT_THRESHOLD` (0.1)

Skips if `since < REFLECTION_EVERY // 2` (10). Runs `decay_beliefs()`, retrieves recent episodes, calls reflection LLM, validates snapshot, consolidates `pending_insights`.

##### `_classify_ess(user_message: str) -> ESSResult`

Calls `classify(client, user_message, sponge.snapshot)`. On exception, returns safe defaults: `score=0.0`, `reasoning_type=NO_ARGUMENT`, `topics=()`, `summary=user_message[:120]`.

##### `_build_structured_traits() -> str`

Formats structured traits for system prompt:

```
Style: {tone}
Top topics: {topic}({count}), ...
Strongest opinions: {topic}={pos:+.2f} c={conf:.1f}, ...
Disagreement rate: {rate:.0%}
```

Top 5 topics by engagement; top 5 opinions by `abs(value)`.

---

## `sonality.ess`

Evidence Strength Score classifier. Evaluates argument quality using a separate LLM call with tool-use structured output.

### Enums

#### `ReasoningType`

```python
class ReasoningType(StrEnum):
    LOGICAL_ARGUMENT = "logical_argument"
    EMPIRICAL_DATA = "empirical_data"
    EXPERT_OPINION = "expert_opinion"
    ANECDOTAL = "anecdotal"
    SOCIAL_PRESSURE = "social_pressure"
    EMOTIONAL_APPEAL = "emotional_appeal"
    NO_ARGUMENT = "no_argument"
```

#### `OpinionDirection`

```python
class OpinionDirection(StrEnum):
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    NEUTRAL = "neutral"

    @property
    def sign(self) -> float:  # +1.0, -1.0, or 0.0
```

#### `SourceReliability`

```python
class SourceReliability(StrEnum):
    PEER_REVIEWED = "peer_reviewed"
    ESTABLISHED_EXPERT = "established_expert"
    INFORMED_OPINION = "informed_opinion"
    CASUAL_OBSERVATION = "casual_observation"
    UNVERIFIED_CLAIM = "unverified_claim"
    NOT_APPLICABLE = "not_applicable"
```

### `ESSResult`

Frozen dataclass containing the classification output:

| Field | Type | Description |
|-------|------|-------------|
| `score` | `float` | Overall argument strength (0.0–1.0) |
| `reasoning_type` | `ReasoningType` | Primary reasoning type |
| `source_reliability` | `SourceReliability` | Source credibility level |
| `internal_consistency` | `bool` | Whether the argument is self-consistent |
| `novelty` | `float` | How new relative to existing views (0.0–1.0) |
| `topics` | `tuple[str, ...]` | 1–3 topic labels |
| `summary` | `str` | One-sentence interaction summary |
| `opinion_direction` | `OpinionDirection` | User supports/opposes/neutral on topic |
| `used_defaults` | `bool` | Whether fallback defaults were used |

### `classify(client, user_message, sponge_snapshot) -> ESSResult`

```python
def classify(
    client: Anthropic,
    user_message: str,
    sponge_snapshot: str,
) -> ESSResult
```

Classify evidence strength of the user's message. Uses separate LLM call with tool-use. **Agent response excluded** to avoid self-judge bias (SYConBench: third-person reduces sycophancy 63.8%).

### `ESS_TOOL` Schema

Tool schema for structured output:

```python
{
    "name": "classify_evidence",
    "description": "Classify the evidence strength and extract metadata from this interaction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {"type": "number", "description": "Overall argument strength 0.0-1.0."},
            "reasoning_type": {"type": "string", "enum": [...]},
            "source_reliability": {"type": "string", "enum": [...]},
            "internal_consistency": {"type": "boolean"},
            "novelty": {"type": "number"},
            "topics": {"type": "array", "items": {"type": "string"}},
            "summary": {"type": "string"},
            "opinion_direction": {"type": "string", "enum": ["supports", "opposes", "neutral"]},
        },
        "required": ["score", "reasoning_type", "source_reliability", "internal_consistency", "novelty", "topics", "summary", "opinion_direction"],
    },
}
```

---

## `sonality.memory.sponge`

Core personality state model. Handles opinion tracking, belief metadata, decay, and persistence.

### `BeliefMeta`

Pydantic model for confidence and provenance:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `confidence` | `float` | 0.0 | Logarithmic with evidence count |
| `evidence_count` | `int` | 1 | Times this belief reinforced |
| `last_reinforced` | `int` | 0 | Interaction number of last reinforcement |
| `provenance` | `str` | `""` | Most recent evidence source description |

### `BehavioralSignature`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `disagreement_rate` | `float` | 0.0 | Cumulative running mean of disagreement events |
| `topic_engagement` | `dict[str, int]` | `{}` | Topic → interaction count |

### `Shift`

| Field | Type | Description |
|-------|------|-------------|
| `interaction` | `int` | Interaction number |
| `timestamp` | `str` | ISO format |
| `description` | `str` | Shift description |
| `magnitude` | `float` | Cumulative magnitude |

### `SpongeState`

Complete personality state. Persisted as JSON.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `version` | `int` | 0 | Monotonically increasing version counter |
| `interaction_count` | `int` | 0 | Total interactions processed |
| `snapshot` | `str` | `SEED_SNAPSHOT` | Natural-language personality narrative |
| `opinion_vectors` | `dict[str, float]` | `{}` | Topic → stance value in [-1, 1] |
| `belief_meta` | `dict[str, BeliefMeta]` | `{}` | Topic → confidence/provenance metadata |
| `tone` | `str` | `"curious, direct, unpretentious"` | Communication style descriptor |
| `behavioral_signature` | `BehavioralSignature` | `{}` | Disagreement rate, topic engagement |
| `recent_shifts` | `list[Shift]` | `[]` | Last 10 personality shifts (`MAX_RECENT_SHIFTS=10`) |
| `pending_insights` | `list[str]` | `[]` | Insights awaiting reflection consolidation |
| `staged_opinion_updates` | `list[StagedOpinionUpdate]` | `[]` | Cooling-period queue for delayed belief commits |
| `last_reflection_at` | `int` | 0 | Interaction count of last reflection |

### Methods

#### `update_opinion(topic, direction, magnitude, provenance="", evidence_increment=1) -> None`

```python
def update_opinion(
    self,
    topic: str,
    direction: float,
    magnitude: float,
    provenance: str = "",
    evidence_increment: int = 1,
) -> None
```

Update opinion vector with Bayesian belief tracking. Clamps to [-1, 1]. Confidence: `min(1.0, log₂(evidence_count + 1) / log₂(20))`. `evidence_increment` lets batched staged updates preserve evidence count.

#### `stage_opinion_update(topic, direction, magnitude, cooling_period, provenance="") -> int`

Adds a signed belief delta to the staged queue and returns the due interaction count.

#### `apply_due_staged_updates() -> list[str]`

Commits due staged updates by topic (netting conflicting deltas), updates beliefs, and returns a compact applied-update summary.

#### `decay_beliefs(decay_rate=0.15, min_confidence=0.05) -> list[str]`

```python
def decay_beliefs(
    self, decay_rate: float = 0.15, min_confidence: float = 0.05
) -> list[str]
```

Power-law decay for unreinforced beliefs. Retention: `R(t) = (1 + gap)^(-β)`. Reinforcement floor: `min(0.6, evidence_count × 0.06)`. Skips `gap < 5`. Returns dropped topic names.

#### `record_shift(description: str, magnitude: float) -> None`

Append `Shift` to `recent_shifts`; trim to `MAX_RECENT_SHIFTS` (10).

#### `track_topic(topic: str) -> None`

Increment `behavioral_signature.topic_engagement[topic]`.

#### `track_disagreement(disagreed: bool) -> None`

Update `behavioral_signature.disagreement_rate` via cumulative running mean.

#### `save(path: Path, history_dir: Path) -> None`

Save state to JSON. Archives previous version to `history_dir/sponge_v{N}.json`. Atomic write (tmp + rename).

#### `load(path: Path) -> SpongeState` (classmethod)

Load state from JSON. Returns seed state if file doesn't exist. Includes backward-compat migration for removed fields (`vibe`, `affect_state`, `commitments`, `personality_ema`).

---

## `sonality.memory.episodes`

ChromaDB-backed episodic memory with ESS-weighted retrieval.

### `EpisodeStore`

#### `__init__(persist_dir: str) -> None`

Initialize ChromaDB `PersistentClient` with collection `"episodes"`, `hnsw:space="cosine"`.

#### `store(user_message, agent_response, ess_score, topics, summary, interaction_count=0, memory_type="episodic") -> None`

```python
def store(
    self,
    user_message: str,
    agent_response: str,
    ess_score: float,
    topics: Sequence[str],
    summary: str,
    interaction_count: int = 0,
    memory_type: str = "episodic",
) -> None
```

Store episode. Document = `summary` or `user_message[:200]`. Metadata: `ess_score`, `topics` (comma-joined), `summary`, `user_message[:500]`, `agent_response[:500]`, `timestamp`, `interaction`.

#### `retrieve(query, n_results=5, min_relevance=0.3, where=None) -> list[str]`

```python
def retrieve(
    self,
    query: str,
    n_results: int = 5,
    min_relevance: float = 0.3,
    where: dict[str, Any] | None = None,
) -> list[str]
```

Retrieve by cosine similarity. Rerank by `similarity × (1 + ess_score)` to prefer high-quality memories. Returns list of summary strings. Supports metadata filtering via `where` (e.g. `{"interaction": {"$gte": N}}`).

#### `retrieve_typed(query, episodic_n=3, semantic_n=2, min_relevance=0.3) -> list[str]`

Two-pass typed retrieval:

- fetch semantic memories (`memory_type=semantic`);
- fetch episodic memories (`memory_type=episodic`);
- merge in semantic-first order with de-duplication.

---

## `sonality.memory.updater`

Magnitude computation, snapshot validation, and insight extraction.

### `compute_magnitude(ess: ESSResult, sponge: SpongeState) -> float`

```python
def compute_magnitude(ess: ESSResult, sponge: SpongeState) -> float
```

Formula:

$$\text{magnitude} = \text{OPINION\_BASE\_RATE} \times \text{score} \times \max(\text{novelty}, 0.1) \times \text{dampening}$$

Dampening: 0.5 if `sponge.interaction_count < BOOTSTRAP_DAMPENING_UNTIL`, else 1.0.

### `validate_snapshot(old: str, new: str) -> bool`

Rejects if:

- `len(new) < 30`
- `len(new) / len(old) < MIN_SNAPSHOT_RETENTION` (0.6)

### `extract_insight(client, ess, user_message, agent_response) -> str | None`

```python
def extract_insight(
    client: Anthropic,
    ess: ESSResult,
    user_message: str,
    agent_response: str,
) -> str | None
```

Extract one-sentence personality insight. Uses `INSIGHT_PROMPT`. Returns `None` if empty or `"NONE"`. Truncates user/agent to 300 chars. Only called when `ess.score > ESS_THRESHOLD`.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `SNAPSHOT_CHAR_LIMIT` | `SPONGE_MAX_TOKENS × 5` (2500) | Max snapshot chars |
| `MIN_SNAPSHOT_RETENTION` | 0.6 | Min length ratio for validation |

---

## `sonality.prompts`

Prompt templates and system prompt assembly.

### `CORE_IDENTITY`

Immutable identity block (~200 tokens). Defines: intellectual honesty, curiosity, independence, explicit disagreement, merit-based evaluation, sycophancy resistance. Injected into every system prompt.

### `build_system_prompt(sponge_snapshot, relevant_episodes, structured_traits="") -> str`

```python
def build_system_prompt(
    sponge_snapshot: str,
    relevant_episodes: list[str],
    structured_traits: str = "",
) -> str
```

Assembles:

1. `<core_identity>` — `CORE_IDENTITY`
2. `<personality_state>` — `sponge_snapshot`
3. `<personality_traits>` — `structured_traits` (optional)
4. `<relevant_memories>` — episodes with "evaluate on merit, not familiarity" (optional)
5. `<instructions>` — response generation, anti-people-please

### `ESS_CLASSIFICATION_PROMPT`

Third-person evidence quality classifier. Placeholders: `{user_message}`, `{sponge_snapshot}`. Calibration examples: greetings 0.02, bare assertions 0.08, social pressure 0.10, rigorous evidence 0.82.

### `INSIGHT_PROMPT`

One-sentence personality insight extraction. Placeholders: `{user_message}`, `{agent_response}`, `{ess_score}`. Focus: reasoning style, communication character, self-discovery — NOT topic opinions.

### `REFLECTION_PROMPT`

PRESERVE-first reflection. Placeholders: `{current_snapshot}`, `{structured_traits}`, `{current_beliefs}`, `{pending_insights}`, `{episode_count}`, `{episode_summaries}`, `{recent_shifts}`, `{max_tokens}`. Tasks: preserve → integrate → synthesize → inject specificity.

---

## `sonality.config`

Environment-based configuration. Loads from `.env` via `dotenv`.

### Paths

| Constant | Value |
|----------|-------|
| `PROJECT_ROOT` | `Path(__file__).resolve().parent.parent` |
| `DATA_DIR` | `PROJECT_ROOT / "data"` |
| `SPONGE_FILE` | `DATA_DIR / "sponge.json"` |
| `SPONGE_HISTORY_DIR` | `DATA_DIR / "sponge_history"` |
| `CHROMADB_DIR` | `DATA_DIR / "chromadb"` |

### Models (from env)

| Constant | Env Var | Default |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` | *(required)* |
| `MODEL` | `SONALITY_MODEL` | *(see .env.example)* |
| `ESS_MODEL` | `SONALITY_ESS_MODEL` | Same as `MODEL` |

### Parameters

| Constant | Env Var | Default | Description |
|----------|---------|---------|-------------|
| `LOG_LEVEL` | `SONALITY_LOG_LEVEL` | `INFO` | Python logging level |
| `ESS_THRESHOLD` | `SONALITY_ESS_THRESHOLD` | 0.3 | Min ESS for updates |
| `SPONGE_MAX_TOKENS` | — | 500 | Snapshot token target |
| `EPISODE_RETRIEVAL_COUNT` | — | 5 | Episodes per retrieval |
| `OPINION_BASE_RATE` | — | 0.1 | Base step for opinion updates |
| `BELIEF_DECAY_RATE` | — | 0.15 | Power-law decay exponent β |
| `BOOTSTRAP_DAMPENING_UNTIL` | `SONALITY_BOOTSTRAP_DAMPENING_UNTIL` | 10 | Interactions with 0.5× dampening |
| `OPINION_COOLING_PERIOD` | `SONALITY_OPINION_COOLING_PERIOD` | 3 | Staging delay before belief commits |
| `SEMANTIC_RETRIEVAL_COUNT` | `SONALITY_SEMANTIC_RETRIEVAL_COUNT` | 2 | Semantic memory retrieval budget |
| `EPISODIC_RETRIEVAL_COUNT` | `SONALITY_EPISODIC_RETRIEVAL_COUNT` | 3 | Episodic memory retrieval budget |
| `MAX_CONVERSATION_CHARS` | — | 100,000 | Conversation truncation limit |
| `REFLECTION_EVERY` | `SONALITY_REFLECTION_EVERY` | 20 | Periodic reflection interval |
| `REFLECTION_SHIFT_THRESHOLD` | — | 0.1 | Cumulative magnitude for early reflection |
