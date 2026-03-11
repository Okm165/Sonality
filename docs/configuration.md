# Configuration

All configuration is via environment variables (set in `.env`) and compile-time constants. One `.env.example`, one source of truth. Values are defined in `sonality/config.py` and `sonality/memory/updater.py`.

---

## Environment Variables

Set these in your `.env` file (copy from `.env.example`):

### Core

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_API_KEY` | *(required)* | API key for the configured provider endpoint |
| `SONALITY_API_VARIANT` | *(required)* | API endpoint variant (`anthropic` or `openrouter`) |
| `SONALITY_MODEL` | *(see .env.example)* | Main reasoning model for response generation |
| `SONALITY_ESS_MODEL` | Same as `SONALITY_MODEL` | Model for ESS classification, insight extraction, and reflection |
| `SONALITY_ESS_THRESHOLD` | `0.3` | Minimum ESS score to trigger personality updates |
| `SONALITY_LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `SONALITY_BOOTSTRAP_DAMPENING_UNTIL` | `10` | First N interactions with 0.5× opinion update dampening |
| `SONALITY_OPINION_COOLING_PERIOD` | `3` | Interactions before staged opinion deltas are committed |
| `SONALITY_SEMANTIC_RETRIEVAL_COUNT` | `2` | Semantic memories retrieved each turn |
| `SONALITY_EPISODIC_RETRIEVAL_COUNT` | `3` | Episodic memories retrieved each turn |
| `SONALITY_REFLECTION_EVERY` | `20` | Periodic reflection interval (interactions) |

### Database (Neo4j + PostgreSQL)

Required for the new memory architecture. When unavailable, the system falls back to ChromaDB.

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_NEO4J_URL` | `bolt://localhost:7687` | Neo4j Bolt connection URL |
| `SONALITY_NEO4J_USER` | `neo4j` | Neo4j username |
| `SONALITY_NEO4J_PASSWORD` | `sonality_password` | Neo4j password |
| `SONALITY_NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `SONALITY_POSTGRES_URL` | `postgresql://sonality:sonality_password@localhost:5432/sonality` | PostgreSQL connection URL (with pgvector) |
| `SONALITY_PG_POOL_MIN_SIZE` | `2` | PostgreSQL connection pool minimum size |
| `SONALITY_PG_POOL_MAX_SIZE` | `10` | PostgreSQL connection pool maximum size |

### Embedding

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_EMBEDDING_PROVIDER` | `openai` | Embedding provider (`openai` or `openrouter`) |
| `SONALITY_EMBEDDING_MODEL` | `text-embedding-3-large` | Embedding model name |
| `SONALITY_EMBEDDING_DIMENSIONS` | `4096` | Embedding vector dimensions |
| `SONALITY_EMBEDDING_API_KEY` | `$OPENAI_API_KEY` | API key for embedding provider |
| `SONALITY_EMBEDDING_BATCH_SIZE` | `32` | Batch size for embedding API calls |
| `SONALITY_EMBEDDING_QUERY_INSTRUCTION` | *(see config.py)* | Instruction prefix for query embeddings |
| `SONALITY_EMBEDDING_DOC_INSTRUCTION` | *(see config.py)* | Instruction prefix for document embeddings |

### Fast LLM (Memory Subsystem)

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_FAST_LLM_MODEL` | `claude-haiku-4-5-20251001` | Fast, cheap model for memory tasks (routing, chunking, forgetting, health) |
| `SONALITY_FAST_LLM_MAX_TOKENS` | `1024` | Max output tokens for fast LLM calls |

### Short-Term Memory

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_STM_BUFFER_CAPACITY` | `64000` | STM buffer character capacity |
| `SONALITY_STM_BATCH_THRESHOLD` | `3` | Minimum evicted messages before triggering background summarization |
| `SONALITY_STM_MAX_BATCH_SIZE` | `10` | Maximum messages per summarization batch |
| `SONALITY_STM_POLL_INTERVAL` | `30.0` | Background summarizer polling interval (seconds) |

### Retrieval

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_RETRIEVAL_MAX_ITERATIONS` | `3` | Max iterations for ChainOfQueryAgent refinement |
| `SONALITY_RETRIEVAL_CONFIDENCE_THRESHOLD` | `0.8` | Confidence threshold to stop iterating in ChainOfQuery |
| `SONALITY_RETRIEVAL_OVER_FETCH_FACTOR` | `3` | Over-fetch multiplier for initial retrieval before reranking |
| `SONALITY_MAX_RERANK_CANDIDATES` | `25` | Max candidates for LLM listwise reranking |

### Provenance

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_MAX_UIDS_PER_BELIEF` | `20` | Max episode UIDs tracked per belief for provenance |

---

## Compile-Time Constants

These are defined in `sonality/config.py` and `sonality/memory/updater.py`. Not configurable via environment. Note: many constants that were previously compile-time only are now configurable via environment variables (see above).

### Personality Parameters

| Constant | Value | Location | Rationale |
|----------|-------|----------|-----------|
| `SPONGE_MAX_TOKENS` | 500 | `config.py` | Target size for personality snapshot. Balances expressiveness with context window budget. ABBEL (2025): compact belief states outperform full context. |
| `OPINION_BASE_RATE` | 0.1 | `config.py` | Conservative per-update step. At max ESS (1.0) and max novelty (1.0), single interaction shifts opinion by 10%. Typical shifts: 2–4%. |
| `BELIEF_DECAY_RATE` | 0.15 | `config.py` | Power-law exponent β for Ebbinghaus-inspired forgetting. Higher = faster forgetting. FadeMem (2026), human memory curve research. |
| `REFLECTION_SHIFT_THRESHOLD` | 0.1 | `config.py` | Cumulative shift magnitude that triggers event-driven reflection before periodic interval. |

### Memory Parameters

| Constant | Value | Location | Rationale |
|----------|-------|----------|-----------|
| `SEMANTIC_RETRIEVAL_COUNT` | 2 | `config.py` | ENGRAM-style semantic memory routing without graph complexity. |
| `EPISODIC_RETRIEVAL_COUNT` | 3 | `config.py` | Keeps concrete interaction recall while preserving prompt budget. |
| `SEMANTIC_RETRIEVAL_COUNT + EPISODIC_RETRIEVAL_COUNT` | 5 | `config.py` (derived) | Combined retrieval budget used per interaction. |
| `MAX_CONVERSATION_CHARS` | 100,000 | `config.py` | Conversation history truncation. Oldest messages removed first when exceeded. |

### Validation Parameters

| Constant | Value | Location | Rationale |
|----------|-------|----------|-----------|
| `SNAPSHOT_CHAR_LIMIT` | `SPONGE_MAX_TOKENS × 5` = 2500 | `updater.py` | Maximum character length for snapshots. Approximate token-to-char ratio. |
| `MIN_SNAPSHOT_RETENTION` | 0.6 | `updater.py` | Minimum `len(new) / len(old)` ratio for snapshot validation. Rejects rewrites losing &gt; 40% of content. Open Character Training (2025): losing a trait sentence = losing a trait. |

### Path Constants

| Constant | Value | Location |
|----------|-------|----------|
| `DATA_DIR` | `<project_root>/data` | `config.py` |
| `SPONGE_FILE` | `data/sponge.json` | `config.py` |
| `SPONGE_HISTORY_DIR` | `data/sponge_history` | `config.py` |
| `CHROMADB_DIR` | `data/chromadb` | `config.py` |
| `ESS_AUDIT_LOG_FILE` | `data/ess_log.jsonl` | `config.py` |
| `BASE_URL` | Derived from `SONALITY_API_VARIANT` | `config.py` |

---

For OpenRouter, set:

```bash
SONALITY_API_VARIANT=openrouter
```

---

## Tuning Guide

### ESS Threshold (`SONALITY_ESS_THRESHOLD`)

Controls what percentage of interactions trigger personality updates:

| Threshold | Approximate Update Rate | Use Case |
|-----------|-------------------------|----------|
| 0.1 | ~70% of interactions | Rapid personality formation; risk of noise absorption |
| 0.2 | ~50% of interactions | Balanced for early development |
| **0.3** (default) | ~30% of interactions | Stable operation; only structured arguments pass |
| 0.4 | ~15% of interactions | Conservative; slow personality formation |
| 0.5 | ~5% of interactions | Very conservative; only well-evidenced arguments |

**Recommendation:** Start with 0.3. Lower to 0.2 if personality develops too slowly. Raise to 0.4 if the agent absorbs too many casual opinions.

**Trade-off:** Lower threshold → faster formation, more noise. Higher threshold → slower formation, higher quality updates.

### Bootstrap Dampening (`SONALITY_BOOTSTRAP_DAMPENING_UNTIL`)

Controls how strongly early interactions are attenuated:

| Setting | Effect |
|---------|--------|
| 1 | No dampening (risky: first user dominates personality) |
| **10** (default) | First 10 interactions at 0.5× magnitude |
| 20 | Extended dampening for slower, more cautious development |

**Recommendation:** Keep at 10. Increase to 20 if the first few users are not representative of intended personality development.

**Trade-off:** Lower → faster early formation, first-impression dominance risk. Higher → slower early formation, more balanced trajectory.

### Reflection Interval (`SONALITY_REFLECTION_EVERY`)

Controls how often the personality narrative is consolidated:

| Setting | Effect |
|---------|--------|
| 5 | Very frequent reflection; higher LLM cost, more consolidation opportunities |
| 10 | Moderate reflection |
| **20** (default) | Standard interval; balances cost with personality coherence |
| 50 | Infrequent reflection; insights accumulate longer before consolidation |

**Recommendation:** Start with 20. Lower if insights accumulate without integration (check `pending_insights` count). Raise if reflection produces low-quality outputs (insufficient data to consolidate).

**Trade-off:** Lower → more consolidation, higher cost, risk of over-consolidation. Higher → lower cost, risk of insight backlog, delayed coherence.

### Model Selection

Using a different model for ESS classification and reflection can reduce costs:

```bash
# Use a cheaper model for ESS (classification task, less reasoning needed)
SONALITY_ESS_MODEL=<model-id>  # see .env.example for examples

# Use the main model for response generation only
SONALITY_MODEL=<model-id>      # see .env.example for defaults

# Runtime overrides (without editing .env)
uv run sonality --model "<main-model-id>" --ess-model "<ess-model-id>"
make run ARGS='--model "<main-model-id>" --ess-model "<ess-model-id>"'
```

**Trade-off:** Cheaper ESS models may produce less calibrated scores. Run IBM-ArgQ correlation test (T2.1) after changing the ESS model.

### Belief Decay Rate (`BELIEF_DECAY_RATE`)

Only change if you have evidence. Default 0.15 matches FadeMem (2026) and Ebbinghaus research.

| Value | Effect |
|-------|--------|
| 0.1 | Slower decay; opinions persist longer |
| **0.15** (default) | Standard power-law decay |
| 0.2 | Faster decay; more aggressive forgetting |

### Opinion Cooling (`SONALITY_OPINION_COOLING_PERIOD`)

Controls how many interactions a belief delta is staged before commit:

| Setting | Effect |
|---------|--------|
| 1 | Near-immediate commits (most reactive) |
| **3** (default) | Balanced responsiveness and anti-sycophancy buffering |
| 4-5 | Stronger resistance to short social pressure bursts |

**Recommendation:** keep `3` unless your runs show either over-reactivity (increase) or under-adaptation (decrease).

---

## Data Paths

Local runtime data is stored under `data/` (gitignored). Database state lives in Neo4j and PostgreSQL:

```
data/
├── sponge.json          # Current personality state
├── sponge_history/      # Archived versions (sponge_v0.json, sponge_v1.json, ...)
├── chromadb/            # Legacy episode vector store (fallback)
└── ess_log.jsonl        # Audit trail (ESS events + reflection events)

Neo4j:
├── EpisodeNode          # Episode graph nodes
├── DerivativeNode       # Semantic chunk nodes
└── Edges                # TEMPORAL, THEMATIC, SUPPORTED_BY, CONSOLIDATED_FROM

PostgreSQL:
├── derivative_embeddings  # pgvector embeddings for derivative chunks
├── stm_state              # Short-Term Memory buffer + running summary
└── semantic_features      # Extracted structured features
```

### Reset Commands

```bash
make reset   # Reset sponge to seed state
make nuke    # Full reset (remove .venv, data, caches)
```

### Rollback

```bash
cp data/sponge_history/sponge_v14.json data/sponge.json
```

---

**Related:** [Architecture Overview — Context Window Budget](architecture/overview.md#context-window-budget) — how these parameters translate into token usage. [Personality Development — Health Monitoring](personality-development.md#health-monitoring-signals) — what healthy parameter ranges look like in practice.
