# Configuration

All configuration is via environment variables (set in `.env`) and compile-time constants. One `.env.example`, one source of truth. Values are defined in `sonality/config.py` and `sonality/memory/updater.py`.

---

## Environment Variables

Set these in your `.env` file (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `SONALITY_API_KEY` | *(required)* | API key for the LLM provider |
| `SONALITY_MODEL` | *(see .env.example)* | Main reasoning model for response generation |
| `SONALITY_ESS_MODEL` | Same as `SONALITY_MODEL` | Model for ESS classification, insight extraction, and reflection |
| `SONALITY_ESS_THRESHOLD` | `0.3` | Minimum ESS score to trigger personality updates |
| `SONALITY_LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `SONALITY_BOOTSTRAP_DAMPENING_UNTIL` | `10` | First N interactions with 0.5× opinion update dampening |
| `SONALITY_OPINION_COOLING_PERIOD` | `3` | Interactions before staged opinion deltas are committed |
| `SONALITY_SEMANTIC_RETRIEVAL_COUNT` | `2` | Semantic memories retrieved each turn |
| `SONALITY_EPISODIC_RETRIEVAL_COUNT` | `3` | Episodic memories retrieved each turn |
| `SONALITY_REFLECTION_EVERY` | `20` | Periodic reflection interval (interactions) |

---

## Compile-Time Constants

These are defined in `sonality/config.py` and `sonality/memory/updater.py`. Not configurable via environment.

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
| `EPISODE_RETRIEVAL_COUNT` | 5 | `config.py` | Episodes retrieved per interaction. Balances context richness with prompt length. |
| `SEMANTIC_RETRIEVAL_COUNT` | 2 | `config.py` | ENGRAM-style semantic memory routing without graph complexity. |
| `EPISODIC_RETRIEVAL_COUNT` | 3 | `config.py` | Keeps concrete interaction recall while preserving prompt budget. |
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

All runtime data is stored under `data/` (gitignored):

```
data/
├── sponge.json          # Current personality state
├── sponge_history/      # Archived versions (sponge_v0.json, sponge_v1.json, ...)
├── chromadb/            # Episode vector store (SQLite + embeddings)
└── ess_log.jsonl        # Audit trail (ESS events + reflection events)
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
