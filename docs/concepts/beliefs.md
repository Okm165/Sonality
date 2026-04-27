# Beliefs

## Schema (Neo4j)

```python
class BeliefNode:
    topic: str           # Unique identifier
    valence: float       # -1 (oppose) to +1 (support)
    confidence: float    # 0 to 1
    evidence_count: int  # Accumulated citations
    belief_text: str     # Natural language summary
```

## Update Pipeline

```mermaid
flowchart LR
    Episode --> Prov[Provenance Assessment]
    Prov -->|direction > 0| S[SUPPORTS_BELIEF]
    Prov -->|direction < 0| C[CONTRADICTS_BELIEF]
    S & C --> Triage[Reflection Triage]
    Triage -->|should_reflect| Deep[Deep Reflection]
    Deep --> Update[Update valence/confidence]
```

**Two phases:**
1. **Provenance** — Creates edges linking episodes to beliefs (audit trail)
2. **Reflection** — LLM evaluates accumulated evidence, updates values

## Edge Properties

```cypher
(Episode)-[:SUPPORTS_BELIEF {strength: 0.75, reasoning: "..."}]->(Belief)
```

## ESS Quality Gating

| ESS Score | Impact |
|-----------|--------|
| 0.0–0.3 | Unlikely to change beliefs |
| 0.3–0.6 | May influence with reasoning |
| 0.6–1.0 | Strong evidence for updates |

## Anti-Manipulation

| Protection | Mechanism |
|------------|-----------|
| ESS gating | Requires `score >= 0.25` |
| Edge-only provenance | Values only change during reflection |
| `debunked_claim` type | Score capped at 0.07 |

## API

```
GET /beliefs           # All beliefs sorted by |valence|
GET /beliefs/{topic}   # Single belief
```

## Prompt Format

```
climate policy: +0.72 (confidence: 0.85), evidence: 12
cryptocurrency: -0.45 (confidence: 0.60), evidence: 5
```
