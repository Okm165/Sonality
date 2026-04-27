# Advanced Retrieval Pipeline

> **Deep-Dive Documentation**: LLM-powered retrieval strategies including reranking, iterative chaining, and query decomposition.

## Overview

The Sonality retrieval system uses three advanced strategies beyond basic vector search:

| Strategy | Module | Purpose |
|----------|--------|---------|
| **Reranker** | `retrieval/reranker.py` | LLM-based relevance ranking with cross-document reasoning |
| **Chain Retrieval** | `retrieval/chain.py` | Iterative search with sufficiency checking |
| **Split Retrieval** | `retrieval/split.py` | Query decomposition with parallel execution |

## System Architecture

```mermaid
flowchart TB
    subgraph Input["Query Input"]
        Q["User Query"]
    end
    
    subgraph Router["Query Router"]
        Classify["Classify query type"]
        Depth["Determine depth"]
    end
    
    subgraph Strategies["Retrieval Strategies"]
        Split["Split Retrieval"]
        Chain["Chain Retrieval"]
        Direct["Direct Search"]
    end
    
    subgraph Rerank["Reranking"]
        LLMRank["LLM Listwise Reranker"]
    end
    
    subgraph Output["Results"]
        Episodes["Ranked Episodes"]
    end
    
    Q --> Router
    Router --> Strategies
    Split --> Rerank
    Chain --> Rerank
    Direct --> Rerank
    Rerank --> Episodes
```

---

## LLM Listwise Reranker

`sonality/memory/retrieval/reranker.py`

Replaces formula-based utility scoring with LLM reasoning for relevance ranking.

### Core Function

```python
def rerank_episodes(
    query: str,
    candidates: list[EpisodeNode],
    *,
    top_k: int = 0,
) -> list[EpisodeNode]:
    """Rerank candidate episodes using LLM Listwise approach.

    Parameters
    ----------
    query:
        The original search query.
    candidates:
        Episodes to rank (max ~25 for context efficiency).
    top_k:
        Number of top results to return. 0 means return all.

    Returns
    -------
    Episodes in LLM-determined relevance order.
    """
```

### Reranking Flow

```mermaid
sequenceDiagram
    participant Caller
    participant Reranker
    participant LLM
    
    Caller->>Reranker: rerank_episodes(query, candidates)
    
    Note over Reranker: Limit to MAX_RERANK_CANDIDATES
    
    Reranker->>Reranker: Format numbered candidates
    Note right of Reranker: [1] (2024-01-15) Summary...<br/>[2] (2024-01-10) Summary...
    
    Reranker->>LLM: RERANK_PROMPT(query, numbered_candidates)
    LLM-->>Reranker: {"ranking": [3, 1, 5, 2, 4]}
    
    Note over Reranker: Map 1-indexed to episodes
    Note over Reranker: Handle missing indices
    
    Reranker-->>Caller: Reordered episodes[:top_k]
```

### Response Model

```python
class _RerankResponse(BaseModel):
    ranking: list[int]  # 1-indexed relevance order
```

### Robustness Handling

```python
# Map 1-indexed ranking to 0-indexed episodes
reranked: list[EpisodeNode] = []
seen: set[int] = set()
for idx in ranking:
    zero_idx = idx - 1
    if 0 <= zero_idx < len(to_rank) and zero_idx not in seen:
        reranked.append(to_rank[zero_idx])
        seen.add(zero_idx)

# Add any candidates not in the ranking (LLM might skip some)
for i, ep in enumerate(to_rank):
    if i not in seen:
        reranked.append(ep)
```

### Configuration

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Max candidates | `MAX_RERANK_CANDIDATES` | 25 | Limit for context efficiency |
| Max tokens | `LLM_MAX_TOKENS` | 1024 | Response token limit |

---

## Chain Retrieval (Iterative Sufficiency)

`sonality/memory/retrieval/chain.py`

Iteratively searches and refines queries until sufficient results are found.

### Core Function

```python
async def chain_retrieve(
    store: DualEpisodeStore, 
    graph: MemoryGraph, 
    query: str, 
    base_n: int = 10,
) -> list[EpisodeNode]:
    """Iteratively search and refine until sufficient results found."""
```

### Iteration Flow

```mermaid
flowchart TB
    Start["Start with query"]
    Search["Vector search (top_k=base_n)"]
    NewUIDs{"New UIDs found?"}
    FetchEpisodes["Fetch episodes from graph"]
    BuildContext["Build context from all episodes"]
    LLMCheck["LLM sufficiency check"]
    Sufficient{"Sufficient + confident?"}
    Refine["Use suggested refinement"]
    MaxIter{"Max iterations?"}
    Return["Return all_episodes"]
    
    Start --> Search
    Search --> NewUIDs
    NewUIDs -->|Yes| FetchEpisodes
    NewUIDs -->|No, iter>1| Return
    FetchEpisodes --> BuildContext
    BuildContext --> LLMCheck
    LLMCheck --> Sufficient
    Sufficient -->|Yes| Return
    Sufficient -->|No| Refine
    Refine --> MaxIter
    MaxIter -->|No| Search
    MaxIter -->|Yes| Return
```

### Sufficiency Response Model

```python
class _SufficiencyDecision(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class _SufficiencyResponse(BaseModel):
    sufficiency_decision: _SufficiencyDecision = _SufficiencyDecision.INSUFFICIENT
    confidence: float = 0.0
    reasoning: str = ""
    suggested_refinement: str = ""
```

### Termination Conditions

```python
# Success: Sufficient with high confidence
if (
    sufficiency.sufficiency_decision is _SufficiencyDecision.SUFFICIENT
    and sufficiency.confidence >= threshold
):
    return all_episodes

# Failure: No new results in later iterations
if not new_uids and iteration > 1:
    break

# Failure: No suggested refinement
if not sufficiency.suggested_refinement:
    break

# Failure: Max iterations reached
if iteration >= max_iter:
    break
```

### Configuration

| Parameter | Config Key | Default | Description |
|-----------|------------|---------|-------------|
| Max iterations | `RETRIEVAL_MAX_ITERATIONS` | 3 | Iteration limit |
| Confidence threshold | `RETRIEVAL_CONFIDENCE_THRESHOLD` | 0.7 | Sufficiency confidence cutoff |

---

## Split Retrieval (Query Decomposition)

`sonality/memory/retrieval/split.py`

Decomposes multi-entity or comparison queries into parallel sub-queries.

### Core Function

```python
async def split_retrieve(
    store: DualEpisodeStore, 
    graph: MemoryGraph, 
    query: str, 
    n_per_sub: int = 10,
) -> list[EpisodeNode]:
    """Decompose query, execute sub-queries in parallel, aggregate."""
```

### Decomposition Response Model

```python
class _AggregationStrategy(StrEnum):
    MERGE = "merge"       # Combine all results, dedupe
    COMPARE = "compare"   # Interleave for side-by-side comparison
    TIMELINE = "timeline" # Sort by creation time


class _DecompositionResponse(BaseModel):
    sub_queries: list[str]
    aggregation_strategy: _AggregationStrategy = _AggregationStrategy.MERGE
```

### Decomposition Flow

```mermaid
sequenceDiagram
    participant Caller
    participant Split
    participant LLM
    participant Store
    participant Graph
    
    Caller->>Split: split_retrieve(query)
    Split->>LLM: DECOMPOSITION_PROMPT(query)
    LLM-->>Split: {sub_queries: [...], aggregation_strategy: "compare"}
    
    par Parallel Execution
        Split->>Store: vector_search(sub_query_1)
        Store-->>Split: results_1
        Split->>Graph: get_episodes(uids_1)
        Graph-->>Split: episodes_1
    and
        Split->>Store: vector_search(sub_query_2)
        Store-->>Split: results_2
        Split->>Graph: get_episodes(uids_2)
        Graph-->>Split: episodes_2
    end
    
    Split->>Split: _aggregate(results, strategy)
    Split-->>Caller: aggregated_episodes
```

### Aggregation Strategies

```python
def _aggregate(sub_results: list[list[EpisodeNode]], strategy: _AggregationStrategy) -> list[EpisodeNode]:
    if strategy is _AggregationStrategy.COMPARE:
        # Interleave for side-by-side comparison
        interleaved: list[EpisodeNode] = []
        max_len = max((len(batch) for batch in sub_results), default=0)
        for index in range(max_len):
            for batch in sub_results:
                if index < len(batch):
                    interleaved.append(batch[index])
        return _dedupe(interleaved)
    
    if strategy is _AggregationStrategy.TIMELINE:
        # Sort by creation time
        return sorted(
            _dedupe([ep for batch in sub_results for ep in batch]),
            key=lambda episode: episode.created_at
        )
    
    # MERGE: Simple concatenation with deduplication
    return _dedupe([ep for batch in sub_results for ep in batch])
```

### Aggregation Visualization

```mermaid
graph TB
    subgraph Input["Sub-Query Results"]
        SQ1["Sub-Query 1: [A, B, C]"]
        SQ2["Sub-Query 2: [D, B, E]"]
        SQ3["Sub-Query 3: [F, A, G]"]
    end
    
    subgraph Merge["MERGE Strategy"]
        M["[A, B, C, D, E, F, G]"]
        Note1["Deduplicated concat"]
    end
    
    subgraph Compare["COMPARE Strategy"]
        C["[A, D, F, B, B, A, C, E, G]"]
        Note2["Interleaved"]
    end
    
    subgraph Timeline["TIMELINE Strategy"]
        T["[A, D, B, C, F, E, G]"]
        Note3["Sorted by created_at"]
    end
    
    Input --> Merge
    Input --> Compare
    Input --> Timeline
```

### Parallel Execution Control

```python
# Limit concurrent sub-queries
sem = asyncio.Semaphore(4)

async def search_one(sq: str) -> list[EpisodeNode]:
    async with sem:
        try:
            results = await store.vector_search(sq, top_k=n_per_sub)
            uids = list({h.episode_uid for h in results})
            return await graph.get_episodes(uids)
        except Exception:
            log.exception("Sub-query failed: %s", sq[:60])
            return []

sub_results = await asyncio.gather(*(search_one(sq) for sq in sub_queries))
```

---

## Strategy Selection

The query router (`retrieval/router.py`) determines which strategy to use:

```mermaid
flowchart TB
    Query["Input Query"]
    
    Router["Query Router"]
    
    Temporal{"Temporal query?"}
    MultiEntity{"Multi-entity comparison?"}
    Complex{"Complex/uncertain?"}
    
    Chain["Chain Retrieval"]
    Split["Split Retrieval"]
    Direct["Direct + Rerank"]
    
    Query --> Router
    Router --> Temporal
    Temporal -->|Yes| Chain
    Temporal -->|No| MultiEntity
    MultiEntity -->|Yes| Split
    MultiEntity -->|No| Complex
    Complex -->|Yes| Chain
    Complex -->|No| Direct
```

### Strategy Characteristics

| Strategy | Best For | Latency | Token Cost |
|----------|----------|---------|------------|
| **Direct + Rerank** | Simple factual queries | Low | Medium |
| **Chain Retrieval** | Uncertain/exploratory queries | High | High |
| **Split Retrieval** | Comparison/multi-entity queries | Medium | Medium-High |

---

## Integration Example

```python
from sonality.memory.retrieval.router import route_query
from sonality.memory.retrieval.chain import chain_retrieve
from sonality.memory.retrieval.split import split_retrieve
from sonality.memory.retrieval.reranker import rerank_episodes

async def retrieve_with_strategy(
    store: DualEpisodeStore,
    graph: MemoryGraph,
    query: str,
) -> list[EpisodeNode]:
    """Execute appropriate retrieval strategy based on query classification."""
    
    decision = route_query(query)
    
    if decision.category is QueryCategory.TEMPORAL:
        episodes = await chain_retrieve(store, graph, query)
    elif len(query.split(" vs ")) > 1 or "compare" in query.lower():
        episodes = await split_retrieve(store, graph, query)
    else:
        results = await store.vector_search(query, top_k=decision.n_results)
        uids = [h.episode_uid for h in results]
        episodes = await graph.get_episodes(uids)
    
    # Always rerank for final ordering
    return rerank_episodes(query, episodes, top_k=10)
```

---

## Performance Considerations

### Latency Optimization

```mermaid
graph LR
    subgraph Fast["Fast Path (~100ms)"]
        V["Vector Search"]
        R["Rerank (1 LLM call)"]
    end
    
    subgraph Medium["Medium Path (~300ms)"]
        D["Decomposition"]
        P["Parallel Search"]
        A["Aggregation"]
        R2["Rerank"]
    end
    
    subgraph Slow["Slow Path (~500ms+)"]
        I1["Iteration 1: Search + Sufficiency"]
        I2["Iteration 2: Refined Search"]
        I3["Iteration 3: Final Check"]
    end
    
    Fast --> Medium --> Slow
```

### Token Budget

| Component | Tokens (Input) | Tokens (Output) |
|-----------|----------------|-----------------|
| Decomposition | ~200 | ~100 |
| Sufficiency Check | ~500/iter | ~100/iter |
| Reranking | ~1000 (25 candidates) | ~50 |

---

## Related Documentation

- [Data Pipeline Trace](data-pipeline-trace.md) — End-to-end retrieval flow
- [Embedder & Consolidation](embedder-consolidation.md) — Vector embedding
- [Database Connections](database-connections.md) — Qdrant integration
- [Agent Core](agent-core.md) — Retrieval integration in respond pipeline
