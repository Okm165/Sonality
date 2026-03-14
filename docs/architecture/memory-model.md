## Memory Model

Sonality memory has two persistent layers and one narrative layer:

- Graph memory (`Neo4j`): causal and temporal structure
- Vector memory (`PostgreSQL + pgvector`): semantic retrieval
- Personality state (`SpongeState`): compact evolving narrative and beliefs

## Graph Memory

Main node types:

- `Episode`
- `Derivative`
- `Topic`
- `Belief`
- `Segment`
- `Summary`

Main edges:

- `DERIVED_FROM`
- `TEMPORAL_NEXT`
- `DISCUSSES`
- `SUPPORTS_BELIEF`
- `CONTRADICTS_BELIEF`
- `BELONGS_TO_SEGMENT`
- `CONSOLIDATES`

This layer supports provenance, temporal traversal, and belief-aware retrieval.

## Vector Memory

`derivatives` table stores derivative embeddings for fast semantic search.

`semantic_features` table stores extracted long-lived features:

- personality
- preferences
- knowledge
- relationships

This layer supports retrieval quality and context enrichment.

## Personality State

`SpongeState` stores:

- narrative snapshot
- topic opinion vectors
- belief metadata (confidence/uncertainty/provenance)
- staged updates
- pending insights
- recent shifts

Belief confidence follows LLM-assessed uncertainty during updates; legacy
evidence-count confidence formulas are removed.

## Consolidation and Forgetting

- Consolidation creates summaries from eligible segments.
- Forgetting archives low-value raw episodes while preserving recoverable graph
  structure and metadata.

Both operations are LLM-guided and run in reflection cycles.
