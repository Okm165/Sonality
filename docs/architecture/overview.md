# Architecture Overview

> **Core Module**: `sonality/agent.py`

LLM agent with self-evolving personality. Stateless per-request — identity loaded from Neo4j each time.

## System Diagram

```mermaid
flowchart LR
    Sources[Chat/Feeds] --> API[FastAPI]
    API --> Agent[SonalityAgent]
    Agent --> Neo4j[(Neo4j)]
    Agent --> Qdrant[(Qdrant)]
    Agent --> Background[SemanticWorker]
```

## Agent Components

```
SonalityAgent
├── MemoryGraph (Neo4j)
├── DualEpisodeStore (Neo4j + Qdrant)
├── Embedder (FastEmbed)
├── BoundaryDetector
└── SemanticWorker (background)
```

## Response Pipeline

```mermaid
flowchart LR
    subgraph Build
        Load[Load identity]
        Trim[Trim context]
    end
    Build --> Loop
    subgraph Loop[Agentic Loop]
        LLM[LLM decides] --> Tools[recall / search / assess / reflect / store / consolidate]
        Tools --> LLM
    end
    Loop --> Bookkeep
    subgraph Bookkeep
        ESS[ESS classify]
        Store[Store episode]
        Prov[Provenance]
    end
```

1. Load personality snapshot + beliefs
2. Trim conversation history to token budget
3. Agentic loop: LLM autonomously calls tools (recall, search, assess, reflect, store, consolidate)
4. Bookkeeping: ESS classify, boundary detection, store episode, provenance, semantic features, forgetting

The agent handles all cognitive work via tools. Bookkeeping runs automatically and silently.

## Retrieval Strategies

| Category | Strategy |
|----------|----------|
| SIMPLE | Vector + topic search |
| TEMPORAL | Chain with sufficiency |
| MULTI_ENTITY | Query decomposition |
| BELIEF_QUERY | Belief edges + topic + vector |

## Ingestion (non-conversational)

```
ESS classify → if update_recommended → store → provenance → semantic features → forgetting
```

## Error Handling

| Component | Fallback |
|-----------|----------|
| LLM | 3 retries → empty |
| ESS | `classifier_exception_fallback` |
| Episode storage | Log and continue |
