# Project Structure

```
sonality/
├── sonality/              # Core package
│   ├── agent.py           # Orchestration (~590 lines)
│   ├── api.py             # FastAPI server
│   ├── ess.py             # ESS classifier (~520 lines)
│   ├── prompts.py         # All prompts (~730 lines)
│   ├── provider.py        # LLM abstraction
│   ├── llm/caller.py      # Structured LLM calls
│   └── memory/
│       ├── graph.py       # Neo4j ops (~730 lines)
│       ├── dual_store.py  # Atomic storage
│       ├── semantic_features.py
│       ├── knowledge_extract.py
│       ├── forgetting.py
│       └── retrieval/     # router, chain, split, reranker
├── chat/                  # Clients
│   ├── client.py          # SonalityClient
│   ├── terminal.py        # REPL
│   └── telegram.py        # Telegram + audio
├── scripts/               # Feed ingestion (RSS, X)
├── tests/                 # pytest suite
├── benches/               # Benchmarks
└── docs/                  # Documentation
```

## Dependency Graph

```mermaid
flowchart LR
    api --> agent
    agent --> ess & prompts
    agent --> dual_store & router
    dual_store --> graph & embedder
    router --> chain & split & reranker
```

## Key Patterns

| Pattern | Location |
|---------|----------|
| Structured LLM calls | `llm/caller.py` |
| Dual-write atomicity | `dual_store.py` |
| Background workers | `semantic_features.py` |
| Centralized prompts | `prompts.py` |
