# Project Structure

```
sonality/
├── sonality/              # Core package
│   ├── agent.py           # Orchestration + agentic loop
│   ├── api.py             # FastAPI server
│   ├── ess.py             # ESS classifier
│   ├── prompts.py         # All prompt templates
│   ├── provider.py        # LLM abstraction
│   ├── progress.py        # AgentEvent and event constants
│   ├── llm/caller.py      # Structured LLM calls
│   ├── tools/             # Symmetric tool system
│   │   ├── __init__.py    # ToolContext, dispatch_tool, get_definitions
│   │   ├── web.py         # web_search, web_extract
│   │   ├── memory.py      # recall_memory, integrate_knowledge
│   │   ├── synthesize.py  # synthesize (evaluate + structure research)
│   │   └── reflect.py     # internal reflection logic + belief graph updates + forgetting
│   ├── web/               # Web I/O (search client, formatting)
│   │   ├── search.py      # WebSearchClient
│   │   └── context.py     # sanitize, format web content
│   └── memory/
│       ├── graph.py              # Neo4j ops (episodes, beliefs, snapshots)
│       ├── dual_store.py         # Atomic Neo4j + Qdrant episode storage
│       ├── belief_provenance.py  # Evidence assessment + provenance tracking
│       ├── consolidation.py      # LLM-based segment consolidation (HEMA)
│       ├── derivatives.py        # Chunking + embedding for episodes
│       ├── segmentation.py       # Event boundary detection
│       ├── semantic_features.py  # Feature extraction + ingestion worker
│       ├── knowledge_extract.py  # Knowledge proposition extraction
│       ├── forgetting.py         # LLM-based memory forgetting
│       └── retrieval/            # router, chain, split, reranker
├── chat/                  # Clients
│   ├── __init__.py        # Package exports
│   ├── __main__.py        # CLI dispatcher (chat / telegram)
│   ├── audio.py           # STT/TTS via Speaches API
│   ├── client.py          # SonalityClient (streaming HTTP)
│   ├── config.py          # Chat-specific env config
│   ├── terminal.py        # Rich TUI REPL
│   └── telegram.py        # Telegram + voice + sendMessageDraft streaming
├── scripts/               # Feed ingestion (RSS, X)
├── tests/                 # pytest suite
├── benches/               # Benchmarks
└── docs/                  # Documentation
```

## Dependency Graph

```mermaid
flowchart LR
    api --> agent
    agent --> tools & ess & prompts
    tools --> web_io[web/] & memory & llm
    agent --> dual_store & router
    dual_store --> graph & embedder
    router --> chain & split & reranker
```

## Key Patterns

| Pattern | Location |
|---------|----------|
| Symmetric tool system | `tools/` (each module: DEFINITIONS + EXECUTORS) |
| Structured LLM calls | `llm/caller.py` |
| Dual-write atomicity | `dual_store.py` |
| Background workers | `semantic_features.py` |
| Centralized prompts | `prompts.py` |
