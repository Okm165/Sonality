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
│   │   ├── memory.py      # recall_memory, store_knowledge
│   │   ├── assess.py      # assess_evidence
│   │   ├── reflect.py     # reflect tool + belief graph updates + forgetting
│   │   └── consolidate.py # consolidate (mid-loop synthesis)
│   ├── web/               # Web I/O (search client, formatting)
│   │   ├── search.py      # WebSearchClient
│   │   └── context.py     # sanitize, format web content
│   └── memory/
│       ├── graph.py       # Neo4j ops
│       ├── dual_store.py  # Atomic storage
│       ├── semantic_features.py
│       ├── knowledge_extract.py
│       ├── forgetting.py
│       └── retrieval/     # router, chain, split, reranker
├── chat/                  # Clients
│   ├── client.py          # SonalityClient
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
