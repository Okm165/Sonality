# System Architecture

This document provides a comprehensive top-down view of the entire Sonality system, tying together all components, data flows, and subsystems.

## High-Level Architecture

```mermaid
flowchart TB
    subgraph external["External World"]
        User[User]
        RSS[RSS Feeds]
        GNews[GNews API]
        XApi[X API v2]
    end
    
    subgraph interfaces["Interface Layer"]
        Terminal[Terminal TUI]
        Telegram[Telegram Bot]
        API[FastAPI Server]
        FeedScripts[Feed Scripts]
    end
    
    subgraph core["Core Agent"]
        Agent[SonalityAgent]
        ESS[ESS Classifier]
        Provider[LLM Provider]
        Caller[LLM Caller]
    end
    
    subgraph memory["Memory System"]
        DualStore[DualEpisodeStore]
        Graph[MemoryGraph]
        Retrieval[Retrieval Pipeline]
        Knowledge[Knowledge Extraction]
    end
    
    subgraph storage["Storage Layer"]
        Neo4j[(Neo4j Graph)]
        Qdrant[(Qdrant Vectors)]
    end
    
    subgraph background["Background Workers"]
        SemanticWorker[Semantic Feature Worker]
        Forgetting[Forgetting System]
        Consolidation[Segment Consolidation]
    end
    
    User --> Terminal
    User --> Telegram
    RSS --> FeedScripts
    GNews --> FeedScripts
    XApi --> FeedScripts
    
    Terminal --> API
    Telegram --> API
    FeedScripts --> API
    
    API --> Agent
    Agent --> ESS
    Agent --> Provider
    ESS --> Caller
    Retrieval --> Caller
    
    Agent --> DualStore
    Agent --> Graph
    Agent --> Retrieval
    Agent --> Knowledge
    
    DualStore --> Neo4j
    DualStore --> Qdrant
    Graph --> Neo4j
    Knowledge --> Qdrant
    
    Agent --> SemanticWorker
    SemanticWorker --> Qdrant
    
    Graph --> Forgetting
    DualStore --> Forgetting
    
    Graph --> Consolidation
```

## Component Inventory

### Entry Points

| Component | Location | Purpose |
|-----------|----------|---------|
| `sonality-server` | `sonality/api.py:serve()` | FastAPI HTTP server |
| `sonality` | `sonality/cli.py:main()` | Interactive REPL |
| `chat/terminal.py` | `chat/terminal.py:main()` | Rich TUI client |
| `chat/telegram.py` | `chat/telegram.py:main()` | Telegram bot |
| `scripts/feed.py` | `scripts/feed.py:main()` | RSS/GNews ingestion |
| `scripts/x_feed.py` | `scripts/x_feed.py:main()` | X/Twitter ingestion |

### Core Modules

| Module | Location | Lines | Responsibility |
|--------|----------|-------|----------------|
| `agent.py` | `sonality/` | ~600 | Central orchestration |
| `ess.py` | `sonality/` | ~280 | Evidence classification |
| `provider.py` | `sonality/` | ~470 | LLM abstraction |
| `prompts.py` | `sonality/` | ~730 | All prompt templates |
| `schema.py` | `sonality/` | ~154 | Database schemas |
| `config.py` | `sonality/` | ~85 | Configuration |

### Memory Subsystem

| Module | Location | Lines | Responsibility |
|--------|----------|-------|----------------|
| `dual_store.py` | `memory/` | ~300 | Atomic Neo4j+Qdrant ops |
| `graph.py` | `memory/` | ~500 | Neo4j graph operations |
| `embedder.py` | `memory/` | ~67 | FastEmbed wrapper |
| `derivatives.py` | `memory/` | ~93 | Semantic chunking |
| `knowledge_extract.py` | `memory/` | ~500 | SLIDE extraction |
| `belief_provenance.py` | `memory/` | ~200 | Belief evidence linking |
| `semantic_features.py` | `memory/` | ~400 | Personality extraction |
| `segmentation.py` | `memory/` | ~119 | Boundary detection |
| `consolidation.py` | `memory/` | ~123 | Segment summarization |
| `forgetting.py` | `memory/` | ~117 | Archive/delete decisions |
| `db.py` | `memory/` | ~67 | Database connections |

### Retrieval Subsystem

| Module | Location | Lines | Responsibility |
|--------|----------|-------|----------------|
| `router.py` | `memory/retrieval/` | ~150 | Query classification |
| `chain.py` | `memory/retrieval/` | ~100 | Iterative retrieval |
| `split.py` | `memory/retrieval/` | ~109 | Parallel sub-queries |
| `reranker.py` | `memory/retrieval/` | ~102 | LLM listwise ranking |

### LLM Subsystem

| Module | Location | Lines | Responsibility |
|--------|----------|-------|----------------|
| `caller.py` | `llm/` | ~251 | Structured calls + repair |

## Data Flow Diagrams

### Conversation Flow

```mermaid
sequenceDiagram
    participant User
    participant API
    participant Agent
    participant ESS
    participant Retrieval
    participant LLM
    participant Memory

    User->>API: POST /chat {message}
    API->>Agent: respond(messages)
    
    Agent->>ESS: classify(message)
    ESS->>LLM: ESS_CLASSIFICATION_PROMPT
    LLM-->>ESS: ESSResult
    ESS-->>Agent: score, type, topics
    
    Agent->>Retrieval: route_query(message)
    Retrieval->>LLM: QUERY_ROUTING_PROMPT
    LLM-->>Retrieval: RoutingDecision
    
    Retrieval->>Memory: vector_search(query)
    Memory-->>Retrieval: episodes
    
    Retrieval->>LLM: RERANK_PROMPT
    LLM-->>Retrieval: ranked_episodes
    Retrieval-->>Agent: context
    
    Agent->>Agent: build_system_prompt()
    Agent->>LLM: chat_completion(prompt)
    LLM-->>Agent: response
    
    Agent->>Memory: store_episode()
    Agent->>Memory: update_beliefs()
    
    Agent-->>API: response_text
    API-->>User: {response, ess_score, topics}
```

### Ingestion Flow

```mermaid
sequenceDiagram
    participant Feed as Feed Script
    participant API
    participant Agent
    participant ESS
    participant Knowledge
    participant Memory

    Feed->>Feed: Fetch articles/posts
    Feed->>Feed: Quality gate
    Feed->>Feed: Enrich text
    
    Feed->>API: POST /ingest {text, topic}
    API->>Agent: ingest(text)
    
    Agent->>ESS: classify(text)
    ESS-->>Agent: ESSResult
    
    alt belief_update_recommended
        Agent->>Memory: store_episode()
        
        opt knowledge_density > NONE
            Agent->>Knowledge: extract_and_store()
            Knowledge->>Knowledge: split_windows()
            Knowledge->>Knowledge: extract_propositions()
            Knowledge->>Knowledge: deduplicate()
            Knowledge->>Memory: persist_proposition()
        end
        
        Agent->>Memory: update_belief_provenance()
    end
    
    Agent-->>API: IngestResponse
    API-->>Feed: {score, reasoning_type, topics}
```

### Background Processing

```mermaid
flowchart TD
    subgraph triggers["Triggers"]
        Interaction[User Interaction]
        Timer[Periodic Timer]
        Threshold[Episode Threshold]
    end
    
    subgraph semantic["Semantic Feature Worker"]
        Queue[Deferred Queue]
        Extract[Feature Extraction]
        Consolidate[Consolidation]
    end
    
    subgraph reflection["Reflection System"]
        Triage[Reflection Triage]
        Deep[Deep Reflection]
        BeliefUpdate[Belief Updates]
        SnapshotUpdate[Snapshot Update]
    end
    
    subgraph cleanup["Cleanup System"]
        Forgetting[Forgetting Assessment]
        Archive[Archive Low-Utility]
        Delete[Delete Redundant]
    end
    
    Interaction --> Queue
    Queue -->|LLM idle| Extract
    Extract --> Consolidate
    Consolidate --> Qdrant[(semantic_features)]
    
    Interaction --> Triage
    Triage -->|should_reflect| Deep
    Deep --> BeliefUpdate
    Deep --> SnapshotUpdate
    BeliefUpdate --> Neo4j[(Neo4j)]
    SnapshotUpdate --> Neo4j
    
    Threshold --> Forgetting
    Forgetting --> Archive
    Forgetting --> Delete
    Archive --> Neo4j
    Delete --> Qdrant
```

## State Management

### Agent State

```mermaid
stateDiagram-v2
    [*] --> Initialized: SonalityAgent()
    Initialized --> Connected: connect_databases()
    Connected --> Ready: load_snapshot()
    
    Ready --> Processing: respond()/ingest()
    Processing --> PostProcess: _post_process()
    PostProcess --> MaybeReflect: check_reflection()
    MaybeReflect --> Ready: done
    MaybeReflect --> Reflecting: should_reflect
    Reflecting --> Ready: reflection_complete
    
    Ready --> Shutdown: shutdown()
    Shutdown --> [*]: close_connections()
```

### Memory State

```mermaid
stateDiagram-v2
    state Episode {
        [*] --> Created: store_episode()
        Created --> HasDerivatives: chunk_and_embed()
        HasDerivatives --> HasProvenance: assess_belief_evidence()
        HasProvenance --> Active
        Active --> Archived: low_utility
        Active --> Deleted: redundant
    }
    
    state Belief {
        [*] --> Formed: new_belief
        Formed --> Updated: evidence_received
        Updated --> Updated: more_evidence
        Updated --> Decayed: reflection_decay
    }
    
    state Segment {
        [*] --> Open: boundary_detected
        Open --> Closed: new_boundary
        Closed --> Consolidated: ready_for_consolidation
    }
```

## Configuration Hierarchy

```mermaid
flowchart TD
    subgraph env["Environment"]
        DotEnv[.env file]
        EnvVars[Environment Variables]
    end
    
    subgraph config["Configuration Modules"]
        SonalityConfig[sonality/config.py]
        ChatConfig[chat/config.py]
    end
    
    subgraph components["Components"]
        Agent[SonalityAgent]
        API[FastAPI]
        Provider[LLMProvider]
        Memory[Memory System]
        Chat[Chat Clients]
    end
    
    DotEnv --> EnvVars
    EnvVars --> SonalityConfig
    EnvVars --> ChatConfig
    
    SonalityConfig --> Agent
    SonalityConfig --> API
    SonalityConfig --> Provider
    SonalityConfig --> Memory
    ChatConfig --> Chat
```

### Configuration Variables

| Category | Key Variables |
|----------|---------------|
| **LLM** | `SONALITY_BASE_URL`, `SONALITY_MODEL`, `SONALITY_API_KEY` |
| **Database** | `SONALITY_NEO4J_URL`, `SONALITY_QDRANT_URL` |
| **Retrieval** | `SONALITY_RETRIEVAL_MAX_ITERATIONS`, `MAX_RERANK_CANDIDATES` |
| **Timeouts** | `SONALITY_LLM_TIMEOUT`, `SONALITY_ASYNC_TIMEOUT` |
| **Chat** | `CHAT_TELEGRAM_TOKEN`, `CHAT_SPEACHES_URL` |
| **STT/TTS** | `CHAT_STT_MODEL`, `CHAT_TTS_MODEL`, `CHAT_TTS_VOICE` |

## Module Dependencies

```mermaid
flowchart TD
    subgraph external["External Dependencies"]
        FastAPI[fastapi]
        Pydantic[pydantic]
        Neo4jDriver[neo4j]
        QdrantClient[qdrant-client]
        FastEmbed[fastembed]
        Aiogram[aiogram]
        Httpx[httpx]
    end
    
    subgraph internal["Internal Modules"]
        Config[config]
        Schema[schema]
        Provider[provider]
        Prompts[prompts]
        Caller[llm/caller]
        Memory[memory/*]
        Agent[agent]
        API[api]
        Chat[chat/*]
    end
    
    Config --> Schema
    Config --> Provider
    Schema --> Memory
    Provider --> Caller
    Prompts --> Caller
    Prompts --> Memory
    Caller --> Memory
    Memory --> Agent
    Provider --> Agent
    Agent --> API
    Agent --> Chat
    
    FastAPI --> API
    Pydantic --> Schema
    Neo4jDriver --> Memory
    QdrantClient --> Memory
    FastEmbed --> Memory
    Aiogram --> Chat
    Httpx --> Chat
```

## Deployment Architecture

```mermaid
flowchart TB
    subgraph docker["Docker Compose"]
        subgraph app["Application"]
            Sonality[sonality:8000]
            Telegram[telegram-bot]
        end
        
        subgraph databases["Databases"]
            Neo4j[neo4j:7474/7687]
            Qdrant[qdrant:6333]
        end
        
        subgraph services["Services"]
            Speaches[speaches:8001]
        end
    end
    
    subgraph volumes["Persistent Volumes"]
        Neo4jData[neo4j_data]
        QdrantData[qdrant_data]
        SonalityData[sonality_data]
    end
    
    subgraph external["External"]
        OpenAI[OpenAI API]
        TelegramAPI[Telegram API]
        XApi[X API]
        GNewsApi[GNews API]
    end
    
    Sonality --> Neo4j
    Sonality --> Qdrant
    Sonality --> OpenAI
    
    Telegram --> Sonality
    Telegram --> Speaches
    Telegram --> TelegramAPI
    
    Sonality --> XApi
    Sonality --> GNewsApi
    
    Neo4j --> Neo4jData
    Qdrant --> QdrantData
    Sonality --> SonalityData
```

## Error Handling Strategy

```mermaid
flowchart TD
    subgraph errors["Error Types"]
        LLMError[LLM Errors]
        DBError[Database Errors]
        ValidationError[Validation Errors]
        NetworkError[Network Errors]
    end
    
    subgraph handling["Handling Strategies"]
        Retry[Retry with Backoff]
        Repair[LLM JSON Repair]
        Fallback[Fallback Values]
        Rollback[Atomic Rollback]
        Log[Structured Logging]
    end
    
    LLMError --> Retry
    LLMError --> Repair
    LLMError --> Fallback
    
    DBError --> Rollback
    DBError --> Log
    
    ValidationError --> Repair
    ValidationError --> Fallback
    
    NetworkError --> Retry
    NetworkError --> Log
```

## Performance Considerations

| Component | Optimization | Impact |
|-----------|--------------|--------|
| **Embeddings** | FastEmbed ONNX, query caching | ~10x faster than API calls |
| **LLM Calls** | Semaphore serialization | Prevents GPU overload |
| **Retrieval** | Over-fetch + rerank | Better relevance at slight cost |
| **Storage** | Async operations | Non-blocking I/O |
| **Background** | Deferred processing | Responsive user interactions |
| **Qdrant** | INT8 quantization | 4x memory reduction |
| **Neo4j** | Composite indexes | Fast temporal queries |
