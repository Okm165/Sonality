# Agent Processing Pipeline

This document provides a detailed breakdown of how `SonalityAgent` processes each interaction, from message receipt through memory persistence.

## Agent Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Init: __init__()
    Init --> Ready: _init_runtime()
    
    state Init {
        [*] --> ConfigCheck: Validate API config
        ConfigCheck --> AsyncLoop: Create event loop
        AsyncLoop --> DBConnect: DatabaseConnections.create()
        DBConnect --> Workers: Start workers
    }
    
    state Ready {
        [*] --> Idle
        Idle --> Processing: respond() / ingest()
        Processing --> Idle: Complete
    }
    
    Ready --> Shutdown: shutdown()
    Shutdown --> [*]
```

## Initialization Sequence

```mermaid
sequenceDiagram
    participant Main as Main Thread
    participant Loop as Async Event Loop
    participant DB as DatabaseConnections
    participant Graph as MemoryGraph
    participant Dual as DualEpisodeStore
    participant Sem as SemanticWorker

    Main->>Loop: Create new event loop
    Main->>Main: Start loop thread (daemon)
    Main->>Loop: _init_runtime()
    
    Loop->>DB: DatabaseConnections.create()
    DB->>DB: Connect Neo4j driver
    DB->>DB: Connect Qdrant client
    DB->>DB: Apply schemas
    DB-->>Loop: connections ready
    
    Loop->>Graph: MemoryGraph(neo4j_driver)
    Loop->>Dual: DualEpisodeStore(graph, qdrant, embedder)
    Loop->>Graph: get_last_episode_uid()
    Graph-->>Loop: last_uid
    Loop->>Dual: restore_last_episode(last_uid)
    
    Loop-->>Main: runtime initialized
    
    Main->>Main: EventBoundaryDetector()
    Main->>Graph: get_latest_segment_counter()
    Main->>Sem: SemanticIngestionWorker(qdrant_url, embedder)
    Main->>Sem: start()
```

## Response Pipeline (`respond`)

The response pipeline processes conversational messages through multiple stages.

### Stage 1: Context Assembly

```mermaid
flowchart TD
    subgraph input["Input"]
        Messages[messages: list of dict]
    end
    
    subgraph extract["Extract"]
        UserMsg[Last user message]
    end
    
    subgraph identity["Load Identity"]
        Snapshot[PersonalitySnapshot]
        Beliefs[Formatted beliefs text]
    end
    
    subgraph retrieval["Retrieval Pipeline"]
        Router[Query Router]
        VecSearch[Vector Search]
        GraphTrav[Graph Traversal]
        Rerank[Listwise Reranker]
        Episodes[Retrieved episodes]
    end
    
    subgraph knowledge["Knowledge Retrieval"]
        KnowledgeSearch[Qdrant semantic_features]
        KnowledgeCtx[Knowledge context]
    end
    
    subgraph prompt["Prompt Assembly"]
        SystemPrompt[build_system_prompt]
    end
    
    Messages --> UserMsg
    UserMsg --> identity
    Snapshot --> SystemPrompt
    Beliefs --> SystemPrompt
    
    UserMsg --> Router
    Router --> VecSearch
    Router --> GraphTrav
    VecSearch --> Rerank
    GraphTrav --> Rerank
    Rerank --> Episodes
    Episodes --> SystemPrompt
    
    UserMsg --> KnowledgeSearch
    KnowledgeSearch --> KnowledgeCtx
    KnowledgeCtx --> SystemPrompt
```

### Stage 2: LLM Generation

```mermaid
sequenceDiagram
    participant Agent as SonalityAgent
    participant Provider as LLMProvider
    participant LLM as External LLM

    loop max 3 attempts
        Agent->>Provider: chat_completion(model, messages, system_prompt)
        Provider->>LLM: HTTP POST /chat/completions
        
        alt success
            LLM-->>Provider: completion response
            Provider->>Provider: strip_thinking_trace()
            Provider-->>Agent: ChatResult(text, tokens)
            Note over Agent: Break retry loop
        else RuntimeError
            LLM-->>Provider: error
            Provider-->>Agent: raise RuntimeError
            Note over Agent: Log warning, retry
        end
    end
    
    alt all attempts failed
        Agent->>Agent: Return empty ChatResult
    end
```

### Stage 3: Post-Processing

```mermaid
flowchart TD
    subgraph post["_post_process()"]
        ESS[ESS Classification]
        Boundary[Event Boundary Detection]
        Store[Episode Storage]
        Consolidate[Segment Consolidation]
        Knowledge[Knowledge Extraction]
        Provenance[Belief Provenance]
        Semantic[Semantic Worker Enqueue]
        Reflect[Two-tier Reflection]
    end
    
    UserMsg[User Message] --> ESS
    AgentResp[Agent Response] --> ESS
    
    ESS --> Boundary
    Boundary --> Store
    
    Store --> |closed segment| Consolidate
    Store --> |episode_uid| Knowledge
    Store --> |episode_uid| Provenance
    Store --> |episode_uid| Semantic
    Store --> Reflect
    
    Knowledge --> |if density != NONE| KnowledgeStore[(Qdrant)]
    Provenance --> |if update_recommended| GraphEdges[(Neo4j edges)]
    Semantic --> SemanticQueue[Background Queue]
```

### Post-Processing Detail

| Step | Condition | Action |
|------|-----------|--------|
| **ESS Classification** | Always | Classify user message quality (score, reasoning_type, topics) |
| **Boundary Detection** | Always | Check if conversation segment has ended |
| **Episode Storage** | Always | Store turn in dual-store (Neo4j + Qdrant) |
| **Segment Consolidation** | If segment closed | Summarize and consolidate closed segment |
| **Knowledge Extraction** | If `knowledge_density != NONE` | Extract propositions to Qdrant |
| **Belief Provenance** | If `belief_update_recommended` | Create SUPPORTS/CONTRADICTS edges |
| **Semantic Enqueue** | Always | Queue for background feature extraction |
| **Reflection** | Always (but gated) | Two-tier: triage → optional deep update |

## Ingest Pipeline (`ingest`)

Non-conversational data ingestion (news, articles, social media) follows a simpler path.

```mermaid
flowchart TD
    Text[Input Text] --> ESS[ESS Classification]
    TopicOverride[topic_override] --> ESS
    
    ESS --> Check{belief_update_recommended?}
    
    Check -->|false| Return[Return ESS only]
    
    Check -->|true| Store[Store Episode]
    Store --> Knowledge[Extract Knowledge]
    Knowledge --> Provenance[Assess Provenance]
    Provenance --> Semantic[Enqueue Semantic]
    Semantic --> Reflect[Reflect]
    Reflect --> Return2[Return ESS]
```

## Two-Tier Reflection System

Reflection balances update frequency against LLM cost through a triage gate.

```mermaid
flowchart TD
    subgraph tier1["Tier 1: Triage"]
        TriagePrompt[REFLECTION_TRIAGE_PROMPT]
        TriageCall[llm_call → _TriageResponse]
        TriageCheck{should_reflect?}
    end
    
    subgraph tier2["Tier 2: Deep Reflection"]
        LoadEpisodes[Load recent episodes]
        DeepPrompt[REFLECTION_DEEP_PROMPT]
        DeepCall[llm_call → _DeepReflectionResponse]
        Apply[_apply_reflection]
    end
    
    subgraph apply["Apply Changes"]
        BeliefUpdates[Upsert belief_updates]
        NewBeliefs[Upsert new_beliefs]
        SnapshotRevision[Update snapshot]
        Forgetting[Assess forgetting candidates]
    end
    
    Input[User message + ESS] --> TriagePrompt
    TriagePrompt --> TriageCall
    TriageCall --> TriageCheck
    
    TriageCheck -->|false| Skip[Skip deep reflection]
    TriageCheck -->|true| LoadEpisodes
    
    LoadEpisodes --> DeepPrompt
    DeepPrompt --> DeepCall
    DeepCall --> Apply
    
    Apply --> BeliefUpdates
    Apply --> NewBeliefs
    Apply --> SnapshotRevision
    Apply --> Forgetting
```

### Triage Response Schema

```python
class _TriageResponse(BaseModel):
    should_reflect: bool = False
    reason: str = ""
```

### Deep Reflection Response Schema

```python
class _DeepReflectionResponse(BaseModel):
    belief_updates: list[_BeliefPatch]  # Updates to existing beliefs
    new_beliefs: list[_BeliefPatch]      # New beliefs to create
    snapshot_revision: str = ""          # New personality narrative
    snapshot_changed: bool = False       # Whether to apply snapshot

class _BeliefPatch(BaseModel):
    topic: str = ""
    valence: float = 0.0        # -1.0 to +1.0
    confidence: float = 0.5     # 0.0 to 1.0
    belief_text: str = ""       # Descriptive text
    reasoning: str = ""         # Why this update
```

## Retrieval Pipeline Detail

```mermaid
flowchart TD
    Query[User Query] --> Router[route_query]
    
    Router --> Category{category}
    
    Category -->|NONE| Empty[Return empty]
    
    Category -->|MULTI_ENTITY| Split[split_retrieve]
    Split --> Decompose[Decompose query]
    Decompose --> Parallel[Parallel sub-queries]
    
    Category -->|TEMPORAL/AGGREGATION| Chain[chain_retrieve]
    Chain --> Iterative[Iterative with sufficiency]
    
    Category -->|BELIEF_QUERY| Belief[Belief Retrieval]
    Belief --> BeliefHits[find_belief_related_episodes]
    Belief --> TopicHits[find_topic_related_episodes]
    Belief --> VectorHits[vector_search]
    
    Category -->|SIMPLE| Simple[Direct Retrieval]
    Simple --> VecSearch[vector_search]
    Simple --> TopicSearch[find_topic_related_episodes]
    
    Parallel --> Dedupe
    Iterative --> Dedupe
    BeliefHits --> Dedupe
    TopicHits --> Dedupe
    VectorHits --> Dedupe
    VecSearch --> Dedupe
    TopicSearch --> Dedupe
    
    Dedupe[Deduplicate by UID] --> Expand{temporal_expansion?}
    
    Expand -->|EXPAND| Temporal[traverse_temporal_context]
    Temporal --> Episodes
    Expand -->|NO_EXPAND| Episodes
    
    Episodes[Candidate Episodes] --> Rerank[rerank_episodes]
    Rerank --> Select[Select top n_results]
    
    Select --> SemanticCheck{semantic_memory?}
    SemanticCheck -->|SEARCH| SemanticSearch[_search_semantic_features]
    SemanticCheck -->|SKIP| Format
    SemanticSearch --> Format
    
    Format[Format episode lines] --> Result[Retrieved Context]
```

### Query Categories and Strategies

| Category | Strategy | Use Case |
|----------|----------|----------|
| `NONE` | Skip retrieval | Greetings, meta-questions |
| `SIMPLE` | Vector + topic search | Direct factual questions |
| `TEMPORAL` | Chain-of-query iterative | "What happened after X?" |
| `MULTI_ENTITY` | Query decomposition | "Compare X and Y" |
| `AGGREGATION` | Chain with sufficiency | "Summarize all views on X" |
| `BELIEF_QUERY` | Belief edges + topics + vectors | "What do you think about X?" |

### Depth-to-Count Mapping

| Depth | n_results |
|-------|-----------|
| `MINIMAL` | 2 |
| `MODERATE` | 7 |
| `DEEP` | 15 |

## Interaction Semaphore

The agent uses a semaphore to manage LLM load and coordinate with background workers.

```mermaid
sequenceDiagram
    participant User as User Request
    participant Agent as SonalityAgent
    participant Sem as LLM Semaphore
    participant Worker as SemanticWorker
    
    User->>Agent: respond()
    Agent->>Sem: interaction_active() context
    Sem->>Sem: Set flag
    
    Note over Worker: Checks interaction_in_progress()
    Worker->>Worker: Defer processing
    
    Agent->>Agent: Process request
    Agent->>Agent: LLM calls
    
    Agent->>Sem: Exit context
    Sem->>Sem: Clear flag
    
    Note over Worker: interaction_in_progress() = False
    Worker->>Worker: Resume processing
```

## Error Handling

| Component | Error | Recovery |
|-----------|-------|----------|
| LLM Generation | RuntimeError | Retry up to 3 times, return empty on failure |
| ESS Classification | Exception | Use `classifier_exception_fallback()` |
| Episode Storage | EpisodeStorageError | Log error, continue without episode |
| Knowledge Extraction | Exception | Log warning, continue |
| Provenance Assessment | Exception | Log exception, continue |
| Reflection | Exception | Log warning, skip reflection |
| Forgetting | Exception | Log warning, skip forgetting cycle |

## Performance Instrumentation

Every `respond()` call logs timing:

```python
log.info("Total: %.1fs", time.perf_counter() - _t0)
```

Key metrics tracked:
- Context assembly time
- LLM wall time
- Post-processing time
- Individual component times via debug logs
