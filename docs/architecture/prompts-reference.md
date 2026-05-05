# Prompts Reference

> **Module**: `sonality/prompts.py`

All LLM prompt templates organized by function.

## Prompt Catalog

| Prompt | Purpose |
|--------|---------|
| **CORE_IDENTITY** | Agent character, anti-sycophancy instructions |
| **ESS_CLASSIFICATION_PROMPT** | Epistemic significance classification |
| **REFLECTION_DEEP_PROMPT** | Belief updates, snapshot revision |
| **QUERY_ROUTING_PROMPT** | Retrieval strategy classification |
| **SUFFICIENCY_PROMPT** | Is retrieved context enough? |
| **RERANK_PROMPT** | Listwise episode re-ranking |
| **CHUNKING_PROMPT** | Semantic episode chunking |
| **BOUNDARY_DETECTION_PROMPT** | Conversation segment detection |
| **BATCH_FORGETTING_PROMPT** | KEEP/ARCHIVE/FORGET decisions |
| **BELIEF_UPDATE_PROMPT** | Evidence → belief impact (single topic) |
| **BATCH_BELIEF_UPDATE_PROMPT** | Evidence → belief impact (batch) |
| **KNOWLEDGE_EXTRACTION_PROMPT** | Proposition extraction from evidence |
| **FEATURE_EXTRACTION_PROMPT** | Personality feature extraction |
| **FEATURE_CONSOLIDATION_PROMPT** | Merge redundant features |
| **STEP_SUMMARY_PROMPT** | One-sentence tool output digest |
| **TOOL_RESULT_DIGEST_PROMPT** | Yield assessment for handoff |
| **QUERY_REFORMULATION_PROMPT** | Reformulate search queries |
| **LOOP_HANDOFF_PROMPT** | Continue/finish decision for agentic loop |
| **STATE_COMPRESSION_PROMPT** | Compress long conversation context |
| **SYNTHESIZE_PROMPT** | Structure and evaluate gathered evidence |
| **KNOWLEDGE_UPDATE_PROMPT** | Identify knowledge worth persisting |
| **CONVERSATION_SUMMARY_PROMPT** | Summarize conversation for context window |

## Query Routing Categories

| Category | Depth | Strategy |
|----------|-------|----------|
| NONE | — | Skip |
| SIMPLE | MODERATE | Vector search |
| TEMPORAL | DEEP | Chain retrieval |
| MULTI_ENTITY | DEEP | Split retrieval |
| AGGREGATION | DEEP | Chain retrieval |
| BELIEF_QUERY | MODERATE | Belief edges + vector |

## Design Patterns

1. **Structured output**: JSON schema + example + `assistant_prefix`
2. **Qualitative guidance**: Descriptive calibration, not numeric bands
3. **Anti-sycophancy**: Social pressure/emotional appeal weighted low
4. **Fallback objects**: Every prompt has safe defaults
