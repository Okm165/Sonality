# Prompts Reference

> **Module**: `sonality/prompts.py`

All LLM prompt templates organized by function.

## Prompt Catalog

| Prompt | Purpose |
|--------|---------|
| **CORE_IDENTITY** | Agent character, anti-sycophancy instructions |
| **ESS_CLASSIFICATION_PROMPT** | Argument quality scoring (0-1) |
| **REFLECTION_TRIAGE_PROMPT** | Should reflect? → bool |
| **REFLECTION_DEEP_PROMPT** | Belief updates, snapshot revision |
| **QUERY_ROUTING_PROMPT** | Retrieval strategy classification |
| **SUFFICIENCY_PROMPT** | Is retrieved context enough? |
| **RERANK_PROMPT** | Listwise episode re-ranking |
| **CHUNKING_PROMPT** | Semantic episode chunking |
| **BOUNDARY_DETECTION_PROMPT** | Conversation segment detection |
| **BATCH_FORGETTING_PROMPT** | KEEP/ARCHIVE/FORGET decisions |
| **BELIEF_UPDATE_PROMPT** | Evidence → belief impact |
| **KNOWLEDGE_EXTRACTION_PROMPT** | 5-stage proposition extraction |
| **FEATURE_EXTRACTION_PROMPT** | Personality feature extraction |
| **FEATURE_CONSOLIDATION_PROMPT** | Merge redundant features |

## ESS Calibration Scale

| Pattern | Score | Type |
|---------|-------|------|
| Greeting | 0.02 | no_argument |
| Bare assertion | 0.08 | no_argument |
| "Everyone knows" | 0.10 | social_pressure |
| Personal anecdote | 0.18 | anecdotal |
| Named expert | 0.22 | expert_opinion |
| Logical argument | 0.40-0.80 | logical_argument |
| Named source + data | 0.75-0.85 | empirical_data |

## Query Routing Categories

| Category | Depth | Strategy |
|----------|-------|----------|
| NONE | — | Skip |
| SIMPLE | 7 | Vector search |
| TEMPORAL | 15 | Chain retrieval |
| MULTI_ENTITY | 15 | Split retrieval |
| BELIEF_QUERY | 7 | Belief edges + vector |

## Design Patterns

1. **Structured output**: JSON schema + example + `assistant_prefix`
2. **Calibration scales**: Numeric ranges by pattern type
3. **Anti-sycophancy**: Social pressure/emotional appeal score low
4. **Fallback objects**: Every prompt has safe defaults
