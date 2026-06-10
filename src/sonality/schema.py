"""Database schema definitions for Neo4j and Qdrant.

Central schema registry: Qdrant collection configs (vectors, indices, quantization),
Neo4j constraints/indices, and shared enums (ToolName, EventType, SemanticCategory,
Collection). Neo4j relationship types live in memory/graph.py alongside the Cypher
queries that use them. ChatRole lives in shared.types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    PayloadSchemaType,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    TextIndexParams,
    TextIndexType,
    VectorParams,
)

from . import config


class Collection(StrEnum):
    """Qdrant collection names — single source of truth."""

    DERIVATIVES = "derivatives"
    SEMANTIC_FEATURES = "semantic_features"


class SemanticCategory(StrEnum):
    """Categories for semantic feature and knowledge extraction."""

    PERSONALITY = "personality"
    PREFERENCES = "preferences"
    KNOWLEDGE = "knowledge"
    RELATIONSHIPS = "relationships"


def normalize_topic(raw: str) -> str:
    """Normalize a topic string to a canonical slug (lowercase, underscores)."""
    return re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_")


class ToolName(StrEnum):
    """Canonical tool names — single source of truth for all tool references."""

    RECALL_MEMORY = "recall_memory"
    WEB_RESEARCH = "web_research"
    INTEGRATE_KNOWLEDGE = "integrate_knowledge"


class EventType(StrEnum):
    """Agent progress event types."""

    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONTEXT_BUILD = "context_build"
    SUMMARIZING = "summarizing"
    TOOL_PROGRESS = "tool_progress"

    DONE = "done"


class Phase(StrEnum):
    """Automaton phases for the agentic loop state machine."""

    THINKING = "thinking"
    ACTING = "acting"
    COMPLETED = "completed"
    FAILED = "failed"


DENSE_VECTOR: Final = "dense"

_SHARED_HNSW: Final = HnswConfigDiff(
    m=16,
    ef_construct=100,
    full_scan_threshold=10000,
    max_indexing_threads=0,
    on_disk=False,
)
_SHARED_QUANTIZATION: Final = ScalarQuantization(
    scalar=ScalarQuantizationConfig(type=ScalarType.INT8, quantile=0.99, always_ram=True),
)
_SHARED_OPTIMIZERS: Final = OptimizersConfigDiff(
    indexing_threshold=20000,
    memmap_threshold=50000,
    default_segment_number=4,
)


@dataclass(frozen=True, slots=True)
class CollectionSpec:
    """Typed Qdrant collection configuration."""

    vectors_config: dict[str, VectorParams]
    payload_schema: dict[str, PayloadSchemaType]
    hnsw_config: HnswConfigDiff = field(default_factory=lambda: _SHARED_HNSW)
    quantization_config: ScalarQuantization = field(default_factory=lambda: _SHARED_QUANTIZATION)
    optimizers_config: OptimizersConfigDiff = field(default_factory=lambda: _SHARED_OPTIMIZERS)
    text_index_field: str = ""


def qdrant_collection_specs(dims: int) -> dict[str, CollectionSpec]:
    """Build Qdrant collection specs for the given embedding dimension."""
    return {
        Collection.DERIVATIVES: CollectionSpec(
            vectors_config={
                DENSE_VECTOR: VectorParams(size=dims, distance=Distance.COSINE, on_disk=False),
            },
            payload_schema={
                "uid": PayloadSchemaType.KEYWORD,
                "episode_uid": PayloadSchemaType.KEYWORD,
                "text": PayloadSchemaType.TEXT,
                "archived": PayloadSchemaType.BOOL,
                "prospective_query": PayloadSchemaType.TEXT,
                "created_at": PayloadSchemaType.DATETIME,
                "specificity": PayloadSchemaType.FLOAT,
                "grounding": PayloadSchemaType.FLOAT,
                "rigor": PayloadSchemaType.FLOAT,
                "source_quality": PayloadSchemaType.FLOAT,
                "objectivity": PayloadSchemaType.FLOAT,
            },
            text_index_field="text",
        ),
        Collection.SEMANTIC_FEATURES: CollectionSpec(
            vectors_config={
                DENSE_VECTOR: VectorParams(size=dims, distance=Distance.COSINE, on_disk=False),
            },
            payload_schema={
                "uid": PayloadSchemaType.KEYWORD,
                "category": PayloadSchemaType.KEYWORD,
                "tag": PayloadSchemaType.KEYWORD,
                "feature_name": PayloadSchemaType.KEYWORD,
                "value": PayloadSchemaType.TEXT,
                "episode_citations": PayloadSchemaType.KEYWORD,
                "confidence": PayloadSchemaType.FLOAT,
                "created_at": PayloadSchemaType.DATETIME,
                "updated_at": PayloadSchemaType.DATETIME,
            },
            text_index_field="value",
        ),
    }


NEO4J_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    "CREATE CONSTRAINT episode_uid IF NOT EXISTS FOR (e:Episode) REQUIRE e.uid IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT belief_topic IF NOT EXISTS FOR (b:Belief) REQUIRE b.topic IS UNIQUE",
    "CREATE CONSTRAINT identity_session IF NOT EXISTS FOR (n:PersonalitySnapshot) REQUIRE n.session_id IS UNIQUE",
    "CREATE INDEX episode_created_at IF NOT EXISTS FOR (e:Episode) ON (e.created_at)",
    "CREATE INDEX episode_archived_created IF NOT EXISTS FOR (e:Episode) ON (e.archived, e.created_at)",
    "CREATE INDEX episode_archived_utility IF NOT EXISTS FOR (e:Episode) ON (e.archived, e.utility_score)",
)


async def init_qdrant_collections(client: AsyncQdrantClient, *, dims: int = 0) -> None:
    """Initialize Qdrant collections with optimized schemas.

    Args:
        dims: Embedding vector dimensions. When 0 (default), reads from config.
    """
    from qdrant_client.models import TokenizerType

    effective_dims = dims or config.settings.embedding_dimensions
    for name, spec in qdrant_collection_specs(effective_dims).items():
        if not await client.collection_exists(name):
            await client.create_collection(
                collection_name=name,
                vectors_config=spec.vectors_config,
                hnsw_config=spec.hnsw_config,
                quantization_config=spec.quantization_config,
                optimizers_config=spec.optimizers_config,
            )
            for field_name, schema_type in spec.payload_schema.items():
                await client.create_payload_index(
                    collection_name=name,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            if spec.text_index_field:
                await client.create_payload_index(
                    collection_name=name,
                    field_name=spec.text_index_field,
                    field_schema=TextIndexParams(
                        type=TextIndexType.TEXT,
                        tokenizer=TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=20,
                        lowercase=True,
                    ),
                )
