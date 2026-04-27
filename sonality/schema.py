"""Database schema definitions for Neo4j and Qdrant.

Shared enums, vector configs, and collection schemas. Neo4j graph relationship
types (EdgeType) live in memory/graph.py alongside the Cypher queries that use them.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Final

from pydantic import BaseModel
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
    """Semantic feature categories for personality extraction."""

    PERSONALITY = "personality"
    PREFERENCES = "preferences"
    KNOWLEDGE = "knowledge"
    RELATIONSHIPS = "relationships"


class ChatRole(StrEnum):
    """Message roles in chat completions."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """Typed chat message for internal use."""

    role: ChatRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to dict for API compatibility."""
        return {"role": self.role, "content": self.content}


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

QDRANT_COLLECTIONS: Final[dict[str, dict[str, Any]]] = {
    Collection.DERIVATIVES: {
        "vectors_config": {
            DENSE_VECTOR: VectorParams(
                size=config.EMBEDDING_DIMENSIONS, distance=Distance.COSINE, on_disk=False
            ),
        },
        "hnsw_config": _SHARED_HNSW,
        "quantization_config": _SHARED_QUANTIZATION,
        "optimizers_config": _SHARED_OPTIMIZERS,
        "payload_schema": {
            "uid": PayloadSchemaType.KEYWORD,
            "episode_uid": PayloadSchemaType.KEYWORD,
            "text": PayloadSchemaType.TEXT,
            "key_concept": PayloadSchemaType.KEYWORD,
            "sequence_num": PayloadSchemaType.INTEGER,
            "archived": PayloadSchemaType.BOOL,
            "created_at": PayloadSchemaType.DATETIME,
        },
        "text_index_field": "text",
    },
    Collection.SEMANTIC_FEATURES: {
        "vectors_config": {
            DENSE_VECTOR: VectorParams(
                size=config.EMBEDDING_DIMENSIONS, distance=Distance.COSINE, on_disk=False
            ),
        },
        "hnsw_config": _SHARED_HNSW,
        "quantization_config": _SHARED_QUANTIZATION,
        "optimizers_config": _SHARED_OPTIMIZERS,
        "payload_schema": {
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
        "text_index_field": "value",
    },
}

NEO4J_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    "CREATE CONSTRAINT episode_uid IF NOT EXISTS FOR (e:Episode) REQUIRE e.uid IS UNIQUE",
    "CREATE CONSTRAINT derivative_uid IF NOT EXISTS FOR (d:Derivative) REQUIRE d.uid IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.segment_id IS UNIQUE",
    "CREATE CONSTRAINT summary_uid IF NOT EXISTS FOR (s:Summary) REQUIRE s.uid IS UNIQUE",
    "CREATE CONSTRAINT belief_topic IF NOT EXISTS FOR (b:Belief) REQUIRE b.topic IS UNIQUE",
    "CREATE CONSTRAINT identity_session IF NOT EXISTS FOR (n:PersonalitySnapshot) REQUIRE n.session_id IS UNIQUE",
    "CREATE INDEX episode_created_at IF NOT EXISTS FOR (e:Episode) ON (e.created_at)",
    "CREATE INDEX episode_segment IF NOT EXISTS FOR (e:Episode) ON (e.segment_id)",
    "CREATE INDEX derivative_episode IF NOT EXISTS FOR (d:Derivative) ON (d.source_episode_uid)",
    "CREATE INDEX episode_archived_created IF NOT EXISTS FOR (e:Episode) ON (e.archived, e.created_at)",
    "CREATE INDEX episode_archived_utility IF NOT EXISTS FOR (e:Episode) ON (e.archived, e.utility_score)",
    "CREATE INDEX episode_segment_ess IF NOT EXISTS FOR (e:Episode) ON (e.segment_id, e.ess_score)",
)


async def init_qdrant_collections(client: AsyncQdrantClient) -> None:
    """Initialize Qdrant collections with optimized schemas."""
    from qdrant_client.models import TokenizerType

    for name, cfg in QDRANT_COLLECTIONS.items():
        if not await client.collection_exists(name):
            await client.create_collection(
                collection_name=name,
                vectors_config=cfg["vectors_config"],
                hnsw_config=cfg.get("hnsw_config"),
                quantization_config=cfg.get("quantization_config"),
                optimizers_config=cfg.get("optimizers_config"),
            )
            for field, schema_type in cfg["payload_schema"].items():
                await client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema_type,
                )
            if text_field := cfg.get("text_index_field"):
                await client.create_payload_index(
                    collection_name=name,
                    field_name=text_field,
                    field_schema=TextIndexParams(
                        type=TextIndexType.TEXT,
                        tokenizer=TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=20,
                        lowercase=True,
                    ),
                )
