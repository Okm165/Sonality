"""Database schema definitions for Neo4j and Qdrant.

Single source of truth for all database schema. Neo4j handles graph relationships
and state persistence (sponge, STM). Qdrant handles vector storage and hybrid search.
"""

from __future__ import annotations

from typing import Final

from qdrant_client.models import Distance, PayloadSchemaType, TextIndexParams, VectorParams

# Qdrant collection configurations
QDRANT_COLLECTIONS: Final[dict[str, dict]] = {
    "derivatives": {
        "vectors_config": VectorParams(size=768, distance=Distance.COSINE),
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
    "semantic_features": {
        "vectors_config": VectorParams(size=768, distance=Distance.COSINE),
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

# Neo4j schema initialization Cypher statements
NEO4J_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    # Graph structure constraints
    "CREATE CONSTRAINT episode_uid IF NOT EXISTS FOR (e:Episode) REQUIRE e.uid IS UNIQUE",
    "CREATE CONSTRAINT derivative_uid IF NOT EXISTS FOR (d:Derivative) REQUIRE d.uid IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.segment_id IS UNIQUE",
    "CREATE CONSTRAINT summary_uid IF NOT EXISTS FOR (s:Summary) REQUIRE s.uid IS UNIQUE",
    "CREATE CONSTRAINT belief_topic IF NOT EXISTS FOR (b:Belief) REQUIRE b.topic IS UNIQUE",
    # State persistence constraints
    "CREATE CONSTRAINT sponge_session IF NOT EXISTS FOR (s:SpongeState) REQUIRE s.session_id IS UNIQUE",
    "CREATE CONSTRAINT stm_session IF NOT EXISTS FOR (s:STMState) REQUIRE s.session_id IS UNIQUE",
    # Indexes for efficient lookups
    "CREATE INDEX episode_created_at IF NOT EXISTS FOR (e:Episode) ON (e.created_at)",
    "CREATE INDEX episode_segment IF NOT EXISTS FOR (e:Episode) ON (e.segment_id)",
    "CREATE INDEX derivative_episode IF NOT EXISTS FOR (d:Derivative) ON (d.source_episode_uid)",
)

# Docker image versions for consistency
NEO4J_IMAGE: Final[str] = "neo4j:5"
QDRANT_IMAGE: Final[str] = "qdrant/qdrant:latest"


async def init_qdrant_collections(client) -> None:
    """Initialize Qdrant collections with proper schemas."""
    from qdrant_client.models import TokenizerType

    for name, cfg in QDRANT_COLLECTIONS.items():
        if not await client.collection_exists(name):
            await client.create_collection(
                collection_name=name,
                vectors_config=cfg["vectors_config"],
            )
            # Create payload indexes
            for field, schema_type in cfg["payload_schema"].items():
                await client.create_payload_index(
                    collection_name=name,
                    field_name=field,
                    field_schema=schema_type,
                )
            # Create text index for hybrid search
            if text_field := cfg.get("text_index_field"):
                await client.create_payload_index(
                    collection_name=name,
                    field_name=text_field,
                    field_schema=TextIndexParams(
                        type="text",
                        tokenizer=TokenizerType.WORD,
                        min_token_len=2,
                        max_token_len=20,
                        lowercase=True,
                    ),
                )
