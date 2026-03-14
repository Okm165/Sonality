"""Database schema definitions for PostgreSQL/pgvector and Neo4j.

Single source of truth for all database schema. Used by:
- Application startup (db.py)
- Docker Compose initialization (scripts/init_postgres.sql)
- Test containers (tests/containers.py)
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

# PostgreSQL/pgvector schema (vector dimension matches EMBEDDING_DIMENSIONS in config)
POSTGRES_SCHEMA_SQL: Final[str] = """
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Derivative embeddings (sentence-level chunks from episodes)
CREATE TABLE IF NOT EXISTS derivatives (
    uid TEXT PRIMARY KEY,
    episode_uid TEXT NOT NULL,
    text TEXT NOT NULL,
    key_concept TEXT NOT NULL DEFAULT '',
    sequence_num INTEGER NOT NULL DEFAULT 0,
    embedding vector(768) NOT NULL,
    archived BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_derivatives_episode ON derivatives (episode_uid);
CREATE INDEX IF NOT EXISTS idx_derivatives_archived ON derivatives (archived) WHERE NOT archived;
CREATE INDEX IF NOT EXISTS idx_derivatives_embedding ON derivatives USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_derivatives_fts ON derivatives USING GIN (to_tsvector('english', text || ' ' || key_concept)) WHERE NOT archived;

-- Semantic features (personality, preferences, knowledge, relationships)
CREATE TABLE IF NOT EXISTS semantic_features (
    uid TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    tag TEXT NOT NULL,
    feature_name TEXT NOT NULL,
    value TEXT NOT NULL,
    episode_citations TEXT[] NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0.0,
    embedding vector(768),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_semantic_features_category ON semantic_features (category);
CREATE INDEX IF NOT EXISTS idx_semantic_features_tag ON semantic_features (category, tag);
CREATE INDEX IF NOT EXISTS idx_semantic_features_embedding ON semantic_features USING hnsw (embedding vector_cosine_ops) WHERE embedding IS NOT NULL;

-- STM state persistence (crash recovery)
CREATE TABLE IF NOT EXISTS stm_state (
    session_id TEXT PRIMARY KEY DEFAULT 'default',
    running_summary TEXT NOT NULL DEFAULT '',
    message_buffer JSONB NOT NULL DEFAULT '[]'::jsonb,
    last_updated TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Insert default session row
INSERT INTO stm_state (session_id) VALUES ('default') ON CONFLICT DO NOTHING;
"""

# Neo4j schema initialization Cypher statements
NEO4J_SCHEMA_STATEMENTS: Final[tuple[str, ...]] = (
    "CREATE CONSTRAINT episode_uid IF NOT EXISTS FOR (e:Episode) REQUIRE e.uid IS UNIQUE",
    "CREATE CONSTRAINT derivative_uid IF NOT EXISTS FOR (d:Derivative) REQUIRE d.uid IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.segment_id IS UNIQUE",
    "CREATE CONSTRAINT summary_uid IF NOT EXISTS FOR (s:Summary) REQUIRE s.uid IS UNIQUE",
    "CREATE CONSTRAINT belief_topic IF NOT EXISTS FOR (b:Belief) REQUIRE b.topic IS UNIQUE",
    "CREATE INDEX episode_created_at IF NOT EXISTS FOR (e:Episode) ON (e.created_at)",
    "CREATE INDEX episode_segment IF NOT EXISTS FOR (e:Episode) ON (e.segment_id)",
    "CREATE INDEX derivative_episode IF NOT EXISTS FOR (d:Derivative) ON (d.source_episode_uid)",
)

# Docker image versions for consistency
PGVECTOR_IMAGE: Final[str] = "pgvector/pgvector:pg16"
NEO4J_IMAGE: Final[str] = "neo4j:5"


def write_postgres_init_script(path: Path | None = None) -> Path:
    """Write PostgreSQL init script to the specified path (or default location)."""
    if path is None:
        from . import config

        path = config.PROJECT_ROOT / "scripts" / "init_postgres.sql"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(POSTGRES_SCHEMA_SQL.strip() + "\n")
    return path


def write_neo4j_init_script(path: Path | None = None) -> Path:
    """Write Neo4j init script (Cypher) to the specified path (or default location)."""
    if path is None:
        from . import config

        path = config.PROJECT_ROOT / "scripts" / "init_neo4j.cypher"
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "// Neo4j schema initialization for Sonality\n"
        "// This file is auto-generated from sonality/schema.py - do not edit directly.\n\n"
    )
    content = "\n".join(f"{stmt};" for stmt in NEO4J_SCHEMA_STATEMENTS)
    path.write_text(header + content + "\n")
    return path


def write_all_init_scripts() -> tuple[Path, Path]:
    """Write both PostgreSQL and Neo4j init scripts to default locations."""
    pg_path = write_postgres_init_script()
    neo4j_path = write_neo4j_init_script()
    return pg_path, neo4j_path
