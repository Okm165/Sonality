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
