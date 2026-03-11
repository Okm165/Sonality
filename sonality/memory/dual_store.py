"""Dual-store episode management: Neo4j (graph) + PostgreSQL/pgvector (vectors).

Handles the complete episode lifecycle: LLM chunking → embedding → Neo4j storage →
pgvector storage with transactional safety and rollback on failure.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from psycopg_pool import AsyncConnectionPool

from .derivatives import DerivativeChunker, DerivativeWithEmbedding
from .embedder import EmbeddingUnavailableError, ExternalEmbedder
from .graph import EpisodeNode, MemoryGraph

log = logging.getLogger(__name__)


class EpisodeStorageError(Exception):
    """Raised when episode storage fails at any phase."""


@dataclass(frozen=True, slots=True)
class StoredEpisode:
    """Result of a successful episode store operation."""

    episode_uid: str
    derivative_uids: list[str]


class DualEpisodeStore:
    """Manages episode storage across Neo4j and pgvector with transactional safety.

    Write order: Neo4j first (ACID, reversible) → pgvector second (with rollback).
    Critical invariant: episodes are NEVER stored without embeddings.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        pg_pool: AsyncConnectionPool,
        chunker: DerivativeChunker,
        embedder: ExternalEmbedder,
    ) -> None:
        self._graph = graph
        self._pg_pool = pg_pool
        self._chunker = chunker
        self._embedder = embedder
        self._last_episode_uid: str | None = None

    async def store(
        self,
        *,
        user_message: str,
        agent_response: str,
        summary: str,
        topics: list[str],
        ess_score: float,
        segment_id: str | None = None,
    ) -> StoredEpisode:
        """Store an episode with derivatives in both Neo4j and pgvector.

        Raises EpisodeStorageError if any critical phase fails.
        """
        episode_uid = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        content = f"User: {user_message}\nAssistant: {agent_response}"

        # Phase 1: LLM chunking + embedding (can fail: LLM timeout, embedding down)
        try:
            derivatives = self._chunker.chunk_and_embed(content, episode_uid)
        except EmbeddingUnavailableError as exc:
            raise EpisodeStorageError(f"Embedding failed: {exc}") from exc
        except Exception as exc:
            raise EpisodeStorageError(f"Chunking failed: {exc}") from exc

        if not derivatives:
            raise EpisodeStorageError("No derivatives produced from chunking")

        # Phase 2: Neo4j graph writes (ACID transaction)
        episode_node = EpisodeNode(
            uid=episode_uid,
            content=content,
            summary=summary,
            topics=topics,
            ess_score=ess_score,
            created_at=now,
            valid_at=now,
            utility_score=0.0,
            access_count=0,
            last_accessed=now,
            segment_id=segment_id,
            consolidation_level=1,
            archived=False,
            user_message=user_message,
            agent_response=agent_response,
        )
        try:
            await self._graph.create_episode(
                episode_node, prev_episode_uid=self._last_episode_uid
            )
            await self._graph.create_derivatives(
                [d.node for d in derivatives], episode_uid
            )
            if topics:
                await self._graph.link_topics(episode_uid, topics)
            if segment_id:
                await self._graph.link_segment(episode_uid, segment_id)
        except Exception as exc:
            log.error("Neo4j write failed for episode %s: %s", episode_uid[:8], exc)
            raise EpisodeStorageError(f"Neo4j write failed: {exc}") from exc

        # Phase 3: pgvector writes (after Neo4j success)
        try:
            await self._insert_derivatives_pgvector(derivatives)
        except Exception as exc:
            log.error("pgvector write failed for episode %s: %s", episode_uid[:8], exc)
            # Rollback: archive the episode in Neo4j
            try:
                await self._graph.archive_episode(episode_uid)
            except Exception:
                log.exception("Failed to rollback Neo4j episode %s", episode_uid[:8])
            raise EpisodeStorageError(f"pgvector write failed: {exc}") from exc

        self._last_episode_uid = episode_uid
        deriv_uids = [d.node.uid for d in derivatives]
        log.info(
            "Stored episode %s with %d derivatives",
            episode_uid[:8], len(deriv_uids),
        )
        return StoredEpisode(episode_uid=episode_uid, derivative_uids=deriv_uids)

    async def vector_search(
        self,
        query: str,
        *,
        top_k: int = 20,
    ) -> list[tuple[str, str, float]]:
        """Search pgvector for similar derivatives. Returns (uid, episode_uid, distance)."""
        query_embedding = self._embedder.embed_query(query)
        async with self._pg_pool.connection() as conn:
            from pgvector.psycopg import register_vector_async

            await register_vector_async(conn)
            async with conn.cursor() as cur:
                await cur.execute("SET hnsw.iterative_scan = 'on'")
                await cur.execute("SET hnsw.ef_search = 150")
                await cur.execute(
                    """
                    SELECT uid, episode_uid, embedding <=> %s::vector AS distance
                    FROM derivatives
                    WHERE NOT archived
                    ORDER BY embedding <=> %s::vector
                    LIMIT %s
                    """,
                    (query_embedding, query_embedding, top_k),
                )
                rows = await cur.fetchall()
                return [(str(r[0]), str(r[1]), float(r[2])) for r in rows]

    async def archive_derivatives(self, episode_uid: str) -> None:
        """Mark derivatives as archived in pgvector (soft delete)."""
        async with self._pg_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                "UPDATE derivatives SET archived = TRUE WHERE episode_uid = %s",
                (episode_uid,),
            )

    async def verify_consistency(self) -> list[str]:
        """Check Neo4j-pgvector sync. Returns list of orphan UIDs found and cleaned."""
        # This would compare derivative UIDs across both stores
        # For now, log a placeholder
        log.debug("Consistency verification placeholder")
        return []

    async def _insert_derivatives_pgvector(
        self, derivatives: list[DerivativeWithEmbedding]
    ) -> None:
        """Insert derivative embeddings into PostgreSQL/pgvector."""
        async with self._pg_pool.connection() as conn:
            from pgvector.psycopg import register_vector_async

            await register_vector_async(conn)
            async with conn.cursor() as cur:
                for d in derivatives:
                    await cur.execute(
                        """
                        INSERT INTO derivatives (uid, episode_uid, text, key_concept,
                                                 sequence_num, embedding)
                        VALUES (%s, %s, %s, %s, %s, %s::vector)
                        ON CONFLICT (uid) DO NOTHING
                        """,
                        (
                            d.node.uid,
                            d.node.source_episode_uid,
                            d.node.text,
                            d.node.key_concept,
                            d.node.sequence_num,
                            d.embedding,
                        ),
                    )
