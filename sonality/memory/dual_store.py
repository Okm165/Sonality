"""Dual-store episode management: Neo4j (graph) + Qdrant (vectors).

Handles the complete episode lifecycle: LLM chunking → embedding → Neo4j storage →
Qdrant storage with transactional safety and rollback on failure.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchText,
    MatchValue,
    PointStruct,
    QuantizationSearchParams,
    SearchParams,
)

from .. import config
from .derivatives import DerivativeChunker, DerivativeWithEmbedding
from .embedder import Embedder, EmbeddingUnavailableError
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
    """Manages episode storage across Neo4j and Qdrant with transactional safety.

    Write order: Neo4j first (ACID, reversible) → Qdrant second (with rollback).
    Critical invariant: episodes are NEVER stored without embeddings.
    """

    def __init__(
        self,
        graph: MemoryGraph,
        qdrant: AsyncQdrantClient,
        chunker: DerivativeChunker,
        embedder: Embedder,
    ) -> None:
        self._graph = graph
        self._qdrant = qdrant
        self._chunker = chunker
        self._embedder = embedder
        self._last_episode_uid = ""
        self.has_episodes = False

    async def store(
        self,
        *,
        user_message: str,
        agent_response: str,
        summary: str,
        topics: list[str],
        ess_score: float,
        segment_id: str = "",
        segment_label: str = "",
        segment_reasoning: str = "",
    ) -> StoredEpisode:
        """Store an episode with derivatives in both Neo4j and Qdrant.

        Raises EpisodeStorageError if any critical phase fails.
        """
        episode_uid = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()
        content = f"User: {user_message}\nAssistant: {agent_response}"

        try:
            derivatives = await asyncio.to_thread(
                self._chunker.chunk_and_embed, content, episode_uid
            )
        except EmbeddingUnavailableError as exc:
            raise EpisodeStorageError(f"Embedding failed: {exc}") from exc
        except Exception as exc:
            raise EpisodeStorageError(f"Chunking failed: {exc}") from exc

        if not derivatives:
            raise EpisodeStorageError("No derivatives produced from chunking")

        log.debug(
            "Episode %s: %d derivatives, dims=%d, concepts=%s",
            episode_uid[:8],
            len(derivatives),
            len(derivatives[0].embedding),
            [d.node.key_concept[:20] for d in derivatives[:3]],
        )
        log.debug(
            "MEMORY_TRACE episode=%s | user_msg_len=%d | agent_resp_len=%d | "
            "topics=%s | ess=%.3f | segment=%s",
            episode_uid[:8],
            len(user_message),
            len(agent_response),
            topics[:3],
            ess_score,
            segment_id[:8] if segment_id else "none",
        )
        for i, d in enumerate(derivatives[:5]):
            log.debug(
                "MEMORY_TRACE deriv[%d]=%s | concept=%s | text=%.80s...",
                i,
                d.node.uid[:8],
                d.node.key_concept,
                d.node.text.replace("\n", " "),
            )

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
            await self._graph.store_episode_atomically(
                episode=episode_node,
                derivatives=[d.node for d in derivatives],
                prev_episode_uid=self._last_episode_uid,
                topics=topics,
                segment_id=segment_id,
                segment_label=segment_label,
                segment_reasoning=segment_reasoning,
            )
        except Exception as exc:
            log.error("Neo4j write failed for episode %s: %s", episode_uid[:8], exc)
            raise EpisodeStorageError(f"Neo4j write failed: {exc}") from exc

        try:
            await self._insert_derivatives_qdrant(derivatives, now)
        except Exception as exc:
            log.error("Qdrant write failed for episode %s: %s", episode_uid[:8], exc)
            try:
                await self._graph.delete_episode(episode_uid)
            except Exception:
                log.exception("Failed to rollback Neo4j episode %s", episode_uid[:8])
            raise EpisodeStorageError(f"Qdrant write failed: {exc}") from exc

        self._last_episode_uid = episode_uid
        self.has_episodes = True
        deriv_uids = [d.node.uid for d in derivatives]
        log.info(
            "Stored episode %s with %d derivatives",
            episode_uid[:8],
            len(deriv_uids),
        )
        return StoredEpisode(episode_uid=episode_uid, derivative_uids=deriv_uids)

    async def vector_search(
        self,
        query: str,
        top_k: int = 20,
        text_filter: bool = False,
    ) -> list[tuple[str, str, float]]:
        """Search Qdrant for similar derivatives using dense vectors.

        Returns (uid, episode_uid, score).
        """
        try:
            query_embedding = await asyncio.to_thread(self._embedder.embed_query, query)
        except Exception:
            log.debug("Embedding unavailable; falling back to text-only search", exc_info=True)
            return await self._text_only_search(query, top_k)

        must = [FieldCondition(key="archived", match=MatchValue(value=False))]
        should = [FieldCondition(key="text", match=MatchText(text=query))] if text_filter else None

        response = await self._qdrant.query_points(
            collection_name="derivatives",
            query=query_embedding,
            using="dense",
            query_filter=Filter(must=must, should=should),
            limit=top_k,
            with_payload=True,
            search_params=SearchParams(
                hnsw_ef=config.QDRANT_SEARCH_EF,
                quantization=QuantizationSearchParams(
                    rescore=config.QDRANT_RESCORE_QUANTIZED,
                    oversampling=2.0,
                ),
            ),
        )
        return [
            (str(p.payload.get("uid", "")), str(p.payload.get("episode_uid", "")), p.score or 0.0)
            for p in response.points
            if p.payload
        ]

    async def _text_only_search(self, query: str, top_k: int) -> list[tuple[str, str, float]]:
        """Keyword-based fallback when embedding service is unavailable."""
        results, _ = await self._qdrant.scroll(
            collection_name="derivatives",
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="archived", match=MatchValue(value=False)),
                    FieldCondition(key="text", match=MatchText(text=query)),
                ]
            ),
            limit=top_k,
            with_payload=True,
        )
        return [
            (str(p.payload.get("uid", "")), str(p.payload.get("episode_uid", "")), 0.5)
            for p in results
            if p.payload
        ]

    async def archive_derivatives(self, episode_uid: str) -> None:
        """Mark derivatives as archived in Qdrant (soft delete)."""
        points, _ = await self._qdrant.scroll(
            collection_name="derivatives",
            scroll_filter=Filter(
                must=[FieldCondition(key="episode_uid", match=MatchValue(value=episode_uid))]
            ),
            limit=1000,
        )
        if points:
            await self._qdrant.set_payload(
                collection_name="derivatives",
                payload={"archived": True},
                points=[p.id for p in points if p.id is not None],
            )

    async def delete_derivatives(self, episode_uid: str) -> None:
        """Hard-delete derivatives by episode UID from Qdrant."""
        await self._qdrant.delete(
            collection_name="derivatives",
            points_selector=Filter(
                must=[FieldCondition(key="episode_uid", match=MatchValue(value=episode_uid))]
            ),
        )

    async def verify_consistency(self) -> list[str]:
        """Check Neo4j-Qdrant sync and clean orphan derivatives."""
        results, _ = await self._qdrant.scroll(
            collection_name="derivatives",
            limit=10000,
            with_payload=["uid"],
        )
        qdrant_uids = {str(p.payload.get("uid", "")) for p in results if p.payload}

        neo4j_uids = await self._graph.list_derivative_uids()

        qdrant_only = sorted(qdrant_uids - neo4j_uids)
        neo4j_only = sorted(neo4j_uids - qdrant_uids)

        if qdrant_only:
            await self._qdrant.delete(
                collection_name="derivatives",
                points_selector=qdrant_only,
            )
            log.warning("Cleaned %d Qdrant-only orphan derivatives", len(qdrant_only))

        if neo4j_only:
            await self._graph.delete_derivatives(neo4j_only)
            log.warning("Cleaned %d Neo4j-only orphan derivatives", len(neo4j_only))

        return [*qdrant_only, *neo4j_only]

    async def _insert_derivatives_qdrant(
        self, derivatives: list[DerivativeWithEmbedding], created_at: str
    ) -> None:
        """Insert derivative embeddings into Qdrant."""
        points = [
            PointStruct(
                id=d.node.uid,
                vector={"dense": d.embedding},
                payload={
                    "uid": d.node.uid,
                    "episode_uid": d.node.source_episode_uid,
                    "text": d.node.text,
                    "key_concept": d.node.key_concept,
                    "sequence_num": d.node.sequence_num,
                    "archived": False,
                    "created_at": created_at,
                },
            )
            for d in derivatives
        ]
        await self._qdrant.upsert(collection_name="derivatives", points=points)
