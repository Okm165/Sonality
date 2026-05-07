"""Dual-store episode management: Neo4j (graph) + Qdrant (vectors).

Handles the complete episode lifecycle: LLM chunking → embedding → Neo4j storage →
Qdrant storage with transactional safety and rollback on failure.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NamedTuple
from uuid import UUID

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    QuantizationSearchParams,
    SearchParams,
)

from shared.errors import EpisodeStorageError
from shared.types import new_id

from .. import config
from ..schema import DENSE_VECTOR, Collection
from .derivatives import DerivativeWithEmbedding, chunk_and_embed
from .embedder import Embedder
from .graph import EpisodeNode, MemoryGraph

log = structlog.get_logger()


class SearchHit(NamedTuple):
    """A derivative match from vector search."""

    uid: str
    episode_uid: str
    score: float


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
        embedder: Embedder,
    ) -> None:
        self._graph = graph
        self._qdrant = qdrant
        self._embedder = embedder
        self._last_episode_uid = ""
        self.has_episodes = False

    def restore_last_episode(self, uid: str) -> None:
        """Re-link to the last stored episode after restart."""
        self._last_episode_uid = uid
        self.has_episodes = True

    async def store(
        self,
        *,
        user_message: str,
        agent_response: str,
        summary: str,
        topics: list[str],
        ess_score: float,
        reasoning_type: str = "",
        segment_id: str = "",
        segment_label: str = "",
    ) -> StoredEpisode:
        """Store an episode with derivatives in both Neo4j and Qdrant.

        Raises EpisodeStorageError if any critical phase fails.
        """
        episode_uid = new_id()
        now = datetime.now(UTC).isoformat()
        content = f"User: {user_message}\nAssistant: {agent_response}"

        try:
            derivatives = await asyncio.to_thread(
                chunk_and_embed, self._embedder, content, episode_uid
            )
        except Exception as exc:
            raise EpisodeStorageError(f"Chunking/embedding failed: {exc}") from exc

        if not derivatives:
            raise EpisodeStorageError("No derivatives produced from chunking")

        log.debug(
            "episode_derivatives_preview",
            episode_uid=episode_uid[:8],
            derivative_count=len(derivatives),
            embedding_dims=len(derivatives[0].embedding),
            key_concepts=[d.node.key_concept[:20] for d in derivatives[:3]],
        )
        log.debug(
            "episode_store_trace",
            episode_uid=episode_uid[:8],
            user_message_len=len(user_message),
            agent_response_len=len(agent_response),
            topics_sample=topics[:3],
            ess_score=ess_score,
            segment_prefix=segment_id[:8] if segment_id else "none",
        )
        for i, d in enumerate(derivatives[:5]):
            log.debug(
                "episode_derivative_trace",
                index=i,
                derivative_uid=d.node.uid[:8],
                key_concept=d.node.key_concept,
                text_preview=d.node.text.replace("\n", " ")[:80],
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
            reasoning_type=reasoning_type,
        )
        try:
            await self._graph.store_episode_atomically(
                episode=episode_node,
                derivatives=[d.node for d in derivatives],
                prev_episode_uid=self._last_episode_uid,
                topics=topics,
                segment_id=segment_id,
                segment_label=segment_label,
            )
        except Exception as exc:
            log.error(
                "neo4j_episode_write_failed",
                episode_uid=episode_uid[:8],
                error=str(exc),
            )
            raise EpisodeStorageError(f"Neo4j write failed: {exc}") from exc

        try:
            await self._insert_derivatives_qdrant(derivatives, now)
        except Exception as exc:
            log.error(
                "qdrant_episode_write_failed",
                episode_uid=episode_uid[:8],
                error=str(exc),
            )
            try:
                await self._graph.delete_episode(episode_uid)
            except Exception:
                log.error(
                    "neo4j_episode_rollback_failed",
                    episode_uid=episode_uid[:8],
                    exc_info=True,
                )
            raise EpisodeStorageError(f"Qdrant write failed: {exc}") from exc

        self._last_episode_uid = episode_uid
        self.has_episodes = True
        deriv_uids = [d.node.uid for d in derivatives]
        log.info(
            "episode_stored",
            episode_uid=episode_uid[:8],
            derivative_count=len(deriv_uids),
        )
        return StoredEpisode(episode_uid=episode_uid, derivative_uids=deriv_uids)

    async def vector_search(
        self,
        query: str,
        top_k: int = 20,
    ) -> list[SearchHit]:
        """Search Qdrant for similar derivatives using dense vectors.

        Returns SearchHit tuples sorted by cosine similarity score (highest first).
        Only non-archived derivatives are searched.
        """
        query_embedding = await asyncio.to_thread(self._embedder.embed_query, query)
        response = await self._qdrant.query_points(
            collection_name=Collection.DERIVATIVES,
            query=query_embedding,
            using=DENSE_VECTOR,
            query_filter=Filter(
                must=[FieldCondition(key="archived", match=MatchValue(value=False))]
            ),
            limit=top_k,
            with_payload=True,
            search_params=SearchParams(
                hnsw_ef=config.settings.qdrant_search_ef,
                quantization=QuantizationSearchParams(
                    rescore=config.settings.qdrant_rescore,
                    oversampling=2.0,
                ),
            ),
        )
        hits = [
            SearchHit(
                str(p.payload.get("uid", "")), str(p.payload.get("episode_uid", "")), p.score or 0.0
            )
            for p in response.points
            if p.payload
        ]
        log.debug(
            "qdrant_vector_search",
            top_k=top_k,
            hit_count=len(hits),
        )
        return hits

    async def archive_derivatives(self, episode_uid: str) -> None:
        """Mark derivatives as archived in Qdrant (soft delete)."""
        filt = Filter(must=[FieldCondition(key="episode_uid", match=MatchValue(value=episode_uid))])
        all_ids: list[int | str | UUID] = []
        offset = None
        while True:
            points, offset = await self._qdrant.scroll(
                collection_name=Collection.DERIVATIVES,
                scroll_filter=filt,
                limit=500,
                offset=offset,
            )
            all_ids.extend(str(p.id) for p in points if p.id is not None)
            if offset is None:
                break
        if all_ids:
            await self._qdrant.set_payload(
                collection_name=Collection.DERIVATIVES,
                payload={"archived": True},
                points=PointIdsList(points=all_ids),
            )
            log.info(
                "qdrant_archive_derivatives",
                episode_uid=episode_uid[:8],
                point_count=len(all_ids),
            )

    async def delete_derivatives(self, episode_uid: str) -> None:
        """Hard-delete derivatives by episode UID from Qdrant."""
        await self._qdrant.delete(
            collection_name=Collection.DERIVATIVES,
            points_selector=Filter(
                must=[FieldCondition(key="episode_uid", match=MatchValue(value=episode_uid))]
            ),
        )
        log.warning(
            "qdrant_delete_derivatives",
            episode_uid=episode_uid[:8],
        )

    async def _insert_derivatives_qdrant(
        self, derivatives: list[DerivativeWithEmbedding], created_at: str
    ) -> None:
        """Insert derivative embeddings into Qdrant."""
        points = [
            PointStruct(
                id=d.node.uid,
                vector={DENSE_VECTOR: d.embedding},
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
        await self._qdrant.upsert(collection_name=Collection.DERIVATIVES, points=points)
