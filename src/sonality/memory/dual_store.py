"""Dual-store episode management: Neo4j (graph) + Qdrant (vectors).

Handles the complete episode lifecycle: LLM chunking → embedding → Neo4j storage →
Qdrant storage with transactional safety and rollback on failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import NamedTuple

import structlog
from pydantic import BaseModel, Field, model_validator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Expression,
    ExtendedPointId,
    FieldCondition,
    Filter,
    FormulaQuery,
    MatchValue,
    MultExpression,
    PointIdsList,
    PointStruct,
    Prefetch,
    QuantizationSearchParams,
    Record,
    SearchParams,
    SumExpression,
)

from shared.config import VECTOR_SEARCH_THRESHOLD
from shared.embedder import Embedder
from shared.errors import EpisodeStorageError
from shared.types import deterministic_id, new_id

from .. import config
from ..caller import async_embed_documents, async_embed_query, async_llm_call, format_prompt
from ..ess import ESS_FALLBACK, SIGNALS_FALLBACK, CredibilitySignals, ESSResult
from ..prompts import PROSPECTIVE_QUERY_PROMPT
from ..schema import DENSE_VECTOR, Collection
from .derivatives import DerivativeWithEmbedding, chunk_and_embed
from .graph import EpisodeNode, MemoryGraph

log = structlog.get_logger(__name__)


class SearchHit(NamedTuple):
    """A derivative match from vector search."""

    episode_uid: str


@dataclass(frozen=True, slots=True)
class StoredEpisode:
    """Result of a successful episode store operation."""

    episode_uid: str


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

    def restore_last_episode(self, uid: str) -> None:
        """Re-link to the last stored episode after restart."""
        self._last_episode_uid = uid

    async def store(
        self,
        *,
        user_message: str,
        agent_response: str,
        ess: ESSResult = ESS_FALLBACK,
    ) -> StoredEpisode:
        """Store an episode with derivatives in both Neo4j and Qdrant.

        Raises EpisodeStorageError if any critical phase fails.
        """
        episode_uid = new_id()
        now = datetime.now(UTC).isoformat()
        content = f"User: {user_message}\nAssistant: {agent_response}"

        try:
            derivatives = await chunk_and_embed(self._embedder, content, episode_uid)
        except Exception as exc:
            raise EpisodeStorageError(f"Chunking/embedding failed: {exc}") from exc

        if not derivatives:
            raise EpisodeStorageError("No derivatives produced from chunking")

        topics = list(ess.topics)
        log.debug(
            "episode_store_trace",
            episode_uid=episode_uid[:8],
            derivative_count=len(derivatives),
            topics=topics[:3],
        )

        episode_node = EpisodeNode(
            uid=episode_uid,
            content=content,
            summary=ess.summary[:300],
            topics=topics,
            ess_score=ess.score,
            created_at=now,
            valid_at=now,
            utility_score=0.0,
            access_count=0,
            last_accessed=now,
            consolidation_level=1,
            archived=False,
            specificity=ess.signals.specificity,
            grounding=ess.signals.grounding,
            rigor=ess.signals.rigor,
            source_quality=ess.signals.source_quality,
            objectivity=ess.signals.objectivity,
        )
        try:
            await self._graph.store_episode_atomically(
                episode=episode_node,
                prev_episode_uid=self._last_episode_uid,
                topics=topics,
            )
        except Exception as exc:
            log.error(
                "neo4j_episode_write_failed",
                episode_uid=episode_uid[:8],
                error=str(exc),
            )
            raise EpisodeStorageError(f"Neo4j write failed: {exc}") from exc

        try:
            await self._insert_derivatives_qdrant(derivatives, ess.signals)
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
        log.info(
            "episode_stored",
            episode_uid=episode_uid[:8],
            derivative_count=len(derivatives),
        )

        try:
            await self._generate_prospective_queries(derivatives, now, ess.signals)
        except Exception:
            log.warning("prospective_indexing_failed", episode_uid=episode_uid[:8], exc_info=True)

        return StoredEpisode(episode_uid=episode_uid)

    async def vector_search(
        self,
        query: str,
        top_k: int = 20,
        score_threshold: float = VECTOR_SEARCH_THRESHOLD,
        signal_weights: dict[str, float] | None = None,
    ) -> list[SearchHit]:
        """Search Qdrant for similar derivatives, boosted by credibility signals.

        signal_weights maps signal names → additive boost weights (0.0–0.3).
        When None or empty, uses pure semantic similarity with no quality boost.
        The LLM router decides per-query which signals to boost.
        """
        query_embedding = await async_embed_query(self._embedder, query)
        prefetch_filter = Filter(
            must=[FieldCondition(key="archived", match=MatchValue(value=False))]
        )
        terms: list[Expression | str] = ["$score"]
        defaults: dict[str, float] = {}
        if signal_weights:
            for signal, weight in signal_weights.items():
                if weight > 0:
                    terms.append(MultExpression(mult=[weight, signal]))
                    defaults[signal] = 0.0
        response = await self._qdrant.query_points(
            collection_name=Collection.DERIVATIVES,
            prefetch=Prefetch(
                query=query_embedding,
                using=DENSE_VECTOR,
                filter=prefetch_filter,
                limit=top_k * 2,
                params=SearchParams(
                    hnsw_ef=config.settings.qdrant_search_ef,
                    quantization=QuantizationSearchParams(rescore=True, oversampling=2.0),
                ),
            ),
            query=FormulaQuery(
                formula=SumExpression(sum=terms),
                defaults=defaults,
            ),
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        hits = [
            SearchHit(str(p.payload.get("episode_uid", "")))
            for p in response.points
            if p.payload and p.payload.get("episode_uid")
        ]
        log.debug(
            "qdrant_vector_search",
            top_k=top_k,
            hit_count=len(hits),
            signal_weights=signal_weights or {},
        )
        return hits

    async def archive_derivatives(self, episode_uid: str) -> None:
        """Mark derivatives as archived in Qdrant (soft delete)."""
        filt = Filter(must=[FieldCondition(key="episode_uid", match=MatchValue(value=episode_uid))])
        all_ids: list[ExtendedPointId] = []
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
        log.debug(
            "qdrant_delete_derivatives",
            episode_uid=episode_uid[:8],
        )

    async def remove_knowledge_citations(self, episode_uid: str) -> None:
        """Remove an episode's citations from semantic_features knowledge store.

        For each knowledge proposition citing this episode: remove the citation.
        If the proposition has no remaining citations, delete it entirely —
        knowledge without provenance is phantom data.
        """
        from ..schema import SemanticCategory

        scroll_filter = Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value=SemanticCategory.KNOWLEDGE)),
                FieldCondition(key="episode_citations", match=MatchValue(value=episode_uid)),
            ]
        )
        results: list[Record] = []
        offset = None
        while True:
            batch, offset = await self._qdrant.scroll(
                collection_name=Collection.SEMANTIC_FEATURES,
                scroll_filter=scroll_filter,
                limit=500,
                offset=offset,
                with_payload=True,
            )
            results.extend(batch)
            if offset is None:
                break
        if not results:
            return

        orphans: list[ExtendedPointId] = []
        for point in results:
            if not point.payload:
                continue
            raw_citations = point.payload.get("episode_citations")
            citations = [
                c
                for c in (raw_citations if isinstance(raw_citations, list) else [])
                if c != episode_uid
            ]
            if not citations:
                orphans.append(point.id)
            else:
                await self._qdrant.set_payload(
                    collection_name=Collection.SEMANTIC_FEATURES,
                    payload={"episode_citations": citations},
                    points=[point.id],
                )
        if orphans:
            await self._qdrant.delete(
                collection_name=Collection.SEMANTIC_FEATURES,
                points_selector=PointIdsList(points=orphans),
            )
        log.info(
            "knowledge_citations_cleaned",
            episode_uid=episode_uid[:8],
            updated=len(results) - len(orphans),
            orphans_deleted=len(orphans),
        )

    async def _insert_derivatives_qdrant(
        self,
        derivatives: list[DerivativeWithEmbedding],
        signals: CredibilitySignals = SIGNALS_FALLBACK,
    ) -> None:
        """Insert derivative embeddings into Qdrant with credibility signals."""
        points = [
            PointStruct(
                id=d.node.uid,
                vector={DENSE_VECTOR: d.embedding},
                payload={
                    "episode_uid": d.node.source_episode_uid,
                    "text": d.node.text,
                    "archived": False,
                    **signals.as_dict(),
                },
            )
            for d in derivatives
        ]
        await self._qdrant.upsert(collection_name=Collection.DERIVATIVES, points=points)

    async def _generate_prospective_queries(
        self,
        derivatives: list[DerivativeWithEmbedding],
        created_at: str,
        signals: CredibilitySignals = SIGNALS_FALLBACK,
    ) -> None:
        """Generate hypothetical future queries for each derivative and store
        as additional Qdrant embeddings (ROXY-inspired prospective indexing).

        This dramatically improves recall for queries phrased differently from
        the stored content — the generated queries bridge the vocabulary gap.
        """
        for d in derivatives:
            try:
                result = await async_llm_call(
                    instructions=format_prompt(PROSPECTIVE_QUERY_PROMPT, text=d.node.text),
                    response_model=_ProspectiveQueries,
                    fallback=_ProspectiveQueries(),
                    model=config.settings.fast_model,
                )
                queries = [q.strip() for q in result.value.queries if q.strip()][:4]
                if not queries:
                    continue
                query_embeddings = await async_embed_documents(self._embedder, queries)
                points = [
                    PointStruct(
                        id=deterministic_id(f"{d.node.uid}:pq:{i}"),
                        vector={DENSE_VECTOR: emb},
                        payload={
                            "uid": d.node.uid,
                            "episode_uid": d.node.source_episode_uid,
                            "text": d.node.text,
                            "key_concept": d.node.key_concept,
                            "sequence_num": d.node.sequence_num,
                            "archived": False,
                            "created_at": created_at,
                            "prospective_query": query,
                            **signals.as_dict(),
                        },
                    )
                    for i, (query, emb) in enumerate(zip(queries, query_embeddings, strict=True))
                ]
                await self._qdrant.upsert(collection_name=Collection.DERIVATIVES, points=points)
                log.debug(
                    "prospective_queries_stored",
                    derivative_uid=d.node.uid[:8],
                    query_count=len(points),
                )
            except Exception:
                log.warning(
                    "prospective_query_generation_skipped",
                    derivative_uid=d.node.uid[:8],
                    exc_info=True,
                )


class _ProspectiveQueries(BaseModel):
    """LLM-generated hypothetical future queries for prospective indexing."""

    queries: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def truncate_queries(cls, data: object) -> object:
        if isinstance(data, dict) and isinstance(data.get("queries"), list):
            data["queries"] = [str(q) for q in data["queries"][:8] if q]
        return data
