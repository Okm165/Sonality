"""Full retrieval orchestration: route → multi-pass search → expand → rerank → format.

Single entry point ``retrieve()`` hides routing decisions, graph/vector search,
temporal expansion, reranking, and semantic feature search behind a simple
``async retrieve(query, ...) -> list[str]`` interface.
"""

from __future__ import annotations

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import QuantizationSearchParams, SearchParams

from shared.config import VECTOR_SEARCH_THRESHOLD
from shared.embedder import Embedder

from ... import config
from ...caller import async_embed_query
from ...schema import DENSE_VECTOR, Collection
from ..dual_store import DualEpisodeStore
from ..graph import EpisodeNode, MemoryGraph, format_episode_line
from .reranker import rerank_episodes
from .router import (
    QueryCategory,
    RoutingDecision,
    SemanticMemoryDecision,
    TemporalExpansionDecision,
    route_query,
)

log = structlog.get_logger(__name__)


async def retrieve(
    query: str,
    *,
    graph: MemoryGraph,
    dual_store: DualEpisodeStore,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
) -> list[str]:
    """Full retrieval pipeline: route → multi-pass search → expand → rerank.

    Returns formatted episode lines and semantic feature strings ready for
    prompt injection. An empty list means nothing relevant was found.
    """
    decision = await route_query(query)
    log.debug(
        "route", category=decision.category, n=decision.n_results, passes=len(decision.passes)
    )
    if decision.category == QueryCategory.NONE:
        return []

    episodes = await _fetch_episodes_multi_pass(decision, query, graph=graph, dual_store=dual_store)

    if decision.temporal_expansion is TemporalExpansionDecision.EXPAND and episodes:
        expanded_uids: set[str] = set()
        for ep in episodes:
            for n in await graph.traverse_temporal_context(ep.uid):
                expanded_uids.add(n.uid)
        new_uids = [u for u in expanded_uids if u not in {e.uid for e in episodes}]
        if new_uids:
            episodes.extend(await graph.get_episodes(new_uids))

    if len(episodes) > 1 and len(episodes) > decision.n_results:
        episodes = await rerank_episodes(query, episodes)

    selected = episodes[: decision.n_results]
    if selected:
        await graph.update_episode_access([ep.uid for ep in selected])

    semantic_context: list[str] = []
    if decision.semantic_memory is SemanticMemoryDecision.SEARCH:
        semantic_context = await _search_semantic_features(
            query, top_k=decision.n_results, qdrant=qdrant, embedder=embedder
        )

    episode_context = [
        format_episode_line(
            created_at=ep.created_at,
            summary=ep.summary,
            content=ep.content,
            content_limit=config.settings.episode_content_limit,
            ess_score=ep.ess_score,
            source_quality=ep.source_quality,
            grounding=ep.grounding,
        )
        for ep in selected
    ]
    log.debug(
        "retrieval",
        category=decision.category,
        episodes=len(selected),
        semantic=len(semantic_context),
    )
    return [*episode_context, *semantic_context]


async def _fetch_episodes_multi_pass(
    decision: RoutingDecision,
    user_message: str,
    *,
    graph: MemoryGraph,
    dual_store: DualEpisodeStore,
) -> list[EpisodeNode]:
    """Run one or more search passes, each with its own query and signal weights.

    Results are merged by UID — earlier passes have priority (appear first).
    Graph-based hits (topic, belief) run once against the original query.
    """
    over_fetch = decision.n_results * config.settings.retrieval_over_fetch_factor
    seen: set[str] = set()
    merged: list[EpisodeNode] = []

    topic_hits = await graph.find_topic_related_episodes(user_message, limit=over_fetch)
    belief_hits = (
        await graph.find_belief_related_episodes(user_message, limit=over_fetch)
        if decision.category == QueryCategory.BELIEF_QUERY
        else []
    )
    for ep in belief_hits + topic_hits:
        if ep.uid not in seen:
            seen.add(ep.uid)
            merged.append(ep)

    for search_pass in decision.passes:
        query = search_pass.query or user_message
        vector_hits = await dual_store.vector_search(
            query,
            top_k=over_fetch,
            signal_weights=search_pass.signal_weights,
        )
        uids = [h.episode_uid for h in vector_hits if h.episode_uid not in seen]
        if uids:
            for ep in await graph.get_episodes(uids):
                if ep.uid not in seen:
                    seen.add(ep.uid)
                    merged.append(ep)

    return merged


async def _search_semantic_features(
    query: str,
    *,
    top_k: int,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
) -> list[str]:
    """Search all semantic features (personality, preferences, knowledge) by similarity."""
    query_embedding = await async_embed_query(embedder, query)
    response = await qdrant.query_points(
        collection_name=Collection.SEMANTIC_FEATURES,
        query=query_embedding,
        using=DENSE_VECTOR,
        limit=top_k,
        with_payload=True,
        score_threshold=VECTOR_SEARCH_THRESHOLD,
        search_params=SearchParams(
            hnsw_ef=config.settings.qdrant_search_ef,
            quantization=QuantizationSearchParams(rescore=True),
        ),
    )
    return [
        f"[semantic/{p.payload.get('category', '')}] "
        f"{p.payload.get('tag', '')}.{p.payload.get('feature_name', '')}: "
        f"{p.payload.get('value', '')} "
        f"(conf={float(p.payload.get('confidence') or 0):.2f}, score={p.score:.3f})"
        for p in response.points
        if p.payload
    ]
