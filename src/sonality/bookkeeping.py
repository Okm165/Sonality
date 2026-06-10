"""Async post-response bookkeeping pipeline.

After each response, episodes are stored, belief provenance assessed,
semantic features extracted, knowledge persisted, and forgetting evaluated.
All operations are async and run on the agent's event loop without blocking
the response thread.
"""

from __future__ import annotations

import asyncio
import dataclasses

import structlog
from qdrant_client import AsyncQdrantClient

from shared.embedder import Embedder

from . import config
from .ess import ESS_FALLBACK, ESSResult, classify
from .memory import (
    DualEpisodeStore,
    EpisodeEvidence,
    MemoryGraph,
    SemanticFeatureExtractor,
    assess_belief_evidence_batch,
)
from .memory.forgetting import assess_and_forget
from .memory.graph import BeliefNode
from .request_identity import get_request_identity
from .schema import SemanticCategory, normalize_topic

log = structlog.get_logger(__name__)

__all__ = [
    "BookkeepingItem",
    "classify_ess",
    "enqueue_bookkeeping",
    "post_ingest",
    "process_bookkeeping",
]


@dataclasses.dataclass(frozen=True, slots=True)
class BookkeepingItem:
    """Payload queued for async post-response bookkeeping."""

    user_message: str
    agent_response: str
    ess: ESSResult
    ltm_content: str = ""


def classify_ess(user_message: str, existing_topics: str = "") -> ESSResult:
    """Run ESS classification. Returns ESS_FALLBACK on error."""
    try:
        result = classify(user_message, existing_topics)
    except Exception:
        log.error("ess_classification_failed", exc_info=True)
        return ESS_FALLBACK
    if result.topics:
        normalized = tuple(normalize_topic(t) for t in result.topics)
        result = dataclasses.replace(result, topics=normalized)
    return result


def enqueue_bookkeeping(
    queue: asyncio.Queue[BookkeepingItem],
    loop: asyncio.AbstractEventLoop,
    user_message: str,
    agent_response: str,
    ess: ESSResult,
    ltm_content: str = "",
) -> None:
    """Submit bookkeeping to the async processing queue.

    If the queue is full (maxsize=64), the item is dropped with a warning
    rather than blocking the response thread.
    """
    item = BookkeepingItem(
        user_message=user_message,
        agent_response=agent_response,
        ess=ess,
        ltm_content=ltm_content,
    )

    def _put() -> None:
        try:
            queue.put_nowait(item)
        except asyncio.QueueFull:
            log.warning("bookkeep_queue_full", qsize=queue.maxsize)

    loop.call_soon_threadsafe(_put)


async def _run_pipeline(
    *,
    episode_uid: str,
    user_message: str,
    agent_response: str,
    ess: ESSResult,
    graph: MemoryGraph,
    dual_store: DualEpisodeStore,
    semantic_worker: SemanticFeatureExtractor,
    qdrant: AsyncQdrantClient | None,
    embedder: Embedder | None,
    ltm_content: str,
    beliefs_override: dict[str, BeliefNode] | None = None,
    semantic_categories: tuple[SemanticCategory, ...] = (),
) -> None:
    """Shared post-episode pipeline: provenance → semantic features → knowledge → forgetting.

    Called by both ``process_bookkeeping`` (queue-driven) and ``post_ingest``
    (synchronous ingest path). ``beliefs_override`` lets callers supply a
    pre-loaded beliefs dict; when *None* beliefs are fetched from graph.
    """
    if ess.belief_update_recommended:
        topics = [normalize_topic(t) for t in ess.topics[:10] if t]
        if topics:
            try:
                if beliefs_override is not None:
                    beliefs_dict = beliefs_override
                else:
                    beliefs_dict = {b.topic: b for b in await graph.get_all_beliefs()}
                await assess_belief_evidence_batch(
                    topics=topics,
                    evidence=EpisodeEvidence(
                        episode_uid=episode_uid,
                        episode_content=user_message,
                        ess_score=ess.score,
                        signals=ess.signals,
                    ),
                    beliefs=beliefs_dict,
                    graph=graph,
                )
            except Exception:
                log.error("provenance_failed", episode_uid=episode_uid[:8], exc_info=True)

    content = (
        f"User: {user_message}\nAssistant: {agent_response}\n"
        f"ESS: {ess.score:.2f} ({ess.signals.summary_str()})"
        if agent_response
        else f"Content: {user_message}\nESS: {ess.score:.2f} ({ess.signals.summary_str()})"
    )
    await semantic_worker.process_episode(episode_uid, content, semantic_categories)

    if ltm_content and qdrant is not None and embedder is not None:
        try:
            from .memory.knowledge_extract import extract_and_store_knowledge

            stored_k, boosted = await extract_and_store_knowledge(
                ltm_content, episode_uid, qdrant, embedder
            )
            log.info("ltm_knowledge_persisted", stored=stored_k, boosted=boosted)
        except Exception:
            log.warning("ltm_knowledge_extraction_failed", exc_info=True)

    try:
        candidates = await graph.get_forgetting_candidates(
            limit=config.settings.forgetting_candidate_limit
        )
        if candidates:
            snapshot = await graph.get_personality_snapshot()
            await assess_and_forget(
                candidates,
                graph,
                dual_store,
                snapshot_excerpt=snapshot.text[:500],
            )
    except Exception:
        log.warning("forgetting_cycle_failed", exc_info=True)


async def process_bookkeeping(
    item: BookkeepingItem,
    *,
    graph: MemoryGraph,
    dual_store: DualEpisodeStore,
    semantic_worker: SemanticFeatureExtractor,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
) -> None:
    """Fully async bookkeeping pipeline.

    LLM calls go through ``async_llm_call`` (gated by sonality's
    ``_llm_gate`` semaphore).  All I/O (graph, Qdrant, embeddings)
    is native async — no thread pool slots consumed.
    """
    ess, user_message, agent_response = item.ess, item.user_message, item.agent_response
    log.debug(
        "bookkeep_start",
        score=round(ess.score, 2),
        signals=ess.signals.summary_str(),
        update=ess.belief_update_recommended,
        topics=list(ess.topics),
    )

    try:
        stored = await dual_store.store(
            user_message=user_message,
            agent_response=agent_response,
            ess=ess,
        )
        episode_uid = stored.episode_uid
        log.info("episode_stored", uid=episode_uid[:8], topics=list(ess.topics))
    except Exception:
        log.error("episode_storage_failed", exc_info=True)
        return

    categories = (SemanticCategory.KNOWLEDGE,) if not agent_response else ()
    await _run_pipeline(
        episode_uid=episode_uid,
        user_message=user_message,
        agent_response=agent_response,
        ess=ess,
        graph=graph,
        dual_store=dual_store,
        semantic_worker=semantic_worker,
        qdrant=qdrant,
        embedder=embedder,
        ltm_content=item.ltm_content,
        semantic_categories=categories,
    )


async def post_ingest(
    text: str,
    agent_response: str,
    ess: ESSResult,
    *,
    graph: MemoryGraph,
    dual_store: DualEpisodeStore,
    semantic_worker: SemanticFeatureExtractor,
    qdrant: AsyncQdrantClient | None = None,
    embedder: Embedder | None = None,
    ltm_content: str = "",
) -> None:
    """Async post-ingest: store → provenance → semantic features → knowledge → forgetting."""
    stored = await dual_store.store(
        user_message=text,
        agent_response=agent_response,
        ess=ess,
    )
    episode_uid = stored.episode_uid
    log.info("episode_stored", uid=episode_uid[:8], topics=list(ess.topics))

    bundle = get_request_identity()
    beliefs_override: dict[str, BeliefNode] | None = (
        {b.topic: b for b in bundle.all_beliefs} if bundle is not None else None
    )
    await _run_pipeline(
        episode_uid=episode_uid,
        user_message=text,
        agent_response=agent_response,
        ess=ess,
        graph=graph,
        dual_store=dual_store,
        semantic_worker=semantic_worker,
        qdrant=qdrant,
        embedder=embedder,
        ltm_content=ltm_content,
        beliefs_override=beliefs_override,
        semantic_categories=(SemanticCategory.KNOWLEDGE,),
    )
