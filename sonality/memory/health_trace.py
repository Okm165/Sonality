"""Debug trace helpers for memory architecture diagnostics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import AsyncSession
    from qdrant_client import AsyncQdrantClient

    from .sponge import SpongeState

log = logging.getLogger(__name__)


def trace_belief_provenance(
    interaction_num: int,
    topic: str,
    episode_uid: str,
    edge_type: str,
    strength: float,
    direction: float,
    update_magnitude: str,
    contraction: str,
    reasoning: str,
) -> None:
    """Trace belief provenance assessment result."""
    log.debug(
        "TRACE_PROVENANCE interaction=%d | topic=%s | episode=%s | "
        "edge=%s | strength=%.2f | dir=%+.2f | mag=%s | contract=%s | reason=%.60s",
        interaction_num,
        topic,
        episode_uid[:12],
        edge_type,
        strength,
        direction,
        update_magnitude,
        contraction,
        reasoning.replace("\n", " "),
    )


def trace_consolidation(
    segment_id: str,
    episode_count: int,
    source_content_len: int,
    summary_len: int,
    topics: list[str],
    readiness_confidence: float,
    summary_focus: str,
) -> None:
    """Trace segment consolidation: compression ratio and topic preservation."""
    compression_ratio = source_content_len / summary_len if summary_len > 0 else 0.0
    log.debug(
        "TRACE_CONSOLIDATION segment=%s | episodes=%d | "
        "source_len=%d | summary_len=%d | compression=%.1fx | "
        "topics=%s | confidence=%.2f | focus=%.50s",
        segment_id[:12],
        episode_count,
        source_content_len,
        summary_len,
        compression_ratio,
        topics[:5],
        readiness_confidence,
        summary_focus.replace("\n", " "),
    )


async def dump_memory_snapshot(
    qdrant: AsyncQdrantClient,
    neo4j_session: AsyncSession,
    sponge: SpongeState,
    label: str = "SNAPSHOT",
) -> None:
    """Dump DB contents to debug log for manual inspection."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    log.debug("=== MEMORY_SNAPSHOT: %s (interaction #%d) ===", label, sponge.interaction_count)

    for topic in sorted(sponge.opinion_vectors):
        b = sponge.get_belief(topic)
        log.debug(
            "SNAP_BELIEF topic=%s | pos=%+.3f | conf=%.2f | ev=%d | support=%d | contra=%d",
            topic,
            b.position,
            b.confidence,
            b.evidence_count,
            b.supporting_count,
            b.contradicting_count,
        )
    for s in sponge.staged_opinion_updates[:10]:
        log.debug(
            "SNAP_STAGED topic=%s | mag=%+.3f | due=%d | prov=%.40s",
            s.topic,
            s.signed_magnitude,
            s.due_interaction,
            s.provenance.replace("\n", " "),
        )

    results, _ = await qdrant.scroll(
        collection_name="semantic_features",
        scroll_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="knowledge"))]
        ),
        limit=20,
        with_payload=["tag", "value", "confidence"],
    )
    for p in results:
        if p.payload:
            log.debug(
                "SNAP_KNOWLEDGE [%s] conf=%.2f | %.100s",
                p.payload.get("tag", ""),
                float(p.payload.get("confidence", 0)),
                str(p.payload.get("value", "")).replace("\n", " "),
            )
    count_result = await qdrant.count(
        collection_name="semantic_features",
        count_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="knowledge"))]
        ),
    )
    total_knowledge = count_result.count
    deriv_count = await qdrant.count(
        collection_name="derivatives",
        count_filter=Filter(must=[FieldCondition(key="archived", match=MatchValue(value=False))]),
    )
    active_derivs = deriv_count.count
    log.debug(
        "SNAP_QDRANT total_knowledge=%d | active_derivatives=%d", total_knowledge, active_derivs
    )

    result = await neo4j_session.run("MATCH (b:Belief) RETURN b.topic AS topic ORDER BY b.topic")
    log.debug("SNAP_GRAPH beliefs=%s", [r["topic"] async for r in result])

    result = await neo4j_session.run(
        "MATCH (t:Topic) RETURN t.name AS name, t.episode_count AS cnt "
        "ORDER BY t.episode_count DESC LIMIT 15"
    )
    log.debug("SNAP_GRAPH topics=%s", [(r["name"], r["cnt"]) async for r in result])

    result = await neo4j_session.run(
        "MATCH (e:Episode) WHERE NOT e.archived "
        "RETURN e.uid AS uid, e.summary AS summary, e.ess_score AS ess "
        "ORDER BY e.created_at DESC LIMIT 10"
    )
    for r in [r async for r in result]:
        log.debug("SNAP_EPISODE %s | ess=%.2f | %s", r["uid"][:8], r["ess"], str(r["summary"])[:60])

    log.debug("=== END MEMORY_SNAPSHOT: %s ===", label)
