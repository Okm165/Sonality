"""Debug trace helpers for memory architecture diagnostics."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neo4j import AsyncSession
    from psycopg_pool import AsyncConnectionPool

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
    pg_pool: AsyncConnectionPool,
    neo4j_session: AsyncSession,
    sponge: SpongeState,
    label: str = "SNAPSHOT",
) -> None:
    """Dump DB contents to debug log for manual inspection."""
    log.debug("=== MEMORY_SNAPSHOT: %s (interaction #%d) ===", label, sponge.interaction_count)

    for topic, pos in sorted(sponge.opinion_vectors.items()):
        meta = sponge.belief_meta.get(topic)
        log.debug(
            "SNAP_BELIEF topic=%s | pos=%+.3f | conf=%.2f | ev=%d | support=%d | contra=%d",
            topic,
            pos,
            meta.confidence if meta else 0.0,
            meta.evidence_count if meta else 0,
            len(meta.supporting_episode_uids) if meta else 0,
            len(meta.contradicting_episode_uids) if meta else 0,
        )
    for s in sponge.staged_opinion_updates[:10]:
        log.debug(
            "SNAP_STAGED topic=%s | mag=%+.3f | due=%d | prov=%.40s",
            s.topic,
            s.signed_magnitude,
            s.due_interaction,
            s.provenance.replace("\n", " "),
        )

    async with pg_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            "SELECT tag, value, confidence FROM semantic_features "
            "WHERE category = 'knowledge' ORDER BY confidence DESC LIMIT 20"
        )
        for tag, value, conf in await cur.fetchall():
            log.debug(
                "SNAP_KNOWLEDGE [%s] conf=%.2f | %.100s",
                tag,
                float(conf),
                str(value).replace("\n", " "),
            )
        await cur.execute("SELECT COUNT(*) FROM semantic_features WHERE category = 'knowledge'")
        row = await cur.fetchone()
        total_knowledge = row[0] if row else 0
        await cur.execute("SELECT COUNT(*) FROM derivatives WHERE NOT archived")
        row = await cur.fetchone()
        active_derivs = row[0] if row else 0
        log.debug(
            "SNAP_PG total_knowledge=%d | active_derivatives=%d", total_knowledge, active_derivs
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
