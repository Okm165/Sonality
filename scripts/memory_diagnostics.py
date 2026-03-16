"""Memory health diagnostics for Neo4j + Qdrant dual-store.

Checks consistency, orphan detection, collection integrity, cross-store sync,
temporal chain integrity, payload completeness, and isolated node detection.

Usage:
    uv run python scripts/memory_diagnostics.py [--fix-orphans]
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from dataclasses import dataclass, field

from neo4j import AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

from sonality import config
from sonality.memory.graph import MemoryGraph

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


@dataclass
class DiagnosticsReport:
    neo4j_episodes: int = 0
    neo4j_derivatives: int = 0
    neo4j_segments: int = 0
    neo4j_topics: int = 0
    neo4j_beliefs: int = 0
    qdrant_derivatives_total: int = 0
    qdrant_derivatives_active: int = 0
    qdrant_derivatives_archived: int = 0
    qdrant_semantic_features: int = 0
    orphan_qdrant_only: list[str] = field(default_factory=list)
    orphan_neo4j_only: list[str] = field(default_factory=list)
    # Chain integrity
    temporal_chain_gaps: int = 0
    episodes_without_segments: int = 0
    # Payload completeness
    qdrant_missing_uid: int = 0
    qdrant_missing_content: int = 0
    # Isolation
    isolated_topics: int = 0
    isolated_episodes: int = 0
    issues: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not self.issues


async def _qdrant_collection_count(qdrant: AsyncQdrantClient, collection: str) -> int:
    try:
        info = await qdrant.get_collection(collection)
        return info.points_count or 0
    except Exception:
        return -1


async def run_diagnostics(fix_orphans: bool = False) -> DiagnosticsReport:
    report = DiagnosticsReport()
    driver = AsyncGraphDatabase.driver(
        config.NEO4J_URL,
        auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
    )
    graph = MemoryGraph(driver)
    qdrant = AsyncQdrantClient(url=config.QDRANT_URL)

    try:
        # ── Neo4j counts ────────────────────────────────────────────────
        async with driver.session(database=config.NEO4J_DATABASE) as session:
            for label, attr in [
                ("Episode", "neo4j_episodes"),
                ("Derivative", "neo4j_derivatives"),
                ("Segment", "neo4j_segments"),
                ("Topic", "neo4j_topics"),
                ("Belief", "neo4j_beliefs"),
            ]:
                result = await session.run(f"MATCH (n:{label}) RETURN count(n) AS cnt")
                record = await result.single()
                setattr(report, attr, record["cnt"] if record else 0)

        # ── Neo4j structural checks ──────────────────────────────────────
        async with driver.session(database=config.NEO4J_DATABASE) as session:
            # Temporal chain gaps: episodes with no TEMPORAL_NEXT edge AND not the latest
            # (episodes that have a successor in time but the edge is missing)
            result = await session.run("""
                MATCH (e:Episode)
                WHERE NOT (e)-[:TEMPORAL_NEXT]->()
                  AND NOT (e)<-[:TEMPORAL_NEXT]-()
                  AND EXISTS { MATCH (other:Episode) WHERE other.created_at > e.created_at }
                RETURN count(e) AS cnt
            """)
            record = await result.single()
            report.temporal_chain_gaps = record["cnt"] if record else 0

            # Episodes without any segments (all episodes should have ≥1 segment)
            result = await session.run("""
                MATCH (e:Episode)
                WHERE NOT (e)-[:BELONGS_TO_SEGMENT]->()
                RETURN count(e) AS cnt
            """)
            record = await result.single()
            report.episodes_without_segments = record["cnt"] if record else 0

            # Isolated topics (not linked to any Episode via DISCUSSES)
            result = await session.run("""
                MATCH (t:Topic)
                WHERE NOT ()-[:DISCUSSES]->(t)
                RETURN count(t) AS cnt
            """)
            record = await result.single()
            report.isolated_topics = record["cnt"] if record else 0

            # Isolated episodes (no edges at all — neither outgoing nor incoming)
            result = await session.run("""
                MATCH (e:Episode)
                WHERE NOT (e)--()
                RETURN count(e) AS cnt
            """)
            record = await result.single()
            report.isolated_episodes = record["cnt"] if record else 0

        # ── Qdrant counts ───────────────────────────────────────────────
        collections = await qdrant.get_collections()
        coll_names = {c.name for c in collections.collections}

        if "derivatives" in coll_names:
            info = await qdrant.get_collection("derivatives")
            report.qdrant_derivatives_total = info.points_count or 0

            # Active vs archived counts
            active_count = 0
            offset = None
            while True:
                results, offset = await qdrant.scroll(
                    collection_name="derivatives",
                    scroll_filter=Filter(
                        must=[FieldCondition(key="archived", match=MatchValue(value=False))]
                    ),
                    limit=1000,
                    with_payload=False,
                    offset=offset,
                )
                active_count += len(results)
                if offset is None:
                    break
            report.qdrant_derivatives_active = active_count
            report.qdrant_derivatives_archived = report.qdrant_derivatives_total - active_count

            # Payload completeness: sample up to 5000 points for missing required fields
            sample_results, _ = await qdrant.scroll(
                collection_name="derivatives",
                limit=5000,
                with_payload=["uid", "text", "episode_uid"],
            )
            for point in sample_results:
                payload = point.payload or {}
                if not payload.get("uid"):
                    report.qdrant_missing_uid += 1
                if not payload.get("text"):
                    report.qdrant_missing_content += 1
        else:
            report.warnings.append("Qdrant 'derivatives' collection does not exist")

        if "semantic_features" in coll_names:
            info = await qdrant.get_collection("semantic_features")
            report.qdrant_semantic_features = info.points_count or 0

        # ── Cross-store consistency check ───────────────────────────────
        if "derivatives" in coll_names:
            qdrant_results, _ = await qdrant.scroll(
                collection_name="derivatives",
                limit=50000,
                with_payload=["uid"],
            )
            qdrant_uids = {str(p.payload.get("uid", "")) for p in qdrant_results if p.payload}
            neo4j_uids = await graph.list_derivative_uids()

            orphan_qdrant = sorted(qdrant_uids - neo4j_uids)
            orphan_neo4j = sorted(neo4j_uids - qdrant_uids)

            report.orphan_qdrant_only = orphan_qdrant[:50]
            report.orphan_neo4j_only = orphan_neo4j[:50]

            if orphan_qdrant:
                msg = f"{len(orphan_qdrant)} orphan derivatives in Qdrant (not in Neo4j)"
                report.issues.append(msg)
                log.warning(msg)
                if fix_orphans:
                    await qdrant.delete(
                        collection_name="derivatives",
                        points_selector=orphan_qdrant[:50],
                    )
                    log.info("Deleted %d Qdrant-only orphans", len(orphan_qdrant[:50]))

            if orphan_neo4j:
                msg = f"{len(orphan_neo4j)} orphan derivatives in Neo4j (not in Qdrant)"
                report.issues.append(msg)
                log.warning(msg)
                if fix_orphans:
                    await graph.delete_derivatives(orphan_neo4j[:50])
                    log.info("Deleted %d Neo4j-only orphans", len(orphan_neo4j[:50]))

        # ── Structural health checks ────────────────────────────────────
        if report.neo4j_episodes > 0 and report.neo4j_derivatives == 0:
            report.issues.append("Episodes exist in Neo4j but no derivatives — broken storage")

        active_ratio = (
            report.qdrant_derivatives_active / max(report.qdrant_derivatives_total, 1)
            if report.qdrant_derivatives_total > 0
            else 1.0
        )
        if active_ratio < 0.5:
            report.warnings.append(
                f"High archive ratio: {1 - active_ratio:.0%} of Qdrant derivatives are archived"
            )

        if report.neo4j_episodes > 0:
            avg_derivs = report.neo4j_derivatives / report.neo4j_episodes
            if avg_derivs < 1.0:
                report.issues.append(
                    f"Low derivative density: {avg_derivs:.1f} derivatives/episode (expected ≥ 1)"
                )
            elif avg_derivs > 50:
                report.warnings.append(
                    f"High derivative density: {avg_derivs:.1f} derivatives/episode"
                )

        if report.temporal_chain_gaps > 0:
            report.warnings.append(
                f"{report.temporal_chain_gaps} episodes appear disconnected from temporal chain"
            )

        if report.episodes_without_segments > 0:
            report.issues.append(
                f"{report.episodes_without_segments} episodes have no segments (broken chunking)"
            )

        if report.qdrant_missing_uid > 0:
            report.issues.append(
                f"{report.qdrant_missing_uid} Qdrant derivatives missing uid payload field"
            )

        if report.qdrant_missing_content > 0:
            report.warnings.append(
                f"{report.qdrant_missing_content} Qdrant derivatives missing text payload field"
            )

        if report.isolated_topics > 5:
            report.warnings.append(
                f"{report.isolated_topics} isolated Topic nodes with no Episode/Belief links"
            )

        if report.isolated_episodes > 0:
            report.issues.append(
                f"{report.isolated_episodes} fully isolated Episode nodes (no edges)"
            )

    finally:
        await driver.close()
        await qdrant.close()

    return report


def _print_report(report: DiagnosticsReport) -> None:
    print("\n── Memory Health Diagnostics ──────────────────────────────────")
    print(f"Status: {'✓ HEALTHY' if report.healthy else '✗ ISSUES DETECTED'}")
    print()
    print("Neo4j:")
    print(f"  Episodes:    {report.neo4j_episodes}")
    print(f"  Derivatives: {report.neo4j_derivatives}")
    print(f"  Segments:    {report.neo4j_segments}")
    print(f"  Topics:      {report.neo4j_topics}")
    print(f"  Beliefs:     {report.neo4j_beliefs}")
    print()
    print("Qdrant:")
    print(f"  derivatives (total):    {report.qdrant_derivatives_total}")
    print(f"  derivatives (active):   {report.qdrant_derivatives_active}")
    print(f"  derivatives (archived): {report.qdrant_derivatives_archived}")
    print(f"  semantic_features:      {report.qdrant_semantic_features}")
    print()
    print("Integrity:")
    print(f"  Temporal chain gaps:      {report.temporal_chain_gaps}")
    print(f"  Episodes without segments:{report.episodes_without_segments}")
    print(f"  Isolated topics:          {report.isolated_topics}")
    print(f"  Isolated episodes:        {report.isolated_episodes}")
    print(f"  Qdrant missing uid:       {report.qdrant_missing_uid}")
    print(f"  Qdrant missing content:   {report.qdrant_missing_content}")
    print()
    if report.orphan_qdrant_only:
        print(f"Qdrant-only orphans ({len(report.orphan_qdrant_only)} shown, may be truncated):")
        for uid in report.orphan_qdrant_only[:10]:
            print(f"  {uid}")
    if report.orphan_neo4j_only:
        print(f"Neo4j-only orphans ({len(report.orphan_neo4j_only)} shown, may be truncated):")
        for uid in report.orphan_neo4j_only[:10]:
            print(f"  {uid}")
    if report.issues:
        print("Issues:")
        for issue in report.issues:
            print(f"  ✗ {issue}")
    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  ⚠ {warning}")
    if report.healthy and not report.warnings:
        print("  No issues or warnings found.")
    print()


async def _main(fix_orphans: bool) -> int:
    report = await run_diagnostics(fix_orphans=fix_orphans)
    _print_report(report)
    return 0 if report.healthy else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory health diagnostics")
    parser.add_argument(
        "--fix-orphans",
        action="store_true",
        help="Auto-delete orphan derivatives from both stores",
    )
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(fix_orphans=args.fix_orphans)))
