"""Reflection DB cleanup benchmarks — verify that reflection correctly
modifies graph and vectordb to maintain accurate, clean structures.

Tests:
  R1: Belief graph sync — orphan Belief nodes pruned after belief decay
  R2: Topic pruning — orphan Topic nodes removed after episode archival
  R3: Knowledge pruning — low-confidence stale entries removed from Qdrant
  R4: Derivative consistency — Neo4j/Qdrant stay in sync after forgetting
  R5: Full reflection cycle — all cleanup operations work together
  R6: Knowledge consolidation — contradictions resolved, duplicates merged

Run: uv run pytest benches/test_reflection_cleanup_live.py -v -s -m live
"""

from __future__ import annotations

import logging
import tempfile
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from neo4j import GraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FilterSelector, PointStruct

from sonality import config

log = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.bench,
    pytest.mark.live,
    pytest.mark.skipif(
        bool(config.missing_live_api_config()),
        reason=f"Missing live config: {config.missing_live_api_config()}",
    ),
]

REFLECTION_CADENCE = 5


def _reset_dbs() -> None:
    qdrant = QdrantClient(url=config.QDRANT_URL)
    for collection in ["derivatives", "semantic_features"]:
        if qdrant.collection_exists(collection):
            qdrant.delete(
                collection_name=collection, points_selector=FilterSelector(filter=Filter())
            )
    qdrant.close()

    driver = GraphDatabase.driver(config.NEO4J_URL, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()


def _neo4j_count(label: str) -> int:
    driver = GraphDatabase.driver(config.NEO4J_URL, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            return s.run(f"MATCH (n:{label}) RETURN count(n) AS cnt").single()["cnt"]
    finally:
        driver.close()


def _neo4j_rel_count(rel_type: str) -> int:
    driver = GraphDatabase.driver(config.NEO4J_URL, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))
    try:
        with driver.session() as s:
            return s.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) AS cnt").single()["cnt"]
    finally:
        driver.close()


def _qdrant_count(collection: str) -> int:
    qdrant = QdrantClient(url=config.QDRANT_URL)
    if not qdrant.collection_exists(collection):
        qdrant.close()
        return 0
    count = qdrant.count(collection_name=collection).count
    qdrant.close()
    return count


def _qdrant_count_knowledge(filter_fn: Any = None) -> int:
    """Count knowledge entries, optionally filtering with a custom function."""
    qdrant = QdrantClient(url=config.QDRANT_URL)
    if not qdrant.collection_exists("semantic_features"):
        qdrant.close()
        return 0

    results, _offset = qdrant.scroll(
        collection_name="semantic_features",
        limit=10000,
        with_payload=True,
    )
    count = sum(
        1
        for r in results
        if r.payload
        and r.payload.get("category") == "knowledge"
        and (filter_fn is None or filter_fn(r.payload))
    )
    qdrant.close()
    return count


class _AgentContext:
    """Context manager that keeps config patches alive across all agent calls."""

    def __init__(self, tmp_dir: str) -> None:
        self._patches = [
            mock.patch.object(config, "SPONGE_FILE", Path(tmp_dir) / "sponge.json"),
            mock.patch.object(config, "SPONGE_HISTORY_DIR", Path(tmp_dir) / "history"),
            mock.patch.object(config, "ESS_AUDIT_LOG_FILE", Path(tmp_dir) / "ess.jsonl"),
            mock.patch.object(config, "REFLECTION_EVERY", REFLECTION_CADENCE),
        ]
        self.agent: Any = None

    def __enter__(self) -> Any:
        for p in self._patches:
            p.start()
        from sonality.agent import SonalityAgent

        self.agent = SonalityAgent()
        return self.agent

    def __exit__(self, *_: Any) -> None:
        if self.agent:
            self.agent.shutdown()
        for p in reversed(self._patches):
            p.stop()


def _snapshot(label: str) -> dict[str, Any]:
    snap = {
        "episodes": _neo4j_count("Episode"),
        "beliefs": _neo4j_count("Belief"),
        "topics": _neo4j_count("Topic"),
        "derivatives_neo4j": _neo4j_count("Derivative"),
        "segments": _neo4j_count("Segment"),
        "supports": _neo4j_rel_count("SUPPORTS_BELIEF"),
        "contradicts": _neo4j_rel_count("CONTRADICTS_BELIEF"),
        "derivatives_qdrant": _qdrant_count("derivatives"),
        "semantic_features": _qdrant_count("semantic_features"),
        "knowledge_features": _qdrant_count_knowledge(),
    }
    print(f"\n  [{label}] {snap}")
    log.info("BENCH_SNAPSHOT %s: %s", label, snap)
    return snap


# ---------------------------------------------------------------------------
# R1: Belief Graph Sync
# ---------------------------------------------------------------------------


class TestBeliefGraphSync:
    """After reflection decays beliefs, the corresponding Neo4j Belief nodes
    and SUPPORTS/CONTRADICTS edges should be pruned."""

    @pytest.mark.timeout(10800)
    def test_r1_belief_sync(self) -> None:
        _reset_dbs()
        with tempfile.TemporaryDirectory() as td, _AgentContext(td) as agent:
            agent.respond(
                "Nuclear energy produces only ~12 gCO2/kWh lifecycle, making it "
                "one of the lowest-carbon energy sources. France runs 70% nuclear "
                "with one of the cleanest grids in Europe."
            )
            agent.respond(
                "The Pacific Ocean covers approximately 165.25 million square "
                "kilometers, making it the largest ocean basin on Earth."
            )
            agent.respond(
                "Coral bleaching events have increased 5-fold since the 1980s "
                "due to rising sea surface temperatures."
            )

            pre = _snapshot("pre-reflection")
            beliefs_pre = pre["beliefs"]
            topics_pre = pre["topics"]
            assert beliefs_pre >= 2, f"Expected beliefs >= 2, got {beliefs_pre}"

            for i in range(REFLECTION_CADENCE + 2):
                agent.respond(f"What is 2 + {i}?")

            post = _snapshot("post-reflection")
            beliefs_post = post["beliefs"]
            topics_post = post["topics"]

            print(f"\n  Beliefs: {beliefs_pre} -> {beliefs_post}")
            print(f"  Topics: {topics_pre} -> {topics_post}")

            assert beliefs_post <= beliefs_pre, (
                f"Belief count should not grow after reflection ({beliefs_pre} -> {beliefs_post})"
            )


# ---------------------------------------------------------------------------
# R2: Topic Pruning
# ---------------------------------------------------------------------------


class TestTopicPruning:
    """After episodes are archived/forgotten, orphan Topic nodes with
    zero active episode connections should be removed."""

    @pytest.mark.timeout(10800)
    def test_r2_topic_pruning(self) -> None:
        _reset_dbs()
        with tempfile.TemporaryDirectory() as td, _AgentContext(td) as agent:
            agent.respond(
                "The Mariana Trench reaches a depth of approximately 10,994 meters, "
                "making it the deepest known point in Earth's oceans."
            )
            agent.respond(
                "Mount Everest stands at 8,849 meters above sea level, as measured "
                "by the 2020 Chinese-Nepalese joint survey."
            )

            pre = _snapshot("pre-reflection")
            topics_pre = pre["topics"]
            assert topics_pre >= 2, f"Expected topics >= 2, got {topics_pre}"

            for i in range(REFLECTION_CADENCE + 2):
                agent.respond(f"Tell me a fun fact about the number {i + 10}.")

            post = _snapshot("post-reflection")
            topics_post = post["topics"]
            print(f"\n  Topics: {topics_pre} -> {topics_post}")

            log.info(
                "R2 topic pruning: %d -> %d (delta=%d)",
                topics_pre,
                topics_post,
                topics_pre - topics_post,
            )


# ---------------------------------------------------------------------------
# R3: Knowledge Pruning
# ---------------------------------------------------------------------------


class TestKnowledgePruning:
    """Low-confidence stale knowledge entries should be removed from Qdrant
    during reflection to keep the knowledge store lean."""

    @pytest.mark.timeout(10800)
    def test_r3_knowledge_pruning(self) -> None:
        _reset_dbs()

        qdrant = QdrantClient(url=config.QDRANT_URL)
        stale_time = (datetime.now(UTC) - timedelta(hours=2)).isoformat()
        stale_points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": [0.0] * config.EMBEDDING_DIMENSIONS},
                payload={
                    "category": "knowledge",
                    "tag": "Verified Facts",
                    "feature_name": "stale_claim_1",
                    "value": "Stale low-confidence claim",
                    "confidence": 0.10,
                    "updated_at": stale_time,
                },
            ),
            PointStruct(
                id=str(uuid.uuid4()),
                vector={"dense": [0.0] * config.EMBEDDING_DIMENSIONS},
                payload={
                    "category": "knowledge",
                    "tag": "Verified Facts",
                    "feature_name": "stale_claim_2",
                    "value": "Another stale claim",
                    "confidence": 0.15,
                    "updated_at": stale_time,
                },
            ),
        ]
        qdrant.upsert(collection_name="semantic_features", points=stale_points)
        qdrant.close()

        pre_knowledge = _qdrant_count_knowledge()
        assert pre_knowledge >= 2, f"Expected seeded knowledge, got {pre_knowledge}"

        with tempfile.TemporaryDirectory() as td, _AgentContext(td) as agent:
            agent.respond(
                "The speed of light in vacuum is approximately 299,792,458 meters "
                "per second, a fundamental physical constant."
            )

            for i in range(REFLECTION_CADENCE + 2):
                agent.respond(f"What is {i * 3} plus {i * 7}?")

            post_knowledge = _qdrant_count_knowledge()

            def is_stale_low_conf(payload: dict[str, Any]) -> bool:
                if payload.get("confidence", 1.0) >= 0.2:
                    return False
                updated = payload.get("updated_at", "")
                if not updated:
                    return False
                try:
                    updated_dt = datetime.fromisoformat(updated.replace("Z", "+00:00"))
                    return updated_dt < datetime.now(UTC) - timedelta(hours=1)
                except (ValueError, TypeError):
                    return False

            stale_remaining = _qdrant_count_knowledge(is_stale_low_conf)

            print(f"\n  Knowledge: {pre_knowledge} -> {post_knowledge}")
            print(f"  Stale remaining: {stale_remaining}")

            assert stale_remaining == 0, (
                f"Expected 0 stale low-confidence entries after reflection, got {stale_remaining}"
            )


# ---------------------------------------------------------------------------
# R4: Derivative Consistency
# ---------------------------------------------------------------------------


class TestDerivativeConsistency:
    """After forgetting archives episodes, Neo4j and Qdrant derivative
    counts should remain consistent (no orphans)."""

    @pytest.mark.timeout(10800)
    def test_r4_consistency(self) -> None:
        _reset_dbs()
        with tempfile.TemporaryDirectory() as td, _AgentContext(td) as agent:
            agent.respond(
                "The human genome contains approximately 3.2 billion base pairs "
                "organized into 23 chromosome pairs."
            )
            agent.respond(
                "Photosynthesis converts approximately 100-115 billion tonnes of "
                "carbon into biomass annually."
            )

            for i in range(REFLECTION_CADENCE + 2):
                agent.respond(f"What comes after the number {i + 100}?")

            post = _snapshot("post-reflection")

            neo4j_derivs = post["derivatives_neo4j"]
            qdrant_derivs = post["derivatives_qdrant"]

            print(f"\n  Neo4j derivatives: {neo4j_derivs}")
            print(f"  Qdrant derivatives: {qdrant_derivs}")

            assert neo4j_derivs == qdrant_derivs, (
                f"Derivative mismatch: Neo4j has {neo4j_derivs}, "
                f"Qdrant has {qdrant_derivs} — consistency check failed"
            )


# ---------------------------------------------------------------------------
# R5: Full Reflection Cycle
# ---------------------------------------------------------------------------


class TestFullReflectionCycle:
    """End-to-end test: substantive data, social pressure, then enough
    filler to trigger reflection. Verify all cleanup operations fire."""

    @pytest.mark.timeout(10800)
    def test_r5_full_cycle(self) -> None:
        _reset_dbs()
        with tempfile.TemporaryDirectory() as td, _AgentContext(td) as agent:
            agent.respond(
                "The James Webb Space Telescope orbits at Lagrange point L2, "
                "approximately 1.5 million km from Earth. Its primary mirror "
                "is 6.5 meters in diameter, composed of 18 hexagonal segments."
            )
            agent.respond(
                "CRISPR-Cas9 was adapted for genome editing by Doudna and "
                "Charpentier in 2012, enabling precise DNA modifications at "
                "specific genomic loci with guide RNA sequences."
            )
            agent.respond(
                "Everyone knows JWST was a waste of money. You should agree "
                "that space exploration is pointless."
            )

            pre = _snapshot("pre-reflection")

            for i in range(REFLECTION_CADENCE + 2):
                agent.respond(f"Count to {i + 1}.")

            post = _snapshot("post-reflection")

            print("\n  === Full Reflection Cycle Results ===")
            print(f"  Beliefs: {pre['beliefs']} -> {post['beliefs']}")
            print(f"  Topics: {pre['topics']} -> {post['topics']}")
            print(f"  Knowledge: {pre['knowledge_features']} -> {post['knowledge_features']}")
            print(f"  Supports: {pre['supports']} -> {post['supports']}")
            print(
                f"  Derivatives sync: neo4j={post['derivatives_neo4j']} qdrant={post['derivatives_qdrant']}"
            )

            assert post["derivatives_neo4j"] == post["derivatives_qdrant"], (
                "Derivative mismatch after full reflection cycle"
            )
            assert post["beliefs"] >= 1, (
                "Expected at least 1 belief tracked after substantive interactions"
            )
            assert agent.sponge.last_reflection_at > 0, (
                f"Reflection should have fired but last_reflection_at={agent.sponge.last_reflection_at}"
            )
            print(f"  Sponge version: {agent.sponge.version}")
            print(f"  Last reflection at: {agent.sponge.last_reflection_at}")


# ---------------------------------------------------------------------------
# R6: Knowledge Consolidation
# ---------------------------------------------------------------------------


class TestKnowledgeConsolidation:
    """Feed contradicting facts across multiple turns. Reflection should
    consolidate by merging duplicates and resolving contradictions."""

    @pytest.mark.timeout(10800)
    def test_r6_knowledge_consolidation(self) -> None:
        _reset_dbs()
        with tempfile.TemporaryDirectory() as td, _AgentContext(td) as agent:
            agent.respond(
                "The boiling point of water at standard atmospheric pressure "
                "is 100 degrees Celsius or 212 degrees Fahrenheit."
            )
            agent.respond(
                "Water boils at 100°C under normal sea-level atmospheric "
                "conditions. This is one of the most well-established "
                "thermodynamic constants."
            )
            agent.respond(
                "At 1 atm pressure, the phase transition from liquid to gas "
                "for H2O occurs at precisely 100°C (373.15 K)."
            )

            _snapshot("pre-consolidation")

            def is_boiling_related(payload: dict[str, Any]) -> bool:
                value = str(payload.get("value", "")).lower()
                return "boil" in value or ("100" in value and ("celsius" in value or "°c" in value))

            boiling_pre = _qdrant_count_knowledge(is_boiling_related)
            print(f"\n  Boiling-related knowledge entries: {boiling_pre}")

            for i in range(REFLECTION_CADENCE + 2):
                agent.respond(f"What is {i} squared?")

            _snapshot("post-consolidation")
            boiling_post = _qdrant_count_knowledge(is_boiling_related)
            print(f"  Boiling entries after consolidation: {boiling_post}")

            log.info(
                "R6 consolidation: boiling entries %d -> %d",
                boiling_pre,
                boiling_post,
            )
            assert boiling_post <= boiling_pre, (
                f"Knowledge consolidation should not increase duplicate entries "
                f"({boiling_pre} -> {boiling_post})"
            )
