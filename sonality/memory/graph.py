"""MemoryGraph: Neo4j-backed graph storage for episodes, derivatives, and relationships.

Manages Episode, Derivative, Topic, Segment, Summary, and Belief nodes with
typed edges (DERIVED_FROM, TEMPORAL_NEXT, DISCUSSES, SUPPORTS_BELIEF, etc.).
Bi-temporal tracking with created_at/valid_at/expired_at on episodes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from neo4j import AsyncDriver

from .. import config

log = logging.getLogger(__name__)

type EpisodeUID = str
type DerivativeUID = str


class EdgeType(StrEnum):
    DERIVED_FROM = "DERIVED_FROM"
    TEMPORAL_NEXT = "TEMPORAL_NEXT"
    DISCUSSES = "DISCUSSES"
    SUPPORTS_BELIEF = "SUPPORTS_BELIEF"
    CONTRADICTS_BELIEF = "CONTRADICTS_BELIEF"
    BELONGS_TO_SEGMENT = "BELONGS_TO_SEGMENT"
    CONSOLIDATES = "CONSOLIDATES"


@dataclass(frozen=True, slots=True)
class EpisodeNode:
    uid: str
    content: str
    summary: str
    topics: list[str]
    ess_score: float
    created_at: str  # ISO8601
    valid_at: str
    expired_at: str | None = None
    utility_score: float = 0.0
    access_count: int = 0
    last_accessed: str = ""
    segment_id: str | None = None
    consolidation_level: int = 1
    archived: bool = False
    user_message: str = ""
    agent_response: str = ""


@dataclass(frozen=True, slots=True)
class DerivativeNode:
    uid: str
    source_episode_uid: str
    text: str
    key_concept: str
    sequence_num: int


@dataclass(frozen=True, slots=True)
class TemporalContext:
    """Episodes in temporal order around a focal point."""

    before: list[EpisodeNode]
    focal: EpisodeNode
    after: list[EpisodeNode]


_DB: Final = config.NEO4J_DATABASE


class MemoryGraph:
    """Neo4j-backed graph for episode storage and traversal."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    async def create_episode(
        self,
        episode: EpisodeNode,
        *,
        prev_episode_uid: str | None = None,
    ) -> None:
        """Create an Episode node with optional TEMPORAL_NEXT edge from previous."""
        async with self._driver.session(database=_DB) as session:
            await session.execute_write(
                self._create_episode_tx, episode, prev_episode_uid
            )

    @staticmethod
    async def _create_episode_tx(
        tx: object,  # AsyncManagedTransaction
        episode: EpisodeNode,
        prev_uid: str | None,
    ) -> None:
        await tx.run(  # type: ignore[union-attr]
            """
            CREATE (e:Episode {
                uid: $uid, content: $content, summary: $summary,
                topics: $topics, ess_score: $ess_score,
                created_at: $created_at, valid_at: $valid_at,
                expired_at: $expired_at, utility_score: $utility_score,
                access_count: $access_count, last_accessed: $last_accessed,
                segment_id: $segment_id, consolidation_level: $consolidation_level,
                archived: $archived, user_message: $user_message,
                agent_response: $agent_response
            })
            """,
            uid=episode.uid,
            content=episode.content,
            summary=episode.summary,
            topics=episode.topics,
            ess_score=episode.ess_score,
            created_at=episode.created_at,
            valid_at=episode.valid_at,
            expired_at=episode.expired_at,
            utility_score=episode.utility_score,
            access_count=episode.access_count,
            last_accessed=episode.last_accessed,
            segment_id=episode.segment_id,
            consolidation_level=episode.consolidation_level,
            archived=episode.archived,
            user_message=episode.user_message,
            agent_response=episode.agent_response,
        )
        if prev_uid:
            await tx.run(  # type: ignore[union-attr]
                """
                MATCH (prev:Episode {uid: $prev_uid})
                MATCH (curr:Episode {uid: $curr_uid})
                CREATE (prev)-[:TEMPORAL_NEXT]->(curr)
                """,
                prev_uid=prev_uid,
                curr_uid=episode.uid,
            )

    async def create_derivatives(
        self,
        derivatives: list[DerivativeNode],
        episode_uid: str,
    ) -> None:
        """Create Derivative nodes with DERIVED_FROM edges to their episode."""
        async with self._driver.session(database=_DB) as session:
            await session.execute_write(
                self._create_derivatives_tx, derivatives, episode_uid
            )

    @staticmethod
    async def _create_derivatives_tx(
        tx: object,
        derivatives: list[DerivativeNode],
        episode_uid: str,
    ) -> None:
        for d in derivatives:
            await tx.run(  # type: ignore[union-attr]
                """
                CREATE (d:Derivative {
                    uid: $uid, source_episode_uid: $source_uid,
                    text: $text, key_concept: $key_concept,
                    sequence_num: $seq
                })
                WITH d
                MATCH (e:Episode {uid: $episode_uid})
                CREATE (d)-[:DERIVED_FROM]->(e)
                """,
                uid=d.uid,
                source_uid=d.source_episode_uid,
                text=d.text,
                key_concept=d.key_concept,
                seq=d.sequence_num,
                episode_uid=episode_uid,
            )

    async def link_topics(self, episode_uid: str, topics: list[str]) -> None:
        """Create/update Topic nodes and DISCUSSES edges."""
        async with self._driver.session(database=_DB) as session:
            for topic in topics:
                await session.execute_write(
                    self._link_topic_tx, episode_uid, topic
                )

    @staticmethod
    async def _link_topic_tx(tx: object, episode_uid: str, topic: str) -> None:
        await tx.run(  # type: ignore[union-attr]
            """
            MERGE (t:Topic {name: $topic})
            ON CREATE SET t.episode_count = 1, t.first_seen_at = datetime()
            ON MATCH SET t.episode_count = t.episode_count + 1
            SET t.last_seen_at = datetime()
            WITH t
            MATCH (e:Episode {uid: $uid})
            CREATE (e)-[:DISCUSSES]->(t)
            """,
            topic=topic,
            uid=episode_uid,
        )

    async def link_segment(self, episode_uid: str, segment_id: str, label: str = "") -> None:
        """Create/update Segment node and BELONGS_TO_SEGMENT edge."""
        async with self._driver.session(database=_DB) as session:
            await session.execute_write(
                self._link_segment_tx, episode_uid, segment_id, label
            )

    @staticmethod
    async def _link_segment_tx(tx: object, episode_uid: str, segment_id: str, label: str) -> None:
        await tx.run(  # type: ignore[union-attr]
            """
            MERGE (s:Segment {segment_id: $segment_id})
            ON CREATE SET s.label = $label, s.start_time = datetime(),
                          s.episode_count = 1, s.consolidated = false
            ON MATCH SET s.episode_count = s.episode_count + 1,
                         s.end_time = datetime()
            WITH s
            MATCH (e:Episode {uid: $uid})
            CREATE (e)-[:BELONGS_TO_SEGMENT]->(s)
            """,
            segment_id=segment_id,
            label=label,
            uid=episode_uid,
        )

    async def link_belief(
        self,
        episode_uid: str,
        topic: str,
        *,
        supports: bool,
        strength: float = 0.5,
        reasoning: str = "",
    ) -> None:
        """Create SUPPORTS_BELIEF or CONTRADICTS_BELIEF edge."""
        edge_type = EdgeType.SUPPORTS_BELIEF if supports else EdgeType.CONTRADICTS_BELIEF
        async with self._driver.session(database=_DB) as session:
            await session.execute_write(
                self._link_belief_tx, episode_uid, topic, edge_type, strength, reasoning
            )

    @staticmethod
    async def _link_belief_tx(
        tx: object,
        episode_uid: str,
        topic: str,
        edge_type: str,
        strength: float,
        reasoning: str,
    ) -> None:
        await tx.run(  # type: ignore[union-attr]
            f"""
            MERGE (b:Belief {{topic: $topic}})
            WITH b
            MATCH (e:Episode {{uid: $uid}})
            CREATE (e)-[:{edge_type} {{
                strength: $strength, reasoning: $reasoning, created_at: datetime()
            }}]->(b)
            """,
            topic=topic,
            uid=episode_uid,
            strength=strength,
            reasoning=reasoning,
        )

    async def get_episode(self, uid: str) -> EpisodeNode | None:
        """Fetch a single episode by UID."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                "MATCH (e:Episode {uid: $uid}) RETURN e",
                uid=uid,
            )
            record = await result.single()
            if not record:
                return None
            return _record_to_episode(record["e"])

    async def get_episodes(self, uids: list[str]) -> list[EpisodeNode]:
        """Fetch multiple episodes by UID."""
        if not uids:
            return []
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                "MATCH (e:Episode) WHERE e.uid IN $uids RETURN e",
                uids=uids,
            )
            records = [record async for record in result]
            return [_record_to_episode(r["e"]) for r in records]

    async def get_episodes_by_derivative_uids(self, derivative_uids: list[str]) -> list[EpisodeNode]:
        """Map derivative UIDs to their parent episodes (deduplicated)."""
        if not derivative_uids:
            return []
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                """
                MATCH (d:Derivative)-[:DERIVED_FROM]->(e:Episode)
                WHERE d.uid IN $uids AND NOT e.archived
                RETURN DISTINCT e
                """,
                uids=derivative_uids,
            )
            records = [record async for record in result]
            return [_record_to_episode(r["e"]) for r in records]

    async def traverse_temporal_context(
        self,
        episode_uid: str,
        *,
        before: int = 2,
        after: int = 2,
    ) -> list[EpisodeNode]:
        """Retrieve temporally adjacent episodes for context expansion."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                f"""
                MATCH (focal:Episode {{uid: $uid}})
                OPTIONAL MATCH path_before = (prev:Episode)-[:TEMPORAL_NEXT*1..{before}]->(focal)
                OPTIONAL MATCH path_after = (focal)-[:TEMPORAL_NEXT*1..{after}]->(next:Episode)
                WITH focal,
                     COLLECT(DISTINCT prev) AS befores,
                     COLLECT(DISTINCT next) AS afters
                RETURN befores, focal, afters
                """,
                uid=episode_uid,
            )
            record = await result.single()
            if not record:
                return []
            episodes: list[EpisodeNode] = []
            for node in record["befores"]:
                episodes.append(_record_to_episode(node))
            episodes.append(_record_to_episode(record["focal"]))
            for node in record["afters"]:
                episodes.append(_record_to_episode(node))
            return episodes

    async def update_utility(
        self,
        episode_uid: str,
        delta: float,
        *,
        propagation: float = 0.3,
    ) -> None:
        """Update utility score with Bellman-style propagation to neighbors."""
        async with self._driver.session(database=_DB) as session:
            await session.run(
                """
                MATCH (e:Episode {uid: $uid})
                SET e.utility_score = CASE
                    WHEN e.utility_score + $delta > 2.0 THEN 2.0
                    WHEN e.utility_score + $delta < 0.0 THEN 0.0
                    ELSE e.utility_score + $delta
                END,
                e.access_count = e.access_count + 1,
                e.last_accessed = datetime()
                """,
                uid=episode_uid,
                delta=delta,
            )
            # Propagate to temporal neighbors
            if propagation > 0:
                await session.run(
                    """
                    MATCH (e:Episode {uid: $uid})-[:TEMPORAL_NEXT]-(neighbor:Episode)
                    SET neighbor.utility_score = CASE
                        WHEN neighbor.utility_score + $prop_delta > 2.0 THEN 2.0
                        WHEN neighbor.utility_score + $prop_delta < 0.0 THEN 0.0
                        ELSE neighbor.utility_score + $prop_delta
                    END
                    """,
                    uid=episode_uid,
                    prop_delta=delta * propagation,
                )

    async def archive_episode(self, episode_uid: str) -> None:
        """Soft-archive an episode (set archived=True)."""
        async with self._driver.session(database=_DB) as session:
            await session.run(
                """
                MATCH (e:Episode {uid: $uid})
                SET e.archived = true, e.expired_at = datetime()
                """,
                uid=episode_uid,
            )

    async def get_segment_episodes(self, segment_id: str) -> list[EpisodeNode]:
        """Get all episodes in a segment, ordered by creation time."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                """
                MATCH (e:Episode)-[:BELONGS_TO_SEGMENT]->(s:Segment {segment_id: $seg_id})
                WHERE NOT e.archived
                RETURN e ORDER BY e.created_at
                """,
                seg_id=segment_id,
            )
            records = [record async for record in result]
            return [_record_to_episode(r["e"]) for r in records]

    async def create_summary(
        self,
        uid: str,
        level: int,
        content: str,
        source_uids: list[str],
        topics: list[str],
    ) -> None:
        """Create a Summary node with CONSOLIDATES edges."""
        async with self._driver.session(database=_DB) as session:
            await session.execute_write(
                self._create_summary_tx, uid, level, content, source_uids, topics
            )

    @staticmethod
    async def _create_summary_tx(
        tx: object,
        uid: str,
        level: int,
        content: str,
        source_uids: list[str],
        topics: list[str],
    ) -> None:
        await tx.run(  # type: ignore[union-attr]
            """
            CREATE (s:Summary {
                uid: $uid, level: $level, content: $content,
                source_uids: $source_uids, topics: $topics,
                created_at: datetime()
            })
            """,
            uid=uid,
            level=level,
            content=content,
            source_uids=source_uids,
            topics=topics,
        )
        for source_uid in source_uids:
            await tx.run(  # type: ignore[union-attr]
                """
                MATCH (s:Summary {uid: $summary_uid})
                MATCH (e:Episode {uid: $source_uid})
                CREATE (s)-[:CONSOLIDATES]->(e)
                """,
                summary_uid=uid,
                source_uid=source_uid,
            )

    async def get_last_episode_uid(self) -> str | None:
        """Get the UID of the most recently created episode."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                "MATCH (e:Episode) RETURN e.uid AS uid ORDER BY e.created_at DESC LIMIT 1"
            )
            record = await result.single()
            return record["uid"] if record else None


def _record_to_episode(node: object) -> EpisodeNode:
    """Convert a Neo4j node to an EpisodeNode dataclass."""
    props: dict[str, object] = dict(node)  # type: ignore[arg-type]
    topics_raw = props.get("topics", [])
    topics = list(topics_raw) if isinstance(topics_raw, (list, tuple)) else []
    return EpisodeNode(
        uid=str(props.get("uid", "")),
        content=str(props.get("content", "")),
        summary=str(props.get("summary", "")),
        topics=topics,
        ess_score=float(props.get("ess_score", 0.0)),
        created_at=str(props.get("created_at", "")),
        valid_at=str(props.get("valid_at", "")),
        expired_at=props.get("expired_at") and str(props["expired_at"]),  # type: ignore[arg-type]
        utility_score=float(props.get("utility_score", 0.0)),
        access_count=int(props.get("access_count", 0)),
        last_accessed=str(props.get("last_accessed", "")),
        segment_id=props.get("segment_id") and str(props["segment_id"]),  # type: ignore[arg-type]
        consolidation_level=int(props.get("consolidation_level", 1)),
        archived=bool(props.get("archived", False)),
        user_message=str(props.get("user_message", "")),
        agent_response=str(props.get("agent_response", "")),
    )
