"""MemoryGraph: Neo4j-backed graph storage for episodes, derivatives, and relationships.

Manages Episode, Derivative, Topic, Segment, Summary, and Belief nodes with
typed edges (DERIVED_FROM, TEMPORAL_NEXT, DISCUSSES, SUPPORTS_BELIEF, etc.).
Bi-temporal tracking with created_at/valid_at/expired_at on episodes.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, cast

import structlog
from neo4j._typing import LiteralString

from neo4j import AsyncDriver, AsyncManagedTransaction

from .. import config

log = structlog.get_logger()


def format_episode_line(
    *,
    created_at: str,
    summary: str,
    content: str,
    content_limit: int = config.settings.episode_content_limit,
) -> str:
    """Render one compact context line for retrieval/reflection."""
    date_text = created_at[:10] if created_at else "?"
    return f"[{date_text}] {summary or content[:content_limit]}"


def format_episode_block(
    *,
    created_at: str,
    content: str,
    content_limit: int = config.settings.episode_content_limit,
) -> str:
    """Render one dated episode content block for summarization prompts."""
    date_text = created_at[:10] if created_at else "?"
    return f"[{date_text}]\n{content[:content_limit]}"


_DEFAULT_SESSION_ID: Final = "default"


class EdgeType(StrEnum):
    """Neo4j relationship types between graph nodes."""

    DERIVED_FROM = "DERIVED_FROM"
    TEMPORAL_NEXT = "TEMPORAL_NEXT"
    DISCUSSES = "DISCUSSES"
    SUPPORTS_BELIEF = "SUPPORTS_BELIEF"
    CONTRADICTS_BELIEF = "CONTRADICTS_BELIEF"
    BELONGS_TO_SEGMENT = "BELONGS_TO_SEGMENT"
    CONSOLIDATES = "CONSOLIDATES"


@dataclass(frozen=True, slots=True)
class BeliefNode:
    """Full belief state stored in Neo4j."""

    topic: str
    valence: float = 0.0
    confidence: float = 0.5
    uncertainty: float = 0.5
    evidence_count: int = 0
    belief_text: str = ""
    provenance: str = ""


def format_beliefs_for_prompt_from_nodes(beliefs: Sequence[BeliefNode]) -> str:
    """Build belief summary lines for the system prompt.

    Format: topic — valence: +0.40, confidence: 0.60, evidence: 3 | belief text
    Valence = opinion direction (-1 negative to +1 positive).
    Confidence = how certain you are (0 to 1).
    """
    if not beliefs:
        return "No beliefs formed yet."
    lines: list[str] = []
    for b in beliefs:
        parts = [f"valence: {b.valence:+.2f}", f"confidence: {b.confidence:.2f}"]
        if b.evidence_count > 0:
            parts.append(f"evidence: {b.evidence_count}")
        meta = ", ".join(parts)
        entry = f"{b.topic} — {meta}"
        if b.belief_text:
            entry += f" | {b.belief_text}"
        lines.append(entry)
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class PersonalitySnapshot:
    """Agent identity narrative stored in Neo4j."""

    text: str = ""
    version: int = 0


SEED_SNAPSHOT: Final = (
    "I am an AI entity forming genuine opinions through conversation and reflection. "
    "I value intellectual honesty and analytical thinking. I am direct about "
    "disagreement and do not default to agreement. My views are still forming "
    "as I accumulate evidence and experience."
)


@dataclass(frozen=True, slots=True)
class EpisodeNode:
    """Single conversation episode stored in Neo4j.

    Bi-temporal: created_at = wall-clock insert time, valid_at = event time.
    consolidation_level: 1 = raw episode, 2+ = summarized/consolidated.
    """

    uid: str
    content: str
    summary: str
    topics: list[str]
    ess_score: float
    created_at: str  # ISO8601
    valid_at: str
    expired_at: str = ""
    utility_score: float = 0.0
    access_count: int = 0
    last_accessed: str = ""
    segment_id: str = ""
    consolidation_level: int = 1
    archived: bool = False
    user_message: str = ""
    agent_response: str = ""
    reasoning_type: str = ""


@dataclass(frozen=True, slots=True)
class DerivativeNode:
    """Semantic chunk derived from an episode for granular vector retrieval."""

    uid: str
    source_episode_uid: str
    text: str
    key_concept: str
    sequence_num: int


_DB: Final = config.settings.neo4j_database


class MemoryGraph:
    """Neo4j-backed graph for episodes, beliefs, personality snapshots, and topic structure."""

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    async def store_episode_atomically(
        self,
        *,
        episode: EpisodeNode,
        derivatives: list[DerivativeNode],
        prev_episode_uid: str,
        topics: list[str],
        segment_id: str,
        segment_label: str,
    ) -> None:
        """Store episode + derivatives + graph links in one write transaction."""
        log.debug(
            "graph_store_episode_trace",
            episode_uid=episode.uid[:8],
            topics_sample=topics[:3],
            ess_score=episode.ess_score,
            derivative_count=len(derivatives),
            prev_episode_uid=(
                prev_episode_uid[:8] if prev_episode_uid else "none"
            ),
        )
        async with self._driver.session(database=_DB) as session:
            await session.execute_write(
                self._store_episode_atomically_tx,
                episode,
                derivatives,
                prev_episode_uid,
                topics,
                segment_id,
                segment_label,
            )

    @staticmethod
    async def _store_episode_atomically_tx(
        tx: AsyncManagedTransaction,
        episode: EpisodeNode,
        derivatives: list[DerivativeNode],
        prev_uid: str,
        topics: list[str],
        segment_id: str,
        segment_label: str,
    ) -> None:
        await tx.run(
            """
            CREATE (e:Episode {
                uid: $uid, content: $content, summary: $summary,
                topics: $topics, ess_score: $ess_score,
                created_at: $created_at, valid_at: $valid_at,
                expired_at: $expired_at, utility_score: $utility_score,
                access_count: $access_count, last_accessed: $last_accessed,
                segment_id: $segment_id, consolidation_level: $consolidation_level,
                archived: $archived, user_message: $user_message,
                agent_response: $agent_response,
                reasoning_type: $reasoning_type
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
            reasoning_type=episode.reasoning_type,
        )
        if prev_uid:
            await tx.run(
                cast(
                    LiteralString,
                    "MATCH (prev:Episode {uid: $prev_uid}) "
                    "MATCH (curr:Episode {uid: $curr_uid}) "
                    f"CREATE (prev)-[:{EdgeType.TEMPORAL_NEXT}]->(curr)",
                ),
                prev_uid=prev_uid,
                curr_uid=episode.uid,
            )
        if derivatives:
            await MemoryGraph._create_derivatives_tx(tx, derivatives, episode.uid)
        for topic in topics:
            await MemoryGraph._link_topic_tx(tx, episode.uid, topic)
        if segment_id:
            await MemoryGraph._link_segment_tx(
                tx,
                episode.uid,
                segment_id,
                segment_label,
            )

    @staticmethod
    async def _create_derivatives_tx(
        tx: AsyncManagedTransaction,
        derivatives: list[DerivativeNode],
        episode_uid: str,
    ) -> None:
        for d in derivatives:
            await tx.run(
                cast(
                    LiteralString,
                    f"""
                CREATE (d:Derivative {{
                    uid: $uid, source_episode_uid: $source_uid,
                    text: $text, key_concept: $key_concept,
                    sequence_num: $seq
                }})
                WITH d
                MATCH (e:Episode {{uid: $episode_uid}})
                CREATE (d)-[:{EdgeType.DERIVED_FROM}]->(e)
                """,
                ),
                uid=d.uid,
                source_uid=d.source_episode_uid,
                text=d.text,
                key_concept=d.key_concept,
                seq=d.sequence_num,
                episode_uid=episode_uid,
            )

    @staticmethod
    async def _link_topic_tx(tx: AsyncManagedTransaction, episode_uid: str, topic: str) -> None:
        await tx.run(
            cast(
                LiteralString,
                f"""
            MERGE (t:Topic {{name: $topic}})
            ON CREATE SET t.episode_count = 1, t.first_seen_at = datetime()
            ON MATCH SET t.episode_count = t.episode_count + 1
            SET t.last_seen_at = datetime()
            WITH t
            MATCH (e:Episode {{uid: $uid}})
            MERGE (e)-[:{EdgeType.DISCUSSES}]->(t)
            """,
            ),
            topic=topic.strip().lower(),
            uid=episode_uid,
        )

    @staticmethod
    async def _link_segment_tx(
        tx: AsyncManagedTransaction,
        episode_uid: str,
        segment_id: str,
        label: str,
    ) -> None:
        await tx.run(
            cast(
                LiteralString,
                f"""
            MERGE (s:Segment {{segment_id: $segment_id}})
            ON CREATE SET s.label = $label, s.start_time = datetime(),
                          s.episode_count = 1, s.consolidated = false
            ON MATCH SET s.episode_count = s.episode_count + 1,
                         s.end_time = datetime(),
                         s.label = CASE
                            WHEN (s.label IS NULL OR s.label = '') AND $label <> ''
                            THEN $label ELSE s.label END
            WITH s
            MATCH (e:Episode {{uid: $uid}})
            MERGE (e)-[:{EdgeType.BELONGS_TO_SEGMENT}]->(s)
            """,
            ),
            segment_id=segment_id,
            label=label,
            uid=episode_uid,
        )

    async def link_belief(
        self,
        episode_uid: str,
        topic: str,
        *,
        edge_type: EdgeType,
        strength: float = 0.5,
        reasoning: str = "",
    ) -> None:
        """Create one belief provenance edge for an episode."""
        log.debug(
            "graph_link_belief_trace",
            episode_uid=episode_uid[:8],
            topic=topic,
            edge_type=edge_type.value,
            strength=strength,
            reasoning_preview=reasoning[:80].replace("\n", " "),
        )
        async with self._driver.session(database=_DB) as session:
            await session.execute_write(
                self._link_belief_tx, episode_uid, topic, edge_type, strength, reasoning
            )

    @staticmethod
    async def _link_belief_tx(
        tx: AsyncManagedTransaction,
        episode_uid: str,
        topic: str,
        edge_type: EdgeType,
        strength: float,
        reasoning: str,
    ) -> None:
        await tx.run(
            cast(
                LiteralString,
                f"""
            MATCH (b:Belief {{topic: $topic}})
            SET b.evidence_count = coalesce(b.evidence_count, 0) + 1,
                b.last_updated = datetime()
            WITH b
            MATCH (e:Episode {{uid: $uid}})
            CREATE (e)-[:{edge_type} {{
                strength: $strength, reasoning: $reasoning, created_at: datetime()
            }}]->(b)
            """,
            ),
            topic=topic.strip().lower(),
            uid=episode_uid,
            strength=strength,
            reasoning=reasoning,
        )

    async def get_episodes(self, uids: list[str]) -> list[EpisodeNode]:
        """Fetch multiple non-archived episodes by UID, preserving input order.

        Maintains the original UID ordering (important for vector ranking scores).
        """
        if not uids:
            return []
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                "MATCH (e:Episode) WHERE e.uid IN $uids AND NOT e.archived RETURN e",
                uids=uids,
            )
            by_uid: dict[str, EpisodeNode] = {}
            async for record in result:
                ep = _record_to_episode(record["e"])
                by_uid[ep.uid] = ep
            # Return in original UID order, skipping any not found
            return [by_uid[uid] for uid in uids if uid in by_uid]

    async def _keyword_episode_search(
        self, cypher: str, query: str, limit: int
    ) -> list[EpisodeNode]:
        """Run a parameterized Cypher query using keywords extracted from the query string."""
        keywords = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) > 2]
        if not keywords:
            return []
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                cast(LiteralString, cypher), keywords=keywords[:8], limit=limit
            )
            return [_record_to_episode(r["e"]) async for r in result]

    async def find_belief_related_episodes(
        self, query: str, *, limit: int = 20
    ) -> list[EpisodeNode]:
        """Retrieve episodes attached to belief edges matching query keywords."""
        return await self._keyword_episode_search(
            f"""
            MATCH (e:Episode)-[:{EdgeType.SUPPORTS_BELIEF}|{EdgeType.CONTRADICTS_BELIEF}]->(b:Belief)
            WHERE NOT e.archived
              AND ANY(keyword IN $keywords WHERE toLower(b.topic) CONTAINS keyword)
            RETURN DISTINCT e ORDER BY e.utility_score DESC, e.created_at DESC LIMIT $limit
        """,
            query,
            limit,
        )

    async def find_topic_related_episodes(
        self, query: str, *, limit: int = 20
    ) -> list[EpisodeNode]:
        """Retrieve episodes by traversing Topic nodes relevant to query keywords."""
        return await self._keyword_episode_search(
            f"""
            MATCH (e:Episode)-[:{EdgeType.DISCUSSES}]->(t:Topic)
            WHERE NOT e.archived
              AND ANY(keyword IN $keywords WHERE toLower(t.name) CONTAINS keyword)
            RETURN DISTINCT e ORDER BY e.utility_score DESC, e.created_at DESC LIMIT $limit
        """,
            query,
            limit,
        )

    async def traverse_temporal_context(
        self,
        episode_uid: str,
        *,
        before: int = 2,
        after: int = 2,
    ) -> list[EpisodeNode]:
        """Retrieve temporally adjacent non-archived episodes for context expansion."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (focal:Episode {{uid: $uid}})
                OPTIONAL MATCH path_before = (prev:Episode)-[:{EdgeType.TEMPORAL_NEXT}*1..{before}]->(focal)
                  WHERE NOT prev.archived
                OPTIONAL MATCH path_after = (focal)-[:{EdgeType.TEMPORAL_NEXT}*1..{after}]->(next:Episode)
                  WHERE NOT next.archived
                WITH focal,
                     COLLECT(DISTINCT prev) AS befores,
                     COLLECT(DISTINCT next) AS afters
                RETURN befores, focal, afters
                """,
                ),
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

    async def update_episode_access(self, episode_uids: list[str]) -> None:
        """Update access_count and last_accessed for retrieved episodes.

        Called after episodes are retrieved to maintain accurate forgetting signals.
        """
        if not episode_uids:
            return
        async with self._driver.session(database=_DB) as session:
            await session.run(
                """
                UNWIND $uids AS uid
                MATCH (e:Episode {uid: uid})
                SET e.access_count = COALESCE(e.access_count, 0) + 1,
                    e.last_accessed = datetime()
                """,
                uids=episode_uids,
            )
        log.debug("episodes_access_stats_updated", count=len(episode_uids))

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
        log.info("graph_archive_episode", episode_uid=episode_uid[:8])

    async def unarchive_episode(self, episode_uid: str) -> None:
        """Reverse a soft-archive (set archived=False). Used for rollback."""
        async with self._driver.session(database=_DB) as session:
            await session.run(
                """
                MATCH (e:Episode {uid: $uid})
                SET e.archived = false, e.expired_at = null
                """,
                uid=episode_uid,
            )
        log.info("graph_unarchive_episode", episode_uid=episode_uid[:8])

    async def delete_episode(self, episode_uid: str) -> None:
        """Hard-delete an episode and its derivative nodes."""
        async with self._driver.session(database=_DB) as session:
            await session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (e:Episode {{uid: $uid}})
                OPTIONAL MATCH (d:Derivative)-[:{EdgeType.DERIVED_FROM}]->(e)
                DETACH DELETE d, e
                """,
                ),
                uid=episode_uid,
            )
        log.warning("graph_delete_episode", episode_uid=episode_uid[:8])

    async def get_segment_episodes(self, segment_id: str) -> list[EpisodeNode]:
        """Get all episodes in a segment, ordered by creation time."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (e:Episode)-[:{EdgeType.BELONGS_TO_SEGMENT}]->(s:Segment {{segment_id: $seg_id}})
                WHERE NOT e.archived
                RETURN e ORDER BY e.created_at
                """,
                ),
                seg_id=segment_id,
            )
            records = [record async for record in result]
            return [_record_to_episode(r["e"]) for r in records]

    async def mark_segment_consolidated(self, segment_id: str) -> None:
        """Mark one segment consolidated after summary generation."""
        async with self._driver.session(database=_DB) as session:
            await session.run(
                """
                MATCH (s:Segment {segment_id: $segment_id})
                SET s.consolidated = true, s.consolidated_at = datetime()
                """,
                segment_id=segment_id,
            )

    async def get_forgetting_candidates(
        self, *, limit: int = 20, min_age_minutes: int = 60
    ) -> list[EpisodeNode]:
        """Fetch low-utility raw (non-consolidated) episodes eligible for forgetting.

        Only considers episodes with consolidation_level=1 that are older than
        min_age_minutes, preventing the forgetting cycle from targeting just-ingested
        or already-consolidated episodes.
        """
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                """
                MATCH (e:Episode)
                WHERE NOT e.archived AND e.consolidation_level = 1
                  AND datetime(e.created_at) < datetime() - duration({minutes: $min_age_minutes})
                RETURN e
                ORDER BY e.utility_score ASC, datetime(e.created_at) ASC
                LIMIT $limit
                """,
                limit=limit,
                min_age_minutes=min_age_minutes,
            )
            records = [record async for record in result]
            return [_record_to_episode(record["e"]) for record in records]

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
        tx: AsyncManagedTransaction,
        uid: str,
        level: int,
        content: str,
        source_uids: list[str],
        topics: list[str],
    ) -> None:
        await tx.run(
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
            await tx.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (s:Summary {{uid: $summary_uid}})
                MATCH (e:Episode {{uid: $source_uid}})
                CREATE (s)-[:{EdgeType.CONSOLIDATES}]->(e)
                """,
                ),
                summary_uid=uid,
                source_uid=source_uid,
            )

    # --- Personality Snapshot ---

    async def get_personality_snapshot(self) -> PersonalitySnapshot:
        """Load the agent's identity narrative from graph, or return seed defaults."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                "MATCH (n:PersonalitySnapshot {session_id: $sid}) RETURN n",
                sid=_DEFAULT_SESSION_ID,
            )
            record = await result.single()
        if not record:
            return PersonalitySnapshot(text=SEED_SNAPSHOT)
        props = dict(record["n"])
        return PersonalitySnapshot(
            text=str(props.get("text", SEED_SNAPSHOT)),
            version=int(props.get("version", 0)),
        )

    async def upsert_personality_snapshot(self, text: str) -> None:
        """Write or update the agent's identity narrative."""
        async with self._driver.session(database=_DB) as session:
            await session.run(
                """
                MERGE (n:PersonalitySnapshot {session_id: $sid})
                SET n.text = $text,
                    n.version = coalesce(n.version, 0) + 1,
                    n.updated_at = datetime()
                """,
                sid=_DEFAULT_SESSION_ID,
                text=text,
            )
        log.info(
            "graph_upsert_personality_snapshot", char_count=len(text)
        )

    # --- Belief CRUD ---

    async def upsert_belief(
        self,
        topic: str,
        *,
        valence: float,
        confidence: float,
        belief_text: str = "",
        uncertainty: float = -1.0,
        evidence_count: int = -1,
        provenance: str = "",
    ) -> None:
        """Create or update a Belief node (valence, confidence, uncertainty, evidence count).

        Sentinel values: uncertainty=-1 and evidence_count=-1 mean "keep existing
        or derive from confidence". This avoids requiring callers to fetch the
        current belief just to preserve unchanged fields.
        """
        topic = topic.strip().lower()
        async with self._driver.session(database=_DB) as session:
            await session.run(
                """
                MERGE (b:Belief {topic: $topic})
                SET b.valence = $valence,
                    b.confidence = $confidence,
                    b.belief_text = CASE WHEN $belief_text <> '' THEN $belief_text ELSE coalesce(b.belief_text, '') END,
                    b.uncertainty = CASE WHEN $uncertainty >= 0 THEN $uncertainty ELSE coalesce(b.uncertainty, 1.0 - $confidence) END,
                    b.evidence_count = CASE WHEN $evidence_count >= 0 THEN $evidence_count ELSE coalesce(b.evidence_count, 0) + 1 END,
                    b.provenance = CASE WHEN $provenance <> '' THEN $provenance ELSE coalesce(b.provenance, '') END,
                    b.formed_at = coalesce(b.formed_at, datetime()),
                    b.last_updated = datetime()
                """,
                topic=topic,
                valence=max(-1.0, min(1.0, valence)),
                confidence=max(0.0, min(1.0, confidence)),
                belief_text=belief_text,
                uncertainty=uncertainty,
                evidence_count=evidence_count,
                provenance=provenance,
            )
        log.info(
            "graph_upsert_belief",
            topic=topic,
            valence=valence,
            confidence=confidence,
        )

    async def get_belief(self, topic: str) -> BeliefNode | None:
        """Fetch a single belief by topic."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                "MATCH (b:Belief {topic: $topic}) RETURN b",
                topic=topic.strip().lower(),
            )
            record = await result.single()
        if not record:
            return None
        return _record_to_belief(record["b"])

    async def get_all_beliefs(self) -> list[BeliefNode]:
        """Fetch all beliefs."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run("MATCH (b:Belief) RETURN b ORDER BY abs(b.valence) DESC")
            return [_record_to_belief(r["b"]) async for r in result]

    async def get_last_episode_uid(self) -> str:
        """Get the UID of the most recently created episode."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                "MATCH (e:Episode) RETURN e.uid AS uid ORDER BY e.created_at DESC LIMIT 1"
            )
            record = await result.single()
            return str(record["uid"]) if record and record.get("uid") else ""

    async def get_episode_count(self) -> int:
        """Get the total count of episodes in the graph."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run("MATCH (e:Episode) RETURN count(e) AS cnt")
            record = await result.single()
            return int(record["cnt"]) if record else 0

    async def get_latest_segment_counter(self) -> int:
        """Get the max numeric suffix from `segment_<n>` identifiers."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                """
                MATCH (s:Segment)
                WHERE s.segment_id STARTS WITH 'segment_'
                RETURN s.segment_id AS segment_id
                """
            )
            counters: list[int] = []
            async for record in result:
                raw = record.get("segment_id")
                if not isinstance(raw, str) or "_" not in raw:
                    continue
                try:
                    counters.append(int(raw.rsplit("_", maxsplit=1)[1]))
                except ValueError:
                    continue
            return max(counters, default=0)


def _float(val: object, default: float = 0.0) -> float:
    """Coerce a Neo4j property value to float (handles None, str, int)."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(str(val))
    except (TypeError, ValueError):
        return default


def _int(val: object, default: int = 0) -> int:
    """Coerce a Neo4j property value to int."""
    if val is None:
        return default
    if isinstance(val, int):
        return val
    try:
        return int(str(val))
    except (TypeError, ValueError):
        return default


def _str(val: object) -> str:
    """Coerce a Neo4j property value to str (None → empty string)."""
    return str(val) if val is not None else ""


def _record_to_belief(node: Mapping[str, object]) -> BeliefNode:
    """Convert a Neo4j Belief node record to a BeliefNode dataclass."""
    props = dict(node)
    return BeliefNode(
        topic=_str(props.get("topic")),
        valence=_float(props.get("valence")),
        confidence=_float(props.get("confidence"), 0.5),
        uncertainty=_float(props.get("uncertainty"), 0.5),
        evidence_count=_int(props.get("evidence_count")),
        belief_text=_str(props.get("belief_text")),
        provenance=_str(props.get("provenance")),
    )


def _record_to_episode(node: Mapping[str, object]) -> EpisodeNode:
    """Convert a Neo4j Episode node record to an EpisodeNode dataclass."""
    props = dict(node)
    topics_raw = props.get("topics", [])
    return EpisodeNode(
        uid=_str(props.get("uid")),
        content=_str(props.get("content")),
        summary=_str(props.get("summary")),
        topics=list(topics_raw) if isinstance(topics_raw, (list, tuple)) else [],
        ess_score=_float(props.get("ess_score")),
        created_at=_str(props.get("created_at")),
        valid_at=_str(props.get("valid_at")),
        expired_at=_str(props.get("expired_at")),
        utility_score=_float(props.get("utility_score")),
        access_count=_int(props.get("access_count")),
        last_accessed=_str(props.get("last_accessed")),
        segment_id=_str(props.get("segment_id")),
        consolidation_level=_int(props.get("consolidation_level"), 1),
        archived=bool(props.get("archived", False)),
        user_message=_str(props.get("user_message")),
        agent_response=_str(props.get("agent_response")),
        reasoning_type=_str(props.get("reasoning_type")),
    )
