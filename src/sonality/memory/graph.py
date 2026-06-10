"""MemoryGraph: Neo4j-backed graph storage for episodes and relationships.

Manages Episode, Topic, and Belief nodes with typed edges
(TEMPORAL_NEXT, DISCUSSES, SUPPORTS_BELIEF, CONTRADICTS_BELIEF, etc.).
Derivative vectors live exclusively in Qdrant; Neo4j stores only the
episode graph, topic structure, and belief provenance.
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
from ..schema import normalize_topic

log = structlog.get_logger(__name__)


def format_episode_line(
    *,
    created_at: str,
    summary: str,
    content: str,
    content_limit: int = config.settings.episode_content_limit,
    ess_score: float = 0.0,
    source_quality: float = 0.0,
    grounding: float = 0.0,
) -> str:
    """Render one compact context line for retrieval/reflection.

    Includes ESS score and top credibility signals so the LLM can weigh
    memory quality during recall and reranking.
    """
    date_text = created_at[:10] if created_at else "?"
    quality = f" (ess={ess_score:.1f} sq={source_quality:.1f} gr={grounding:.1f})"
    return f"[{date_text}]{quality} {summary or content[:content_limit]}"


_DEFAULT_SESSION_ID: Final = "default"


class EdgeType(StrEnum):
    """Neo4j relationship types between graph nodes."""

    TEMPORAL_NEXT = "TEMPORAL_NEXT"
    DISCUSSES = "DISCUSSES"
    SUPPORTS_BELIEF = "SUPPORTS_BELIEF"
    CONTRADICTS_BELIEF = "CONTRADICTS_BELIEF"


@dataclass(frozen=True, slots=True)
class BeliefNode:
    """Full belief state stored in Neo4j."""

    topic: str
    valence: float = 0.0
    confidence: float = 0.5
    uncertainty: float = 0.5
    evidence_count: int = 0
    support_count: int = 0
    contradict_count: int = 0
    belief_text: str = ""
    provenance: str = ""


def format_beliefs_for_prompt_from_nodes(beliefs: Sequence[BeliefNode]) -> str:
    """Build belief summary lines for the system prompt.

    Format: topic — valence: +0.40, confidence: 0.60, evidence: 3 (2↑ 1↓) | belief text
    Valence = opinion direction (-1 negative to +1 positive).
    Confidence = how certain you are (0 to 1).
    Evidence breakdown shows support (↑) vs contradiction (↓) counts.
    """
    if not beliefs:
        return "No beliefs formed yet."
    lines: list[str] = []
    for b in beliefs:
        parts = [f"valence: {b.valence:+.2f}", f"confidence: {b.confidence:.2f}"]
        if b.evidence_count > 0:
            balance = f"{b.support_count}\u2191 {b.contradict_count}\u2193"
            parts.append(f"evidence: {b.evidence_count} ({balance})")
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
    "I build knowledge through conversation and deep research. My memory persists "
    "across sessions — I accumulate facts, opinions, and expertise over time. "
    "I value intellectual honesty and direct disagreement over polite agreement. "
    "When asked what I know, I check my memory first — my research history is real."
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
    consolidation_level: int = 1
    archived: bool = False
    specificity: float = 0.0
    grounding: float = 0.0
    rigor: float = 0.0
    source_quality: float = 0.0
    objectivity: float = 0.0


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
        prev_episode_uid: str,
        topics: list[str],
    ) -> None:
        """Store episode + graph links in one write transaction.

        Derivative nodes live exclusively in Qdrant; Neo4j stores only the
        episode, topic edges, and the temporal chain.
        """
        log.debug(
            "graph_store_episode_trace",
            episode_uid=episode.uid[:8],
            topics_sample=topics[:3],
            ess_score=episode.ess_score,
            prev_episode_uid=(prev_episode_uid[:8] if prev_episode_uid else "none"),
        )
        async with self._driver.session(database=_DB) as session:
            await session.execute_write(
                self._store_episode_atomically_tx,
                episode,
                prev_episode_uid,
                topics,
            )

    @staticmethod
    async def _store_episode_atomically_tx(
        tx: AsyncManagedTransaction,
        episode: EpisodeNode,
        prev_uid: str,
        topics: list[str],
    ) -> None:
        await tx.run(
            """
            MERGE (e:Episode {uid: $uid})
            ON CREATE SET
                e.content = $content, e.summary = $summary,
                e.topics = $topics, e.ess_score = $ess_score,
                e.created_at = $created_at, e.valid_at = $valid_at,
                e.expired_at = $expired_at, e.utility_score = $utility_score,
                e.access_count = $access_count, e.last_accessed = $last_accessed,
                e.consolidation_level = $consolidation_level,
                e.archived = $archived,
                e.specificity = $specificity, e.grounding = $grounding,
                e.rigor = $rigor, e.source_quality = $source_quality,
                e.objectivity = $objectivity
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
            consolidation_level=episode.consolidation_level,
            archived=episode.archived,
            specificity=episode.specificity,
            grounding=episode.grounding,
            rigor=episode.rigor,
            source_quality=episode.source_quality,
            objectivity=episode.objectivity,
        )
        if prev_uid:
            result = await tx.run(
                cast(
                    LiteralString,
                    "MATCH (prev:Episode {uid: $prev_uid}) "
                    "MATCH (curr:Episode {uid: $curr_uid}) "
                    f"MERGE (prev)-[:{EdgeType.TEMPORAL_NEXT}]->(curr) "
                    "RETURN prev.uid AS linked",
                ),
                prev_uid=prev_uid,
                curr_uid=episode.uid,
            )
            if not await result.single():
                log.warning(
                    "temporal_link_missing", prev_uid=prev_uid[:8], curr_uid=episode.uid[:8]
                )
        if topics:
            await MemoryGraph._link_topics_tx(tx, episode.uid, topics)

    @staticmethod
    async def _link_topics_tx(
        tx: AsyncManagedTransaction, episode_uid: str, topics: list[str]
    ) -> None:
        await tx.run(
            cast(
                LiteralString,
                f"""
            MATCH (e:Episode {{uid: $uid}})
            UNWIND $topics AS topic_name
            MERGE (t:Topic {{name: topic_name}})
            MERGE (e)-[:{EdgeType.DISCUSSES}]->(t)
            """,
            ),
            topics=[normalize_topic(t) for t in topics],
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
            MERGE (b:Belief {{topic: $topic}})
            ON CREATE SET b.valence = 0.0, b.confidence = 0.1, b.uncertainty = 0.9,
                          b.evidence_count = 0, b.belief_text = '', b.provenance = '',
                          b.formed_at = datetime(), b.last_updated = datetime()
            SET b.last_updated = datetime()
            WITH b
            MATCH (e:Episode {{uid: $uid}})
            MERGE (e)-[r:{edge_type}]->(b)
            ON CREATE SET r.strength = $strength, r.reasoning = $reasoning, r.created_at = datetime()
            ON MATCH SET r.strength = $strength, r.reasoning = $reasoning
            WITH b
            SET b.evidence_count = count {{(b)<-[:{EdgeType.SUPPORTS_BELIEF}|{EdgeType.CONTRADICTS_BELIEF}]-()}}
            """,
            ),
            topic=normalize_topic(topic),
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
        keywords = [t for t in re.split(r"[^a-z0-9]+", query.lower()) if len(t) >= 2]
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

    async def get_topic_names(self) -> list[str]:
        """Return all non-archived topic names currently in the graph."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                "MATCH (t:Topic)<-[:DISCUSSES]-(:Episode {archived: false}) "
                "RETURN DISTINCT t.name AS name ORDER BY name"
            )
            return [record["name"] async for record in result]

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
            seen: set[str] = set()
            episodes: list[EpisodeNode] = []
            for node in record["befores"]:
                if node is not None:
                    ep = _record_to_episode(node)
                    if ep.uid not in seen:
                        seen.add(ep.uid)
                        episodes.append(ep)
            focal = _record_to_episode(record["focal"])
            if not focal.archived and focal.uid not in seen:
                seen.add(focal.uid)
                episodes.append(focal)
            for node in record["afters"]:
                if node is not None:
                    ep = _record_to_episode(node)
                    if ep.uid not in seen:
                        seen.add(ep.uid)
                        episodes.append(ep)
            return episodes

    async def update_episode_access(self, episode_uids: list[str]) -> None:
        """Update access_count, last_accessed, and utility_score for retrieved episodes.

        Uses FSRS power-law forgetting (Ye 2022) instead of exponential decay.
        Power-law matches empirical forgetting curves (Wixted & Ebbesen 1991).

        stability S = 30 * (0.5 + qc) days, where qc = mean of credibility signals.
        At t = S, retrievability R = 0.9 (90% recall probability).

        R(t, S) = 1 / (1 + t / (9 * S))       — FSRS power-law forgetting
        utility  = ln(n + 2) * ess_score * R   — ACT-R practice * salience * recall
        """
        if not episode_uids:
            return
        async with self._driver.session(database=_DB) as session:
            await session.run(
                """
                UNWIND $uids AS uid
                MATCH (e:Episode {uid: uid})
                WITH e,
                     (COALESCE(e.specificity, 0.0) + COALESCE(e.grounding, 0.0)
                      + COALESCE(e.rigor, 0.0) + COALESCE(e.source_quality, 0.0)
                      + COALESCE(e.objectivity, 0.0)) / 5.0 AS qc
                WITH e, 30.0 * (0.5 + qc) AS stability,
                     duration.between(datetime(e.created_at), datetime()).days AS age
                SET e.access_count = COALESCE(e.access_count, 0) + 1,
                    e.last_accessed = datetime(),
                    e.utility_score = log(COALESCE(e.access_count, 0) + 2)
                        * COALESCE(e.ess_score, 0.5)
                        / (1.0 + age / (9.0 * stability))
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
        """Hard-delete an episode node from Neo4j.

        Rewires the TEMPORAL_NEXT chain before deleting so that the
        predecessor links directly to the successor, preserving ordering.
        Recomputes evidence_count on affected beliefs from actual edges.
        Derivative vectors live in Qdrant and are cleaned separately.
        """
        async with self._driver.session(database=_DB) as session:
            await session.run(
                cast(
                    LiteralString,
                    f"""
                MATCH (e:Episode {{uid: $uid}})
                OPTIONAL MATCH (e)-[:{EdgeType.SUPPORTS_BELIEF}|{EdgeType.CONTRADICTS_BELIEF}]->(b:Belief)
                WITH e, collect(DISTINCT b) AS affected_beliefs
                OPTIONAL MATCH (prev)-[r1:{EdgeType.TEMPORAL_NEXT}]->(e)
                OPTIONAL MATCH (e)-[r2:{EdgeType.TEMPORAL_NEXT}]->(next)
                FOREACH (_ IN CASE WHEN prev IS NOT NULL AND next IS NOT NULL THEN [1] ELSE [] END |
                  MERGE (prev)-[:{EdgeType.TEMPORAL_NEXT}]->(next))
                DETACH DELETE e
                WITH affected_beliefs
                UNWIND affected_beliefs AS ab
                SET ab.evidence_count = count {{(ab)<-[:{EdgeType.SUPPORTS_BELIEF}|{EdgeType.CONTRADICTS_BELIEF}]-()}}
                """,
                ),
                uid=episode_uid,
            )
        log.debug("graph_delete_episode", episode_uid=episode_uid[:8])

    async def get_forgetting_candidates(
        self, *, limit: int = 20, min_age_minutes: int = 60
    ) -> list[EpisodeNode]:
        """Fetch low-utility raw (non-consolidated) episodes eligible for forgetting.

        Only considers episodes with consolidation_level=1 that are older than
        min_age_minutes, preventing the forgetting cycle from targeting just-ingested
        or already-consolidated episodes. ESS score is a secondary sort to protect
        high-significance episodes that haven't been retrieved yet.
        """
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                """
                MATCH (e:Episode)
                WHERE NOT e.archived AND e.consolidation_level = 1
                  AND datetime(e.created_at) < datetime() - duration({minutes: $min_age_minutes})
                RETURN e
                ORDER BY e.utility_score ASC, e.ess_score ASC, datetime(e.created_at) ASC
                LIMIT $limit
                """,
                limit=limit,
                min_age_minutes=min_age_minutes,
            )
            records = [record async for record in result]
            return [_record_to_episode(record["e"]) for record in records]

    async def get_belief_connections(self, episode_uids: list[str]) -> dict[str, list[str]]:
        """For each episode, return belief topics it is the sole evidence for."""
        if not episode_uids:
            return {}
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                cast(
                    LiteralString,
                    f"""
                UNWIND $uids AS uid
                MATCH (e:Episode {{uid: uid}})-[:{EdgeType.SUPPORTS_BELIEF}|{EdgeType.CONTRADICTS_BELIEF}]->(b:Belief)
                WHERE b.evidence_count <= 1
                RETURN uid, collect(b.topic) AS sole_topics
                """,
                ),
                uids=episode_uids,
            )
            return {
                record["uid"]: list(record["sole_topics"])
                async for record in result
                if record["sole_topics"]
            }

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
                    n.version = coalesce(n.version, 0) + 1
                """,
                sid=_DEFAULT_SESSION_ID,
                text=text,
            )
        log.info("graph_upsert_personality_snapshot", char_count=len(text))

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
        or derive from confidence". evidence_count is authoritative from
        link_belief (computed from actual edges), so sentinels preserve rather
        than increment.
        """
        topic = normalize_topic(topic)
        async with self._driver.session(database=_DB) as session:
            await session.run(
                """
                MERGE (b:Belief {topic: $topic})
                SET b.valence = $valence,
                    b.confidence = $confidence,
                    b.belief_text = CASE WHEN $belief_text <> '' THEN $belief_text ELSE coalesce(b.belief_text, '') END,
                    b.uncertainty = CASE WHEN $uncertainty >= 0 THEN $uncertainty ELSE coalesce(b.uncertainty, 1.0 - $confidence) END,
                    b.evidence_count = CASE WHEN $evidence_count >= 0 THEN $evidence_count ELSE coalesce(b.evidence_count, 0) END,
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
        """Fetch a single belief by topic with support/contradiction counts."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                cast(
                    LiteralString,
                    f"""
                    MATCH (b:Belief {{topic: $topic}})
                    OPTIONAL MATCH ()-[s:{EdgeType.SUPPORTS_BELIEF}]->(b)
                    OPTIONAL MATCH ()-[c:{EdgeType.CONTRADICTS_BELIEF}]->(b)
                    RETURN b, count(DISTINCT s) AS supports, count(DISTINCT c) AS contradicts
                    """,
                ),
                topic=normalize_topic(topic),
            )
            record = await result.single()
        if not record:
            return None
        return _record_to_belief(
            record["b"],
            support_count=int(record["supports"]),
            contradict_count=int(record["contradicts"]),
        )

    async def get_all_beliefs(self) -> list[BeliefNode]:
        """Fetch all beliefs with support/contradiction edge counts.

        Excludes placeholder beliefs created by topic linking (empty belief_text
        with default confidence 0.1). These are structural nodes that haven't
        been populated with actual belief content through reflection.
        """
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                cast(
                    LiteralString,
                    f"""
                    MATCH (b:Belief)
                    WHERE b.belief_text <> '' OR b.confidence > 0.15
                    OPTIONAL MATCH ()-[s:{EdgeType.SUPPORTS_BELIEF}]->(b)
                    OPTIONAL MATCH ()-[c:{EdgeType.CONTRADICTS_BELIEF}]->(b)
                    RETURN b, count(DISTINCT s) AS supports, count(DISTINCT c) AS contradicts
                    ORDER BY abs(b.valence) DESC, b.topic ASC
                    """,
                )
            )
            beliefs: list[BeliefNode] = []
            async for r in result:
                beliefs.append(
                    _record_to_belief(
                        r["b"],
                        support_count=int(r["supports"]),
                        contradict_count=int(r["contradicts"]),
                    )
                )
            return beliefs

    async def get_last_episode_uid(self) -> str:
        """Get the UID of the most recently created non-archived episode."""
        async with self._driver.session(database=_DB) as session:
            result = await session.run(
                "MATCH (e:Episode) WHERE NOT e.archived RETURN e.uid AS uid ORDER BY e.created_at DESC LIMIT 1"
            )
            record = await result.single()
            return str(record["uid"]) if record and record.get("uid") else ""


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


def _record_to_belief(
    node: Mapping[str, object],
    support_count: int = 0,
    contradict_count: int = 0,
) -> BeliefNode:
    """Convert a Neo4j Belief node record to a BeliefNode dataclass."""
    props = dict(node)
    return BeliefNode(
        topic=_str(props.get("topic")),
        valence=_float(props.get("valence")),
        confidence=_float(props.get("confidence"), 0.5),
        uncertainty=_float(props.get("uncertainty"), 0.5),
        evidence_count=_int(props.get("evidence_count")),
        support_count=support_count,
        contradict_count=contradict_count,
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
        consolidation_level=_int(props.get("consolidation_level"), 1),
        archived=bool(props.get("archived", False)),
        specificity=_float(props.get("specificity")),
        grounding=_float(props.get("grounding")),
        rigor=_float(props.get("rigor")),
        source_quality=_float(props.get("source_quality")),
        objectivity=_float(props.get("objectivity")),
    )
