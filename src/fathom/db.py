"""Neo4j graph database — all fathom I/O in one place.

Graph model (separate from sonality's Episode/Belief labels):
  (:ResearchSession)-[:HAS_URL]->(:FrontierURL)
  (:ResearchSession)-[:HAS_FACT]->(:ResearchFact)
  (:ResearchSession)-[:HAS_QUESTION]->(:ChecklistQuestion)
  (:ResearchFact)-[:EXTRACTED_FROM]->(:FrontierURL)

Shares the same Neo4j instance as sonality; node labels prevent collision.
"""

from __future__ import annotations

from typing import TypedDict, cast

from neo4j._typing import LiteralString

from neo4j import AsyncDriver
from shared.neo4j import connect as _neo4j_connect
from shared.neo4j import ping as _neo4j_ping
from shared.types import new_id

from .config import settings
from .models import ChecklistItem, Fact


class SessionRow(TypedDict):
    """Shape of a Neo4j ResearchSession node unpacked to dict."""

    id: str
    goal: str
    status: str
    pages_scraped: int


class FrontierRow(TypedDict):
    """Shape of a pending frontier URL row."""

    url: str
    anchor_text: str
    context: str


class FactRow(TypedDict):
    """Shape of a research fact row."""

    claim: str
    confidence: float
    source_quality: float
    source_url: str
    topic: str


_driver: AsyncDriver | None = None

_DB = settings.neo4j_database

SCHEMA_STATEMENTS: tuple[str, ...] = (
    # Session-scoped nodes
    "CREATE CONSTRAINT rs_id IF NOT EXISTS FOR (s:ResearchSession) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT fu_session_url IF NOT EXISTS FOR (u:FrontierURL) REQUIRE (u.session_id, u.url) IS UNIQUE",
    "CREATE INDEX fu_pending IF NOT EXISTS FOR (u:FrontierURL) ON (u.session_id, u.status)",
    "CREATE CONSTRAINT rf_id IF NOT EXISTS FOR (f:ResearchFact) REQUIRE f.id IS UNIQUE",
    "CREATE INDEX rf_session IF NOT EXISTS FOR (f:ResearchFact) ON (f.session_id)",
    # Persistent source map (cross-session learning)
    "CREATE CONSTRAINT sd_domain IF NOT EXISTS FOR (d:SourceDomain) REQUIRE d.domain IS UNIQUE",
    "CREATE CONSTRAINT tc_name IF NOT EXISTS FOR (t:TopicCluster) REQUIRE t.name IS UNIQUE",
    "CREATE INDEX sd_quality IF NOT EXISTS FOR (d:SourceDomain) ON (d.quality_rate)",
    "CREATE INDEX tc_fact_count IF NOT EXISTS FOR (t:TopicCluster) ON (t.fact_count)",
)


async def get_driver() -> AsyncDriver:
    global _driver
    if _driver is None:
        _driver = await _neo4j_connect(
            settings.neo4j_url,
            settings.neo4j_user,
            settings.neo4j_password,
            database=_DB,
            schema_statements=SCHEMA_STATEMENTS,
            max_pool_size=settings.neo4j_max_pool_size,
            connection_timeout=settings.neo4j_connection_timeout,
        )
    return _driver


async def close_driver() -> None:
    global _driver
    if _driver is not None:
        await _driver.close()
        _driver = None


async def check() -> None:
    """Verify Neo4j is still reachable."""
    await _neo4j_ping(await get_driver(), _DB)


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


async def create_session(driver: AsyncDriver, goal: str) -> str:
    session_id = new_id()
    async with driver.session(database=_DB) as session:
        await session.run(
            """
            CREATE (s:ResearchSession {
                id: $id, goal: $goal, status: 'active',
                pages_scraped: 0,
                created_at: datetime(), completed_at: null
            })
            """,
            id=session_id,
            goal=goal,
        )
    return session_id


async def get_session(driver: AsyncDriver, session_id: str) -> SessionRow | None:
    async with driver.session(database=_DB) as session:
        result = await session.run(
            "MATCH (s:ResearchSession {id: $id}) RETURN s",
            id=session_id,
        )
        record = await result.single()
    if record is None:
        return None
    return cast(SessionRow, dict(record["s"]))


async def update_session(
    driver: AsyncDriver,
    session_id: str,
    *,
    status: str | None = None,
    pages_scraped: int | None = None,
    fact_count: int | None = None,
) -> None:
    sets: list[str] = []
    params: dict[str, str | int | None] = {"id": session_id}
    if status is not None:
        sets.append("s.status = $status")
        params["status"] = status
    if pages_scraped is not None:
        sets.append("s.pages_scraped = $pages_scraped")
        params["pages_scraped"] = pages_scraped
    if fact_count is not None:
        sets.append("s.fact_count = $fact_count")
        params["fact_count"] = fact_count
    if status in ("completed", "timed_out", "failed"):
        sets.append("s.completed_at = datetime()")
    if not sets:
        return
    cypher = f"MATCH (s:ResearchSession {{id: $id}}) SET {', '.join(sets)}"
    async with driver.session(database=_DB) as session:
        await session.run(cast(LiteralString, cypher), parameters=params)  # type: ignore[arg-type]


async def cleanup_stale_sessions(driver: AsyncDriver, *, max_age_s: int = 3600) -> int:
    """Mark 'active' sessions older than max_age_s as 'timed_out'."""
    async with driver.session(database=_DB) as session:
        result = await session.run(
            """
            MATCH (s:ResearchSession {status: 'active'})
            WHERE s.created_at < datetime() - duration({seconds: $max_age})
            SET s.status = 'timed_out', s.completed_at = datetime()
            RETURN count(s) AS cleaned
            """,
            max_age=max_age_s,
        )
        record = await result.single()
        return record["cleaned"] if record else 0


# ---------------------------------------------------------------------------
# Frontier
# ---------------------------------------------------------------------------


async def add_urls(
    driver: AsyncDriver,
    session_id: str,
    urls: list[tuple[str, str, str, str]],
) -> None:
    """Insert URLs into frontier graph. Ignores duplicates (MERGE)."""
    if not urls:
        return
    rows = [{"url": u, "anchor": a, "ctx": c, "src": s} for u, a, c, s in urls]
    async with driver.session(database=_DB) as session:
        await session.run(
            """
            MATCH (s:ResearchSession {id: $sid})
            UNWIND $rows AS row
            MERGE (u:FrontierURL {session_id: $sid, url: row.url})
            ON CREATE SET u.anchor_text = row.anchor, u.context = row.ctx,
                          u.source = row.src, u.status = 'pending',
                          u.created_at = datetime()
            MERGE (s)-[:HAS_URL]->(u)
            """,
            sid=session_id,
            rows=rows,
        )


async def get_pending_urls(driver: AsyncDriver, session_id: str) -> list[FrontierRow]:
    async with driver.session(database=_DB) as session:
        result = await session.run(
            """
            MATCH (s:ResearchSession {id: $sid})-[:HAS_URL]->(u:FrontierURL {status: 'pending'})
            RETURN u.url AS url, u.anchor_text AS anchor_text, u.context AS context
            ORDER BY rand()
            """,
            sid=session_id,
        )
        return [cast(FrontierRow, dict(r)) for r in await result.data()]


async def mark_urls_fetched(driver: AsyncDriver, session_id: str, urls: list[str]) -> None:
    """Mark selected URLs as fetched."""
    if not urls:
        return
    async with driver.session(database=_DB) as session:
        await session.run(
            """
            UNWIND $urls AS url
            MATCH (u:FrontierURL {session_id: $sid, url: url})
            SET u.status = 'fetched'
            """,
            sid=session_id,
            urls=urls,
        )


# ---------------------------------------------------------------------------
# Facts
# ---------------------------------------------------------------------------


async def insert_facts(
    driver: AsyncDriver,
    session_id: str,
    facts: list[Fact],
    source_url: str,
) -> int:
    """Insert facts as graph nodes linked to session and source URL."""
    if not facts:
        return 0
    rows = [
        {
            "fid": f.id,
            "claim": f.claim,
            "conf": f.confidence,
            "sq": f.source_quality,
            "topic": f.topic,
        }
        for f in facts
    ]
    async with driver.session(database=_DB) as session:
        await session.run(
            """
            MATCH (s:ResearchSession {id: $sid})
            OPTIONAL MATCH (u:FrontierURL {session_id: $sid, url: $source_url})
            UNWIND $rows AS row
            MERGE (f:ResearchFact {id: row.fid})
            ON CREATE SET f.session_id = $sid, f.claim = row.claim,
                f.confidence = row.conf, f.source_quality = row.sq,
                f.source_url = $source_url, f.topic = row.topic,
                f.extracted_at = datetime()
            MERGE (s)-[:HAS_FACT]->(f)
            FOREACH (_ IN CASE WHEN u IS NOT NULL THEN [1] ELSE [] END |
                MERGE (f)-[:EXTRACTED_FROM]->(u)
            )
            """,
            sid=session_id,
            source_url=source_url,
            rows=rows,
        )
    return len(facts)


async def get_facts(driver: AsyncDriver, session_id: str) -> list[FactRow]:
    async with driver.session(database=_DB) as session:
        result = await session.run(
            """
            MATCH (s:ResearchSession {id: $sid})-[:HAS_FACT]->(f:ResearchFact)
            RETURN f.claim AS claim, f.confidence AS confidence,
                   f.source_quality AS source_quality,
                   f.source_url AS source_url, f.topic AS topic
            ORDER BY f.extracted_at
            """,
            sid=session_id,
        )
        return [cast(FactRow, dict(r)) for r in await result.data()]


# ---------------------------------------------------------------------------
# Checklist
# ---------------------------------------------------------------------------


async def save_checklist(
    driver: AsyncDriver,
    session_id: str,
    items: list[ChecklistItem],
) -> None:
    """Replace entire checklist for a session."""
    rows = [{"question": i.question} for i in items]
    async with driver.session(database=_DB) as session:
        await session.run(
            """
            MATCH (s:ResearchSession {id: $sid})
            OPTIONAL MATCH (s)-[:HAS_QUESTION]->(old:ChecklistQuestion)
            DETACH DELETE old
            WITH DISTINCT s
            UNWIND $rows AS row
            CREATE (q:ChecklistQuestion {question: row.question})
            CREATE (s)-[:HAS_QUESTION]->(q)
            """,
            sid=session_id,
            rows=rows,
        )


async def get_checklist(driver: AsyncDriver, session_id: str) -> list[ChecklistItem]:
    async with driver.session(database=_DB) as session:
        result = await session.run(
            """
            MATCH (s:ResearchSession {id: $sid})-[:HAS_QUESTION]->(q:ChecklistQuestion)
            RETURN q.question AS question
            """,
            sid=session_id,
        )
        rows = await result.data()
    return [ChecklistItem(question=r["question"]) for r in rows]
