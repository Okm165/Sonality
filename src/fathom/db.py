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

import structlog
from neo4j._typing import LiteralString

from neo4j import AsyncDriver, AsyncManagedTransaction
from shared.neo4j import connect as _neo4j_connect
from shared.types import new_id

from .config import settings
from .models import ChecklistItem, Fact


class SessionRow(TypedDict):
    """Shape of a Neo4j ResearchSession node unpacked to dict."""

    id: str
    goal: str
    status: str
    document: str
    pages_scraped: int


class FrontierRow(TypedDict):
    """Shape of a pending frontier URL row."""

    url: str
    anchor_text: str
    context: str
    source: str


class FactRow(TypedDict):
    """Shape of a research fact row."""

    claim: str
    confidence: float
    has_evidence: bool
    source_url: str
    topic: str

log = structlog.get_logger()

_driver: AsyncDriver | None = None

_DB = settings.neo4j_database

SCHEMA_STATEMENTS: tuple[str, ...] = (
    "CREATE CONSTRAINT rs_id IF NOT EXISTS FOR (s:ResearchSession) REQUIRE s.id IS UNIQUE",
    "CREATE CONSTRAINT fu_session_url IF NOT EXISTS FOR (u:FrontierURL) REQUIRE (u.session_id, u.url) IS UNIQUE",
    "CREATE INDEX fu_pending IF NOT EXISTS FOR (u:FrontierURL) ON (u.session_id, u.status)",
    "CREATE INDEX rf_session IF NOT EXISTS FOR (f:ResearchFact) ON (f.session_id)",
    "CREATE INDEX cq_session IF NOT EXISTS FOR (q:ChecklistQuestion) ON (q.session_id)",
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
                document: '', pages_scraped: 0,
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
    document: str | None = None,
    pages_scraped: int | None = None,
) -> None:
    sets: list[str] = []
    params: dict[str, str | int | None] = {"id": session_id}
    if status is not None:
        sets.append("s.status = $status")
        params["status"] = status
    if document is not None:
        sets.append("s.document = $document")
        params["document"] = document
    if pages_scraped is not None:
        sets.append("s.pages_scraped = $pages_scraped")
        params["pages_scraped"] = pages_scraped
    if status == "completed":
        sets.append("s.completed_at = datetime()")
    if not sets:
        return
    cypher = f"MATCH (s:ResearchSession {{id: $id}}) SET {', '.join(sets)}"
    async with driver.session(database=_DB) as session:
        await session.run(cast(LiteralString, cypher), parameters=params)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Frontier
# ---------------------------------------------------------------------------


async def add_urls(
    driver: AsyncDriver,
    session_id: str,
    urls: list[tuple[str, str, str, str]],
) -> int:
    """Insert URLs into frontier graph. Ignores duplicates (MERGE)."""
    if not urls:
        return 0

    async def _tx(tx: AsyncManagedTransaction) -> int:
        added = 0
        for url, anchor, ctx, src in urls:
            result = await tx.run(
                """
                MATCH (s:ResearchSession {id: $sid})
                MERGE (u:FrontierURL {session_id: $sid, url: $url})
                ON CREATE SET u.anchor_text = $anchor, u.context = $ctx,
                              u.source = $src, u.status = 'pending',
                              u.created_at = datetime()
                WITH s, u
                MERGE (s)-[:HAS_URL]->(u)
                RETURN u.status AS status
                """,
                sid=session_id,
                url=url,
                anchor=anchor,
                ctx=ctx,
                src=src,
            )
            record = await result.single()
            if record and record["status"] == "pending":
                added += 1
        return added

    async with driver.session(database=_DB) as session:
        return await session.execute_write(_tx)


async def get_pending_urls(driver: AsyncDriver, session_id: str) -> list[FrontierRow]:
    async with driver.session(database=_DB) as session:
        result = await session.run(
            """
            MATCH (s:ResearchSession {id: $sid})-[:HAS_URL]->(u:FrontierURL {status: 'pending'})
            WHERE u.session_id = $sid
            RETURN u.url AS url, u.anchor_text AS anchor_text,
                   u.context AS context, u.source AS source
            ORDER BY u.created_at
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

    async def _tx(tx: AsyncManagedTransaction) -> int:
        for fact in facts:
            await tx.run(
                """
                MATCH (s:ResearchSession {id: $sid})
                CREATE (f:ResearchFact {
                    id: $fid, session_id: $sid, claim: $claim,
                    confidence: $conf, has_evidence: $evidence,
                    source_url: $source_url, topic: $topic,
                    extracted_at: datetime()
                })
                CREATE (s)-[:HAS_FACT]->(f)
                WITH f
                OPTIONAL MATCH (u:FrontierURL {session_id: $sid, url: $source_url})
                FOREACH (_ IN CASE WHEN u IS NOT NULL THEN [1] ELSE [] END |
                    CREATE (f)-[:EXTRACTED_FROM]->(u)
                )
                """,
                sid=session_id,
                fid=fact.id,
                claim=fact.claim,
                conf=fact.confidence,
                evidence=fact.has_evidence,
                source_url=source_url,
                topic=fact.topic,
            )
        return len(facts)

    async with driver.session(database=_DB) as session:
        return await session.execute_write(_tx)


async def get_facts(driver: AsyncDriver, session_id: str) -> list[FactRow]:
    async with driver.session(database=_DB) as session:
        result = await session.run(
            """
            MATCH (s:ResearchSession {id: $sid})-[:HAS_FACT]->(f:ResearchFact)
            RETURN f.claim AS claim, f.confidence AS confidence,
                   f.has_evidence AS has_evidence, f.source_url AS source_url,
                   f.topic AS topic
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

    async def _tx(tx: AsyncManagedTransaction) -> None:
        await tx.run(
            """
            MATCH (s:ResearchSession {id: $sid})-[:HAS_QUESTION]->(q:ChecklistQuestion)
            DETACH DELETE q
            """,
            sid=session_id,
        )
        for item in items:
            await tx.run(
                """
                MATCH (s:ResearchSession {id: $sid})
                CREATE (q:ChecklistQuestion {
                    session_id: $sid, question: $question,
                    answered: $answered, evidence: $evidence
                })
                CREATE (s)-[:HAS_QUESTION]->(q)
                """,
                sid=session_id,
                question=item.question,
                answered=item.answered,
                evidence=item.evidence,
            )

    async with driver.session(database=_DB) as session:
        await session.execute_write(_tx)


async def get_checklist(driver: AsyncDriver, session_id: str) -> list[ChecklistItem]:
    async with driver.session(database=_DB) as session:
        result = await session.run(
            """
            MATCH (s:ResearchSession {id: $sid})-[:HAS_QUESTION]->(q:ChecklistQuestion)
            RETURN q.question AS question, q.answered AS answered, q.evidence AS evidence
            """,
            sid=session_id,
        )
        rows = await result.data()
    return [
        ChecklistItem(
            question=r["question"],
            answered=r["answered"],
            evidence=list(r["evidence"]) if r["evidence"] else [],
        )
        for r in rows
    ]
