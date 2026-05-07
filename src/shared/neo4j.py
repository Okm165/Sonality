"""Neo4j bootstrap helpers shared by sonality and fathom.

Provides connect → verify → apply-schema in one call, reducing duplication
across packages that share the same Neo4j instance.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from neo4j import AsyncGraphDatabase

if TYPE_CHECKING:
    from neo4j import AsyncDriver

log = structlog.get_logger()


async def connect(
    url: str,
    user: str,
    password: str,
    database: str = "neo4j",
    schema_statements: tuple[str, ...] = (),
    *,
    max_pool_size: int = 50,
    connection_timeout: float = 30.0,
) -> AsyncDriver:
    """Create an AsyncDriver, verify connectivity, and apply schema.

    Returns the ready-to-use driver.
    """
    driver = AsyncGraphDatabase.driver(
        url,
        auth=(user, password),
        max_connection_pool_size=max_pool_size,
        connection_timeout=connection_timeout,
    )
    async with driver.session(database=database) as session:
        await session.run("RETURN 1")
    log.info("neo4j_connected", url=url)

    if schema_statements:
        async with driver.session(database=database) as session:
            for stmt in schema_statements:
                await session.run(stmt)  # type: ignore[arg-type]
        log.info("neo4j_schema_applied", statements=len(schema_statements))

    return driver
