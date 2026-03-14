"""Database connection management for Neo4j and PostgreSQL/pgvector.

Single driver/pool instances created at startup, reused across all operations,
and closed gracefully on shutdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from neo4j import AsyncDriver, AsyncGraphDatabase
from psycopg_pool import AsyncConnectionPool

from .. import config
from ..schema import NEO4J_SCHEMA_STATEMENTS

log = logging.getLogger(__name__)

# Neo4j emits INFORMATION-level GqlStatus notifications when IF NOT EXISTS constraints
# already exist. These are harmless but very verbose in test/startup logs.
logging.getLogger("neo4j.notifications").setLevel(logging.WARNING)


async def _configure_pgvector(conn: object) -> None:
    """Register pgvector adapters for each pooled PostgreSQL connection."""
    from pgvector.psycopg import register_vector_async  # type: ignore[import-untyped]

    await register_vector_async(conn)


@dataclass
class DatabaseConnections:
    """Holds Neo4j driver and PostgreSQL pool for the application lifetime."""

    neo4j_driver: AsyncDriver = field(init=False)
    pg_pool: AsyncConnectionPool = field(init=False)

    @classmethod
    async def create(cls) -> DatabaseConnections:
        """Create and verify all database connections."""
        self = cls()
        log.info("Connecting to Neo4j at %s", config.NEO4J_URL)
        self.neo4j_driver = AsyncGraphDatabase.driver(
            config.NEO4J_URL,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        # Verify Neo4j connectivity
        async with self.neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            await session.run("RETURN 1")
        log.info("Neo4j connected")

        # Initialize Neo4j schema
        async with self.neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            for stmt in NEO4J_SCHEMA_STATEMENTS:
                await session.run(stmt)
        log.info("Neo4j schema initialized")

        log.info("Connecting to PostgreSQL at %s", config.POSTGRES_URL.split("@")[-1])
        self.pg_pool = AsyncConnectionPool(
            conninfo=config.POSTGRES_URL,
            min_size=config.PG_POOL_MIN_SIZE,
            max_size=config.PG_POOL_MAX_SIZE,
            configure=_configure_pgvector,
            open=False,
        )
        await self.pg_pool.open()
        await self.pg_pool.wait()
        log.info("PostgreSQL connected (pgvector configured)")

        return self

    async def close(self) -> None:
        """Gracefully close all connections."""
        log.info("Closing database connections")
        await self.neo4j_driver.close()
        await self.pg_pool.close()
        log.info("Database connections closed")
