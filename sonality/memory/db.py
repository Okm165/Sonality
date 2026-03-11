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

log = logging.getLogger(__name__)

# Neo4j schema initialization Cypher statements
_NEO4J_INIT_STATEMENTS: list[str] = [
    "CREATE CONSTRAINT episode_uid IF NOT EXISTS FOR (e:Episode) REQUIRE e.uid IS UNIQUE",
    "CREATE CONSTRAINT derivative_uid IF NOT EXISTS FOR (d:Derivative) REQUIRE d.uid IS UNIQUE",
    "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT segment_id IF NOT EXISTS FOR (s:Segment) REQUIRE s.segment_id IS UNIQUE",
    "CREATE CONSTRAINT summary_uid IF NOT EXISTS FOR (s:Summary) REQUIRE s.uid IS UNIQUE",
    "CREATE CONSTRAINT belief_topic IF NOT EXISTS FOR (b:Belief) REQUIRE b.topic IS UNIQUE",
    "CREATE INDEX episode_created_at IF NOT EXISTS FOR (e:Episode) ON (e.created_at)",
    "CREATE INDEX episode_segment IF NOT EXISTS FOR (e:Episode) ON (e.segment_id)",
    "CREATE INDEX derivative_episode IF NOT EXISTS FOR (d:Derivative) ON (d.source_episode_uid)",
]


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
            for stmt in _NEO4J_INIT_STATEMENTS:
                await session.run(stmt)
        log.info("Neo4j schema initialized")

        log.info("Connecting to PostgreSQL at %s", config.POSTGRES_URL.split("@")[-1])
        self.pg_pool = AsyncConnectionPool(
            conninfo=config.POSTGRES_URL,
            min_size=config.PG_POOL_MIN_SIZE,
            max_size=config.PG_POOL_MAX_SIZE,
            open=False,
        )
        await self.pg_pool.open()
        await self.pg_pool.wait()

        # Register pgvector type
        async with self.pg_pool.connection() as conn:
            from pgvector.psycopg import register_vector_async

            await register_vector_async(conn)
        log.info("PostgreSQL connected (pgvector registered)")

        return self

    async def close(self) -> None:
        """Gracefully close all connections."""
        log.info("Closing database connections")
        await self.neo4j_driver.close()
        await self.pg_pool.close()
        log.info("Database connections closed")
