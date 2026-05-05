"""Database connection management for Neo4j and Qdrant.

Single driver/client instances created at startup, reused across all operations,
and closed gracefully on shutdown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from neo4j import AsyncDriver, AsyncGraphDatabase
from qdrant_client import AsyncQdrantClient

from .. import config
from ..schema import NEO4J_SCHEMA_STATEMENTS, init_qdrant_collections

log = logging.getLogger(__name__)


@dataclass
class DatabaseConnections:
    """Holds Neo4j driver and Qdrant client for the application lifetime."""

    neo4j_driver: AsyncDriver = field(init=False)
    qdrant: AsyncQdrantClient = field(init=False)

    @classmethod
    async def create(cls) -> DatabaseConnections:
        """Create and verify all database connections."""
        self = cls()
        log.info("Connecting to Neo4j at %s", config.NEO4J_URL)
        self.neo4j_driver = AsyncGraphDatabase.driver(
            config.NEO4J_URL,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
            max_connection_pool_size=config.NEO4J_MAX_POOL_SIZE,
            connection_timeout=config.NEO4J_CONNECTION_TIMEOUT,
        )
        # Verify Neo4j connectivity
        async with self.neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            await session.run("RETURN 1")
        log.info("Neo4j connected")

        async with self.neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            for stmt in NEO4J_SCHEMA_STATEMENTS:
                await session.run(stmt)
        log.info("Neo4j schema initialized")

        log.info("Connecting to Qdrant at %s", config.QDRANT_URL)
        self.qdrant = AsyncQdrantClient(url=config.QDRANT_URL)
        await init_qdrant_collections(self.qdrant)
        log.info("Qdrant connected and collections initialized")

        return self

    async def close(self) -> None:
        """Gracefully close all connections."""
        log.info("Closing database connections")
        await self.neo4j_driver.close()
        await self.qdrant.close()
        log.info("Database connections closed")
