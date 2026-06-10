"""Database connection management for Neo4j and Qdrant.

Single driver/client instances created at startup, reused across all operations,
and closed gracefully on shutdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog
from qdrant_client import AsyncQdrantClient

from neo4j import AsyncDriver
from shared.neo4j import connect as neo4j_connect

from .. import config
from ..schema import NEO4J_SCHEMA_STATEMENTS, init_qdrant_collections

log = structlog.get_logger(__name__)


@dataclass
class DatabaseConnections:
    """Holds Neo4j driver and Qdrant client for the application lifetime."""

    neo4j_driver: AsyncDriver = field(init=False)
    qdrant: AsyncQdrantClient = field(init=False)

    @classmethod
    async def create(cls, *, embedding_dims: int = 0) -> DatabaseConnections:
        """Create and verify all database connections.

        Args:
            embedding_dims: Actual embedding dimensions (from server probe).
                When 0, falls back to config.settings.embedding_dimensions.
        """
        self = cls()
        self.neo4j_driver = await neo4j_connect(
            config.settings.neo4j_url,
            config.settings.neo4j_user,
            config.settings.neo4j_password,
            database=config.settings.neo4j_database,
            schema_statements=NEO4J_SCHEMA_STATEMENTS,
            max_pool_size=config.settings.neo4j_max_pool_size,
            connection_timeout=config.settings.neo4j_connection_timeout,
        )

        log.info("qdrant_connecting", url=config.settings.qdrant_url)
        self.qdrant = AsyncQdrantClient(url=config.settings.qdrant_url)
        await init_qdrant_collections(self.qdrant, dims=embedding_dims)
        log.info("qdrant_ready", dims=embedding_dims or config.settings.embedding_dimensions)

        return self

    async def close(self) -> None:
        """Gracefully close all connections."""
        log.info("db_closing")
        if hasattr(self, "neo4j_driver"):
            try:
                await self.neo4j_driver.close()
            except Exception:
                log.warning("neo4j_close_failed", exc_info=True)
        if hasattr(self, "qdrant"):
            try:
                await self.qdrant.close()
            except Exception:
                log.warning("qdrant_close_failed", exc_info=True)
        log.info("db_closed")
