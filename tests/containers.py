"""Testcontainers support for isolated database testing.

Provides Neo4j and Qdrant containers that spin up on demand.
Use the session-scoped fixtures to share containers across tests, or
function-scoped fixtures for complete isolation.

Usage:
    pytest tests/ --use-containers  # Force testcontainers instead of local DBs
    pytest tests/                   # Uses local DBs if available, containers as fallback
"""

from __future__ import annotations

import logging
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass

from sonality.schema import (
    NEO4J_SCHEMA_STATEMENTS,
    Collection,
    init_qdrant_collections,
)

log = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Connection details for running containers."""

    qdrant_url: str
    neo4j_url: str
    neo4j_user: str
    neo4j_password: str


def _wait_for_qdrant(url: str, max_attempts: int = 30) -> bool:
    """Wait for Qdrant to accept connections."""
    import httpx

    for _attempt in range(max_attempts):
        try:
            resp = httpx.get(f"{url}/readyz", timeout=5)
            if resp.status_code == 200:
                return True
        except Exception:
            time.sleep(1)
    return False


def _wait_for_neo4j(url: str, auth: tuple[str, str], max_attempts: int = 30) -> bool:
    """Wait for Neo4j to accept connections."""
    from neo4j import GraphDatabase

    for _attempt in range(max_attempts):
        try:
            driver = GraphDatabase.driver(url, auth=auth)
            with driver.session() as session:
                session.run("RETURN 1").single()
            driver.close()
            return True
        except Exception:
            time.sleep(1)
    return False


async def _init_qdrant_schema(url: str) -> None:
    """Initialize Qdrant collections for testing."""
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=url)
    await init_qdrant_collections(client)
    await client.close()
    log.info("Qdrant collections initialized")


def _init_neo4j_schema(url: str, auth: tuple[str, str]) -> None:
    """Initialize Neo4j schema for testing."""
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(url, auth=auth)
    try:
        with driver.session() as session:
            for stmt in NEO4J_SCHEMA_STATEMENTS:
                session.run(stmt)
    finally:
        driver.close()
    log.info("Neo4j schema initialized")


@contextmanager
def qdrant_container() -> Generator[str, None, None]:
    """Start a Qdrant container and return connection URL."""
    import asyncio

    from testcontainers.qdrant import QdrantContainer

    log.info("Starting Qdrant container...")
    container = QdrantContainer(image="qdrant/qdrant:latest")
    with container:
        url = f"http://{container.get_container_host_ip()}:{container.get_exposed_port(6333)}"
        if not _wait_for_qdrant(url):
            raise RuntimeError("Qdrant container failed to start")
        asyncio.run(_init_qdrant_schema(url))
        log.info("Qdrant container ready at %s", url)
        yield url


@contextmanager
def neo4j_container() -> Generator[tuple[str, str, str], None, None]:
    """Start a Neo4j container and return (url, user, password)."""
    from testcontainers.neo4j import Neo4jContainer

    log.info("Starting Neo4j container...")
    container = Neo4jContainer(image="neo4j:5")
    with container:
        url = container.get_connection_url()
        user = "neo4j"
        password = container.password
        auth = (user, password)
        if not _wait_for_neo4j(url, auth):
            raise RuntimeError("Neo4j container failed to start")
        _init_neo4j_schema(url, auth)
        log.info("Neo4j container ready at %s", url)
        yield url, user, password


@contextmanager
def both_containers() -> Generator[ContainerConfig, None, None]:
    """Start both database containers and yield config."""
    with qdrant_container() as qdrant_url, neo4j_container() as neo4j_info:
        neo4j_url, neo4j_user, neo4j_password = neo4j_info
        yield ContainerConfig(
            qdrant_url=qdrant_url,
            neo4j_url=neo4j_url,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )


def patch_config_for_containers(container_config: ContainerConfig) -> None:
    """Monkey-patch sonality.config with container connection details."""
    import sonality.config as cfg

    object.__setattr__(cfg, "QDRANT_URL", container_config.qdrant_url)
    object.__setattr__(cfg, "NEO4J_URL", container_config.neo4j_url)
    object.__setattr__(cfg, "NEO4J_USER", container_config.neo4j_user)
    object.__setattr__(cfg, "NEO4J_PASSWORD", container_config.neo4j_password)


async def clear_databases(qdrant_url: str, neo4j_url: str, neo4j_auth: tuple[str, str]) -> None:
    """Clear all data from both databases while preserving schema."""
    from neo4j import GraphDatabase
    from qdrant_client import AsyncQdrantClient

    client = AsyncQdrantClient(url=qdrant_url)
    for collection in Collection:
        if await client.collection_exists(collection):
            await client.delete_collection(collection)
    await init_qdrant_collections(client)
    await client.close()

    driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()

    log.info("Databases cleared")
