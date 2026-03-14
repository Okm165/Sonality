"""Testcontainers support for isolated database testing.

Provides Neo4j and PostgreSQL/pgvector containers that spin up on demand.
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from sonality.schema import (
    NEO4J_IMAGE,
    NEO4J_SCHEMA_STATEMENTS,
    PGVECTOR_IMAGE,
    POSTGRES_SCHEMA_SQL,
)

log = logging.getLogger(__name__)


@dataclass
class ContainerConfig:
    """Connection details for a running container."""

    postgres_url: str
    neo4j_url: str
    neo4j_user: str
    neo4j_password: str


def _wait_for_postgres(url: str, max_attempts: int = 30) -> bool:
    """Wait for PostgreSQL to accept connections."""
    import psycopg

    for _attempt in range(max_attempts):
        try:
            with psycopg.connect(url, autocommit=True) as conn:
                conn.execute("SELECT 1")
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


def _init_postgres_schema(url: str) -> None:
    """Initialize PostgreSQL schema for testing."""
    import psycopg

    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(POSTGRES_SCHEMA_SQL)
    log.info("PostgreSQL schema initialized")


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
def postgres_container() -> Generator[str, None, None]:
    """Start a PostgreSQL/pgvector container and return connection URL."""
    from testcontainers.postgres import PostgresContainer

    log.info("Starting PostgreSQL/pgvector container...")
    container = PostgresContainer(
        image=PGVECTOR_IMAGE,
        username="sonality",
        password="sonality_password",
        dbname="sonality",
    )
    with container:
        url = container.get_connection_url()
        url = url.replace("postgresql+psycopg2://", "postgresql://")
        if not _wait_for_postgres(url):
            raise RuntimeError("PostgreSQL container failed to start")
        _init_postgres_schema(url)
        log.info("PostgreSQL container ready at %s", url.split("@")[-1])
        yield url


@contextmanager
def neo4j_container() -> Generator[tuple[str, str, str], None, None]:
    """Start a Neo4j container and return (url, user, password)."""
    from testcontainers.neo4j import Neo4jContainer

    log.info("Starting Neo4j container...")
    container = Neo4jContainer(image=NEO4J_IMAGE)
    with container:
        url = container.get_connection_url()
        user = "neo4j"
        password = container.NEO4J_ADMIN_PASSWORD
        auth = (user, password)
        if not _wait_for_neo4j(url, auth):
            raise RuntimeError("Neo4j container failed to start")
        _init_neo4j_schema(url, auth)
        log.info("Neo4j container ready at %s", url)
        yield url, user, password


@contextmanager
def both_containers() -> Generator[ContainerConfig, None, None]:
    """Start both database containers and yield config."""
    with postgres_container() as pg_url, neo4j_container() as neo4j_info:
        neo4j_url, neo4j_user, neo4j_password = neo4j_info
        yield ContainerConfig(
            postgres_url=pg_url,
            neo4j_url=neo4j_url,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
        )


def patch_config_for_containers(config_module: object, container_config: ContainerConfig) -> None:
    """Monkey-patch sonality.config with container connection details."""
    import sonality.config as cfg

    object.__setattr__(cfg, "POSTGRES_URL", container_config.postgres_url)
    object.__setattr__(cfg, "NEO4J_URL", container_config.neo4j_url)
    object.__setattr__(cfg, "NEO4J_USER", container_config.neo4j_user)
    object.__setattr__(cfg, "NEO4J_PASSWORD", container_config.neo4j_password)


def clear_databases(postgres_url: str, neo4j_url: str, neo4j_auth: tuple[str, str]) -> None:
    """Clear all data from both databases while preserving schema."""
    import psycopg
    from neo4j import GraphDatabase

    with psycopg.connect(postgres_url, autocommit=True) as conn:
        conn.execute("TRUNCATE derivatives, semantic_features RESTART IDENTITY CASCADE")
        conn.execute("DELETE FROM stm_state WHERE session_id != 'default'")
        conn.execute(
            "UPDATE stm_state SET running_summary = '', message_buffer = '[]'::jsonb "
            "WHERE session_id = 'default'"
        )

    driver = GraphDatabase.driver(neo4j_url, auth=neo4j_auth)
    try:
        with driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    finally:
        driver.close()

    log.info("Databases cleared")
