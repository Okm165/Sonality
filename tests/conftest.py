"""Pytest configuration and fixtures for Sonality tests.

Container fixtures:
- `db_containers` — Session-scoped, shared across all tests (efficient)
- `isolated_pg` — Function-scoped PostgreSQL (fresh per test)
- `isolated_neo4j` — Function-scoped Neo4j (fresh per test)

Enable containers with: pytest --use-containers
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Generator
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import BaseModel

from sonality.llm.caller import LLMCallResult

if TYPE_CHECKING:
    from tests.containers import ContainerConfig

log = logging.getLogger(__name__)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add testcontainers command-line option."""
    parser.addoption(
        "--use-containers",
        action="store_true",
        default=False,
        help="Use testcontainers for Neo4j and PostgreSQL instead of local DBs.",
    )


@pytest.fixture(scope="session")
def use_containers(pytestconfig: pytest.Config) -> bool:
    """Whether to use testcontainers for database isolation."""
    return bool(pytestconfig.getoption("--use-containers"))


@pytest.fixture(scope="session")
def db_containers(use_containers: bool) -> Generator[dict[str, Any] | None, None, None]:
    """Session-scoped database containers (only started if --use-containers is set)."""
    if not use_containers:
        yield None
        return

    from tests.containers import both_containers, patch_config_for_containers

    log.info("Starting testcontainers for isolated database testing...")
    with both_containers() as config:
        import sonality.config as cfg
        patch_config_for_containers(cfg, config)
        yield {
            "postgres_url": config.postgres_url,
            "neo4j_url": config.neo4j_url,
            "neo4j_user": config.neo4j_user,
            "neo4j_password": config.neo4j_password,
        }


@pytest.fixture(autouse=True)
def clear_db_between_tests(
    use_containers: bool, db_containers: dict[str, Any] | None
) -> Generator[None, None, None]:
    """Clear databases between tests when using containers."""
    yield
    if use_containers and db_containers:
        from tests.containers import clear_databases
        clear_databases(
            db_containers["postgres_url"],
            db_containers["neo4j_url"],
            (db_containers["neo4j_user"], db_containers["neo4j_password"]),
        )


@pytest.fixture(scope="function")
def isolated_pg() -> Generator[str, None, None]:
    """Function-scoped PostgreSQL container for complete isolation.

    Use this when a test requires a completely fresh database with no
    state from other tests. More expensive than session-scoped fixtures.
    """
    from tests.containers import postgres_container

    with postgres_container() as url:
        yield url


@pytest.fixture(scope="function")
def isolated_neo4j() -> Generator[tuple[str, str, str], None, None]:
    """Function-scoped Neo4j container for complete isolation.

    Returns (url, user, password). Use when a test requires a completely
    fresh graph database with no state from other tests.
    """
    from tests.containers import neo4j_container

    with neo4j_container() as info:
        yield info


@pytest.fixture(scope="function")
def isolated_both() -> Generator[ContainerConfig, None, None]:
    """Function-scoped both containers for complete test isolation.

    Use for tests that need fresh instances of both databases.
    """
    from tests.containers import both_containers

    with both_containers() as config:
        yield config


@pytest.fixture
def mock_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[dict[str, dict[str, object]]], None]:
    """Patch llm_call across modules with deterministic prompt-keyed responses."""

    responses: dict[str, dict[str, object]] = {}

    def configure(mapping: dict[str, dict[str, object]]) -> None:
        responses.clear()
        responses.update(mapping)

    def fake_call[T: BaseModel](
        *,
        prompt: str,
        response_model: type[T],
        fallback: T,
        **_: object,
    ) -> LLMCallResult[T]:
        for key, response in responses.items():
            if key in prompt:
                return LLMCallResult(
                    value=response_model.model_validate(response),
                    success=True,
                    attempts=1,
                    raw_text=json.dumps(response),
                )
        return LLMCallResult(
            value=fallback,
            success=False,
            error=f"No canned response for prompt: {prompt[:40]}",
            attempts=1,
            raw_text="",
        )

    targets = (
        "sonality.llm.caller.llm_call",
        "sonality.memory.retrieval.router.llm_call",
        "sonality.memory.retrieval.reranker.llm_call",
        "sonality.memory.retrieval.chain.llm_call",
        "sonality.memory.retrieval.split.llm_call",
        "sonality.agent.llm_call",
        "sonality.memory.belief_provenance.llm_call",
        "sonality.memory.forgetting.llm_call",
        "sonality.memory.updater.llm_call",
        "sonality.memory.knowledge_extract.llm_call",
    )
    for target in targets:
        monkeypatch.setattr(target, fake_call, raising=False)
    return configure
