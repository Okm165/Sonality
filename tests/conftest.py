"""Pytest configuration and fixtures for Sonality tests.

Enable containers with: pytest --use-containers
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Generator
from typing import Final

import pytest
from pydantic import BaseModel

from sonality.caller import LLMCallResult

log = logging.getLogger(__name__)
NO_DB_CONTAINERS: Final[dict[str, str]] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add testcontainers command-line option."""
    parser.addoption(
        "--use-containers",
        action="store_true",
        default=False,
        help="Use testcontainers for Neo4j and Qdrant instead of local DBs.",
    )


@pytest.fixture(scope="session")
def db_containers(pytestconfig: pytest.Config) -> Generator[dict[str, str], None, None]:
    """Session-scoped database containers (only started if --use-containers is set)."""
    if not bool(pytestconfig.getoption("--use-containers")):
        yield NO_DB_CONTAINERS
        return

    from tests.containers import both_containers, patch_config_for_containers

    log.info("Starting testcontainers for isolated database testing...")
    with both_containers() as config:
        patch_config_for_containers(config)
        yield {
            "qdrant_url": config.qdrant_url,
            "neo4j_url": config.neo4j_url,
            "neo4j_user": config.neo4j_user,
            "neo4j_password": config.neo4j_password,
        }


@pytest.fixture(autouse=True)
def clear_db_between_tests(db_containers: dict[str, str]) -> Generator[None, None, None]:
    """Clear databases between tests when using containers."""
    yield
    if db_containers:
        import asyncio

        from tests.containers import clear_databases

        asyncio.run(
            clear_databases(
                db_containers["qdrant_url"],
                db_containers["neo4j_url"],
                (db_containers["neo4j_user"], db_containers["neo4j_password"]),
            )
        )


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
        "sonality.caller.llm_call",
        "sonality.agent.llm_call",
        "sonality.memory.derivatives.llm_call",
        "sonality.memory.knowledge_extract.llm_call",
    )
    for target in targets:
        monkeypatch.setattr(target, fake_call, raising=False)
    return configure
