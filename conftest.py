"""Root conftest: configures debug-level file logging for all test sessions.

Every test run writes a timestamped log file to data/ with full DEBUG output
from the sonality package — ESS decisions, belief updates, knowledge extraction,
memory retrieval, reflection cycles, and health diagnostics.
"""
from __future__ import annotations

import logging
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pytest

_LOG_DIR = Path(__file__).parent / "data"


def _embedding_service_available() -> bool:
    """Ping the configured embedding endpoint; return True if reachable."""
    from sonality import config as cfg
    url = cfg.EMBEDDING_BASE_URL.rstrip("/").replace("/v1", "") + "/api/tags"
    try:
        urllib.request.urlopen(url, timeout=3)  # noqa: S310
        return True
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Attach a rotating file handler to the root 'sonality' logger."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    worker = getattr(config, "workerinput", {}).get("workerid", "")
    suffix = f"_{worker}" if worker else ""
    log_file = _LOG_DIR / f"test_run_{ts}{suffix}.log"

    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)s:%(lineno)d  %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    for logger_name in ("sonality", "benches", "tests"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(handler)

    config._sonality_log_file = log_file  # type: ignore[attr-defined]
    config._sonality_log_handler = handler  # type: ignore[attr-defined]


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip live tests that require embeddings if the embedding service is down."""
    if not any(item.get_closest_marker("live") for item in items):
        return
    if _embedding_service_available():
        return
    skip = pytest.mark.skip(reason="Embedding service unavailable — start Ollama first")
    for item in items:
        if item.get_closest_marker("live"):
            item.add_marker(skip)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Flush and close the file handler at session end."""
    handler = getattr(config, "_sonality_log_handler", None)
    if handler:
        handler.flush()
        handler.close()
        for logger_name in ("sonality", "benches", "tests"):
            logging.getLogger(logger_name).removeHandler(handler)
