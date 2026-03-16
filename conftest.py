"""Root conftest: configures debug-level file logging for all test sessions.

Every test run writes a timestamped log file to data/ with full DEBUG output
from the sonality package — ESS decisions, belief updates, knowledge extraction,
memory retrieval, reflection cycles, and health diagnostics.

A console handler at INFO level is also attached so key events stream to the
terminal when running with -s (no-capture mode).
"""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

_LOG_DIR = Path(__file__).parent / "data"

# Loggers that should emit INFO-level messages to the console during test runs.
_CONSOLE_LOGGERS = ("sonality.agent", "sonality.memory.knowledge_extract", "benches")


def _embedding_service_available() -> bool:
    """Check if embedding is available. Always True with local FastEmbed."""
    return True


def pytest_configure(config: pytest.Config) -> None:
    """Attach file and console log handlers to sonality loggers."""
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    worker = getattr(config, "workerinput", {}).get("workerid", "")
    suffix = f"_{worker}" if worker else ""
    log_file = _LOG_DIR / f"test_run_{ts}{suffix}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-5s %(name)s:%(lineno)d  %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(
        logging.Formatter("%(levelname)-5s  %(name)s  %(message)s")
    )

    for logger_name in ("sonality", "benches", "tests"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    for logger_name in _CONSOLE_LOGGERS:
        logging.getLogger(logger_name).addHandler(console_handler)

    config._sonality_log_file = log_file  # type: ignore[attr-defined]
    config._sonality_log_handler = file_handler  # type: ignore[attr-defined]
    config._sonality_console_handler = console_handler  # type: ignore[attr-defined]


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
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
    """Flush and close log handlers at session end."""
    file_handler = getattr(config, "_sonality_log_handler", None)
    if file_handler:
        file_handler.flush()
        file_handler.close()
        for logger_name in ("sonality", "benches", "tests"):
            logging.getLogger(logger_name).removeHandler(file_handler)

    console_handler = getattr(config, "_sonality_console_handler", None)
    if console_handler:
        console_handler.flush()
        for logger_name in _CONSOLE_LOGGERS:
            logging.getLogger(logger_name).removeHandler(console_handler)
