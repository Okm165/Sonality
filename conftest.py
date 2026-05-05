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

# Module-level storage for log handlers (avoids dynamic attrs on pytest.Config)
_log_state: dict[str, logging.Handler | Path | None] = {}


def pytest_configure(config: pytest.Config) -> None:
    """Attach file and console log handlers to sonality loggers."""
    del config  # unused - we use module-level storage
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    log_name = f"test_run_{ts}.log"

    log_file: Path | None = None
    for log_dir in (_LOG_DIR, Path("/tmp/sonality_test_logs")):
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            candidate = log_dir / log_name
            candidate.touch()
            log_file = candidate
            break
        except (PermissionError, OSError):
            continue

    if log_file is None:
        return

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
    console_handler.setFormatter(logging.Formatter("%(levelname)-5s  %(name)s  %(message)s"))

    for logger_name in ("sonality", "benches", "tests"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)

    for logger_name in _CONSOLE_LOGGERS:
        logging.getLogger(logger_name).addHandler(console_handler)

    _log_state["file_handler"] = file_handler
    _log_state["console_handler"] = console_handler




@pytest.fixture
def suppress_expected_logs():
    """Temporarily suppress ERROR-level logs during tests that expect failures.

    Use this fixture in tests that intentionally trigger exceptions to prevent
    alarming tracebacks from appearing in test output.
    """
    loggers = [logging.getLogger(name) for name in ("sonality", "benches", "tests")]
    original_levels = [logger.level for logger in loggers]
    for logger in loggers:
        logger.setLevel(logging.CRITICAL)
    yield
    for logger, level in zip(loggers, original_levels, strict=True):
        logger.setLevel(level)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Flush and close log handlers at session end."""
    del config  # unused
    file_handler = _log_state.get("file_handler")
    if isinstance(file_handler, logging.FileHandler):
        file_handler.flush()
        file_handler.close()
        for logger_name in ("sonality", "benches", "tests"):
            logging.getLogger(logger_name).removeHandler(file_handler)

    console_handler = _log_state.get("console_handler")
    if isinstance(console_handler, logging.Handler):
        console_handler.flush()
        for logger_name in _CONSOLE_LOGGERS:
            logging.getLogger(logger_name).removeHandler(console_handler)
