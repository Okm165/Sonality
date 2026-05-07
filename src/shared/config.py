"""Shared configuration — project root and noisy-logger suppression.

All three modules (sonality, fathom, chat) use pydantic-settings for env vars.
This module only provides structural helpers that pydantic-settings doesn't cover.
"""

from __future__ import annotations

import logging
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


def quiet_third_party_loggers() -> None:
    """Suppress noisy library loggers (httpx, neo4j, etc.)."""
    for lib in ("httpcore", "httpx", "neo4j", "neo4j.io", "neo4j.pool"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)
    logging.getLogger("trafilatura").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("ddgs").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
