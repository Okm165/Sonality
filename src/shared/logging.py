"""Unified structlog configuration for all modules.

Both sonality and fathom use this single setup_logging() call.
Console rendering when TTY, JSON when piped (Docker/production).
stdlib logging is bridged into structlog so all output is consistent.
"""

from __future__ import annotations

import logging
import sys

import structlog

from .config import quiet_third_party_loggers

_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}

_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure structlog globally. Call once at startup.

    Bridges stdlib logging into structlog so modules using either
    produce identical output format.
    """
    global _configured
    if _configured:
        return
    _configured = True

    quiet_third_party_loggers()
    log_level = _LOG_LEVELS.get(level.upper(), 20)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.UnicodeDecoder(),
    ]
    renderer: structlog.types.Processor = (
        structlog.dev.ConsoleRenderer()
        if sys.stderr.isatty()
        else structlog.processors.JSONRenderer()
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Bridge stdlib logging → structlog for consistent output
    root = logging.getLogger()
    root.setLevel(log_level)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
