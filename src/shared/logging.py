"""Unified logging — human-readable console by default, JSON only if LOG_FORMAT=json.

Uses structlog with ProcessorFormatter to unify both structlog and stdlib
logging (uvicorn, third-party) through the same rendering pipeline.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any

import structlog
from structlog.stdlib import ProcessorFormatter

from .config import quiet_third_party_loggers

_LOG_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
_configured = False

_DIM = "\033[2m"
_BOLD = "\033[1m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"

_LEVEL_STYLE = {
    "debug": f"{_DIM}debug{_RESET}",
    "info": "info ",
    "warning": f"{_YELLOW}warn {_RESET}",
    "error": f"{_RED}{_BOLD}ERROR{_RESET}",
    "critical": f"{_RED}{_BOLD}CRIT {_RESET}",
}

_IMPORTANT_FIRST = ("elapsed_s", "status", "error", "tools", "facts", "pages")
_META_KEYS = frozenset({"event", "timestamp", "level", "logger", "_record", "_from_structlog"})
_MAX_VALUE_LEN = 120


def _minimal_renderer(_logger: Any, _method: str, event_dict: structlog.types.EventDict) -> str:
    """Render a log line: dim timestamp, colored level, bold event, plain kv pairs."""
    ts = event_dict.pop("timestamp", "")
    level = event_dict.pop("level", "info")
    event = event_dict.pop("event", "")
    logger_name = event_dict.pop("logger", "")
    event_dict.pop("_record", None)
    event_dict.pop("_from_structlog", None)

    level_str = _LEVEL_STYLE.get(level, level)

    # Reorder: important fields first
    ordered: list[tuple[str, Any]] = []
    for key in _IMPORTANT_FIRST:
        if key in event_dict:
            ordered.append((key, event_dict.pop(key)))
    ordered.extend(event_dict.items())

    # Format kv pairs — truncate long strings
    parts: list[str] = []
    for k, v in ordered:
        if k in _META_KEYS:
            continue
        s = str(v)
        if len(s) > _MAX_VALUE_LEN:
            s = s[:_MAX_VALUE_LEN] + "…"
        parts.append(f"{k}={s}")
    kv = " ".join(parts)

    line = f"{_DIM}{ts}{_RESET} {level_str} {_BOLD}{event:<30}{_RESET} {_DIM}{logger_name}{_RESET}"
    if kv:
        line += f"  {kv}"
    return line


def setup_logging(level: str = "DEBUG") -> None:
    """Configure logging once at startup.

    Bridges stdlib logging into structlog so uvicorn, httpx, neo4j, etc.
    all render through the same pipeline.
    """
    global _configured
    if _configured:
        return
    _configured = True

    quiet_third_party_loggers()
    log_level = _LOG_LEVELS.get(level.upper(), 20)
    use_json = os.environ.get("LOG_FORMAT", "").lower() == "json"

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if use_json:
        shared_processors.append(structlog.processors.TimeStamper(fmt="iso"))
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        shared_processors.append(structlog.processors.TimeStamper(fmt="%H:%M:%S"))
        renderer = _minimal_renderer

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        ProcessorFormatter(
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
            foreign_pre_chain=shared_processors,
        )
    )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
