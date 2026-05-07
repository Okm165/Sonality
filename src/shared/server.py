"""Shared FastAPI server bootstrap — used by sonality and fathom."""

from __future__ import annotations

import argparse
import logging
import os

from .logging import setup_logging

_NOISY_PATHS = frozenset({"/health", "/readyz"})


class _HealthCheckFilter(logging.Filter):
    """Suppress uvicorn access logs for health check endpoints."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(p in msg for p in _NOISY_PATHS)


def make_server_parser(
    description: str,
    *,
    default_port: int = 8000,
) -> argparse.ArgumentParser:
    """Build a standard CLI parser for uvicorn-backed services."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=default_port, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default=None, help="Log level")
    return parser


def run_uvicorn(app_path: str, args: argparse.Namespace, *, env_log_key: str = "") -> None:
    """Start uvicorn with parsed CLI args."""
    import uvicorn

    log_level = (
        args.log_level or os.environ.get(env_log_key, "info")
        if env_log_key
        else (args.log_level or "info")
    )
    setup_logging(log_level)
    logging.getLogger("uvicorn.access").addFilter(_HealthCheckFilter())
    uvicorn.run(
        app_path,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level.lower(),
    )
