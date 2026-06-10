"""Shared FastAPI server bootstrap — used by sonality and fathom."""

from __future__ import annotations

import argparse
import logging
import os
from urllib.request import Request as _UrlRequest
from urllib.request import urlopen as _urlopen

from pydantic import BaseModel, Field

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


def http_dependency_status(
    url: str,
    *,
    timeout: float = 4.0,
    user_agent: str = "health/1.0",
) -> str:
    """Lightweight HTTP probe for health checks — shared by all services."""
    try:
        req = _UrlRequest(url, headers={"User-Agent": user_agent})
        with _urlopen(req, timeout=timeout) as resp:
            status = int(resp.status)
        return "ok" if 200 <= status < 400 else f"error: HTTP {status}"
    except Exception as exc:
        return f"error: {exc}"


class HealthResponse(BaseModel):
    """Shared health response shape — services extend with extra fields."""

    status: str = "ok"
    version: str = ""
    dependencies: dict[str, str] = Field(default_factory=dict)


def check_health_dependencies(dependencies: dict[str, str]) -> None:
    """Raise 503 if any dependency reports an error."""
    if any(not v.startswith("ok") for v in dependencies.values()):
        from fastapi import HTTPException

        raise HTTPException(status_code=503, detail={"dependencies": dependencies})


class RequestIDMiddleware:
    """ASGI middleware: capture or generate trace_id and bind to structlog.

    Checks ``X-Trace-ID``, then ``X-Request-ID``; falls back to a short
    random UUID fragment.  Sets ``X-Trace-ID`` on the response.

    Compatible with ``app.add_middleware(RequestIDMiddleware)``.
    """

    def __init__(self, app: object) -> None:
        import uuid

        import structlog
        from starlette.middleware.base import BaseHTTPMiddleware

        mw_app = app  # capture for closure

        class _Inner(BaseHTTPMiddleware):
            async def dispatch(inner_self, request, call_next):  # type: ignore[override] # noqa: N805
                trace_id = (
                    request.headers.get("X-Trace-ID")
                    or request.headers.get("X-Request-ID")
                    or str(uuid.uuid4())[:12]
                )
                structlog.contextvars.clear_contextvars()
                structlog.contextvars.bind_contextvars(trace_id=trace_id)
                response = await call_next(request)
                response.headers["X-Trace-ID"] = trace_id
                return response

        self._inner = _Inner(mw_app)  # type: ignore[arg-type]

    async def __call__(self, scope: object, receive: object, send: object) -> None:
        await self._inner(scope, receive, send)  # type: ignore[arg-type]


def run_uvicorn(app_path: str, args: argparse.Namespace, *, env_log_key: str = "") -> None:
    """Start uvicorn with parsed CLI args."""
    import uvicorn

    log_level = (
        args.log_level or os.environ.get(env_log_key, "debug")
        if env_log_key
        else (args.log_level or "debug")
    )
    setup_logging(log_level)
    logging.getLogger("uvicorn.access").addFilter(_HealthCheckFilter())
    uvicorn.run(
        app_path,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=log_level.lower(),
        log_config=None,
        access_log=False,
    )
