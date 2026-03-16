"""Sonality API server entry point.

Run with: sonality-server
Or: uvicorn sonality.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import logging

import uvicorn


def main() -> None:
    """Start the Sonality API server."""
    parser = argparse.ArgumentParser(description="Sonality API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    uvicorn.run(
        "sonality.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
