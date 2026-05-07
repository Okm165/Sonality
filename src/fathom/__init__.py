"""Fathom — autonomous web research agent."""

from __future__ import annotations

import os

from shared.logging import setup_logging

__version__ = "0.1.0"

__all__: list[str] = []

setup_logging(os.environ.get("FATHOM_LOG_LEVEL", "INFO"))
