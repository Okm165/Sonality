"""Sonality — LLM agent with a self-evolving personality."""

from __future__ import annotations

import os

from shared.logging import setup_logging

__version__ = "0.1.0"

__all__: list[str] = []

setup_logging(os.environ.get("SONALITY_LOG_LEVEL", "INFO"))
