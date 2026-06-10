"""Chat module — terminal TUI and Telegram bot interfaces for Sonality.

Run via ``python -m chat terminal`` or ``python -m chat telegram``.
"""

from __future__ import annotations

from shared.logging import setup_logging

from .config import settings

setup_logging(settings.log_level)
