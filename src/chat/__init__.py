"""Chat module — terminal TUI and Telegram bot interfaces for Sonality.

Public API: SonalityClient (streaming HTTP), AudioProcessor (STT/TTS),
and audio utilities (chunk_text, optimize_for_speech, mp3_to_ogg_opus).
Run via ``python -m chat terminal`` or ``python -m chat telegram``.
"""

from __future__ import annotations

import os

from shared.logging import setup_logging

from .audio import AudioProcessor, chunk_text, mp3_to_ogg_opus, optimize_for_speech
from .client import Belief, HealthStatus, SonalityClient

__version__ = "0.1.0"

__all__ = [
    "AudioProcessor",
    "Belief",
    "HealthStatus",
    "SonalityClient",
    "chunk_text",
    "mp3_to_ogg_opus",
    "optimize_for_speech",
]

setup_logging(os.environ.get("CHAT_LOG_LEVEL", "INFO"))
