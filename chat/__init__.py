"""Chat module — terminal TUI and Telegram bot interfaces for Sonality.

Public API: SonalityClient (streaming HTTP), AudioProcessor (STT/TTS),
and audio utilities (chunk_text, optimize_for_speech, mp3_to_ogg_opus).
Run via ``python -m chat terminal`` or ``python -m chat telegram``.
"""

from __future__ import annotations

from .audio import AudioProcessor, chunk_text, llm_call, mp3_to_ogg_opus, optimize_for_speech
from .client import Belief, HealthStatus, SonalityClient

__all__ = [
    "AudioProcessor",
    "Belief",
    "HealthStatus",
    "SonalityClient",
    "chunk_text",
    "llm_call",
    "mp3_to_ogg_opus",
    "optimize_for_speech",
]
