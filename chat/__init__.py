"""Chat module for Sonality with terminal and Telegram interfaces."""

from __future__ import annotations

from .audio import AudioProcessor, chunk_text, mp3_to_ogg_opus, optimize_for_speech
from .client import Belief, ChatResponse, HealthStatus, SonalityClient
from .llm import llm_call

__all__ = [
    "AudioProcessor",
    "Belief",
    "ChatResponse",
    "HealthStatus",
    "SonalityClient",
    "chunk_text",
    "llm_call",
    "mp3_to_ogg_opus",
    "optimize_for_speech",
]
