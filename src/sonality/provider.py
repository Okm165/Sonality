"""Sonality LLM provider instance and interaction state.

Thin wrapper that creates the default provider from sonality config and
manages the interaction state (semantic worker defers during user turns).
The heavy lifting (HTTP transport, retries, response parsing) lives in
``shared.llm.provider``.
"""

from __future__ import annotations

import contextlib
import threading
import time as _time
from collections.abc import Iterator

from shared.llm.provider import ChatResult, LLMProvider, StreamChunk

from . import config

__all__ = [
    "ChatResult",
    "LLMProvider",
    "StreamChunk",
    "default_provider",
    "interaction_active",
    "interaction_in_progress",
    "llm_semaphore_idle",
]

# --- Default provider instance (sonality endpoint) ---
default_provider: LLMProvider = LLMProvider(
    config.settings.base_url,
    config.settings.api_key,
    config.settings.llm_timeout,
    max_retries=config.settings.llm_max_retries,
    backoff_base=config.settings.llm_backoff_base,
    concurrency=config.settings.llm_concurrency,
)

# --- Interaction state (global; used by semantic worker to defer during user turns) ---
_interaction_event = threading.Event()
_last_interaction_end: float = 0.0
_COOLDOWN_SECONDS: float = 180.0

interaction_in_progress = _interaction_event.is_set


@contextlib.contextmanager
def interaction_active() -> Iterator[None]:
    global _last_interaction_end
    _interaction_event.set()
    try:
        yield
    finally:
        _interaction_event.clear()
        _last_interaction_end = _time.monotonic()


def llm_semaphore_idle() -> bool:
    """True when the LLM is safe for low-priority background work.

    Requires: no active interaction, semaphore free, AND a cooldown period
    since the last interaction ended (gives Fathom time to use the LLM).
    """
    if interaction_in_progress():
        return False
    if not default_provider.semaphore_idle():
        return False
    return not (
        _last_interaction_end
        and (_time.monotonic() - _last_interaction_end) < _COOLDOWN_SECONDS
    )
