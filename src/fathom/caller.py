"""Fathom-specific structured LLM call wrappers.

Mirrors sonality/caller.py — wraps ``shared.llm.caller.llm_call`` with
fathom config defaults.
"""

from __future__ import annotations

from pydantic import BaseModel

from shared.llm.caller import JSON_SYSTEM_PROMPT, LLMCallResult
from shared.llm.caller import llm_call as _shared_llm_call

from .config import settings
from .provider import default_provider

__all__ = [
    "LLMCallResult",
    "llm_call",
]


def llm_call[T: BaseModel](
    *,
    prompt: str,
    response_model: type[T],
    fallback: T,
    model: str = settings.model,
    max_tokens: int = settings.llm_max_tokens,
    system: str = JSON_SYSTEM_PROMPT,
    max_retries: int = settings.llm_max_retries,
) -> LLMCallResult[T]:
    """Execute a structured LLM call with retry, repair, and fallback.

    Thin wrapper that supplies fathom's default provider and config values
    to the shared implementation.
    """
    return _shared_llm_call(
        default_provider,
        prompt=prompt,
        response_model=response_model,
        fallback=fallback,
        model=model,
        max_tokens=max_tokens,
        repair_model=model,
        backoff_base=settings.llm_backoff_base,
        system=system,
        max_retries=max_retries,
    )
