"""Fathom LLM provider instance.

Mirrors sonality/provider.py — creates the default provider from fathom config.
The heavy lifting (HTTP transport, retries) lives in ``shared.llm.provider``.
"""

from __future__ import annotations

from shared.llm.provider import ChatResult, LLMProvider

from .config import settings

__all__ = [
    "ChatResult",
    "LLMProvider",
    "default_provider",
]

default_provider: LLMProvider = LLMProvider(
    settings.base_url,
    settings.api_key,
    settings.llm_request_timeout,
    max_retries=settings.llm_max_retries,
    backoff_base=settings.llm_backoff_base,
    concurrency=settings.llm_concurrency,
)
