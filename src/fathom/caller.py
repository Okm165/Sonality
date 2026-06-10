"""Fathom-specific structured LLM and embedding call wrappers.

Wraps ``shared.llm.caller`` and ``shared.embedder`` with fathom config
defaults so call sites stay clean.  Symmetric with ``sonality.caller``.

  ``async_llm_call``        — gated by ``_llm_gate`` (FATHOM_LLM_CONCURRENCY).
  ``async_embed_query``     — gated by ``_embedding_gate``
  ``async_embed_documents``   (FATHOM_EMBEDDING_CONCURRENCY).
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from shared.embedder import make_gated_embedders
from shared.llm.caller import LLMCallResult
from shared.llm.caller import async_llm_call as _shared_async_llm_call
from shared.llm.caller import format_prompt as _shared_format_prompt
from shared.llm.provider import LLMProvider

from .config import settings

# --- Provider (symmetric with sonality.caller.provider) ---

provider: LLMProvider = settings.make_llm_provider(timeout=settings.llm_request_timeout)

# --- Concurrency gates (InfraSettings already clamps ≥ 1) ---

_llm_gate = asyncio.Semaphore(settings.llm_concurrency)
_embedding_gate = asyncio.Semaphore(settings.embedding_concurrency)


def format_prompt(template: str, **kwargs: object) -> str:
    """Format template with per-value compression guard using fathom defaults."""
    return _shared_format_prompt(provider, template, model=settings.model, **kwargs)  # type: ignore[arg-type]


async def async_llm_call[T: BaseModel](
    *,
    instructions: str,
    response_model: type[T],
    fallback: T,
    model: str = settings.model,
    max_tokens: int = settings.llm_max_tokens,
    max_retries: int = settings.llm_max_retries,
) -> LLMCallResult[T]:
    """Async LLM call — gated by ``_llm_gate`` (FATHOM_LLM_CONCURRENCY)."""
    async with _llm_gate:
        return await _shared_async_llm_call(
            provider,
            instructions=instructions,
            response_model=response_model,
            fallback=fallback,
            model=model,
            max_tokens=max_tokens,
            repair_model=model,
            backoff_base=settings.llm_backoff_base,
            max_retries=max_retries,
        )


async_embed_query, async_embed_documents = make_gated_embedders(_embedding_gate)
