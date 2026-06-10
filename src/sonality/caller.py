"""Sonality-specific LLM provider, structured call wrappers, and embedding gates.

Owns the ``provider`` instance and concurrency semaphores.  Symmetric with
``fathom.caller``.

  ``llm_call``              — synchronous (request threads, agentic loop).
  ``async_llm_call``        — gated by ``_llm_gate`` (SONALITY_LLM_CONCURRENCY).
  ``async_embed_query``     — gated by ``_embedding_gate``
  ``async_embed_documents``   (SONALITY_EMBEDDING_CONCURRENCY).
"""

from __future__ import annotations

import asyncio

import structlog
from pydantic import BaseModel

from shared.embedder import make_gated_embedders
from shared.llm.caller import LLMCallResult
from shared.llm.caller import async_llm_call as _shared_async_llm_call
from shared.llm.caller import format_prompt as _shared_format_prompt
from shared.llm.caller import llm_call as _shared_llm_call
from shared.llm.provider import ChatResult as ChatResult
from shared.llm.provider import LLMProvider
from shared.llm.provider import StreamChunk as StreamChunk

from . import config

log = structlog.get_logger(__name__)

# --- Provider (symmetric with fathom.caller.provider) ---

provider: LLMProvider = config.settings.make_llm_provider(timeout=config.settings.llm_timeout)

# --- Concurrency gates (InfraSettings already clamps ≥ 1) ---

_llm_gate = asyncio.Semaphore(config.settings.llm_concurrency)
_embedding_gate = asyncio.Semaphore(config.settings.embedding_concurrency)


def format_prompt(template: str, **kwargs: object) -> str:
    """Format template with per-value compression guard using sonality defaults."""
    return _shared_format_prompt(
        provider,
        template,
        model=config.settings.agent_model,
        **kwargs,  # type: ignore[arg-type]
    )


def llm_call[T: BaseModel](
    *,
    instructions: str,
    response_model: type[T],
    fallback: T,
    model: str = config.settings.agent_model,
    max_tokens: int = config.settings.llm_max_tokens,
    max_retries: int = config.settings.llm_max_retries,
) -> LLMCallResult[T]:
    """Synchronous structured LLM call with retry, repair, and fallback."""
    schema = response_model.__name__
    log.debug(
        "llm_call",
        schema=schema,
        model=model,
        instructions_chars=len(instructions),
        max_tokens=max_tokens,
    )
    result = _shared_llm_call(
        provider,
        instructions=instructions,
        response_model=response_model,
        fallback=fallback,
        model=model,
        max_tokens=max_tokens,
        repair_model=model,
        backoff_base=config.settings.llm_backoff_base,
        max_retries=max_retries,
    )
    log.debug(
        "llm_call_done",
        schema=schema,
        ok=result.success,
        elapsed_s=result.elapsed_s,
        in_tok=result.input_tokens,
        out_tok=result.output_tokens,
        attempts=result.attempts,
    )
    return result


async def async_llm_call[T: BaseModel](
    *,
    instructions: str,
    response_model: type[T],
    fallback: T,
    model: str = config.settings.agent_model,
    max_tokens: int = config.settings.llm_max_tokens,
    max_retries: int = config.settings.llm_max_retries,
) -> LLMCallResult[T]:
    """Async LLM call — gated by ``_llm_gate`` (SONALITY_LLM_CONCURRENCY)."""
    async with _llm_gate:
        return await _shared_async_llm_call(
            provider,
            instructions=instructions,
            response_model=response_model,
            fallback=fallback,
            model=model,
            max_tokens=max_tokens,
            repair_model=model,
            backoff_base=config.settings.llm_backoff_base,
            max_retries=max_retries,
        )


# --- Embedding wrappers (from shared factory, symmetric with fathom.caller) ---

async_embed_query, async_embed_documents = make_gated_embedders(_embedding_gate)
