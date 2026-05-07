"""Sonality-specific structured LLM call wrappers.

Wraps ``shared.llm.caller.llm_call`` with sonality config defaults so that
existing call sites remain unchanged.  Also includes consensus_call
(multi-model merge).
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

import structlog
from pydantic import BaseModel

from shared.llm.caller import JSON_SYSTEM_PROMPT
from shared.llm.caller import LLMCallResult as _SharedLLMCallResult
from shared.llm.caller import llm_call as _shared_llm_call

from . import config
from .provider import default_provider

log = structlog.get_logger()

LLMCallResult = _SharedLLMCallResult


def llm_call[T: BaseModel](
    *,
    prompt: str,
    response_model: type[T],
    fallback: T,
    model: str = config.settings.agent_model,
    max_tokens: int = config.settings.llm_max_tokens,
    system: str = JSON_SYSTEM_PROMPT,
    max_retries: int = config.settings.llm_max_retries,
) -> LLMCallResult[T]:
    """Execute a structured LLM call with retry, repair, and fallback.

    Thin wrapper that supplies sonality's default provider and config values
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
        backoff_base=config.settings.llm_backoff_base,
        system=system,
        max_retries=max_retries,
    )


def consensus_call[T: BaseModel](
    *,
    prompt: str,
    response_model: type[T],
    fallback: T,
    models: tuple[str, str],
    merge: Callable[[T, T], T] | None = None,
    max_tokens: int = config.settings.llm_max_tokens,
    system: str = JSON_SYSTEM_PROMPT,
) -> LLMCallResult[T]:
    """Call two models in parallel and merge results for higher confidence."""
    model_a, model_b = models
    if model_a == model_b:
        return llm_call(
            prompt=prompt,
            response_model=response_model,
            fallback=fallback,
            model=model_a,
            max_tokens=max_tokens,
            system=system,
        )

    log.info(
        "consensus_call_started",
        model_a=model_a,
        model_b=model_b,
        response_model=response_model.__name__,
    )
    results: dict[str, LLMCallResult[T]] = {}
    pool = ThreadPoolExecutor(max_workers=2)
    futures = {
        pool.submit(
            llm_call,
            prompt=prompt,
            response_model=response_model,
            fallback=fallback,
            model=m,
            max_tokens=max_tokens,
            system=system,
        ): m
        for m in (model_a, model_b)
    }
    for future in as_completed(futures):
        model_name = futures[future]
        try:
            results[model_name] = future.result()
        except Exception as exc:
            log.warning(
                "consensus_call_model_failed",
                model_name=model_name,
                error=str(exc),
            )
    pool.shutdown(wait=True)

    result_a = results.get(model_a)
    result_b = results.get(model_b)

    if result_a and result_a.success and result_b and result_b.success:
        if merge is not None:
            merged = merge(result_a.value, result_b.value)
            log.info(
                "consensus_call_merged",
                model_a=model_a,
                model_b=model_b,
            )
            return LLMCallResult(value=merged, success=True, attempts=2, raw_text=result_a.raw_text)
        return result_a

    if result_a and result_a.success:
        return result_a
    if result_b and result_b.success:
        return result_b

    log.warning("consensus_call: both models failed")
    return LLMCallResult(value=fallback, success=False, error="Both models failed", attempts=2)


