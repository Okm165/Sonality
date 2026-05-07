"""Structured LLM call wrapper with retry and JSON repair.

All functions take an explicit ``LLMProvider`` — no module-level globals or
config coupling.  Each entry-point package (sonality, fathom, chat) wraps
these with its own defaults.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import structlog
from pydantic import BaseModel, ValidationError

from ..errors import (
    LLMParseError,
    ProviderHTTPError,
    ProviderTransportError,
)
from ..types import ChatRole
from .parse import decode_llm_json
from .provider import LLMProvider

log = structlog.get_logger()

JSON_SYSTEM_PROMPT: Final = (
    "Respond with valid JSON only — no explanation, no markdown fences, no preamble. "
    "Fill in actual values, not schema annotations or placeholders."
)
_JSON_REPAIR_PROMPT: Final = (
    "Fix this malformed JSON to match the schema. Return only the corrected JSON.\n\n"
    "Schema: {schema}\n\nBroken JSON:\n{broken}\n\n"
    "Error: {error}"
)


class LLMErrorCategory(StrEnum):
    """Discriminator for LLMCallResult failure mode — avoids string-matching on error text."""

    NONE = "none"
    PARSE = "parse"
    VALIDATION = "validation"
    TRANSPORT = "transport"
    HTTP = "http"
    UNEXPECTED = "unexpected"


@dataclass(frozen=True, slots=True)
class LLMCallResult[T: BaseModel]:
    """Result of a structured LLM call."""

    value: T
    success: bool
    error: str = ""
    error_category: LLMErrorCategory = LLMErrorCategory.NONE
    attempts: int = 0
    repaired: bool = False
    raw_text: str = ""


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def raw_call(
    provider: LLMProvider,
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    system: str = "",
    temperature: float = -1.0,
) -> str:
    """Execute a single provider chat completion and return plain text."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": ChatRole.SYSTEM, "content": system})
    messages.append({"role": ChatRole.USER, "content": prompt})
    log.debug("raw_call", model=model, prompt_chars=len(prompt), max_tokens=max_tokens)
    completion = provider.chat_completion(
        model=model, messages=tuple(messages), max_tokens=max_tokens, temperature=temperature
    )
    log.debug("raw_call_done", chars=len(completion.text), in_tokens=completion.input_tokens, out_tokens=completion.output_tokens)
    return completion.text


def _attempt_repair[T: BaseModel](
    provider: LLMProvider,
    raw_text: str,
    response_model: type[T],
    repair_model: str,
    max_tokens: int,
    error: Exception,
    *,
    max_repair_tries: int = 2,
) -> LLMCallResult[T] | None:
    """Repair malformed JSON via an LLM call. Returns None when all tries fail."""
    schema_name = response_model.__name__
    current_error = error
    current_text = raw_text

    for attempt in range(1, max_repair_tries + 1):
        log.info(
            "json_repair_attempt",
            attempt=attempt,
            max_attempts=max_repair_tries,
            schema=schema_name,
            repair_model=repair_model,
            error=str(current_error)[:120],
        )
        try:
            repair_prompt = _JSON_REPAIR_PROMPT.format(
                schema=response_model.model_json_schema(),
                broken=current_text[:2000],
                error=str(current_error)[:500],
            )
            repaired_text = raw_call(
                provider, prompt=repair_prompt, model=repair_model, max_tokens=max_tokens
            )
            repaired_data = decode_llm_json(repaired_text)
            value = response_model.model_validate(repaired_data)
            log.info(
                "json_repair_ok",
                schema=schema_name,
                attempt=attempt,
                max_attempts=max_repair_tries,
            )
            return LLMCallResult(
                value=value,
                success=True,
                attempts=attempt,
                repaired=True,
                raw_text=repaired_text,
            )
        except Exception as exc:
            log.warning(
                "json_repair_attempt_failed",
                attempt=attempt,
                max_attempts=max_repair_tries,
                schema=schema_name,
                exc=str(exc),
            )
            current_error = exc

    log.error("json_repair_exhausted", tries=max_repair_tries, schema=schema_name)
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def llm_call[T: BaseModel](
    provider: LLMProvider,
    *,
    prompt: str,
    response_model: type[T],
    fallback: T,
    model: str,
    max_tokens: int,
    repair_model: str | None = None,
    backoff_base: float = 2.0,
    system: str = JSON_SYSTEM_PROMPT,
    max_retries: int = 3,
) -> LLMCallResult[T]:
    """Execute a structured LLM call with retry, repair, and fallback.

    Retries on JSON parse errors and validation failures (with LLM-based
    repair).  Network/transport errors already retried by provider are treated
    as terminal.
    """
    effective_repair_model = repair_model or model
    schema_name = response_model.__name__
    log.debug("llm_call_start", schema=schema_name, model=model)
    last_error = ""
    last_category = LLMErrorCategory.NONE
    raw_text = ""
    attempts_made = 0

    for attempt in range(1, max_retries + 1):
        attempts_made = attempt
        try:
            raw_text = raw_call(
                provider, prompt=prompt, model=model, max_tokens=max_tokens, system=system
            )
            data = decode_llm_json(raw_text)
            value = response_model.model_validate(data)
            return LLMCallResult(value=value, success=True, attempts=attempt, raw_text=raw_text)

        except ValidationError as exc:
            last_error = f"Schema validation: {exc}"
            last_category = LLMErrorCategory.VALIDATION
            log.warning(
                "llm_call_validation_error",
                attempt=attempt,
                max_retries=max_retries,
                schema=schema_name,
                exc=str(exc),
            )
            repaired = _attempt_repair(
                provider, raw_text, response_model, effective_repair_model, max_tokens, exc
            )
            if repaired:
                return repaired

        except LLMParseError as exc:
            last_error = f"JSON parse: {exc}"
            last_category = LLMErrorCategory.PARSE
            log.warning(
                "llm_call_json_error",
                attempt=attempt,
                max_retries=max_retries,
                schema=schema_name,
                error=last_error,
            )
            repaired = _attempt_repair(
                provider, raw_text, response_model, effective_repair_model, max_tokens, exc
            )
            if repaired:
                return repaired

        except ProviderHTTPError as exc:
            last_error = f"HTTP {exc.status}: {exc.detail}"
            last_category = LLMErrorCategory.HTTP
            is_terminal = exc.status in (400, 401, 403, 404, 422)
            if is_terminal:
                log.warning(
                    "llm_call_terminal_error",
                    attempt=attempt,
                    max_retries=max_retries,
                    schema=schema_name,
                    http_status=exc.status,
                )
                break
            if attempt < max_retries:
                base_wait = backoff_base**attempt
                jitter = random.uniform(0, base_wait * 0.5)
                wait = base_wait + jitter
                log.warning(
                    "llm_call_retrying",
                    attempt=attempt,
                    max_retries=max_retries,
                    schema=schema_name,
                    exc=str(exc),
                    retry_in=f"{wait:.1f}s",
                )
                time.sleep(wait)
                continue

        except ProviderTransportError as exc:
            last_error = f"Transport: {exc}"
            last_category = LLMErrorCategory.TRANSPORT
            log.warning(
                "llm_call_terminal_error",
                attempt=attempt,
                max_retries=max_retries,
                schema=schema_name,
                exc=str(exc),
            )
            break

        except Exception as exc:
            last_error = f"Unexpected: {exc}"
            last_category = LLMErrorCategory.UNEXPECTED
            log.error(
                "llm_call_unexpected_error",
                attempt=attempt,
                max_retries=max_retries,
                schema=schema_name,
                exc_info=True,
            )
            break

    log.error(
        "llm_call_failed",
        attempts=attempts_made,
        max_retries=max_retries,
        schema=schema_name,
        error=last_error,
    )
    return LLMCallResult(
        value=fallback,
        success=False,
        error=last_error,
        error_category=last_category,
        attempts=attempts_made,
        raw_text=raw_text,
    )
