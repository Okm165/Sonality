"""Structured LLM call wrapper with retry and JSON repair.

All functions take an explicit ``LLMProvider`` — no module-level globals or
config coupling.  Each entry-point package (sonality, fathom, chat) wraps
these with its own defaults.

Two call variants:
  ``llm_call``        — synchronous, for use from request threads / sync code.
  ``async_llm_call``  — async bridge, runs blocking I/O in the default executor.
                        Concurrency gating is the caller's responsibility —
                        sonality and fathom each own their own ``asyncio.Semaphore``.

Context-overflow architecture:
- ``format_prompt`` — the single entry point for formatting prompt templates.
  Escapes braces in values and compresses individual values proportionally
  when total content would overflow.  Template text is preserved verbatim.
- ``compose_guarded`` — message-level compression for multi-turn chat windows
  (agent loop tool results, TTS optimization) where individual messages are
  compressed proportionally to fit the context budget.
"""

from __future__ import annotations

import asyncio
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

log = structlog.get_logger(__name__)

JSON_SYSTEM_PROMPT: Final = (
    "Output a single JSON object. No markdown, no preamble, no conversational "
    "text. Exception: when the task explicitly includes <analysis> tags, you "
    "may reason inside those tags before the JSON. Otherwise start with { and "
    "end with }. Nothing else."
)

_CTX_TOKENS: Final = 262_144
"""Target model context window size in tokens.  Matches llama.cpp --ctx-size.
Used to derive character budgets: chars ≈ tokens × 3.5 (UTF-8 average).
"""

_EFFECTIVE_CTX_RATIO: Final = 0.65
"""Fraction of context window safely usable before quality degradation.
Research shows models degrade 30-50% before advertised limits (arXiv:2601.15300).
Conservative 65% avoids "lost in the middle" and attention dilution effects.
"""

_COMPLETION_RESERVE_TOKENS: Final = 16_384
"""Tokens reserved for model output.  With large context, reserve more for
comprehensive agent responses (2× the old 8K reserve for 32K context).
"""

CONTEXT_CHAR_LIMIT: Final = 540_000
"""Default input character budget.  Calibrated to ctx-size=262144:
262k tokens × 0.65 effective - 16k completion reserve = 154k input tokens × 3.5 ≈ 540k.
Derived: (_CTX_TOKENS * _EFFECTIVE_CTX_RATIO - _COMPLETION_RESERVE_TOKENS) * 3.5
"""


# ---------------------------------------------------------------------------
# Compression engine: overlapping moving-window LLM summarization
# ---------------------------------------------------------------------------

_WINDOW_SIZE: Final = 50_000
"""Chars per compression window (~14k tokens).  Larger windows reduce LLM
calls and preserve coherence across related content.  Scaled from 20k (32K ctx)
to 50k (262K ctx) — 2.5× increase balances efficiency vs. manageability.
"""

_WINDOW_OVERLAP: Final = 8_000
"""Overlap between adjacent windows.  Prevents information loss at boundaries.
Scaled from 3k to 8k to match larger window size (~16% overlap ratio).
"""

_COMPRESS_SYSTEM: Final = """\
Compress this text while preserving every fact. Your output replaces the \
original — anything you drop is permanently lost.

Preserve (in order of importance): specific numbers and statistics, proper \
nouns (people, organizations, protocols, products), dates and time references, \
direct quotes and attributions, URLs and source references, causal claims \
and conclusions, technical terms and definitions.

Remove: filler words, redundant restatements, formatting artifacts, \
conversational padding, obvious context-setting. Prefer concise rephrasing \
over deletion of substantive content."""


def _build_compress_prompt(target: int, chunk: str, digest: str) -> str:
    """Build compression user prompt without .format() on untrusted text."""
    parts = [f"Distill the following into ≤{target} characters of plain text.\n"]
    if digest:
        parts.append(f"Continuation of prior content (merge, don't repeat):\n{digest}\n")
    parts.append(f"---\n{chunk}")
    return "\n".join(parts)


def _compress_text(
    provider: LLMProvider,
    text: str,
    *,
    model: str,
    target_chars: int,
) -> str:
    """Overlapping moving-window LLM compression.

    Windows of _WINDOW_SIZE chars advance by (_WINDOW_SIZE - _WINDOW_OVERLAP),
    so adjacent windows share _WINDOW_OVERLAP chars of trailing context.
    Each pass receives the running digest as continuation context and merges
    new information into it.  Guarantees no fact is split across a boundary
    without the compressor seeing both halves.
    """
    if len(text) <= target_chars:
        return text

    stride = _WINDOW_SIZE - _WINDOW_OVERLAP
    max_tok = max(512, target_chars // 3)

    def _pass(chunk: str, digest: str) -> str:
        user_prompt = _build_compress_prompt(target_chars, chunk, digest)
        try:
            completion = provider.chat_completion(
                model=model,
                messages=(
                    {"role": ChatRole.SYSTEM, "content": _COMPRESS_SYSTEM},
                    {"role": ChatRole.USER, "content": user_prompt},
                ),
                max_tokens=max_tok,
            )
            compressed = completion.text.strip()
            if compressed:
                return compressed[:target_chars]
        except Exception:
            log.warning("compress_call_failed", exc_info=True)
        if digest:
            return f"{digest} {chunk}"[:target_chars]
        return chunk[:target_chars]

    if len(text) <= _WINDOW_SIZE:
        return _pass(text, "")

    digest = ""
    n_windows = 0
    offset = 0
    while offset < len(text):
        window = text[offset : offset + _WINDOW_SIZE]
        digest = _pass(window, digest)
        n_windows += 1
        offset += stride

    log.info(
        "text_compressed",
        raw_chars=len(text),
        digest_chars=len(digest),
        windows=n_windows,
        target=target_chars,
    )
    return digest


# ---------------------------------------------------------------------------
# format_prompt: single entry point for prompt template formatting + guarding
# ---------------------------------------------------------------------------


def format_prompt(
    provider: LLMProvider,
    template: str,
    *,
    model: str,
    budget: int = CONTEXT_CHAR_LIMIT,
    **kwargs: object,
) -> str:
    """Format a prompt template with per-value compression guard.

    Escapes braces in values (prevents KeyError from JSON/code content) and
    compresses individual values proportionally when total would overflow
    *budget*.  Template text (instructions) is preserved verbatim — only
    dynamic values are compressed, each independently so detail is preserved.
    """
    escaped = {k: str(v).replace("{", "{{").replace("}", "}}") for k, v in kwargs.items()}

    placeholder_chars = sum(len(f"{{{k}}}") for k in kwargs)
    template_chars = len(template) - placeholder_chars
    value_budget = max(2_000, budget - template_chars)

    total = sum(len(v) for v in escaped.values())
    if total <= value_budget:
        return template.format(**escaped)

    log.info(
        "format_prompt_guarding", total_chars=total, budget=value_budget, n_values=len(escaped)
    )
    for k, v in escaped.items():
        v_budget = max(200, int(value_budget * len(v) / total))
        if len(v) > v_budget:
            escaped[k] = _compress_text(provider, v, model=model, target_chars=v_budget)

    return template.format(**escaped)


type _Msg = dict[str, object]


def _guard_inputs(
    provider: LLMProvider,
    inputs: list[_Msg],
    *,
    model: str,
    budget: int,
) -> list[_Msg]:
    """Compress input messages proportionally to fit within *budget* chars.

    This is the internal compression engine.  Callers must NEVER pass
    scaffolding (hard-defined prompts) here — only dynamic content that
    is safe to compress.  Scaffolding budget is subtracted externally by
    ``compose_guarded``.
    """
    total = sum(len(str(m.get("content", ""))) for m in inputs)
    if total <= budget:
        return inputs

    sizes = [len(str(m.get("content", ""))) for m in inputs]

    log.info(
        "context_guard_triggered",
        input_chars=total,
        budget=budget,
        input_msgs=len(inputs),
    )

    guarded = list(inputs)
    compressed_count = 0
    for idx, (msg, msg_size) in enumerate(zip(inputs, sizes, strict=True)):
        msg_budget = max(500, int(budget * msg_size / total))
        if msg_size <= msg_budget:
            continue
        content = str(msg.get("content", ""))
        compressed = _compress_text(provider, content, model=model, target_chars=msg_budget)
        guarded[idx] = {**msg, "content": compressed}
        compressed_count += 1

    if compressed_count:
        after_total = sum(len(str(m.get("content", ""))) for m in guarded)
        log.info(
            "context_guard_done",
            messages_compressed=compressed_count,
            before_chars=total,
            after_chars=after_total,
            reduction_pct=round(100 * (1 - after_total / total), 1) if total else 0,
        )
    return guarded


def compose_guarded(
    provider: LLMProvider,
    *,
    scaffolding: list[_Msg],
    inputs: list[_Msg],
    model: str,
    context_char_limit: int = CONTEXT_CHAR_LIMIT,
) -> list[_Msg]:
    """Message-level compression gate for multi-turn chat windows.

    *scaffolding* — hard-defined prompts (system messages, user instructions).
    NEVER compressed.

    *inputs* — dynamic content (tool results, formatted data, user-supplied text).
    Compressed proportionally when total exceeds ``context_char_limit``.

    Returns ``scaffolding + guarded_inputs`` ready for ``chat_completion``.
    """
    scaffolding_chars = sum(len(str(m.get("content", ""))) for m in scaffolding)
    input_budget = max(2_000, context_char_limit - scaffolding_chars)
    guarded = _guard_inputs(provider, inputs, model=model, budget=input_budget)
    return scaffolding + guarded


def _build_repair_prompt(schema: object, broken: str, error: str) -> str:
    """Build repair prompt without .format() on untrusted broken JSON."""
    return (
        "Fix this malformed JSON to match the schema. Return only the corrected JSON.\n\n"
        f"Schema: {schema}\n\nBroken JSON:\n{broken}\n\n"
        f"Error: {error}"
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
    raw_text: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    elapsed_s: float = 0.0


def _backoff_retry(
    attempt: int,
    max_retries: int,
    backoff_base: float,
    schema_name: str,
    kind: str,
    exc: Exception,
) -> bool:
    """Sleep with jittered exponential backoff if retries remain. Returns True to continue."""
    if attempt >= max_retries:
        return False
    base_wait = backoff_base**attempt
    wait = base_wait + random.uniform(0, base_wait * 0.5)
    log.warning(
        f"llm_call_{kind}_retrying",
        attempt=attempt,
        max_retries=max_retries,
        schema=schema_name,
        exc=str(exc),
        retry_in_s=round(wait, 1),
    )
    time.sleep(wait)
    return True


def _attempt_repair[T: BaseModel](
    provider: LLMProvider,
    raw_text: str,
    response_model: type[T],
    repair_model: str,
    max_tokens: int,
    error: Exception,
    t0: float,
    *,
    max_repair_tries: int = 2,
) -> LLMCallResult[T] | None:
    """Repair malformed JSON via an LLM call. Returns None when all tries fail."""
    schema_name = response_model.__name__
    current_error = error
    current_text = raw_text

    for attempt in range(1, max_repair_tries + 1):
        log.debug(
            "json_repair_attempt",
            attempt=attempt,
            max_attempts=max_repair_tries,
            schema=schema_name,
            repair_model=repair_model,
            error=str(current_error)[:120],
        )
        try:
            repair_prompt = _build_repair_prompt(
                schema=response_model.model_json_schema(),
                broken=current_text[:2000],
                error=str(current_error)[:500],
            )
            repair_result = provider.chat_completion(
                model=repair_model,
                messages=(
                    {"role": ChatRole.SYSTEM, "content": JSON_SYSTEM_PROMPT},
                    {"role": ChatRole.USER, "content": repair_prompt},
                ),
                max_tokens=max_tokens,
            )
            repaired_text = repair_result.text
            repaired_data = decode_llm_json(repaired_text)
            value = response_model.model_validate(repaired_data)
            log.debug(
                "json_repair_ok",
                schema=schema_name,
                attempt=attempt,
                max_attempts=max_repair_tries,
            )
            return LLMCallResult(
                value=value,
                success=True,
                attempts=attempt,
                raw_text=repaired_text,
                input_tokens=repair_result.input_tokens,
                output_tokens=repair_result.output_tokens,
                elapsed_s=round(time.time() - t0, 1),
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
    instructions: str,
    response_model: type[T],
    fallback: T,
    model: str,
    max_tokens: int,
    repair_model: str | None = None,
    backoff_base: float = 2.0,
    max_retries: int = 3,
) -> LLMCallResult[T]:
    """Execute a structured LLM call with retry, repair, and fallback.

    ``instructions`` — fully formatted prompt (use ``format_prompt`` to build it).
    Values are already guarded at format time; no message-level compression here.
    ``JSON_SYSTEM_PROMPT`` is always prepended as the system message.
    """
    effective_repair_model = repair_model or model
    schema_name = response_model.__name__
    last_error = ""
    last_category = LLMErrorCategory.NONE
    raw_text = ""
    attempts_made = 0
    in_tokens = out_tokens = 0
    t0 = time.time()

    messages: tuple[_Msg, ...] = (
        {"role": ChatRole.SYSTEM, "content": JSON_SYSTEM_PROMPT},
        {"role": ChatRole.USER, "content": instructions},
    )

    for attempt in range(1, max_retries + 1):
        attempts_made = attempt
        try:
            completion = provider.chat_completion(
                model=model, messages=tuple(messages), max_tokens=max_tokens
            )
            raw_text = completion.text
            in_tokens = completion.input_tokens
            out_tokens = completion.output_tokens

            if completion.finish_reason == "length":
                log.warning("llm_call_truncated", attempt=attempt, schema=schema_name)
                repaired = _attempt_repair(
                    provider,
                    raw_text,
                    response_model,
                    effective_repair_model,
                    max_tokens,
                    LLMParseError("Output truncated"),
                    t0,
                )
                if repaired:
                    return repaired
                last_error = "Output truncated (max_tokens exhausted)"
                last_category = LLMErrorCategory.PARSE
                break

            data = decode_llm_json(raw_text)
            value = response_model.model_validate(data)
            return LLMCallResult(
                value=value,
                success=True,
                attempts=attempt,
                raw_text=raw_text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                elapsed_s=round(time.time() - t0, 1),
            )

        except (ValidationError, LLMParseError) as exc:
            is_validation = isinstance(exc, ValidationError)
            last_error = f"{'Schema validation' if is_validation else 'JSON parse'}: {exc}"
            last_category = LLMErrorCategory.VALIDATION if is_validation else LLMErrorCategory.PARSE
            log.warning(
                "llm_call_parse_error",
                attempt=attempt,
                max_retries=max_retries,
                schema=schema_name,
                category=last_category.value,
                error=str(exc),
            )
            repaired = _attempt_repair(
                provider, raw_text, response_model, effective_repair_model, max_tokens, exc, t0
            )
            if repaired:
                return repaired

        except ProviderHTTPError as exc:
            last_error = f"HTTP {exc.status}: {exc.detail}"
            last_category = LLMErrorCategory.HTTP
            if exc.status in (400, 401, 403, 404, 422):
                log.warning(
                    "llm_call_terminal_error",
                    attempt=attempt,
                    max_retries=max_retries,
                    schema=schema_name,
                    http_status=exc.status,
                )
                break
            if _backoff_retry(attempt, max_retries, backoff_base, schema_name, "http", exc):
                continue

        except ProviderTransportError as exc:
            last_error = f"Transport: {exc}"
            last_category = LLMErrorCategory.TRANSPORT
            if _backoff_retry(attempt, max_retries, backoff_base, schema_name, "transport", exc):
                continue
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
        input_tokens=in_tokens,
        output_tokens=out_tokens,
        elapsed_s=round(time.time() - t0, 1),
    )


async def async_llm_call[T: BaseModel](
    provider: LLMProvider,
    *,
    instructions: str,
    response_model: type[T],
    fallback: T,
    model: str,
    max_tokens: int,
    repair_model: str | None = None,
    backoff_base: float = 2.0,
    max_retries: int = 3,
) -> LLMCallResult[T]:
    """Async bridge for ``llm_call`` — blocking I/O in default executor.

    Concurrency gating is the caller's responsibility.  Sonality and fathom
    each own an ``asyncio.Semaphore`` sized to their ``llm_concurrency`` setting.
    """
    return await asyncio.to_thread(
        llm_call,
        provider,
        instructions=instructions,
        response_model=response_model,
        fallback=fallback,
        model=model,
        max_tokens=max_tokens,
        repair_model=repair_model,
        backoff_base=backoff_base,
        max_retries=max_retries,
    )
