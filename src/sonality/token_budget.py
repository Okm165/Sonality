"""Token estimation, chat input trimming, and LLM-based context summarization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import structlog
from pydantic import BaseModel, Field, model_validator

from shared.llm.parse import coerce_string_fields, render_labeled
from shared.types import ChatRole

log = structlog.get_logger(__name__)


class _ConversationSummarySchema(BaseModel):
    """Structured conversation summary."""

    intent: str = Field(default="", max_length=1000)
    key_facts: str = Field(default="", max_length=2000)
    decisions: str = Field(default="", max_length=2000)
    open_threads: str = Field(default="", max_length=1000)

    @model_validator(mode="before")
    @classmethod
    def coerce_lists(cls, data: object) -> object:
        return coerce_string_fields(data, ("key_facts", "decisions", "open_threads"), sep="; ")

    def render(self) -> str:
        return render_labeled(
            self,
            {
                "intent": "Intent",
                "key_facts": "Key facts",
                "decisions": "Decisions",
                "open_threads": "Open threads",
            },
        )


def estimate_tokens_utf8(text: str) -> int:
    """Approximate token count for budgeting chat input (not billing).

    Formula: ``len(text.encode("utf-8")) // 4 + 1`` (non-empty text), else ``0``.

    This is a fast, provider-agnostic heuristic (sometimes called "Claw-style"):
    English-like text often lands near ~4 bytes per token on byte-oriented
    subword tokenizers, so UTF-8 byte length divided by 4 approximates tokens
    from below well enough for *trimming* and budget checks.

    **When to use:** ``trim_chat_messages_for_budget``, ``message_tokens_budget_for_system``,
    and any code that must cap *main* chat history size without calling a tokenizer API.

    **Not interchangeable with:** ad-hoc ``len(text) // 4`` character heuristics used
    elsewhere (e.g. prompt-size spot checks). Those can differ by up to ~4x on non-Latin
    scripts. Prefer this function when the goal is consistent budget math with trimming.
    """
    if not text:
        return 0
    return len(text.encode("utf-8")) // 4 + 1


def _estimate_messages_tokens(messages: Sequence[dict[str, str]]) -> int:
    """Sum :func:`estimate_tokens_utf8` over each message's content field."""
    return sum(estimate_tokens_utf8(m.get("content", "")) for m in messages)


def trim_chat_messages_for_budget(
    messages: list[dict[str, str]],
    *,
    max_message_tokens: int,
    min_tail_messages: int = 1,
) -> list[dict[str, str]]:
    """Drop oldest messages until estimated tokens <= ``max_message_tokens``.

    Uses :func:`estimate_tokens_utf8` per message body. Always keeps at least the last
    ``min_tail_messages`` entries (if present).
    """
    if max_message_tokens <= 0 or not messages:
        return messages
    trimmed: list[dict[str, str]] = list(messages)
    while len(trimmed) > min_tail_messages:
        est = _estimate_messages_tokens(trimmed)
        if est <= max_message_tokens:
            return trimmed
        trimmed.pop(0)
    return trimmed


_DEFAULT_COMPLETION_RESERVE: Final = 16_384
"""Tokens reserved for model completion when max_tokens is unspecified.
Scaled from 8K (32K context) to 16K (262K context) for comprehensive responses.
"""


def message_tokens_budget_for_system(
    *,
    total_budget: int,
    system_prompt: str,
    reserve_completion: int,
) -> int:
    """Headroom for user/assistant turns after system prompt and completion reserve.

    ``total_budget`` is typically :data:`sonality.config.settings.chat_input_token_budget`.
    Subtracts :func:`estimate_tokens_utf8` of ``system_prompt`` and ``reserve_completion``,
    then returns at least 256 so a minimal tail can remain.

    When ``reserve_completion <= 0`` (server-decides mode), uses 8192 as a
    safe default so context math still works.
    """
    effective_reserve = (
        reserve_completion if reserve_completion > 0 else _DEFAULT_COMPLETION_RESERVE
    )
    sys_tok = estimate_tokens_utf8(system_prompt)
    return max(256, total_budget - sys_tok - effective_reserve)


SUMMARIZE_THRESHOLD: Final = 20
"""Minimum message count before LLM summarization triggers.
Scaled from 8 (32K context) to 20 (262K context) to leverage larger window.
"""
_SUMMARY_PREFIX: Final = "[Earlier conversation summary]"


def summarize_and_trim(
    messages: list[dict[str, str]],
    *,
    max_message_tokens: int,
    recent_keep: int = 4,
) -> list[dict[str, str]]:
    """Anchored iterative summarization: merge new turns into existing summary.

    If a previous summary exists in messages, extracts it and merges new older turns.
    If no summary exists, generates fresh from older messages. Falls back to simple
    trimming when history is short or LLM summarization fails.
    """
    if len(messages) < SUMMARIZE_THRESHOLD:
        log.debug(
            "history_short_simple_trim",
            message_count=len(messages),
            summarize_threshold=SUMMARIZE_THRESHOLD,
        )
        return trim_chat_messages_for_budget(messages, max_message_tokens=max_message_tokens)

    est = _estimate_messages_tokens(messages)
    if est <= max_message_tokens:
        log.debug(
            "history_fits_token_budget",
            estimated_tokens=est,
            max_message_tokens=max_message_tokens,
        )
        return messages

    recent = messages[-recent_keep:]
    older = messages[:-recent_keep]

    if not older:
        return trim_chat_messages_for_budget(messages, max_message_tokens=max_message_tokens)

    previous_summary = ""
    non_summary_older: list[dict[str, str]] = []
    for m in older:
        content = m.get("content", "")
        if content.startswith(_SUMMARY_PREFIX):
            previous_summary = content[len(_SUMMARY_PREFIX) :].strip()
        else:
            non_summary_older.append(m)

    log.info(
        "conversation_summarize_start",
        older_message_count=len(non_summary_older),
        previous_summary_chars=len(previous_summary),
        recent_message_count=len(recent),
    )
    try:
        summary = _llm_summarize_messages(non_summary_older, previous_summary)
    except Exception:
        log.warning("summarization_failed_fallback_trim", exc_info=True)
        return trim_chat_messages_for_budget(messages, max_message_tokens=max_message_tokens)
    if not summary:
        return trim_chat_messages_for_budget(messages, max_message_tokens=max_message_tokens)

    summary_msg: dict[str, str] = {
        "role": ChatRole.SYSTEM,
        "content": f"{_SUMMARY_PREFIX}\n{summary}",
    }
    result = [summary_msg, *recent]

    if _estimate_messages_tokens(result) > max_message_tokens:
        return trim_chat_messages_for_budget(result, max_message_tokens=max_message_tokens)

    return result


def _llm_summarize_messages(messages: list[dict[str, str]], previous_summary: str = "") -> str:
    """Summarize messages using anchored iterative pattern via LLM.

    If previous_summary is provided, the LLM merges new information into it
    rather than regenerating from scratch (anchored approach, arXiv:2603.29193).
    """
    from . import config
    from .caller import format_prompt, llm_call
    from .prompts import CONVERSATION_SUMMARY_PROMPT

    formatted = "\n".join(
        f"{m.get('role', ChatRole.USER).capitalize()}: {m.get('content', '')[:800]}"
        for m in messages
    )
    prev_section = (
        f"Previous summary to update:\n{previous_summary}"
        if previous_summary
        else "No previous summary."
    )
    r = llm_call(
        instructions=format_prompt(
            CONVERSATION_SUMMARY_PROMPT, previous_summary=prev_section, messages=formatted
        ),
        response_model=_ConversationSummarySchema,
        fallback=_ConversationSummarySchema(),
        model=config.settings.fast_model,
    )
    rendered = r.value.render()
    log.debug(
        "conversation_summarized",
        source_message_count=len(messages),
        summary_char_count=len(rendered),
        anchored=bool(previous_summary),
        llm_success=r.success,
    )
    return rendered
