"""Token estimation, chat input trimming, and LLM-based context summarization."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, model_validator

from .llm.parse import coerce_string_fields
from .schema import ChatRole

log = logging.getLogger(__name__)


class _ConversationSummarySchema(BaseModel):
    """Structured conversation summary."""

    intent: str = ""
    key_facts: str = ""
    decisions: str = ""
    open_threads: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_lists(cls, data: object) -> object:
        return coerce_string_fields(data, ("key_facts", "decisions", "open_threads"), sep="; ")

    def render(self) -> str:
        parts: list[str] = []
        if self.intent:
            parts.append(f"Intent: {self.intent}")
        if self.key_facts:
            parts.append(f"Key facts: {self.key_facts}")
        if self.decisions:
            parts.append(f"Decisions: {self.decisions}")
        if self.open_threads:
            parts.append(f"Open threads: {self.open_threads}")
        return "\n".join(parts)


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


def _extract_text_from_content(content: object) -> str:
    """Extract text from message content, handling both string and multipart formats."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                parts.append(item["text"])
        return " ".join(parts)
    return ""


def _estimate_messages_tokens(messages: Sequence[dict[str, object] | dict[str, str]]) -> int:
    """Sum :func:`estimate_tokens_utf8` over each message's content field.

    Handles both string content and multipart content (list of text/image blocks).
    """
    return sum(
        estimate_tokens_utf8(_extract_text_from_content(m.get("content", "")))
        for m in messages
    )


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


def message_tokens_budget_for_system(
    *,
    total_budget: int,
    system_prompt: str,
    reserve_completion: int,
) -> int:
    """Headroom for user/assistant turns after system prompt and completion reserve.

    ``total_budget`` is typically :data:`sonality.config.CHAT_INPUT_TOKEN_BUDGET`.
    Subtracts :func:`estimate_tokens_utf8` of ``system_prompt`` and ``reserve_completion``,
    then returns at least 256 so a minimal tail can remain.
    """
    sys_tok = estimate_tokens_utf8(system_prompt)
    return max(256, total_budget - sys_tok - reserve_completion)


SUMMARIZE_THRESHOLD: Final = 8
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
            "History short (%d msgs < %d), using simple trim", len(messages), SUMMARIZE_THRESHOLD
        )
        return trim_chat_messages_for_budget(messages, max_message_tokens=max_message_tokens)

    est = _estimate_messages_tokens(messages)
    if est <= max_message_tokens:
        log.debug("History fits budget (%d est tokens <= %d max)", est, max_message_tokens)
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
        "Summarizing: %d older messages, previous_summary=%d chars, keeping %d recent",
        len(non_summary_older),
        len(previous_summary),
        len(recent),
    )
    summary = _llm_summarize_messages(non_summary_older, previous_summary)
    if not summary:
        log.warning("Summarization failed, falling back to simple trim")
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
    from .llm.caller import llm_call
    from .prompts import CONVERSATION_SUMMARY_PROMPT

    formatted = "\n".join(
        f"{m.get('role', ChatRole.USER).capitalize()}: {m.get('content', '')[:300]}"
        for m in messages
    )
    prev_section = (
        f"Previous summary to update:\n{previous_summary}"
        if previous_summary
        else "No previous summary."
    )
    prompt = CONVERSATION_SUMMARY_PROMPT.format(
        previous_summary=prev_section, messages=formatted[:3000]
    )
    try:
        r = llm_call(
            prompt=prompt,
            response_model=_ConversationSummarySchema,
            fallback=_ConversationSummarySchema(),
            model=config.FAST_MODEL,
        )
        rendered = r.value.render()
        log.debug(
            "Summarized %d messages → %d chars (anchored=%s success=%s)",
            len(messages),
            len(rendered),
            bool(previous_summary),
            r.success,
        )
        return rendered
    except Exception:
        log.warning("Context summarization failed", exc_info=True)
        return ""
