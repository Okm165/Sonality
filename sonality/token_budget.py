"""Token estimation and chat input trimming for main completion calls."""

from __future__ import annotations

from collections.abc import Sequence


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


def estimate_messages_tokens(messages: Sequence[dict[str, str]]) -> int:
    """Sum :func:`estimate_tokens_utf8` over each message's string ``content`` field."""
    total = 0
    for m in messages:
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens_utf8(content)
    return total


def trim_chat_messages_for_budget(
    messages: list[dict[str, str]],
    *,
    max_message_tokens: int,
    min_tail_messages: int = 1,
) -> list[dict[str, str]]:
    """Drop oldest messages until :func:`estimate_messages_tokens` <= ``max_message_tokens``.

    Uses :func:`estimate_tokens_utf8` per message body. Always keeps at least the last
    ``min_tail_messages`` entries (if present).
    """
    if max_message_tokens <= 0 or not messages:
        return messages
    trimmed: list[dict[str, str]] = list(messages)
    while len(trimmed) > min_tail_messages:
        est = estimate_messages_tokens(trimmed)
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
