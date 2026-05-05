"""LLM output parsing: JSON extraction, thinking trace removal, tool call parsing.

Pure functions with no I/O or provider dependency.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
from collections.abc import Callable, Mapping
from typing import Final, NamedTuple

log = logging.getLogger(__name__)

_THINKING_ANSWER_MARKERS: Final = (
    "Final Output:",
    "Output:",
    "Answer:",
    "Final Answer:",
    "Response:",
)
_THINK_BLOCK_RE: Final = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_CODE_BLOCK_RE: Final = re.compile(
    r"```(?:thinking|thought|reasoning)[\s\S]*?```", re.IGNORECASE
)
_ASTERISK_THOUGHT_RE: Final = re.compile(r"^\s*\*[^*\n]+\*?\s*$", re.MULTILINE)
_SEMANTIC_MARKER_RE: Final = re.compile(r"\[semantic/[^\]]*\][^\n]*\n?", re.IGNORECASE)
_ANALYSIS_COMPLETE_RE: Final = re.compile(r"<analysis>[\s\S]*?</analysis>", re.IGNORECASE)
_ANALYSIS_UNCLOSED_RE: Final = re.compile(r"<analysis>[\s\S]*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Thinking trace removal
# ---------------------------------------------------------------------------


def strip_thinking_trace(text: str) -> str:
    """Remove chain-of-thought traces from model-visible output."""
    result = _THINK_BLOCK_RE.sub("", text)
    result = _THINK_CODE_BLOCK_RE.sub("", result)
    result = _ASTERISK_THOUGHT_RE.sub("", result)
    result = _SEMANTIC_MARKER_RE.sub("", result)
    return result.strip()


_THINK_TAG_RE: Final = re.compile(r"</?think>", re.IGNORECASE)


def unwrap_think_tags(text: str) -> str:
    """Remove <think>/</think> tags but preserve the content within.

    Used when a model wraps its entire output in think blocks even with
    enable_thinking=False (e.g. gemma4). strip_thinking_trace would discard
    everything; this preserves the inner content so structured output (JSON,
    answer markers) can be extracted from it.
    """
    return _THINK_TAG_RE.sub("", text).strip()


def extract_answer_from_reasoning(reasoning: str) -> str:
    """Extract structured answer (JSON, marked output) from reasoning text.

    Unlike strip_thinking_trace, this preserves bullet-point content since
    reasoning/thinking blocks legitimately use asterisk lists.
    """
    cleaned = _THINK_BLOCK_RE.sub("", reasoning)
    cleaned = _THINK_CODE_BLOCK_RE.sub("", cleaned)
    cleaned = _ANALYSIS_COMPLETE_RE.sub("", cleaned)
    cleaned = _ANALYSIS_UNCLOSED_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    for marker in _THINKING_ANSWER_MARKERS:
        idx = cleaned.lower().rfind(marker.lower())
        if idx != -1 and (answer := cleaned[idx + len(marker) :].strip()):
            return answer
    for opener, closer in (("{", "}"), ("[", "]")):
        end = cleaned.rfind(closer)
        if end != -1:
            depth = 0
            for i in range(end, -1, -1):
                if cleaned[i] == closer:
                    depth += 1
                elif cleaned[i] == opener:
                    depth -= 1
                    if depth == 0 and len(candidate := cleaned[i : end + 1].strip()) > 2:
                        return candidate
    for line in reversed(cleaned.strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


# ---------------------------------------------------------------------------
# JSON extraction from messy LLM output
# ---------------------------------------------------------------------------


def _strip_markdown_fences(text: str) -> str:
    return text.replace("```json", "").replace("```", "").strip()


def _strip_analysis_block(text: str) -> str:
    """Strip <analysis>...</analysis> scratchpad blocks from LLM output."""
    text = _ANALYSIS_COMPLETE_RE.sub("", text)
    text = _ANALYSIS_UNCLOSED_RE.sub("", text)
    return text.strip()


def extract_last_json_object(text: str) -> dict[str, object] | None:
    """Find the largest (by character span) valid JSON object in text."""
    cleaned = _strip_markdown_fences(text.strip())
    # Fix malformed "+N" JSON values (e.g. "+3" → "3") that some models emit.
    cleaned = re.sub(r":\s*\+(\d)", r": \1", cleaned)
    decoder = json.JSONDecoder()

    candidates: list[tuple[int, int, dict[str, object]]] = []
    i = 0
    while i < len(cleaned):
        if cleaned[i] != "{":
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(cleaned, i)
            if isinstance(obj, dict):
                candidates.append((i, end, obj))
            i = end
        except json.JSONDecodeError:
            i += 1

    if candidates:
        return max(candidates, key=lambda c: c[1] - c[0])[2]

    try:
        arr = json.loads(cleaned)
        if isinstance(arr, list) and arr and all(isinstance(x, int) for x in arr):
            return {"ranking": arr}
    except json.JSONDecodeError:
        pass
    return None


def decode_llm_json(text: str) -> dict[str, object] | list[object]:
    """Extract JSON (object or array) from LLM response text.

    Strips markdown fences and <analysis> blocks, attempts whole-text parse
    first (preserving bare arrays), then falls back to extract_last_json_object
    for messy output. Raises ValueError if no JSON can be extracted.
    """
    cleaned = _strip_markdown_fences(_strip_analysis_block(text))
    if not cleaned:
        raise ValueError("Empty LLM response")
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass
    if '\\"' in cleaned:
        try:
            parsed = json.loads(cleaned.replace('\\"', '"'))
            if isinstance(parsed, (dict, list)):
                return parsed
        except json.JSONDecodeError:
            pass
    obj = extract_last_json_object(cleaned)
    if obj is not None:
        return obj
    raise ValueError(f"No valid JSON in LLM response: {text[:120]!r}")


def parse_json_object(text: str) -> dict[str, object]:
    """Extract a JSON object from LLM response text; returns {} on failure."""
    try:
        result = decode_llm_json(text)
        return result if isinstance(result, dict) else {}
    except ValueError:
        return {}


# ---------------------------------------------------------------------------
# Tool call parsing from OpenAI-compatible completion payloads
# ---------------------------------------------------------------------------


class ParsedToolCall(NamedTuple):
    """Parsed tool call from an LLM chat-completion payload."""

    name: str
    args: dict[str, object]
    id: str


def get_raw_tool_calls(raw: Mapping[str, object]) -> list[dict[str, object]]:
    """Navigate choices[0].message.tool_calls safely, returning raw dicts."""
    choices = raw.get("choices")
    if not isinstance(choices, list) or not choices:
        return []
    first = choices[0]
    if not isinstance(first, dict):
        return []
    message = first.get("message")
    if not isinstance(message, dict):
        return []
    calls = message.get("tool_calls")
    return [c for c in calls if isinstance(c, dict)] if isinstance(calls, list) else []


def extract_tool_calls(raw: Mapping[str, object]) -> list[ParsedToolCall]:
    """Parse all tool calls from a raw LLM completion payload."""
    parsed: list[ParsedToolCall] = []
    for call in get_raw_tool_calls(raw):
        func = call.get("function")
        if not isinstance(func, dict):
            continue
        name = str(func.get("name", ""))
        raw_args = func.get("arguments", "{}")
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, TypeError):
            log.warning("Malformed tool-call arguments for %s, using empty args", name)
            args = {}
        if not isinstance(args, dict):
            args = {}
        parsed.append(ParsedToolCall(name=name, args=args, id=str(call.get("id", ""))))
    return parsed


def extract_tool_call_arguments(
    raw_payload: Mapping[str, object], function_name: str
) -> dict[str, object]:
    """Extract arguments for a specific tool call by function name."""
    for call in get_raw_tool_calls(raw_payload):
        func = call.get("function")
        if not isinstance(func, dict) or func.get("name") != function_name:
            continue
        arguments = func.get("arguments")
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            with contextlib.suppress(json.JSONDecodeError):
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return dict(parsed)
    return {}


# ---------------------------------------------------------------------------
# Message content helpers
# ---------------------------------------------------------------------------


def message_content_text(message: object) -> str:
    """Extract text from a message content field (string or multipart list)."""
    if isinstance(message, str):
        return message
    if not isinstance(message, list):
        return ""
    parts: list[str] = []
    for item in message:
        if isinstance(item, str):
            parts.append(item)
        elif (
            isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ):
            parts.append(item["text"])
    return "".join(parts)


def to_nonnegative_int(value: object) -> int:
    """Parse a non-negative integer from various types."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


# ---------------------------------------------------------------------------
# Pydantic model_validator helper for LLM response normalization
# ---------------------------------------------------------------------------


def coerce_string_fields(data: object, fields: tuple[str, ...], sep: str = "\n") -> object:
    """Coerce list/dict values to strings in a dict — use in Pydantic model_validators.

    LLMs frequently return list or dict where a string field is expected.
    This normalizes them: lists are joined with ``sep``, dicts are flattened
    to ``key: value`` pairs joined with ``; ``.
    """
    if not isinstance(data, dict):
        return data
    for k in fields:
        v = data.get(k)
        if v is None:
            data[k] = ""
        elif isinstance(v, bool):
            data[k] = str(v).lower()
        elif isinstance(v, (int, float)):
            data[k] = str(v)
        elif isinstance(v, list):
            data[k] = sep.join(str(x) for x in v)
        elif isinstance(v, dict):
            data[k] = "; ".join(f"{dk}: {dv}" for dk, dv in v.items())
    return data


def normalize_llm_list_response(
    data: object,
    *,
    list_key: str,
    item_required_key: str = "",
    item_filter: Callable[[object], bool] | None = None,
) -> object:
    """Normalize messy LLM JSON into ``{list_key: [...]}``.

    Handles three common LLM output shapes:
    1. Bare list: ``[{...}, {...}]`` → ``{list_key: [...]}}``
    2. Bare item dict (truncated wrapper): ``{"text": ...}`` → ``{list_key: [{...}]}``
    3. Normal ``{list_key: [...]}`` — filters out empty/malformed items.

    Use in a Pydantic ``@model_validator(mode="before")`` to avoid repeating
    this pattern across every LLM response model.
    """
    is_valid = item_filter or (
        (lambda x: isinstance(x, dict) and item_required_key in x)
        if item_required_key
        else (lambda x: isinstance(x, dict) and bool(x))
    )
    if isinstance(data, list):
        return {list_key: [x for x in data if is_valid(x)]}
    if isinstance(data, dict):
        if list_key not in data and item_required_key and item_required_key in data:
            return {list_key: [data]}
        items = data.get(list_key)
        if isinstance(items, list):
            return {**data, list_key: [x for x in items if is_valid(x)]}
    return data


# ---------------------------------------------------------------------------
# Plain-text cleanup for LLM responses
# ---------------------------------------------------------------------------

_MD_BOLD: Final = re.compile(r"\*\*(.+?)\*\*")
_MD_HEADER: Final = re.compile(r"^#{1,4}\s+", re.MULTILINE)
_MD_CODE_FENCE: Final = re.compile(r"^```\w*\n?|^```$", re.MULTILINE)


def strip_markdown(text: str) -> str:
    """Remove markdown bold, headers, and code fences from LLM output."""
    text = _MD_CODE_FENCE.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_HEADER.sub("", text)
    return text.strip()
