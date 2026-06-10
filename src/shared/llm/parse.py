"""LLM output cleanup: thinking removal, JSON extraction, tool call parsing.

All cleanup of raw LLM text lives here — provider and caller import from
this module exclusively.  Pure functions, no I/O.

Cleanup pipeline (top to bottom):
  clean_completion   — entry point for raw chat-completion fields
    _strip_thinking  — remove <think> blocks, reasoning fences, noise
    _extract_answer  — find structured output buried in reasoning text
  decode_llm_json    — extract JSON from cleaned text
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Final, NamedTuple

import structlog

from ..errors import LLMParseError

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Regex constants — shared across cleanup stages
# ---------------------------------------------------------------------------

_THINK_BLOCK_RE: Final = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_UNCLOSED_RE: Final = re.compile(r"<think>(?:(?!</think>).)*$", re.IGNORECASE | re.DOTALL)
_THINK_TAG_RE: Final = re.compile(r"</?think>", re.IGNORECASE)
_THINK_CODE_BLOCK_RE: Final = re.compile(
    r"```(?:thinking|thought|reasoning)[\s\S]*?```", re.IGNORECASE
)
_ASTERISK_THOUGHT_RE: Final = re.compile(r"^\s*\*[^*\n]+\*?\s*$", re.MULTILINE)
_SEMANTIC_MARKER_RE: Final = re.compile(r"\[semantic/[^\]]*\][^\n]*\n?", re.IGNORECASE)
_ANALYSIS_COMPLETE_RE: Final = re.compile(r"<analysis>[\s\S]*?</analysis>", re.IGNORECASE)
_ANALYSIS_UNCLOSED_RE: Final = re.compile(r"<analysis>[\s\S]*$", re.IGNORECASE)
_INTERNAL_XML_TAGS: Final = (
    "research_plan",
    "planning",
    "internal",
    "reasoning",
    "reflection",
    "scratch_pad",
    "scratchpad",
    "notes",
    "thought_process",
    "chain_of_thought",
)
_INTERNAL_XML_RE: Final = re.compile(
    r"<(?:"
    + "|".join(_INTERNAL_XML_TAGS)
    + r")>[\s\S]*?</(?:"
    + "|".join(_INTERNAL_XML_TAGS)
    + r")>",
    re.IGNORECASE,
)
_INTERNAL_XML_UNCLOSED_RE: Final = re.compile(
    r"<(?:" + "|".join(_INTERNAL_XML_TAGS) + r")>[\s\S]*$",
    re.IGNORECASE,
)
_ANSWER_MARKERS: Final = (
    "Final Output:",
    "Output:",
    "Answer:",
    "Final Answer:",
    "Response:",
)


# ---------------------------------------------------------------------------
# Completion cleanup — the single entry point for provider.py
# ---------------------------------------------------------------------------


def _strip_thinking(text: str) -> str:
    """Remove all thinking/reasoning traces, return only the answer portion.

    Handles closed <think>...</think> blocks, unclosed <think> tails
    (truncated by max_tokens), reasoning code fences, asterisk thoughts,
    and semantic markers.
    """
    result = _THINK_BLOCK_RE.sub("", text)
    result = _THINK_UNCLOSED_RE.sub("", result)
    result = _THINK_CODE_BLOCK_RE.sub("", result)
    result = _INTERNAL_XML_RE.sub("", result)
    result = _INTERNAL_XML_UNCLOSED_RE.sub("", result)
    result = _ASTERISK_THOUGHT_RE.sub("", result)
    result = _SEMANTIC_MARKER_RE.sub("", result)
    return result.strip()


def _extract_answer(text: str) -> str:
    """Find a structured answer (JSON, marked output) in reasoning text.

    Preserves bullet-point content (reasoning blocks legitimately use
    asterisk lists).  Tries answer markers first, then outermost JSON
    brackets, then last non-empty line.
    """
    cleaned = _THINK_BLOCK_RE.sub("", text)
    cleaned = _THINK_CODE_BLOCK_RE.sub("", cleaned)
    cleaned = _ANALYSIS_COMPLETE_RE.sub("", cleaned)
    cleaned = _ANALYSIS_UNCLOSED_RE.sub("", cleaned)
    cleaned = _INTERNAL_XML_RE.sub("", cleaned)
    cleaned = _INTERNAL_XML_UNCLOSED_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    if not cleaned:
        return ""
    for marker in _ANSWER_MARKERS:
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


def clean_completion(content: str, reasoning: str = "") -> str:
    """Extract the meaningful answer from raw LLM completion fields.

    This is the single entry point for all thinking/reasoning recovery.
    Handles models that:
    - Wrap output in <think>...</think> blocks
    - Put the entire answer inside reasoning_content
    - Get truncated mid-think by max_tokens (unclosed <think>)
    """
    text = _strip_thinking(content)

    if (not text or text.lstrip().startswith("<think")) and content.strip():
        text = _extract_answer(_THINK_TAG_RE.sub("", content))
        if text:
            log.debug("thinking_wrap_recovered", chars=len(text))

    if not text and reasoning:
        text = _extract_answer(reasoning)
        if text:
            log.debug("recovered_from_reasoning", chars=len(text))

    return text


# ---------------------------------------------------------------------------
# JSON extraction from messy LLM output
# ---------------------------------------------------------------------------


_MD_FENCE_LINE_RE: Final = re.compile(r"^\s*```\w*\s*$", re.MULTILINE)


def _strip_markdown_fences(text: str) -> str:
    return _MD_FENCE_LINE_RE.sub("", text).strip()


def _strip_analysis_block(text: str) -> str:
    text = _ANALYSIS_COMPLETE_RE.sub("", text)
    text = _ANALYSIS_UNCLOSED_RE.sub("", text)
    return text.strip()


def _extract_last_json_object(text: str) -> dict[str, object] | list[object] | None:
    """Find the last valid JSON object or array in text.

    Expects input already stripped of markdown fences and analysis blocks.
    LLMs emit reasoning/examples first and the answer last, so the final
    successfully-parsed structure is almost always the intended output.
    Prefers the last dict; falls back to the last array if no dict found.
    """
    cleaned = re.sub(r":\s*\+(\d)", r": \1", text.strip())
    decoder = json.JSONDecoder()

    last_dict: dict[str, object] | None = None
    last_array: list[object] | None = None
    i = 0
    while i < len(cleaned):
        if cleaned[i] not in ("{", "["):
            i += 1
            continue
        try:
            obj, end = decoder.raw_decode(cleaned, i)
            if isinstance(obj, dict):
                last_dict = obj
            elif isinstance(obj, list):
                last_array = obj
            i = end
        except json.JSONDecodeError:
            i += 1

    return last_dict if last_dict is not None else last_array


def decode_llm_json(text: str) -> dict[str, object] | list[object]:
    """Extract JSON from LLM response text.

    Strips thinking traces, analysis blocks, and markdown fences before
    parsing. Falls back to scanning for the last valid JSON object.
    """
    text = _strip_thinking(text) or _THINK_TAG_RE.sub("", text).strip() or text
    cleaned = _strip_markdown_fences(_strip_analysis_block(text))
    if not cleaned:
        raise LLMParseError("Empty LLM response")
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
    obj = _extract_last_json_object(cleaned)
    if obj is not None:
        return obj
    raise LLMParseError(f"No valid JSON in LLM response: {text[:120]!r}")


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
            log.warning("malformed_tool_call_args", tool=name)
            args = {}
        if not isinstance(args, dict):
            args = {}
        parsed.append(ParsedToolCall(name=name, args=args, id=str(call.get("id", ""))))
    return parsed


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
    """Parse a non-negative integer from various types including numeric strings."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    if isinstance(value, str):
        try:
            return max(int(float(value)), 0)
        except (ValueError, TypeError):
            return 0
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


def render_labeled(obj: object, labels: dict[str, str]) -> str:
    """Render a Pydantic model (or any object) as labeled lines.

    ``labels`` maps attribute names to display labels. Empty/missing values
    are skipped. Returns newline-joined string.
    """
    parts: list[str] = []
    for attr, label in labels.items():
        val = getattr(obj, attr, "")
        if val:
            parts.append(f"{label}: {val}")
    return "\n".join(parts)


def normalize_llm_list_response(
    data: object,
    *,
    list_key: str,
    item_required_key: str = "",
    item_filter: Callable[[object], bool] | None = None,
) -> object:
    """Normalize messy LLM JSON into ``{list_key: [...]}``.

    Handles three common LLM output shapes:
    1. Bare list: ``[{...}, {...}]`` -> ``{list_key: [...]}``
    2. Bare item dict (truncated wrapper): ``{"text": ...}`` -> ``{list_key: [{...}]}``
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
_MD_CODE_FENCE_MARKERS: Final = re.compile(r"^```\w*\s*$", re.MULTILINE)
_MD_HORIZONTAL_RULE: Final = re.compile(r"^---+\s*$", re.MULTILINE)
_MD_BLOCKQUOTE: Final = re.compile(r"^>\s+", re.MULTILINE)
_FAKE_TOOL_XML_RE: Final = re.compile(
    r"</?(?:tool_?\w*|function(?:_call)?|parameter\b)[^>]*>.*?(?:</(?:tool_?\w*|function(?:_call)?|parameter\b)[^>]*>|$)",
    re.IGNORECASE | re.DOTALL,
)
_FAKE_TOOL_BRACKET_RE: Final = re.compile(
    r"\[/?tool_?\w*\].*?(?:\[/tool_?\w*\]|$)",
    re.IGNORECASE | re.DOTALL,
)
_LEAKED_SYNTHESIS_RE: Final = re.compile(
    r"\[Research (?:complete|:\s*\d+ facts)\.?.*?\]\s*", re.DOTALL
)
_LEAKED_RESEARCH_NOTES_HEADER_RE: Final = re.compile(
    r"^(?:#+\s*)?Research notes\s*(?:\([^)]*\))?\s*$", re.MULTILINE | re.IGNORECASE
)
_LEAKED_STEPS_HEADER_RE: Final = re.compile(
    r"^(?:#+\s*)?Steps taken\s*$", re.MULTILINE | re.IGNORECASE
)
_BARE_TOOL_CALL_RE: Final = re.compile(
    r"^\s*(?:recall_memory|web_research|integrate_knowledge)\s*\(.*?\)\s*$",
    re.MULTILINE,
)
_LEAKED_BELIEF_RE: Final = re.compile(r"^.+ — valence:\s*[+-]?\d.*$", re.MULTILINE)
_LEAKED_STEP_HISTORY_RE: Final = re.compile(
    r"^\d+\.\s*(?:recall_memory|web_research|integrate_knowledge)\(.*?\)\s*→.*$",
    re.MULTILINE,
)
_LEAKED_PLAN_RE: Final = re.compile(r"^Research Plan:\s*$", re.MULTILINE)
_LEAKED_PLAN_STEP_RE: Final = re.compile(
    r"^\d+\.\s*(?:Web research on|Integrate findings into|Recall memory|Look up in memory)\s.*$",
    re.MULTILINE | re.IGNORECASE,
)
_LEAKED_RESEARCHING_RE: Final = re.compile(r"^Researching\s+\S.*\.\.\.\s*$", re.MULTILINE)
_LEAKED_MEMORY_META_RE: Final = re.compile(
    r"^(?:Note:\s*)?Memory recall (?:did not|returned).*$", re.MULTILINE | re.IGNORECASE
)
_LEAKED_INTERNAL_HEADER_RE: Final = re.compile(
    r"^(?:#+\s*)?(?:Prior Knowledge|Research Findings|Your Plan|Your Beliefs|Actions Taken)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
_LEAKED_EPISODE_REF_RE: Final = re.compile(r"^-\s*\[\d{4}-\d{2}-\d{2}\]\s+.*$", re.MULTILINE)
_LEAKED_SEMANTIC_MARKER_RE: Final = re.compile(
    r"^\[semantic/\w+\]\s*.*$", re.MULTILINE | re.IGNORECASE
)
_LEAKED_ESS_METADATA_RE: Final = re.compile(r"\[?ESS[=:]\s*\d+\.?\d*.*?\]?", re.IGNORECASE)
_LEAKED_ITERATION_RE: Final = re.compile(
    r"^\[?Iteration\s+\d+(?:\s+of\s+\d+)?\]?\s*$", re.MULTILINE | re.IGNORECASE
)


def strip_markdown(text: str) -> str:
    """Remove markdown bold, headers, code fences, fake tool-call XML, and leaked prompts."""
    text = _MD_CODE_FENCE_MARKERS.sub("", text)
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_HEADER.sub("", text)
    text = _MD_HORIZONTAL_RULE.sub("", text)
    text = _MD_BLOCKQUOTE.sub("", text)
    text = _FAKE_TOOL_XML_RE.sub("", text)
    text = _FAKE_TOOL_BRACKET_RE.sub("", text)
    text = _LEAKED_SYNTHESIS_RE.sub("", text)
    text = _LEAKED_RESEARCH_NOTES_HEADER_RE.sub("", text)
    text = _LEAKED_STEPS_HEADER_RE.sub("", text)
    text = _BARE_TOOL_CALL_RE.sub("", text)
    text = _LEAKED_BELIEF_RE.sub("", text)
    text = _LEAKED_STEP_HISTORY_RE.sub("", text)
    text = _LEAKED_PLAN_RE.sub("", text)
    text = _LEAKED_PLAN_STEP_RE.sub("", text)
    text = _LEAKED_RESEARCHING_RE.sub("", text)
    text = _LEAKED_MEMORY_META_RE.sub("", text)
    text = _LEAKED_INTERNAL_HEADER_RE.sub("", text)
    text = _LEAKED_EPISODE_REF_RE.sub("", text)
    text = _LEAKED_SEMANTIC_MARKER_RE.sub("", text)
    text = _LEAKED_ESS_METADATA_RE.sub("", text)
    text = _LEAKED_ITERATION_RE.sub("", text)
    return text.strip()
