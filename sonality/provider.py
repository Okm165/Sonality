from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import config

log = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUSES: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_EMPTY_MAPPING: Final[Mapping[str, object]] = {}
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
# Qwen 3 sometimes outputs asterisk-prefixed reasoning without <think> tags
_ASTERISK_THOUGHT_RE: Final = re.compile(r"^\s*\*[^*\n]+\*?\s*$", re.MULTILINE)
_SEMANTIC_MARKER_RE: Final = re.compile(r"\[semantic/[^\]]*\][^\n]*\n?", re.IGNORECASE)


class StreamChunk(NamedTuple):
    """A single streaming chunk with content and reasoning deltas."""

    content: str
    reasoning: str


def _normalize_schema_notation(text: str) -> str:
    """Coerce common model schema-template patterns into valid JSON."""
    text = re.sub(r':\s*"\.\.\."', ': ""', text)
    text = re.sub(r'\{\s*"\.\.\."(?:\s*,)?\s*', "{", text)
    text = re.sub(r",\s*\.\.\.\s*(?=[}\]])", "", text)
    text = re.sub(r"(?<=[{,\[])\s*\.\.\.\s*(?=[,}\]])", "", text)
    text = re.sub(r',\s*\.\.\.\s*(?=")', ", ", text)
    text = re.sub(r'"\s*\.\.\.\s*(?=")', '", ', text)
    text = re.sub(r"\s+\.\.\.\s*(?=[}\]])", "", text)
    text = re.sub(r"\[\s*\.\.\.\s*\]", "[]", text)
    text = re.sub(r":\s*<float>", ": 0.5", text)
    text = re.sub(r":\s*<string>", ': ""', text)
    text = re.sub(r":\s*<int>", ": 0", text)
    text = re.sub(r'"([^"]+)"\s*(?:\|\s*"[^"]*"\s*)+', r'"\1"', text)
    text = re.sub(r":\s*([A-Z_]+)\s*(?:\|\s*[A-Z_]+\s*)+(?=[,}\s])", r': "\1"', text)
    text = re.sub(r'"([^"]+)"\s*(?:or\s+"[^"]*"\s*)+', r'"\1"', text)
    text = re.sub(r'"([^"/]+)(?:/[^"/]+)+"', r'"\1"', text)
    text = re.sub(r":\s*\bfloat\b", ": 0.5", text)
    text = re.sub(r":\s*\bint\b", ": 0", text)
    text = re.sub(r":\s*\bbool\b", ": false", text)
    text = re.sub(r":\s*\bstring\b", ': ""', text)
    text = re.sub(r":\s*\bnull\b\s*(?=[,}])", ": null", text)
    text = re.sub(r'"[-+]?\d+\.?\d*\s*(?:to|-)\s*[-+]?\d+\.?\d*"', '"0.5"', text)
    text = re.sub(r":\s*[-+]?\d+\.?\d*\s*(?:to|-)\s*[-+]?\d+\.?\d*(?=[,}\s])", ": 0.5", text)
    text = re.sub(r"(\d)\.\.\.", r"\1", text)
    text = re.sub(r":\s*\.\.\.\s*(?=[,}])", ": 0.0", text)
    return text


def extract_last_json_object(text: str) -> dict[str, object] | None:
    """Find the largest (by character span) valid JSON object in text."""
    stripped = text.strip()
    cleaned = stripped.replace("```json", "").replace("```", "").strip()
    cleaned = re.sub(r":\s*\+(\d)", r": \1", cleaned)
    decoder = json.JSONDecoder()

    def _try_parse(source: str) -> dict[str, object] | None:
        candidates: list[tuple[int, int, dict[str, object]]] = []
        i = 0
        while i < len(source):
            if source[i] != "{":
                i += 1
                continue
            try:
                obj, end = decoder.raw_decode(source, i)
                if isinstance(obj, dict):
                    candidates.append((i, end, obj))
                i = end
            except json.JSONDecodeError:
                i += 1
        return max(candidates, key=lambda c: c[1] - c[0])[2] if candidates else None

    result = _try_parse(cleaned)
    if result is not None:
        return result
    normalized = _normalize_schema_notation(cleaned)
    if normalized != cleaned and (result := _try_parse(normalized)) is not None:
        return result
    try:
        arr = json.loads(cleaned)
        if isinstance(arr, list) and arr and all(isinstance(x, int) for x in arr):
            return {"ranking": arr}
    except json.JSONDecodeError:
        pass
    return None


def _extract_answer_from_reasoning(reasoning: str) -> str:
    """Extract answer from thinking model reasoning_content field."""
    cleaned = strip_thinking_trace(reasoning)
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


def strip_thinking_trace(text: str) -> str:
    """Remove chain-of-thought traces from model-visible output."""
    result = _THINK_BLOCK_RE.sub("", text)
    result = _THINK_CODE_BLOCK_RE.sub("", result)
    result = _ASTERISK_THOUGHT_RE.sub("", result)
    result = _SEMANTIC_MARKER_RE.sub("", result)
    return result.strip()


def _message_content_text(message: object) -> str:
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


def _to_nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Normalized chat completion payload."""

    text: str
    input_tokens: int
    output_tokens: int
    raw: dict[str, object]


class LLMProvider:
    """LLM provider with per-instance config and semaphore.

    Each instance has its own:
    - base_url, api_key, timeout configuration
    - threading.Semaphore for serializing calls (single-GPU servers)

    Usage:
        provider = LLMProvider("https://api.openai.com/v1", "sk-...")
        result = provider.chat_completion(model="gpt-4", messages=[...], max_tokens=1000)
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._semaphore = threading.Semaphore(config.LLM_CONCURRENCY)

    def _post_json(self, path: str, payload: Mapping[str, object]) -> dict[str, object]:
        """POST JSON to the provider with retries."""
        body = json.dumps(payload).encode("utf-8")
        normalized = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{normalized}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        log.debug("POST %s (%d bytes)", normalized, len(body))
        for attempt in range(1, 4):
            request = Request(url, data=body, headers=headers, method="POST")
            t0 = time.time()
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                elapsed = time.time() - t0
                if isinstance(parsed, dict):
                    log.debug("POST %s OK in %.1fs", normalized, elapsed)
                    return parsed
                raise RuntimeError("Provider returned a non-object JSON payload")
            except HTTPError as exc:
                elapsed = time.time() - t0
                if attempt < 3 and int(exc.code) in _RETRYABLE_HTTP_STATUSES:
                    log.warning(
                        "POST %s HTTP %d (attempt %d/3, %.1fs), retrying",
                        normalized,
                        exc.code,
                        attempt,
                        elapsed,
                    )
                    time.sleep(config.LLM_BACKOFF_BASE**attempt)
                    continue
                detail = exc.read().decode("utf-8") if exc.fp else ""
                log.error(
                    "POST %s HTTP %d after %.1fs: %s", normalized, exc.code, elapsed, detail[:200]
                )
                raise RuntimeError(f"Provider HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                elapsed = time.time() - t0
                reason = getattr(exc, "reason", None)
                if isinstance(reason, TimeoutError):
                    log.error("POST %s timeout after %.1fs", normalized, elapsed)
                    raise RuntimeError(f"Provider timeout: {reason}") from exc
                reason_str = str(reason) if reason else str(exc)
                if "name resolution" in reason_str.lower():
                    log.error("POST %s DNS failure: %s", normalized, reason_str)
                    raise RuntimeError(f"Provider network error: {exc}") from exc
                if attempt < 3:
                    log.warning(
                        "POST %s network error (attempt %d/3, %.1fs): %s",
                        normalized,
                        attempt,
                        elapsed,
                        reason_str,
                    )
                    time.sleep(config.LLM_BACKOFF_BASE**attempt)
                    continue
                log.error("POST %s network error after 3 attempts: %s", normalized, reason_str)
                raise RuntimeError(f"Provider network error: {exc}") from exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                elapsed = time.time() - t0
                if attempt < 3:
                    log.warning(
                        "POST %s transport error (attempt %d/3, %.1fs): %s",
                        normalized,
                        attempt,
                        elapsed,
                        exc,
                    )
                    time.sleep(config.LLM_BACKOFF_BASE**attempt)
                    continue
                log.error("POST %s transport error after 3 attempts: %s", normalized, exc)
                raise RuntimeError(f"Provider transport error: {exc}") from exc
        raise RuntimeError("Provider request failed after retries")

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, object]],
        max_tokens: int,
        temperature: float = -1.0,
        tools: Sequence[Mapping[str, object]] = (),
        tool_choice: str | Mapping[str, object] = _EMPTY_MAPPING,
        enable_thinking: bool = False,
    ) -> ChatResult:
        """Synchronous chat completion with semaphore protection."""
        log.info(
            "LLM call: model=%s msgs=%d max_tokens=%d tools=%d temp=%.1f",
            model,
            len(messages),
            max_tokens,
            len(tools),
            temperature,
        )
        payload: dict[str, object] = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if temperature >= 0.0:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [dict(t) for t in tools]
        if tool_choice:
            payload["tool_choice"] = (
                tool_choice if isinstance(tool_choice, str) else dict(tool_choice)
            )

        t0 = time.time()
        with self._semaphore:
            raw = self._post_json("/chat/completions", payload)

        text = ""
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict) and isinstance(message := first.get("message"), dict):
                raw_content = _message_content_text(message.get("content", ""))
                raw_reasoning = message.get("reasoning_content", "")
                log.debug(
                    "LLM_RAW content=%d chars, reasoning=%d chars, finish=%s",
                    len(raw_content),
                    len(raw_reasoning) if raw_reasoning else 0,
                    first.get("finish_reason", "unknown"),
                )
                text = strip_thinking_trace(raw_content)
                if not text and isinstance(raw_reasoning, str) and raw_reasoning:
                    log.debug("LLM_RAW extracting from reasoning_content (content was empty)")
                    text = _extract_answer_from_reasoning(raw_reasoning)

        usage = raw.get("usage")
        input_tokens = output_tokens = 0
        if isinstance(usage, dict):
            input_tokens = _to_nonnegative_int(
                usage.get("prompt_tokens", usage.get("input_tokens", 0))
            )
            output_tokens = _to_nonnegative_int(
                usage.get("completion_tokens", usage.get("output_tokens", 0))
            )

        elapsed = time.time() - t0
        log.info(
            "LLM done: in=%d out=%d text=%d elapsed=%.1fs",
            input_tokens,
            output_tokens,
            len(text),
            elapsed,
        )
        return ChatResult(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens, raw=raw
        )

    def semaphore_idle(self) -> bool:
        """Return True if the semaphore is free (advisory, subject to TOCTOU)."""
        acquired = self._semaphore.acquire(blocking=False)
        if acquired:
            self._semaphore.release()
            return True
        return False


# --- Default provider instance for sonality modules ---
default_provider: LLMProvider = LLMProvider(
    config.BASE_URL, config.API_KEY, config.LLM_REQUEST_TIMEOUT
)

_interaction_event = threading.Event()

interaction_in_progress = _interaction_event.is_set


@contextlib.contextmanager
def interaction_active() -> Iterator[None]:
    _interaction_event.set()
    try:
        yield
    finally:
        _interaction_event.clear()


def llm_semaphore_idle() -> bool:
    """Return True if the default LLM semaphore is free AND no interaction is active."""
    return not interaction_in_progress() and default_provider.semaphore_idle()


_ANALYSIS_COMPLETE_RE: Final = re.compile(r"<analysis>[\s\S]*?</analysis>", re.IGNORECASE)
_ANALYSIS_UNCLOSED_RE: Final = re.compile(r"<analysis>[\s\S]*$", re.IGNORECASE)


def _strip_analysis_block(text: str) -> str:
    """Strip <analysis>...</analysis> scratchpad blocks from LLM output."""
    text = _ANALYSIS_COMPLETE_RE.sub("", text)
    text = _ANALYSIS_UNCLOSED_RE.sub("", text)
    return text.strip()


def decode_llm_json(text: str) -> dict[str, object] | list[object]:
    """Extract JSON (object or array) from LLM response text.

    Strips markdown fences and <analysis> blocks, attempts whole-text parse
    first (preserving bare arrays), then falls back to extract_last_json_object
    for messy output. Raises ValueError if no JSON can be extracted.
    """
    cleaned = _strip_analysis_block(text)
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("Empty LLM response")
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, (dict, list)):
            return parsed
    except json.JSONDecodeError:
        pass
    # Fix double-escaped quotes: \\" → " inside JSON strings
    if '\\"' in cleaned:
        try:
            fixed = cleaned.replace('\\"', '"')
            parsed = json.loads(fixed)
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


class ParsedToolCall(NamedTuple):
    """Parsed tool call from an LLM chat-completion payload."""

    name: str
    args: dict[str, object]
    id: str


def _get_raw_tool_calls(raw: Mapping[str, object]) -> list[dict[str, object]]:
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
    for call in _get_raw_tool_calls(raw):
        func = call.get("function")
        if not isinstance(func, dict):
            continue
        name = str(func.get("name", ""))
        raw_args = func.get("arguments", "{}")
        args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        if not isinstance(args, dict):
            args = {}
        parsed.append(ParsedToolCall(name=name, args=args, id=str(call.get("id", ""))))
    return parsed


def extract_tool_call_arguments(
    raw_payload: Mapping[str, object], function_name: str
) -> dict[str, object]:
    """Extract arguments for a specific tool call by function name."""
    for call in _get_raw_tool_calls(raw_payload):
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
