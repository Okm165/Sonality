from __future__ import annotations

import contextlib
import http.client
import json
import logging
import re
import ssl
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from . import config

log = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_EMPTY_MAPPING: dict[str, object] = {}
_THINKING_ANSWER_MARKERS = ("Final Output:", "Output:", "Answer:", "Final Answer:", "Response:")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_CODE_BLOCK_RE = re.compile(r"```(?:thinking|thought|reasoning)[\s\S]*?```", re.IGNORECASE)
# Qwen 3 sometimes outputs asterisk-prefixed reasoning without <think> tags
# Matches: "*Wait, I need to...", "*thinking*", "*checks instruction*"
_ASTERISK_THOUGHT_RE = re.compile(r"^\s*\*[^*\n]+\*?\s*$", re.MULTILINE)
# Leaked context markers from retrieval system
_SEMANTIC_MARKER_RE = re.compile(r"\[semantic/[^\]]*\][^\n]*\n?", re.IGNORECASE)


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
    """Find the last valid JSON object in text."""
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

        for attempt in range(1, 4):
            request = Request(url, data=body, headers=headers, method="POST")
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                if isinstance(parsed, dict):
                    return parsed
                raise RuntimeError("Provider returned a non-object JSON payload")
            except HTTPError as exc:
                if attempt < 3 and int(exc.code) in _RETRYABLE_HTTP_STATUSES:
                    time.sleep(1.5**attempt)
                    continue
                detail = exc.read().decode("utf-8") if exc.fp else ""
                raise RuntimeError(f"Provider HTTP {exc.code}: {detail}") from exc
            except URLError as exc:
                reason = getattr(exc, "reason", None)
                if isinstance(reason, TimeoutError):
                    raise RuntimeError(f"Provider timeout: {reason}") from exc
                reason_str = str(reason) if reason else str(exc)
                if "name resolution" in reason_str.lower():
                    raise RuntimeError(f"Provider network error: {exc}") from exc
                if attempt < 3:
                    time.sleep(1.5**attempt)
                    continue
                raise RuntimeError(f"Provider network error: {exc}") from exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                if attempt < 3:
                    time.sleep(1.5**attempt)
                    continue
                raise RuntimeError(f"Provider transport error: {exc}") from exc
        raise RuntimeError("Provider request failed after retries")

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
        temperature: float = -1.0,
        tools: Sequence[Mapping[str, object]] = (),
        tool_choice: Mapping[str, object] = _EMPTY_MAPPING,
        enable_thinking: bool = False,
    ) -> ChatResult:
        """Synchronous chat completion with semaphore protection."""
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
            payload["tool_choice"] = dict(tool_choice)

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

        return ChatResult(
            text=text, input_tokens=input_tokens, output_tokens=output_tokens, raw=raw
        )

    def chat_completion_stream(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, str]],
        max_tokens: int,
        temperature: float = -1.0,
        enable_thinking: bool = False,
    ) -> Iterator[StreamChunk]:
        """Stream chat completion, yielding content and reasoning deltas."""
        payload: dict[str, object] = {
            "model": model,
            "messages": [dict(m) for m in messages],
            "max_tokens": max_tokens,
            "stream": True,
            "chat_template_kwargs": {"enable_thinking": enable_thinking},
        }
        if temperature >= 0.0:
            payload["temperature"] = temperature

        parsed = urlparse(self.base_url)
        host = parsed.hostname or "localhost"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        with self._semaphore:
            conn = (
                http.client.HTTPSConnection(
                    host, port, timeout=self.timeout, context=ssl.create_default_context()
                )
                if parsed.scheme == "https"
                else http.client.HTTPConnection(host, port, timeout=self.timeout)
            )
            try:
                conn.request(
                    "POST",
                    (parsed.path or "") + "/chat/completions",
                    json.dumps(payload).encode(),
                    headers,
                )
                resp = conn.getresponse()
                if resp.status != 200:
                    raise RuntimeError(f"Provider HTTP {resp.status}: {resp.read().decode()}")

                buf = ""
                while data := resp.read(4096).decode("utf-8", errors="replace"):
                    buf += data
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        if not (line := line.strip()) or not line.startswith("data:"):
                            continue
                        if (payload_str := line[5:].strip()) == "[DONE]":
                            return
                        try:
                            parsed_data = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        choices = parsed_data.get("choices")
                        if isinstance(choices, list) and choices:
                            delta = choices[0].get("delta", {})
                            content, reasoning = (
                                delta.get("content") or "",
                                delta.get("reasoning_content") or "",
                            )
                            if content or reasoning:
                                yield StreamChunk(content, reasoning)
            finally:
                conn.close()

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


def strip_analysis_block(text: str) -> str:
    """Strip <analysis>...</analysis> scratchpad blocks from LLM output.

    The analysis block is a chain-of-thought drafting area that improves output
    quality but should not be parsed as part of the structured response.
    """
    import re

    # Strip complete <analysis>...</analysis> blocks
    text = re.sub(r"<analysis>[\s\S]*?</analysis>", "", text, flags=re.IGNORECASE)
    # Strip unclosed <analysis> blocks (model cut off mid-thought)
    text = re.sub(r"<analysis>[\s\S]*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def decode_llm_json(text: str) -> dict[str, object] | list[object]:
    """Extract JSON (object or array) from LLM response text.

    Strips markdown fences and <analysis> blocks, attempts whole-text parse
    first (preserving bare arrays), then falls back to extract_last_json_object
    for messy output. Raises ValueError if no JSON can be extracted.
    """
    # Strip analysis scratchpad blocks before JSON extraction
    cleaned = strip_analysis_block(text)
    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
    if not cleaned:
        raise ValueError("Empty LLM response")
    try:
        parsed = json.loads(cleaned)
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


def extract_tool_call_arguments(
    raw_payload: Mapping[str, object], function_name: str
) -> dict[str, object]:
    """Extract tool call arguments from raw completion payload."""
    choices = raw_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first = choices[0]
    if not isinstance(first, dict):
        return {}
    message = first.get("message")
    if not isinstance(message, dict):
        return {}
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return {}
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict) or function.get("name") != function_name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                if isinstance(parsed, dict):
                    return dict(parsed)
            except json.JSONDecodeError:
                continue
    return {}
