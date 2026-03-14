from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import config

log = logging.getLogger(__name__)

# Serialize all LLM HTTP calls. llama.cpp / single-GPU servers process one request
# at a time; concurrent calls pile up in the queue and trigger cascading timeouts.
# Embedding calls are excluded (different endpoint, much faster).
_LLM_SEMAPHORE: threading.Semaphore = threading.Semaphore(1)

_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_EMPTY_MAPPING: dict[str, object] = {}
_THINKING_ANSWER_MARKERS = ("Final Output:", "Output:", "Answer:", "Final Answer:", "Response:")


def _normalize_schema_notation(text: str) -> str:
    """Coerce common model schema-template patterns into valid JSON.

    Some models (especially quantized variants) output schema notation like
      "field": "OPTION_A" | "OPTION_B" | "OPTION_C"
      "field": float
      "field": string
      "field": 0.0-1.0
      {"field": "value", ...}         ← ellipsis placeholder
      {"field": [...]}                ← array placeholder
      "value/alternative"             ← slash-separated alternatives
    instead of filling in actual values.  We normalize these to the first
    concrete option or a sensible default so downstream parsing succeeds.
    """
    # Bare ellipsis between delimiters  →  remove (handles {}, {key:val,...}, [...])
    text = re.sub(r",\s*\.\.\.\s*(?=[}\]])", "", text)  # trailing ", ..."
    text = re.sub(r"(?<=[{,\[])\s*\.\.\.\s*(?=[,}\]])", "", text)  # inner "..."
    # Standalone "..." value in a string  →  ""
    text = re.sub(r':\s*"\.\.\."', ': ""', text)
    # Quoted ellipsis as a JSON key placeholder  {"..."}  {"...", ...}  →  strip it
    text = re.sub(r'\{\s*"\.\.\."(?:\s*,)?\s*', "{", text)
    # Array placeholder  [...]  →  []
    text = re.sub(r"\[\s*\.\.\.\s*\]", "[]", text)
    # Angle-bracket placeholders  <float>  <string>  <int>  →  sensible defaults
    text = re.sub(r":\s*<float>", ": 0.5", text)
    text = re.sub(r":\s*<string>", ': ""', text)
    text = re.sub(r":\s*<int>", ": 0", text)
    # "X" | "Y" | "Z"  →  "X"  (keep first enum option)
    text = re.sub(r'"([^"]+)"\s*(?:\|\s*"[^"]*"\s*)+', r'"\1"', text)
    # Unquoted pipe-separated options: RETAIN | DECAY | FORGET → "RETAIN"
    # (matches after colon, before comma/brace)
    text = re.sub(r":\s*([A-Z_]+)\s*(?:\|\s*[A-Z_]+\s*)+(?=[,}\s])", r': "\1"', text)
    # "X" or "Y" or "Z"  →  "X"  (same pattern with English "or")
    text = re.sub(r'"([^"]+)"\s*(?:or\s+"[^"]*"\s*)+', r'"\1"', text)
    # "option1/option2/..."  →  "option1"  (slash-separated alternatives in strings)
    text = re.sub(r'"([^"/]+)(?:/[^"/]+)+"', r'"\1"', text)
    # bare type names as values  →  sensible defaults
    text = re.sub(r":\s*\bfloat\b", ": 0.5", text)
    text = re.sub(r":\s*\bint\b", ": 0", text)
    text = re.sub(r":\s*\bbool\b", ": false", text)
    text = re.sub(r":\s*\bstring\b", ': ""', text)
    text = re.sub(r":\s*\bnull\b\s*(?=[,}])", ": null", text)
    # range notation  "0.0-1.0"  or  "-1.0 to +1.0"  →  "0.5"
    text = re.sub(r'"[-+]?\d+\.?\d*\s*(?:to|-)\s*[-+]?\d+\.?\d*"', '"0.5"', text)
    # bare range as value  :  0.0-1.0  →  : 0.5
    text = re.sub(r":\s*[-+]?\d+\.?\d*\s*(?:to|-)\s*[-+]?\d+\.?\d*(?=[,}\s])", ": 0.5", text)
    # trailing ellipsis after a digit  0.3...  →  0.3  (template placeholder in numbers)
    text = re.sub(r"(\d)\.\.\.", r"\1", text)
    return text


def extract_last_json_object(text: str) -> dict[str, object] | None:
    """Find the last valid JSON object in text, scanning left-to-right with raw_decode.

    Uses json.JSONDecoder.raw_decode to parse JSON starting at each '{' position and
    return where it ends. Scanning all '{' positions and keeping the last successful
    parse means a model self-correction ("actually: {...}") produces the corrected value.
    Handles markdown fences, trailing prose, nested braces, and multiple JSON blocks.
    Also normalizes common schema-template patterns output by quantized thinking models.
    """
    stripped = text.strip()
    # Strip markdown code fences
    cleaned = stripped.replace("```json", "").replace("```", "").strip()
    # +0.42 and +1 are not valid JSON; strip leading + from numeric literals
    cleaned = re.sub(r":\s*\+(\d)", r": \1", cleaned)

    decoder = json.JSONDecoder()

    def _try_parse(source: str) -> dict[str, object] | None:
        last_good: dict[str, object] | None = None
        i = 0
        while i < len(source):
            if source[i] != "{":
                i += 1
                continue
            try:
                obj, end = decoder.raw_decode(source, i)
                if isinstance(obj, dict):
                    last_good = obj
                i = end
            except json.JSONDecodeError:
                i += 1
        return last_good

    result = _try_parse(cleaned)
    if result is not None:
        return result
    # Second pass: normalize schema-template notation and retry
    normalized = _normalize_schema_notation(cleaned)
    if normalized != cleaned:
        result = _try_parse(normalized)
    if result is not None:
        return result
    # Third pass: bare integer array → {"ranking": [...]} for listwise reranker
    try:
        arr = json.loads(cleaned)
        if isinstance(arr, list) and arr and all(isinstance(x, int) for x in arr):
            return {"ranking": arr}
    except json.JSONDecodeError:
        pass
    return None


def _extract_answer_from_reasoning(reasoning: str) -> str:
    """Extract answer from thinking model reasoning_content field.

    Thinking models like Qwen 3.5 A3B output their chain-of-thought in reasoning_content
    and may leave content empty when max_tokens is insufficient. This function attempts
    to extract the final answer portion from the reasoning.

    Priority order:
    1. Text after a known answer marker (captures both single-line and multi-line answers)
    2. Last JSON object or array block in the reasoning
    3. Last non-empty line (plain-text fallback)
    """
    # 1. Marker-based extraction — take everything after the last marker
    for marker in _THINKING_ANSWER_MARKERS:
        idx = reasoning.lower().rfind(marker.lower())
        if idx != -1:
            answer = reasoning[idx + len(marker) :].strip()
            if answer:
                return answer
    # 2. Last balanced JSON object/array in the reasoning
    for opener, closer in (("{", "}"), ("[", "]")):
        end = reasoning.rfind(closer)
        if end != -1:
            depth = 0
            for i in range(end, -1, -1):
                if reasoning[i] == closer:
                    depth += 1
                elif reasoning[i] == opener:
                    depth -= 1
                    if depth == 0:
                        candidate = reasoning[i : end + 1].strip()
                        if len(candidate) > 2:
                            return candidate
    # 3. Last non-empty line
    for line in reversed(reasoning.strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Normalized chat completion payload used by all model call sites."""

    text: str
    input_tokens: int
    output_tokens: int
    raw: dict[str, object]


def _to_nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


def _post_json(
    path: str, payload: Mapping[str, object], base_url: str = "", api_key: str = ""
) -> dict[str, object]:
    body = json.dumps(payload).encode("utf-8")
    base = (base_url or config.BASE_URL).rstrip("/")
    normalized = path if path.startswith("/") else f"/{path}"
    url = f"{base}{normalized}"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    key = api_key or config.API_KEY
    if key:
        headers["Authorization"] = f"Bearer {key}"

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        request = Request(
            url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                decoded = response.read().decode("utf-8")
            parsed = json.loads(decoded)
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError("Provider returned a non-object JSON payload")
        except HTTPError as exc:
            status = int(exc.code)
            if attempt < max_attempts and status in _RETRYABLE_HTTP_STATUSES:
                wait = 1.5**attempt
                log.warning(
                    "Provider HTTP %d on attempt %d/%d; retrying in %.1fs",
                    status,
                    attempt,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue
            detail = ""
            try:
                detail = exc.read().decode("utf-8")
            except Exception:
                detail = "<failed to read error payload>"
            raise RuntimeError(f"Provider HTTP {status}: {detail}") from exc
        except URLError as exc:
            if attempt < max_attempts:
                wait = 1.5**attempt
                log.warning(
                    "Provider network error on attempt %d/%d; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"Provider network error: {exc}") from exc
        except (TimeoutError, ConnectionError, OSError) as exc:
            if attempt < max_attempts:
                wait = 1.5**attempt
                log.warning(
                    "Provider transport error on attempt %d/%d; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"Provider transport error: {exc}") from exc
    raise RuntimeError("Provider request failed after retries")


def _message_content_text(message: object) -> str:
    if isinstance(message, str):
        return message
    if not isinstance(message, list):
        return ""
    parts: list[str] = []
    for item in message:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def chat_completion(
    *,
    model: str,
    messages: Sequence[Mapping[str, str]],
    max_tokens: int,
    temperature: float = -1.0,
    tools: Sequence[Mapping[str, object]] = (),
    tool_choice: Mapping[str, object] = _EMPTY_MAPPING,
    disable_thinking: bool = True,
) -> ChatResult:
    payload: dict[str, object] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": max_tokens,
    }
    if temperature >= 0.0:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = [dict(tool) for tool in tools]
    if tool_choice:
        payload["tool_choice"] = dict(tool_choice)
    if disable_thinking:
        # Qwen3 and similar thinking models: suppress chain-of-thought so the
        # JSON answer lands directly in `content` without token budget waste.
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    with _LLM_SEMAPHORE:
        raw = _post_json("/chat/completions", payload)
    text = ""
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                text = _message_content_text(message.get("content", ""))
                if not text:
                    reasoning = message.get("reasoning_content")
                    if isinstance(reasoning, str) and reasoning:
                        text = _extract_answer_from_reasoning(reasoning)
    usage = raw.get("usage")
    input_tokens = 0
    output_tokens = 0
    if isinstance(usage, dict):
        input_tokens = _to_nonnegative_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        output_tokens = _to_nonnegative_int(
            usage.get("completion_tokens", usage.get("output_tokens", 0))
        )
    return ChatResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw=raw,
    )


def embed(
    *,
    model: str,
    texts: list[str],
    dimensions: int = 0,
) -> list[list[float]]:
    payload: dict[str, object] = {"model": model, "input": texts}
    if dimensions > 0 and config.EMBEDDING_SEND_DIMENSIONS:
        payload["dimensions"] = dimensions
    api_key = config.EMBEDDING_API_KEY or config.API_KEY
    raw = _post_json("/embeddings", payload, config.EMBEDDING_BASE_URL, api_key)
    data = raw.get("data")
    if not isinstance(data, list):
        raise RuntimeError("Embedding response is missing `data` array")
    sorted_rows = sorted(
        (row for row in data if isinstance(row, dict)),
        key=lambda row: int(row.get("index", 0)),
    )
    vectors: list[list[float]] = []
    for row in sorted_rows:
        embedding = row.get("embedding")
        if not isinstance(embedding, list):
            continue
        vectors.append([float(v) for v in embedding])
    return vectors


def parse_json_object(text: str) -> dict[str, object]:
    """Extract a JSON object from LLM response text.

    Tries direct parse first (fast path), then falls through to extract_last_json_object
    which handles markdown fences, multiple JSON blocks, trailing prose, and nested braces.
    """
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return extract_last_json_object(stripped) or {}


def extract_tool_call_arguments(
    raw_payload: Mapping[str, object], function_name: str
) -> dict[str, object]:
    choices = raw_payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return {}
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        return {}
    message = first_choice.get("message")
    if not isinstance(message, dict):
        return {}
    tool_calls = message.get("tool_calls")
    if not isinstance(tool_calls, list):
        return {}
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue
        if function.get("name") != function_name:
            continue
        arguments = function.get("arguments")
        if isinstance(arguments, dict):
            return dict(arguments)
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return dict(parsed)
    return {}
