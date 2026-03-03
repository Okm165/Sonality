from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import config

log = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True, slots=True)
class OpenRouterChatResult:
    """Normalized chat completion payload used by runtime and ESS paths."""

    text: str
    input_tokens: int
    output_tokens: int
    raw: dict[str, object]


def _to_nonnegative_int(value: object) -> int:
    """Convert mixed numeric values to a non-negative integer."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


def _request_headers() -> dict[str, str]:
    """Build headers for OpenRouter API requests."""
    if not config.API_KEY:
        raise ValueError("Missing required API config: SONALITY_API_KEY")
    return {
        "Authorization": f"Bearer {config.API_KEY}",
        "Content-Type": "application/json",
    }


def _provider_preferences() -> dict[str, object]:
    """Build provider-routing preferences for OpenRouter chat completions."""
    prefs: dict[str, object] = {
        "allow_fallbacks": config.OPENROUTER_ALLOW_FALLBACKS,
        "zdr": config.OPENROUTER_FORCE_ZDR,
    }
    if config.OPENROUTER_PROVIDER_ORDER:
        prefs["order"] = list(config.OPENROUTER_PROVIDER_ORDER)
    if config.OPENROUTER_DATA_COLLECTION is not None:
        prefs["data_collection"] = config.OPENROUTER_DATA_COLLECTION
    return prefs


def _openrouter_url() -> str:
    """Return OpenRouter chat completions endpoint URL."""
    return f"{config.BASE_URL.rstrip('/')}/v1/chat/completions"


def _post_json(payload: Mapping[str, object]) -> dict[str, object]:
    """POST a JSON payload to OpenRouter with bounded retries."""
    body = json.dumps(payload).encode("utf-8")
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        request = Request(
            _openrouter_url(),
            data=body,
            headers=_request_headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                decoded = response.read().decode("utf-8")
            parsed = json.loads(decoded)
            if isinstance(parsed, dict):
                return parsed
            raise RuntimeError("OpenRouter returned a non-object JSON payload")
        except HTTPError as exc:
            status = int(exc.code)
            if attempt < max_attempts and status in _RETRYABLE_HTTP_STATUSES:
                wait = 1.5**attempt
                log.warning(
                    "OpenRouter HTTP %d on attempt %d/%d; retrying in %.1fs",
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
            raise RuntimeError(f"OpenRouter HTTP {status}: {detail}") from exc
        except URLError as exc:
            if attempt < max_attempts:
                wait = 1.5**attempt
                log.warning(
                    "OpenRouter network error on attempt %d/%d; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenRouter network error: {exc}") from exc
        except (TimeoutError, ConnectionError, OSError) as exc:
            if attempt < max_attempts:
                wait = 1.5**attempt
                log.warning(
                    "OpenRouter transport error on attempt %d/%d; retrying in %.1fs",
                    attempt,
                    max_attempts,
                    wait,
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"OpenRouter transport error: {exc}") from exc

    raise RuntimeError("OpenRouter request failed after retries")


def _message_content_text(message: object) -> str:
    """Extract plain text from OpenRouter message content variants."""
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
    temperature: float | None = None,
    tools: Sequence[Mapping[str, object]] | None = None,
    tool_choice: Mapping[str, object] | None = None,
) -> OpenRouterChatResult:
    """Execute one OpenRouter chat completion with provider policy controls."""
    payload: dict[str, object] = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "max_tokens": max_tokens,
        "provider": _provider_preferences(),
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if tools:
        payload["tools"] = [dict(tool) for tool in tools]
    if tool_choice is not None:
        payload["tool_choice"] = dict(tool_choice)
    raw = _post_json(payload)

    text = ""
    choices = raw.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                text = _message_content_text(message.get("content", ""))
    usage = raw.get("usage")
    input_tokens = 0
    output_tokens = 0
    if isinstance(usage, dict):
        input_tokens = _to_nonnegative_int(usage.get("prompt_tokens", usage.get("input_tokens", 0)))
        output_tokens = _to_nonnegative_int(
            usage.get("completion_tokens", usage.get("output_tokens", 0))
        )
    return OpenRouterChatResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        raw=raw,
    )


def parse_json_object(text: str) -> dict[str, object]:
    """Parse a JSON object from model text, tolerating fenced wrappers."""
    stripped = text.strip()
    if not stripped:
        return {}
    for candidate in (
        stripped,
        stripped.replace("```json", "").replace("```", "").strip(),
    ):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    block_match = _JSON_BLOCK_RE.search(stripped)
    if block_match is None:
        return {}
    try:
        parsed = json.loads(block_match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}
