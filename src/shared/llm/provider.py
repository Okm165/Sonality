"""LLM HTTP transport: provider, chat completion, concurrency control.

All JSON parsing and thinking trace removal live in ``parse``.  This module
handles only HTTP communication.  Each ``LLMProvider`` instance is
self-contained: base_url, api_key, timeout, retry budget, and concurrency
semaphore are all per-instance — no module-level globals or config coupling.
"""

from __future__ import annotations

import json
import random
import socket
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import structlog

from ..errors import LLMParseError, ProviderHTTPError, ProviderTransportError
from .parse import clean_completion, message_content_text, to_nonnegative_int

log = structlog.get_logger(__name__)

_RETRYABLE_HTTP_STATUSES: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_EMPTY_MAPPING: Final[Mapping[str, object]] = {}

__all__ = [
    "ChatResult",
    "LLMProvider",
    "StreamChunk",
]


class StreamChunk(NamedTuple):
    """A single streaming text chunk from the agentic loop."""

    content: str


@dataclass(frozen=True, slots=True)
class ChatResult:
    """Normalized chat completion payload."""

    text: str
    input_tokens: int
    output_tokens: int
    raw: dict[str, object]
    finish_reason: str = ""


class LLMProvider:
    """LLM provider with per-instance config and semaphore.

    Each instance has its own base_url, api_key, timeout, retry/backoff
    budget, and a threading.Semaphore for serializing calls (single-GPU
    servers or rate-limited APIs).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        timeout: int = 300,
        *,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        concurrency: int = 1,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self._semaphore = threading.Semaphore(concurrency)

    # -----------------------------------------------------------------
    # HTTP transport
    # -----------------------------------------------------------------

    def _post_json(self, path: str, payload: Mapping[str, object]) -> dict[str, object]:
        """POST JSON to the provider with retries on transient failures."""
        body = json.dumps(payload).encode("utf-8")
        normalized = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{normalized}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        max_attempts = self.max_retries
        for attempt in range(1, max_attempts + 1):
            request = Request(url, data=body, headers=headers, method="POST")
            t0 = time.time()
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    raw_bytes = response.read()
                    try:
                        parsed = json.loads(raw_bytes.decode("utf-8"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as decode_exc:
                        elapsed = time.time() - t0
                        log.error(
                            "http_post_decode_failed",
                            path=normalized,
                            elapsed_s=round(elapsed, 1),
                            error=str(decode_exc),
                        )
                        raise LLMParseError(f"Response decode error: {decode_exc}") from decode_exc
                elapsed = time.time() - t0
                if isinstance(parsed, dict):
                    return parsed
                raise LLMParseError("Provider returned a non-object JSON payload")
            except HTTPError as exc:
                elapsed = time.time() - t0
                retryable = int(exc.code) in _RETRYABLE_HTTP_STATUSES
                if retryable and attempt < max_attempts:
                    self._retry_wait(normalized, f"HTTP {exc.code}", attempt, max_attempts, elapsed)
                    continue
                detail = exc.read().decode("utf-8") if exc.fp else ""
                log.error(
                    "http_post_http_error",
                    path=normalized,
                    http_status=int(exc.code),
                    elapsed_s=round(elapsed, 1),
                    detail=detail[:200],
                )
                raise ProviderHTTPError(int(exc.code), detail) from exc
            except URLError as exc:
                elapsed = time.time() - t0
                reason = getattr(exc, "reason", None)
                if isinstance(reason, socket.gaierror):
                    raise ProviderTransportError(f"DNS failure: {reason}") from exc
                reason_str = str(reason) if reason else str(exc)
                if attempt < max_attempts:
                    self._retry_wait(
                        normalized,
                        f"network: {reason_str}",
                        attempt,
                        max_attempts,
                        elapsed,
                    )
                    continue
                raise ProviderTransportError(f"Network error: {reason_str}") from exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                elapsed = time.time() - t0
                if attempt < max_attempts:
                    self._retry_wait(
                        normalized, f"transport: {exc}", attempt, max_attempts, elapsed
                    )
                    continue
                raise ProviderTransportError(f"Transport error: {exc}") from exc
        raise ProviderTransportError("Request failed after retries")

    def _retry_wait(
        self, path: str, reason: str, attempt: int, max_attempts: int, elapsed: float
    ) -> None:
        base_wait = self.backoff_base**attempt
        jitter = random.uniform(0, base_wait * 0.5)
        wait = base_wait + jitter
        log.warning(
            "http_post_retry_scheduled",
            path=path,
            reason=reason,
            attempt=attempt,
            max_attempts=max_attempts,
            elapsed_s=round(elapsed, 1),
            retry_in_s=round(wait, 1),
        )
        time.sleep(wait)

    # -----------------------------------------------------------------
    # Chat completion
    # -----------------------------------------------------------------

    def chat_completion(
        self,
        *,
        model: str,
        messages: Sequence[Mapping[str, object]],
        max_tokens: int = -1,
        temperature: float = -1.0,
        tools: Sequence[Mapping[str, object]] = (),
        tool_choice: str | Mapping[str, object] = _EMPTY_MAPPING,
    ) -> ChatResult:
        """Synchronous chat completion with semaphore protection."""
        payload: dict[str, object] = {
            "model": model,
            "messages": [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages],
        }
        if max_tokens > 0:
            payload["max_tokens"] = max_tokens
        if temperature >= 0.0:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [dict(t) for t in tools]
        if tool_choice:
            payload["tool_choice"] = (
                tool_choice if isinstance(tool_choice, str) else dict(tool_choice)
            )

        with self._semaphore:
            raw = self._post_json("/chat/completions", payload)

        text = ""
        choices = raw.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict) and isinstance(message := first.get("message"), dict):
                raw_content = message_content_text(message.get("content", ""))
                raw_reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
                if isinstance(raw_reasoning, list):
                    raw_reasoning = " ".join(
                        str(x.get("text", x)) if isinstance(x, dict) else str(x)
                        for x in raw_reasoning
                    )
                elif not isinstance(raw_reasoning, str):
                    raw_reasoning = str(raw_reasoning) if raw_reasoning else ""
                text = clean_completion(raw_content, raw_reasoning)

        usage = raw.get("usage")
        input_tokens = output_tokens = 0
        if isinstance(usage, dict):
            input_tokens = to_nonnegative_int(
                usage.get("prompt_tokens", usage.get("input_tokens", 0))
            )
            output_tokens = to_nonnegative_int(
                usage.get("completion_tokens", usage.get("output_tokens", 0))
            )

        finish_reason = ""
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            fr = choices[0].get("finish_reason")
            if isinstance(fr, str):
                finish_reason = fr

        return ChatResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=raw,
            finish_reason=finish_reason,
        )
