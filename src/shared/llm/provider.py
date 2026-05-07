"""LLM HTTP transport: provider, chat completion, concurrency control.

All JSON parsing and thinking trace removal live in ``parse``.  This module
handles only HTTP communication.  Each ``LLMProvider`` instance is
self-contained: base_url, api_key, timeout, retry budget, and concurrency
semaphore are all per-instance — no module-level globals or config coupling.
"""

from __future__ import annotations

import json
import random
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import structlog

from ..errors import LLMParseError, ProviderHTTPError, ProviderTransportError
from .parse import (
    extract_answer_from_reasoning,
    message_content_text,
    strip_thinking_trace,
    to_nonnegative_int,
    unwrap_think_tags,
)

log = structlog.get_logger()

_RETRYABLE_HTTP_STATUSES: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_EMPTY_MAPPING: Final[Mapping[str, object]] = {}

__all__ = [
    "ChatResult",
    "LLMProvider",
    "StreamChunk",
    "supports_thinking",
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


_THINKING_MODELS: Final = frozenset({"qwq", "qwen3", "deepseek-r1", "gemma4", "gemma-4", "glm-4.7"})


def supports_thinking(model: str) -> bool:
    """Return True if the model supports enable_thinking=True."""
    return any(name in model.lower() for name in _THINKING_MODELS)


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
        self._semaphore = threading.Semaphore(max(1, concurrency))

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

        log.debug("http_post", path=normalized, bytes=len(body))
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
                            elapsed=f"{elapsed:.1f}s",
                            error=str(decode_exc),
                        )
                        raise LLMParseError(
                            f"Response decode error: {decode_exc}"
                        ) from decode_exc
                elapsed = time.time() - t0
                if isinstance(parsed, dict):
                    log.debug("http_post_ok", path=normalized, elapsed=f"{elapsed:.1f}s")
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
                    elapsed=f"{elapsed:.1f}s",
                    detail=detail[:200],
                )
                raise ProviderHTTPError(int(exc.code), detail) from exc
            except URLError as exc:
                elapsed = time.time() - t0
                reason = getattr(exc, "reason", None)
                if isinstance(reason, TimeoutError):
                    raise ProviderTransportError(f"Timeout after {elapsed:.1f}s") from exc
                reason_str = str(reason) if reason else str(exc)
                if "name resolution" in reason_str.lower():
                    raise ProviderTransportError(f"DNS failure: {reason_str}") from exc
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
            elapsed=f"{elapsed:.1f}s",
            retry_in=f"{wait:.1f}s",
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
        max_tokens: int,
        temperature: float = -1.0,
        tools: Sequence[Mapping[str, object]] = (),
        tool_choice: str | Mapping[str, object] = _EMPTY_MAPPING,
    ) -> ChatResult:
        """Synchronous chat completion with semaphore protection."""
        log.info(
            "llm_completion_requested",
            model=model,
            message_count=len(messages),
            max_tokens=max_tokens,
            tool_count=len(tools),
            temperature=temperature if temperature >= 0 else "default",
        )
        payload: dict[str, object] = {
            "model": model,
            "messages": [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages],
            "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": supports_thinking(model)},
        }
        if temperature >= 0.0:
            payload["temperature"] = temperature
        if tools:
            payload["tools"] = [dict(t) for t in tools]
        if tool_choice:
            payload["tool_choice"] = (
                tool_choice if isinstance(tool_choice, str) else dict(tool_choice)
            )

        with self._semaphore:
            t0 = time.time()
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
                log.debug(
                    "llm_raw_response",
                    content_chars=len(raw_content),
                    reasoning_chars=len(raw_reasoning),
                    finish_reason=first.get("finish_reason", "unknown"),
                )
                text = strip_thinking_trace(raw_content)
                if not text and raw_content.strip():
                    unwrapped = unwrap_think_tags(raw_content)
                    text = extract_answer_from_reasoning(unwrapped)
                    if text:
                        log.info(
                            "thinking_wrap_recovered",
                            chars=len(text),
                        )
                    else:
                        log.warning(
                            "thinking_wrap_extract_failed",
                            raw_chars=len(raw_content),
                            unwrapped_chars=len(unwrapped),
                        )
                if not text and raw_reasoning:
                    text = extract_answer_from_reasoning(raw_reasoning)
                    if text:
                        log.info("recovered_from_reasoning", chars=len(text))
                    else:
                        log.warning(
                            "reasoning_extract_failed",
                            reasoning_chars=len(raw_reasoning),
                        )

        usage = raw.get("usage")
        input_tokens = output_tokens = 0
        if isinstance(usage, dict):
            input_tokens = to_nonnegative_int(
                usage.get("prompt_tokens", usage.get("input_tokens", 0))
            )
            output_tokens = to_nonnegative_int(
                usage.get("completion_tokens", usage.get("output_tokens", 0))
            )

        elapsed = time.time() - t0
        has_tool_calls = bool(
            isinstance(choices, list)
            and choices
            and isinstance(choices[0], dict)
            and isinstance(choices[0].get("message"), dict)
            and choices[0]["message"].get("tool_calls")
        )
        if not text and output_tokens > 100 and not has_tool_calls:
            log.warning(
                "llm_empty_output_large_token_count",
                model=model,
                output_tokens=output_tokens,
            )
        log.info(
            "llm_completion_finished",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            text_chars=len(text),
            elapsed=f"{elapsed:.1f}s",
        )
        return ChatResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            raw=raw,
        )

    def semaphore_idle(self) -> bool:
        """Return True if the semaphore is free (advisory, subject to TOCTOU)."""
        acquired = self._semaphore.acquire(blocking=False)
        if acquired:
            self._semaphore.release()
            return True
        return False
