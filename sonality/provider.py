"""LLM HTTP transport: provider, chat completion, concurrency control.

All JSON parsing, thinking trace removal, and tool call extraction live in
``llm.parse``. This module handles only HTTP communication and the global
provider instance + interaction state.
"""

from __future__ import annotations

import contextlib
import json
import logging
import random
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, NamedTuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from . import config
from .llm.parse import (
    extract_answer_from_reasoning,
    message_content_text,
    strip_thinking_trace,
    to_nonnegative_int,
    unwrap_think_tags,
)

log = logging.getLogger(__name__)

_RETRYABLE_HTTP_STATUSES: Final = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_EMPTY_MAPPING: Final[Mapping[str, object]] = {}

__all__ = [
    "ChatResult",
    "LLMProvider",
    "StreamChunk",
    "default_provider",
    "interaction_active",
    "interaction_in_progress",
    "llm_semaphore_idle",
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


class LLMProvider:
    """LLM provider with per-instance config and semaphore.

    Each instance has its own base_url, api_key, timeout, and a
    threading.Semaphore for serializing calls (single-GPU servers).
    """

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 300) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self._semaphore = threading.Semaphore(config.LLM_CONCURRENCY)

    def _post_json(self, path: str, payload: Mapping[str, object]) -> dict[str, object]:
        """POST JSON to the provider with retries on transient failures."""
        body = json.dumps(payload).encode("utf-8")
        normalized = path if path.startswith("/") else f"/{path}"
        url = f"{self.base_url}{normalized}"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        log.debug("POST %s (%d bytes)", normalized, len(body))
        max_attempts = config.LLM_MAX_RETRIES
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
                            "POST %s decode failed after %.1fs: %s",
                            normalized,
                            elapsed,
                            decode_exc,
                        )
                        raise RuntimeError(f"Provider response decode error: {decode_exc}") from decode_exc
                elapsed = time.time() - t0
                if isinstance(parsed, dict):
                    log.debug("POST %s OK in %.1fs", normalized, elapsed)
                    return parsed
                raise RuntimeError("Provider returned a non-object JSON payload")
            except HTTPError as exc:
                elapsed = time.time() - t0
                retryable = int(exc.code) in _RETRYABLE_HTTP_STATUSES
                if retryable and attempt < max_attempts:
                    self._retry_wait(normalized, f"HTTP {exc.code}", attempt, max_attempts, elapsed)
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
                    raise RuntimeError(f"Provider timeout after {elapsed:.1f}s") from exc
                reason_str = str(reason) if reason else str(exc)
                if "name resolution" in reason_str.lower():
                    raise RuntimeError(f"Provider DNS failure: {reason_str}") from exc
                if attempt < max_attempts:
                    self._retry_wait(
                        normalized, f"network: {reason_str}", attempt, max_attempts, elapsed
                    )
                    continue
                raise RuntimeError(f"Provider network error: {reason_str}") from exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                elapsed = time.time() - t0
                if attempt < max_attempts:
                    self._retry_wait(
                        normalized, f"transport: {exc}", attempt, max_attempts, elapsed
                    )
                    continue
                raise RuntimeError(f"Provider transport error: {exc}") from exc
        raise RuntimeError("Provider request failed after retries")

    @staticmethod
    def _retry_wait(
        path: str, reason: str, attempt: int, max_attempts: int, elapsed: float
    ) -> None:
        base_wait = config.LLM_BACKOFF_BASE**attempt
        jitter = random.uniform(0, base_wait * 0.5)
        wait = base_wait + jitter
        log.warning(
            "POST %s %s (attempt %d/%d, %.1fs), retrying in %.1fs",
            path,
            reason,
            attempt,
            max_attempts,
            elapsed,
            wait,
        )
        time.sleep(wait)

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
            "LLM call: model=%s msgs=%d max_tokens=%d tools=%d temp=%.1f",
            model,
            len(messages),
            max_tokens,
            len(tools),
            temperature,
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

        t0 = time.time()
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
                log.debug(
                    "LLM_RAW content=%d chars, reasoning=%d chars, finish=%s",
                    len(raw_content),
                    len(raw_reasoning),
                    first.get("finish_reason", "unknown"),
                )
                text = strip_thinking_trace(raw_content)
                if not text and raw_content.strip():
                    unwrapped = unwrap_think_tags(raw_content)
                    text = extract_answer_from_reasoning(unwrapped)
                    if text:
                        log.info("Recovered %d chars from thinking-wrapped content", len(text))
                    else:
                        log.warning(
                            "Failed to extract from thinking-wrapped content (%d raw, %d unwrapped)",
                            len(raw_content),
                            len(unwrapped),
                        )
                if not text and raw_reasoning:
                    text = extract_answer_from_reasoning(raw_reasoning)
                    if text:
                        log.info("Recovered %d chars from reasoning_content", len(text))
                    else:
                        log.warning(
                            "Failed to extract from reasoning_content (%d chars)",
                            len(raw_reasoning),
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
                "LLM %s: %d output tokens produced no extractable text", model, output_tokens
            )
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


# --- Default provider instance ---
default_provider: LLMProvider = LLMProvider(
    config.BASE_URL, config.API_KEY, config.LLM_REQUEST_TIMEOUT
)

# Models known to support extended thinking (Ollama: chat_template_kwargs enable_thinking).
# Patterns must match llama.cpp GGUF filenames — note gemma-4 (hyphen) vs Ollama gemma4.
_THINKING_MODELS: Final = frozenset({"qwq", "qwen3", "deepseek-r1", "gemma4", "gemma-4", "glm-4.7"})


def supports_thinking(model: str) -> bool:
    """Return True if the model supports enable_thinking=True."""
    return any(name in model.lower() for name in _THINKING_MODELS)


# --- Interaction state (global; used by semantic worker to defer during user turns) ---
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
