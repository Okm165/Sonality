"""Async Sonality API client with client-side conversation history.

SonalityClient manages a sliding window of messages and streams responses
via SSE from the /v1/chat/completions endpoint. Progress events (tool calls,
thinking, reviewing) are yielded alongside content deltas so UIs can render
real-time agent status.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final

import httpx

from . import config

log = logging.getLogger(__name__)

# Shared tool labels for UX display (no emojis)
TOOL_LABELS: Final[dict[str, str]] = {
    "recall_memory": "Recalling",
    "web_search": "Searching web",
    "web_extract": "Reading page",
    "integrate_knowledge": "Integrating",
    "synthesize": "Synthesizing",
    "quorum_critique": "Cross-checking",
}


def extract_tool_arg_summary(tool_args: str) -> str:
    """Extract a concise human-readable summary from tool call arguments."""
    if not tool_args:
        return ""
    try:
        parsed = json.loads(tool_args)
        raw = parsed.get(
            "query",
            parsed.get("url", parsed.get("focus", parsed.get("topic", parsed.get("text", "")))),
        )
        return str(raw)[:60] if raw else ""
    except json.JSONDecodeError:
        return tool_args[:60]


def pipeline_summary(tool_names: list[str]) -> str:
    """Build compact pipeline: recall > search x3 > integrate."""
    if not tool_names:
        return ""
    parts: list[str] = []
    i = 0
    while i < len(tool_names):
        name = tool_names[i]
        count = 1
        while i + count < len(tool_names) and tool_names[i + count] == name:
            count += 1
        short = TOOL_LABELS.get(name, name.replace("_", " "))
        parts.append(f"{short} x{count}" if count > 1 else short)
        i += count
    return " > ".join(parts)


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Response from ``/health`` endpoint."""

    belief_count: int
    snapshot_version: int
    uptime_seconds: float = 0.0
    version: str = ""


@dataclass(frozen=True, slots=True)
class Belief:
    """Single belief from ``/beliefs`` endpoint."""

    topic: str
    valence: float
    confidence: float
    belief_text: str = ""


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    """Agent progress event parsed from SSE."""

    type: str
    detail: str = ""
    tool_name: str = ""
    tool_args: str = ""
    tool_result_summary: str = ""
    iteration: int = 0
    sources_count: int = 0


class SonalityClient:
    """Async client that owns conversation history and sends it with each request."""

    def __init__(self, max_history: int = 40) -> None:
        self._client = httpx.AsyncClient(
            base_url=config.SONALITY_URL, timeout=httpx.Timeout(config.HTTP_TIMEOUT, connect=10.0)
        )
        self._history: list[dict[str, str]] = []
        self._max_history = max_history

    def clear_history(self) -> None:
        self._history.clear()

    async def health(self) -> HealthStatus:
        r = await self._client.get("/health")
        r.raise_for_status()
        d = r.json()
        return HealthStatus(
            belief_count=d.get("belief_count", 0),
            snapshot_version=d.get("snapshot_version", 0),
            uptime_seconds=float(d.get("uptime_seconds", 0)),
            version=d.get("version", ""),
        )

    async def beliefs(self) -> list[Belief]:
        """Fetch all beliefs from the API. Server returns a JSON list."""
        r = await self._client.get("/beliefs")
        r.raise_for_status()
        return [
            Belief(
                topic=b.get("topic", ""),
                valence=float(b.get("valence", 0)),
                confidence=float(b.get("confidence", 0)),
                belief_text=b.get("belief_text", ""),
            )
            for b in r.json()
        ]

    async def chat_stream(self, message: str) -> AsyncIterator[str | ProgressEvent]:
        """Stream chat response with full history via SSE.

        Yields str for content deltas and ProgressEvent for agent progress.
        """
        self._history.append({"role": "user", "content": message})
        self._trim_history()
        log.debug(
            "chat_stream: sending %d messages (latest=%d chars)", len(self._history), len(message)
        )
        chunks: list[str] = []
        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json={
                    "model": config.MODEL_ID,
                    "messages": list(self._history),
                    "stream": True,
                },
            ) as r:
                r.raise_for_status()
                event_type = ""
                async for line in r.aiter_lines():
                    if not line:
                        event_type = ""
                        continue
                    if line.startswith("event: "):
                        event_type = line[7:].strip()
                        continue
                    if line == "data: [DONE]":
                        continue
                    if not line.startswith("data: "):
                        continue
                    try:
                        data = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue

                    if event_type:
                        yield ProgressEvent(
                            type=event_type,
                            detail=data.get("detail", ""),
                            tool_name=data.get("tool_name", ""),
                            tool_args=data.get("tool_args", ""),
                            tool_result_summary=data.get("tool_result_summary", ""),
                            iteration=data.get("iteration", 0),
                            sources_count=data.get("sources_count", 0),
                        )
                        event_type = ""
                        continue

                    if content := data.get("choices", [{}])[0].get("delta", {}).get("content"):
                        chunks.append(content)
                        yield content
                    event_type = ""
        except Exception:
            self._history.pop()
            raise
        full_response = "".join(chunks)
        log.debug("chat_stream: received %d chars in %d chunks", len(full_response), len(chunks))
        self._history.append({"role": "assistant", "content": full_response})

    def _trim_history(self) -> None:
        while len(self._history) > self._max_history:
            self._history.pop(0)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> SonalityClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
