"""Async Sonality API client with client-side conversation history.

SonalityClient manages a sliding window of messages and streams responses
via SSE from the /v1/chat/completions endpoint. Progress events (tool calls,
thinking, reviewing) are yielded alongside content deltas so UIs can render
real-time agent status.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Final

import httpx
import structlog

from . import config

log = structlog.get_logger(__name__)

# Shared tool labels for UX display (no emojis)
TOOL_LABELS: Final[dict[str, str]] = {
    "recall_memory": "Recalling",
    "web_research": "Researching",
    "integrate_knowledge": "Integrating",
}


def extract_tool_arg_summary(tool_args: str) -> str:
    """Extract a concise human-readable summary from tool call arguments."""
    if not tool_args:
        return ""
    try:
        parsed = json.loads(tool_args)
        for key in ("query", "goal", "url", "focus", "topic", "text"):
            if raw := parsed.get(key):
                return str(raw)[:60]
        return ""
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
    sources_count: int = 0


class SonalityClient:
    """Async client that owns conversation history and sends it with each request."""

    def __init__(self, max_history: int = 40) -> None:
        self._client = httpx.AsyncClient(
            base_url=config.settings.sonality_url,
            timeout=httpx.Timeout(config.settings.http_timeout, connect=10.0),
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
            belief_count=int(d.get("belief_count") or 0),
            snapshot_version=int(d.get("snapshot_version") or 0),
            uptime_seconds=float(d.get("uptime_seconds") or 0),
            version=str(d.get("version") or ""),
        )

    async def beliefs(self) -> list[Belief]:
        """Fetch all beliefs from the API. Server returns a JSON list."""
        r = await self._client.get("/beliefs")
        r.raise_for_status()
        items = r.json()
        if not isinstance(items, list):
            return []
        return [
            Belief(
                topic=str(b.get("topic", "")),
                valence=float(b.get("valence") or 0),
                confidence=float(b.get("confidence") or 0),
                belief_text=str(b.get("belief_text", "")),
            )
            for b in items
            if isinstance(b, dict)
        ]

    async def chat_stream(self, message: str) -> AsyncIterator[str | ProgressEvent]:
        """Stream chat response with full history via SSE.

        Yields str for content deltas and ProgressEvent for agent progress.
        """
        self._history.append({"role": "user", "content": message})
        self._trim_history()
        trace_id = str(uuid.uuid4())[:12]
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(trace_id=trace_id)
        log.debug("chat_stream_send", messages=len(self._history), latest_chars=len(message))
        chunks: list[str] = []
        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                headers={"X-Trace-ID": trace_id},
                json={
                    "model": config.settings.model_id,
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
                        log.debug("sse_json_skip", line=line[:60])
                        continue

                    if event_type:
                        detail = data.get("detail", "")
                        if event_type == "tool_progress":
                            log.debug(
                                "tool_progress_received",
                                detail=detail[:60],
                                tool=data.get("tool_name", ""),
                            )
                        yield ProgressEvent(
                            type=event_type,
                            detail=detail,
                            tool_name=data.get("tool_name", ""),
                            tool_args=data.get("tool_args", ""),
                            tool_result_summary=data.get("tool_result_summary", ""),
                            sources_count=int(data.get("sources_count") or 0),
                        )
                        event_type = ""
                        continue

                    choices = data.get("choices") or []
                    if content := (choices[0].get("delta", {}).get("content") if choices else None):
                        chunks.append(content)
                        yield content
                    event_type = ""
        except Exception:
            self._history.pop()
            raise
        full_response = "".join(chunks)
        log.debug("chat_stream_received", chars=len(full_response), chunks=len(chunks))
        if full_response:
            self._history.append({"role": "assistant", "content": full_response})
        else:
            self._history.pop()
            log.warning("empty_assistant_response")

    def _trim_history(self) -> None:
        while len(self._history) > self._max_history:
            self._history.pop(0)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> SonalityClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
