"""Async Sonality API client with client-side conversation history."""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx

from . import config

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChatResponse:
    text: str
    ess_score: float
    topics: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HealthStatus:
    belief_count: int
    snapshot_version: int


@dataclass(frozen=True, slots=True)
class Belief:
    topic: str
    valence: float
    confidence: float
    belief_text: str = ""


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
        )

    async def beliefs(self) -> list[Belief]:
        r = await self._client.get("/beliefs")
        r.raise_for_status()
        data = r.json()
        items = data if isinstance(data, list) else data.get("beliefs", [])
        return [
            Belief(
                topic=b.get("topic", ""),
                valence=float(b.get("valence", 0)),
                confidence=float(b.get("confidence", 0)),
                belief_text=b.get("belief_text", ""),
            )
            for b in items
        ]

    async def chat(self, message: str) -> ChatResponse:
        """Send message with full conversation history."""
        self._history.append({"role": "user", "content": message})
        self._trim_history()
        try:
            r = await self._client.post(
                "/chat", json={"message": message, "context": self._history[:-1]}
            )
            r.raise_for_status()
        except Exception:
            self._history.pop()
            raise
        d = r.json()
        response_text = d.get("response", "")
        self._history.append({"role": "assistant", "content": response_text})
        return ChatResponse(
            text=response_text,
            ess_score=float(d.get("ess_score", 0)),
            topics=tuple(d.get("topics", [])),
        )

    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        """Stream chat response with full history via SSE."""
        self._history.append({"role": "user", "content": message})
        self._trim_history()
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
                async for line in r.aiter_lines():
                    if not line or line == "data: [DONE]" or not line.startswith("data: "):
                        continue
                    try:
                        chunk = json.loads(line[6:])
                        if content := chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                            chunks.append(content)
                            yield content
                    except (json.JSONDecodeError, IndexError):
                        continue
        except Exception:
            self._history.pop()
            raise
        self._history.append({"role": "assistant", "content": "".join(chunks)})

    def _trim_history(self) -> None:
        while len(self._history) > self._max_history:
            self._history.pop(0)

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> SonalityClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
