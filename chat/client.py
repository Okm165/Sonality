"""Async Sonality API client."""

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
    version: int
    interaction_count: int
    belief_count: int
    topic_count: int
    staged_updates: int


@dataclass(frozen=True, slots=True)
class Belief:
    topic: str
    position: float
    confidence: float


class SonalityClient:
    """Async client for Sonality API."""

    def __init__(self) -> None:
        self._client = httpx.AsyncClient(
            base_url=config.SONALITY_URL, timeout=httpx.Timeout(config.HTTP_TIMEOUT, connect=10.0)
        )
        log.debug("SonalityClient: url=%s timeout=%.0fs", config.SONALITY_URL, config.HTTP_TIMEOUT)

    async def health(self) -> HealthStatus:
        log.debug("Health request")
        r = await self._client.get("/health")
        if r.status_code != 200:
            log.error("Health failed: status=%d body=%s", r.status_code, r.text[:500])
        r.raise_for_status()
        d = r.json()
        status = HealthStatus(
            version=d.get("version", 0),
            interaction_count=d.get("interaction_count", 0),
            belief_count=d.get("belief_count", 0),
            topic_count=d.get("topic_count", 0),
            staged_updates=d.get("staged_updates", 0),
        )
        log.debug("Health: v%d interactions=%d beliefs=%d", status.version, status.interaction_count, status.belief_count)
        return status

    async def beliefs(self) -> list[Belief]:
        log.debug("Beliefs request")
        r = await self._client.get("/beliefs")
        if r.status_code != 200:
            log.error("Beliefs failed: status=%d body=%s", r.status_code, r.text[:500])
        r.raise_for_status()
        data = r.json()
        items = data if isinstance(data, list) else data.get("beliefs", [])
        beliefs = [
            Belief(
                topic=b.get("topic", ""),
                position=float(b.get("position", 0)),
                confidence=float(b.get("confidence", 0)),
            )
            for b in items
        ]
        log.debug("Beliefs: %d items", len(beliefs))
        return beliefs

    async def chat(self, message: str) -> ChatResponse:
        log.debug("Chat request: %d chars", len(message))
        r = await self._client.post("/chat", json={"message": message})
        if r.status_code != 200:
            log.error("Chat failed: status=%d body=%s", r.status_code, r.text[:500])
        r.raise_for_status()
        d = r.json()
        result = ChatResponse(
            text=d.get("response", ""),
            ess_score=float(d.get("ess_score", 0)),
            topics=tuple(d.get("topics", [])),
        )
        log.info("Chat: %.50s... -> %.50s... (ess=%.2f)", message, result.text, result.ess_score)
        return result

    async def chat_stream(self, message: str) -> AsyncIterator[str]:
        """Stream chat response (SSE)."""
        async with self._client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "sonality",
                "messages": [{"role": "user", "content": message}],
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
                        yield content
                except (json.JSONDecodeError, IndexError):
                    continue

    async def close(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> SonalityClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
