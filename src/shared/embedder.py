"""Shared embedding infrastructure — GPU-accelerated via llama.cpp server.

Calls a dedicated llama.cpp embedding server (Qwen3-Embedding) over HTTP.
Instruction-aware asymmetric embedding: queries get an instruction prefix
("Instruct: ... Query: ..."), documents are embedded raw.
MRL dimension truncation applied client-side for flexible vector sizes.

Two call variants (same pattern as ``shared.llm.caller``):
  ``embed_query`` / ``embed_documents``     — synchronous.
  ``async_embed_query`` / ``async_embed_documents`` — async bridge, blocking
    I/O in the default executor.  Concurrency gating is the caller's
    responsibility — sonality and fathom each own their own semaphore.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Final, Protocol

import httpx
import structlog

log = structlog.get_logger(__name__)

DEFAULT_EMBEDDING_URL: Final = "http://localhost:8090"
DEFAULT_EMBEDDING_DIMS: Final = 2560

QUERY_INSTRUCTION: Final = "Given a query, retrieve relevant passages that answer the query"

_MAX_RETRIES: Final = 3
_RETRY_BACKOFF: Final = 0.5
_EMBED_BATCH_SIZE: Final = 64


class EmbedderProtocol(Protocol):
    """Protocol for embedding implementations."""

    @property
    def dims(self) -> int: ...
    def embed_query(self, query: str) -> list[float]: ...
    def embed_documents(self, documents: list[str]) -> list[list[float]]: ...


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    """Configuration for embedder initialization."""

    url: str = DEFAULT_EMBEDDING_URL
    dims: int = DEFAULT_EMBEDDING_DIMS
    timeout: float = 300.0


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    if len(a) != len(b):
        log.warning("cosine_dim_mismatch", a_dims=len(a), b_dims=len(b))
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


def probe_embedding_dims(url: str) -> int:
    """Probe the embedding server for native output dimensions.

    Single canonical probe — used by both ``Embedder.__init__`` and
    ``fathom.api.get_qdrant`` so there is one probe implementation.
    Raises RuntimeError if the server is unreachable.
    """
    base = url.rstrip("/")
    try:
        with httpx.Client(base_url=base, timeout=30.0) as client:
            response = client.post(
                "/v1/embeddings",
                json={"input": ["dim_probe"], "model": "embedding"},
            )
            response.raise_for_status()
            body = response.json()
            items = body.get("data") if isinstance(body, dict) else None
            if not isinstance(items, list) or not items:
                raise ValueError("response missing 'data' array")
            emb = items[0].get("embedding") if isinstance(items[0], dict) else None
            if not isinstance(emb, list):
                raise ValueError("response missing 'embedding' vector")
            return len(emb)
    except Exception as exc:
        log.error("embedding_probe_failed", url=base, error=str(exc)[:200])
        raise RuntimeError(f"Cannot reach embedding server at {base}: {exc}") from exc


class Embedder:
    """GPU-accelerated embedder via llama.cpp /v1/embeddings endpoint.

    Asymmetric instruction-aware embedding (Qwen3-Embedding):
    - embed_query: prepends "Instruct: ... Query: ..." for retrieval queries.
    - embed_documents: embeds raw text for stored content.

    No caching — the embedding server is local and fast.  Qdrant already
    stores computed vectors; duplicating that in an app-level LRU adds
    staleness risk, memory pressure, and threading complexity for no gain.
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        cfg = config or EmbeddingConfig()
        self._url = cfg.url.rstrip("/")
        self._client = httpx.Client(base_url=self._url, timeout=cfg.timeout)

        native = probe_embedding_dims(self._url)
        self._dims = min(cfg.dims, native)
        if self._dims < cfg.dims:
            log.warning(
                "embedding_dims_capped",
                configured=cfg.dims,
                native=native,
                using=self._dims,
            )
        log.info("embedder_ready", url=self._url, dims=self._dims, native=native)

    @property
    def dims(self) -> int:
        return self._dims

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with instruction prefix."""
        if not query.strip():
            return [0.0] * self._dims
        full_query = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {query}"
        return self._embed_batch([full_query])[0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Batch embed documents without instruction prefix."""
        if not documents:
            return []
        return self._embed_batch(documents)

    async def async_embed_query(self, query: str) -> list[float]:
        """Async bridge — runs blocking HTTP in the default executor."""
        return await asyncio.to_thread(self.embed_query, query)

    async def async_embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Async bridge — runs blocking HTTP in the default executor."""
        return await asyncio.to_thread(self.embed_documents, documents)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Call llama.cpp /v1/embeddings with retry, batching, and MRL truncation.

        Splits large inputs into chunks of _EMBED_BATCH_SIZE to avoid HTTP body
        limits and embedding server OOM on large requests.
        """
        if len(texts) <= _EMBED_BATCH_SIZE:
            return self._embed_batch_single(texts)
        results: list[list[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH_SIZE):
            results.extend(self._embed_batch_single(texts[i : i + _EMBED_BATCH_SIZE]))
        return results

    def _embed_batch_single(self, texts: list[str]) -> list[list[float]]:
        """Single batch call to embedding server with retry."""
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                response = self._client.post(
                    "/v1/embeddings",
                    json={"input": texts, "model": "embedding"},
                )
                response.raise_for_status()
                body = response.json()
                data = body.get("data") if isinstance(body, dict) else None
                if not isinstance(data, list):
                    raise ValueError("embedding response missing 'data' array")
                sorted_data = sorted(data, key=lambda x: x.get("index", 0))
                return [item["embedding"][: self._dims] for item in sorted_data]
            except (httpx.HTTPStatusError, httpx.TransportError, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt < _MAX_RETRIES - 1:
                    time.sleep(_RETRY_BACKOFF * (2**attempt))
                    log.warning(
                        "embed_retry",
                        attempt=attempt + 1,
                        error=str(exc)[:120],
                    )
        raise RuntimeError(f"Embedding failed after {_MAX_RETRIES} attempts: {last_exc}")


# ---------------------------------------------------------------------------
# Gated wrappers — one factory, used by both sonality.caller and fathom.caller
# ---------------------------------------------------------------------------


def make_gated_embedders(
    gate: asyncio.Semaphore,
) -> tuple[
    Callable[[Embedder, str], Coroutine[None, None, list[float]]],
    Callable[[Embedder, list[str]], Coroutine[None, None, list[list[float]]]],
]:
    """Return ``(async_embed_query, async_embed_documents)`` gated by *gate*.

    Each module (sonality, fathom) calls this once with its own semaphore.
    """

    async def async_embed_query(embedder: Embedder, query: str) -> list[float]:
        async with gate:
            return await embedder.async_embed_query(query)

    async def async_embed_documents(
        embedder: Embedder,
        documents: list[str],
    ) -> list[list[float]]:
        async with gate:
            return await embedder.async_embed_documents(documents)

    return async_embed_query, async_embed_documents
