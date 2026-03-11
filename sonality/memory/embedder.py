"""External embedding provider with batching, retry, and fallback.

Supports OpenAI text-embedding-3-large (default) and OpenRouter-proxied models.
Instruction-aware embeddings for queries vs documents (GRIT, arXiv 2402.09906).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Final
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .. import config

log = logging.getLogger(__name__)

_MAX_RETRIES: Final = 3
_BACKOFF_BASE: Final = 2.0
_TIMEOUT_SEC: Final = 30


class EmbeddingUnavailableError(Exception):
    """Raised when all embedding providers fail."""


@dataclass(frozen=True, slots=True)
class EmbedderConfig:
    """Resolved embedding configuration."""

    provider: str
    model: str
    dimensions: int
    api_key: str
    query_instruction: str
    doc_instruction: str
    batch_size: int


def _resolve_config() -> EmbedderConfig:
    return EmbedderConfig(
        provider=config.EMBEDDING_PROVIDER,
        model=config.EMBEDDING_MODEL,
        dimensions=config.EMBEDDING_DIMENSIONS,
        api_key=config.EMBEDDING_API_KEY,
        query_instruction=config.EMBEDDING_QUERY_INSTRUCTION,
        doc_instruction=config.EMBEDDING_DOC_INSTRUCTION,
        batch_size=config.EMBEDDING_BATCH_SIZE,
    )


def _openai_embed(
    texts: list[str],
    *,
    model: str,
    dimensions: int,
    api_key: str,
) -> list[list[float]]:
    """Call OpenAI embeddings API."""
    payload = {
        "model": model,
        "input": texts,
        "dimensions": dimensions,
    }
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        "https://api.openai.com/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=_TIMEOUT_SEC) as response:
        result = json.loads(response.read().decode("utf-8"))

    # Sort by index to ensure order matches input
    data = sorted(result["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


def _openrouter_embed(
    texts: list[str],
    *,
    model: str,
    dimensions: int,
    api_key: str,
) -> list[list[float]]:
    """Call OpenRouter embeddings endpoint."""
    payload = {
        "model": model,
        "input": texts,
        "dimensions": dimensions,
    }
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        f"{config.BASE_URL.rstrip('/')}/v1/embeddings",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=_TIMEOUT_SEC) as response:
        result = json.loads(response.read().decode("utf-8"))

    data = sorted(result["data"], key=lambda d: d["index"])
    return [d["embedding"] for d in data]


class ExternalEmbedder:
    """Embedding provider with batching, retry, and instruction-aware embeddings."""

    def __init__(self, cfg: EmbedderConfig | None = None) -> None:
        self._cfg = cfg or _resolve_config()
        if not self._cfg.api_key:
            raise ValueError(
                "Embedding API key not configured. Set SONALITY_EMBEDDING_API_KEY or OPENAI_API_KEY."
            )

    @property
    def dimensions(self) -> int:
        return self._cfg.dimensions

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Batch embed texts. Returns embeddings in same order as input."""
        if not texts:
            return []
        return self._embed_with_retry(texts)

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query with retrieval-optimized instruction."""
        prefixed = f"{self._cfg.query_instruction} {query}"
        result = self._embed_with_retry([prefixed])
        return result[0]

    def embed_document(self, document: str) -> list[float]:
        """Embed a document for storage with indexing-optimized instruction."""
        prefixed = f"{self._cfg.doc_instruction} {document}"
        result = self._embed_with_retry([prefixed])
        return result[0]

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Batch embed documents with storage-optimized instruction."""
        if not documents:
            return []
        prefixed = [f"{self._cfg.doc_instruction} {doc}" for doc in documents]
        return self._embed_with_retry(prefixed)

    def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Embed with automatic sub-batching and retry."""
        all_embeddings: list[list[float]] = []
        batch_size = self._cfg.batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self._call_provider(batch)
            all_embeddings.extend(embeddings)

        if len(all_embeddings) != len(texts):
            raise EmbeddingUnavailableError(
                f"Embedding count mismatch: expected {len(texts)}, got {len(all_embeddings)}"
            )
        return all_embeddings

    def _call_provider(self, texts: list[str]) -> list[list[float]]:
        """Call the embedding provider with retry logic."""
        provider = self._cfg.provider
        last_error: Exception | None = None

        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                if provider == "openai":
                    return _openai_embed(
                        texts,
                        model=self._cfg.model,
                        dimensions=self._cfg.dimensions,
                        api_key=self._cfg.api_key,
                    )
                if provider == "openrouter":
                    return _openrouter_embed(
                        texts,
                        model=self._cfg.model,
                        dimensions=self._cfg.dimensions,
                        api_key=self._cfg.api_key,
                    )
                raise ValueError(f"Unknown embedding provider: {provider}")

            except HTTPError as exc:
                status = int(exc.code)
                last_error = exc
                if status == 429:
                    wait = _BACKOFF_BASE**attempt
                    log.warning(
                        "Embedding rate limited (attempt %d/%d); backing off %.1fs",
                        attempt, _MAX_RETRIES, wait,
                    )
                    time.sleep(wait)
                    continue
                if status >= 500 and attempt < _MAX_RETRIES:
                    log.warning("Embedding server error %d (attempt %d/%d)", status, attempt, _MAX_RETRIES)
                    time.sleep(1.0)
                    continue
                break

            except (URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt < _MAX_RETRIES:
                    wait = _BACKOFF_BASE**attempt
                    log.warning(
                        "Embedding network error (attempt %d/%d): %s; retrying in %.1fs",
                        attempt, _MAX_RETRIES, exc, wait,
                    )
                    time.sleep(wait)
                    continue
                break

        raise EmbeddingUnavailableError(
            f"All {_MAX_RETRIES} embedding attempts failed. Last error: {last_error}"
        )
