"""Local ONNX-optimized embedding via FastEmbed.

Uses bge-large-en-v1.5 (1024 dims) for dense vectors.
"""

from __future__ import annotations

import hashlib
import logging
import math
from typing import Final

from .. import config

log = logging.getLogger(__name__)

DENSE_MODEL: Final = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIMS: Final = 1024


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b) if norm_a > 0 and norm_b > 0 else 0.0


class EmbeddingUnavailableError(Exception):
    """Raised when embedding fails."""


class Embedder:
    """Local ONNX-optimized embedder with query caching.

    Singleton pattern avoids reloading the 335M parameter model.
    Thread-safe: FastEmbed handles internal synchronization.
    """

    _model: object = None

    def __init__(self, cache_size: int = 10000) -> None:
        from fastembed import TextEmbedding

        if Embedder._model is None:
            log.info("Loading dense embedding model: %s", DENSE_MODEL)
            Embedder._model = TextEmbedding(model_name=DENSE_MODEL)
        self._model_ref = Embedder._model
        self._cache: dict[str, list[float]] = {}
        self._max_cache = cache_size

    @property
    def dimensions(self) -> int:
        return EMBEDDING_DIMS

    def embed_query(self, query: str) -> list[float]:
        """Embed search query (cached)."""
        key = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()
        if key in self._cache:
            return self._cache[key]
        result = next(iter(self._model_ref.query_embed(query))).tolist()  # type: ignore[union-attr]
        if len(self._cache) < self._max_cache:
            self._cache[key] = result
        return result

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Batch embed documents (not cached - one-time indexing)."""
        if not documents:
            return []
        truncated = [doc[: config.EMBEDDING_MAX_CHARS] for doc in documents]
        return [emb.tolist() for emb in self._model_ref.passage_embed(truncated)]  # type: ignore[union-attr]
