"""Shared configuration — project root, logger tuning, and settings base class.

All three modules (sonality, fathom, chat) use pydantic-settings for env vars.
``InfraSettings`` defines the common infrastructure fields (Neo4j, Qdrant,
embedding, LLM) inherited by both ``sonality.config.Settings`` and
``fathom.config.Settings`` — one canonical location, zero drift.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import model_validator
from pydantic_settings import BaseSettings

from .errors import ConfigError

if TYPE_CHECKING:
    from .embedder import Embedder
    from .llm.provider import LLMProvider

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

VECTOR_SEARCH_THRESHOLD: Final = 0.3
"""Default cosine similarity floor for Qdrant ANN queries.

Used by both sonality (episode retrieval, semantic features, knowledge)
and fathom (source suggestion).  A single constant prevents silent
divergence across modules.
"""


class InfraSettings(BaseSettings):
    """Common infrastructure fields inherited by sonality and fathom settings.

    Subclasses MUST set ``model_config`` with their own ``env_prefix``
    (``SONALITY_`` or ``FATHOM_``) so pydantic-settings resolves the right
    environment variables.  This class is never instantiated directly.
    """

    # --- LLM endpoint ---
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    llm_max_tokens: int = -1
    llm_max_retries: int = 3
    llm_backoff_base: float = 2.0
    llm_concurrency: int = 4

    # --- Neo4j ---
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "sonality_password"
    neo4j_database: str = "neo4j"
    neo4j_max_pool_size: int = 50
    neo4j_connection_timeout: float = 30.0

    # --- Qdrant + Embedding ---
    qdrant_url: str = "http://localhost:6333"
    embedding_url: str = "http://localhost:8090"
    embedding_dimensions: int = 2560
    embedding_concurrency: int = 4

    @model_validator(mode="after")
    def _clamp_infra_bounds(self) -> InfraSettings:
        """Ensure sane floors for infrastructure settings."""
        self.llm_concurrency = max(1, self.llm_concurrency)
        self.embedding_concurrency = max(1, self.embedding_concurrency)
        self.llm_max_retries = max(0, self.llm_max_retries)
        self.llm_backoff_base = max(1.0, self.llm_backoff_base)
        self.neo4j_max_pool_size = max(1, self.neo4j_max_pool_size)
        self.neo4j_connection_timeout = max(1.0, self.neo4j_connection_timeout)
        self.embedding_dimensions = max(1, self.embedding_dimensions)
        return self

    def make_embedder(self) -> Embedder:
        """Create an Embedder from this config's embedding_url and embedding_dimensions."""
        from .embedder import Embedder, EmbeddingConfig

        return Embedder(EmbeddingConfig(url=self.embedding_url, dims=self.embedding_dimensions))

    def make_llm_provider(self, *, timeout: int) -> LLMProvider:
        """Create an LLMProvider from shared infra fields + caller-specific timeout."""
        from .llm.provider import LLMProvider

        return LLMProvider(
            self.base_url,
            self.api_key,
            timeout,
            max_retries=self.llm_max_retries,
            backoff_base=self.llm_backoff_base,
            concurrency=self.llm_concurrency,
        )

    @model_validator(mode="after")
    def _validate_llm_required(self) -> InfraSettings:
        """Fail fast if required LLM config is missing."""
        if not self.base_url:
            prefix = self.model_config.get("env_prefix", "")
            raise ConfigError(f"{prefix}BASE_URL must be set")
        if not self.model:
            prefix = self.model_config.get("env_prefix", "")
            raise ConfigError(f"{prefix}MODEL must be set")
        return self


def quiet_third_party_loggers() -> None:
    """Aggressively suppress noisy library loggers.

    HTTP clients and search engines at ERROR to eliminate the
    'response: https://...' spam. Database/browser at WARNING.
    Uvicorn access logs suppressed — redundant with app-level request logging.
    """
    for lib in ("httpcore", "httpx", "urllib3", "urllib3.connectionpool", "primp"):
        logging.getLogger(lib).setLevel(logging.ERROR)

    logging.getLogger("ddgs").setLevel(logging.ERROR)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    for lib in ("neo4j", "neo4j.io", "neo4j.pool"):
        logging.getLogger(lib).setLevel(logging.WARNING)
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    logging.getLogger("trafilatura").setLevel(logging.ERROR)
    logging.getLogger("charset_normalizer").setLevel(logging.WARNING)
    logging.getLogger("playwright").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    for lib in ("reqwest", "hyper_util", "rustls", "htmldate", "h2", "cookie_store"):
        logging.getLogger(lib).setLevel(logging.WARNING)
