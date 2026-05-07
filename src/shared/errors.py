"""Shared error hierarchy used across all modules.

Three base categories cover every failure mode:
  - LLMError: anything wrong with an LLM call (transport, parse, validation)
  - StorageError: database write/read failures (Neo4j, Qdrant)
  - ConfigError: missing or invalid configuration at startup

Modules raise the most specific subclass that applies. Callers catch at
whatever granularity they need — catching LLMError handles all LLM failures,
catching ProviderTransportError handles only network-level issues.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# LLM errors
# ---------------------------------------------------------------------------


class LLMError(Exception):
    """Base for all LLM-related failures."""


class ProviderTransportError(LLMError):
    """Network, DNS, timeout, or HTTP transport failure talking to the LLM provider."""


class ProviderHTTPError(ProviderTransportError):
    """Non-retryable HTTP error from the LLM provider (4xx, non-transient 5xx)."""

    def __init__(self, status: int, detail: str = "") -> None:
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}")


class LLMParseError(LLMError):
    """LLM returned text that could not be parsed as valid JSON."""


class LLMValidationError(LLMError):
    """LLM JSON parsed but failed Pydantic schema validation."""


# ---------------------------------------------------------------------------
# Storage errors
# ---------------------------------------------------------------------------


class StorageError(Exception):
    """Base for all persistent-store failures (Neo4j, Qdrant, etc.)."""


class EpisodeStorageError(StorageError):
    """Episode write pipeline failed (chunking, embedding, Neo4j, or Qdrant)."""


class KnowledgeStorageError(StorageError):
    """Knowledge proposition storage failed."""


class BeliefUpdateError(StorageError):
    """Belief graph upsert or provenance recording failed."""


# ---------------------------------------------------------------------------
# Configuration errors
# ---------------------------------------------------------------------------


class ConfigError(ValueError):
    """Required configuration missing or invalid at startup."""


# ---------------------------------------------------------------------------
# Service errors
# ---------------------------------------------------------------------------


class ServiceUnavailableError(Exception):
    """An external service (browser, fathom, etc.) is not reachable or misconfigured."""
