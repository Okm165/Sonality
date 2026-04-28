"""Web access layer for Sonality: search, extract, format.

Public API:
    get_client()         — lazily initialize the shared WebSearchClient
    SearchResult/Response — data types
    format_web_context() — safe formatting of results for prompt injection
    sanitize_web_content() — strip injection patterns from web text
"""

from __future__ import annotations

from .. import config
from .context import format_web_context, sanitize_web_content
from .search import ExtractResult, SearchResponse, SearchResult, WebSearchClient

__all__ = [
    "ExtractResult",
    "SearchResponse",
    "SearchResult",
    "WebSearchClient",
    "format_web_context",
    "get_client",
    "sanitize_web_content",
]

_client: WebSearchClient | None = None


def get_client() -> WebSearchClient | None:
    """Return shared WebSearchClient, or None if web search is disabled."""
    global _client
    if not config.WEB_SEARCH_ENABLED or not config.WEB_SEARCH_URL:
        return None
    if _client is None:
        _client = WebSearchClient(
            config.WEB_SEARCH_URL,
            cache_ttl=config.WEB_CACHE_TTL,
        )
    return _client
