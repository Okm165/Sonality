"""Async web client backed by Firecrawl (self-hosted, Docker).

Firecrawl provides:
  /v1/search — web search via SearXNG, each result scraped to markdown via Playwright
  /v1/scrape — single URL → clean markdown via Playwright

All methods return empty results on failure — web access is additive, never required.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

_SCRAPE_OPTIONS: dict[str, object] = {
    "formats": ["markdown"],
    "onlyMainContent": True,
}


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Single web search result with Playwright-rendered markdown content."""

    title: str
    url: str
    description: str
    markdown: str
    domain: str

    @staticmethod
    def from_raw(item: dict[str, object]) -> SearchResult:
        """Parse a result dict from the Firecrawl /v1/search response."""
        url = str(item.get("url", ""))
        return SearchResult(
            title=str(item.get("title", "")),
            url=url,
            description=str(item.get("description", "")),
            markdown=str(item.get("markdown", "") or ""),
            domain=urlparse(url).netloc.removeprefix("www."),
        )


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Complete search response from Firecrawl /v1/search.

    results is an immutable tuple of SearchResult (each with Playwright-rendered markdown).
    error: non-empty string if the search failed (vs legitimately empty results).
    """

    results: tuple[SearchResult, ...]
    error: str = ""

    @property
    def failed(self) -> bool:
        """True if search failed due to error (not just empty results)."""
        return bool(self.error)


@dataclass(frozen=True, slots=True)
class ExtractResult:
    """Content extracted from a single URL (markdown)."""

    url: str
    content: str


class WebSearchClient:
    """Web search and scrape via Firecrawl (self-hosted Docker service).

    Every search call includes scrapeOptions so results carry full
    Playwright-rendered markdown with nav/footer stripped (onlyMainContent).
    """

    def __init__(
        self,
        base_url: str,
        *,
        cache_ttl: int = 14400,
        max_concurrent: int = 3,
    ) -> None:
        """Initialize with Firecrawl base URL.

        cache_ttl: seconds to cache search results (default 4 hours).
        max_concurrent: semaphore limit for parallel HTTP requests.
        """
        self._base_url = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base_url, timeout=60.0)
        self._cache: dict[str, tuple[float, SearchResponse]] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._cache_ttl = cache_ttl
        self._call_count = 0

    async def search(
        self,
        query: str,
        *,
        max_results: int = 8,
    ) -> SearchResponse:
        """Search the web via Firecrawl; each result is scraped to markdown.

        Returns empty SearchResponse on any failure.
        """
        cache_key = f"{query}:{max_results}"
        now = time.monotonic()
        if cache_key in self._cache:
            ts, cached = self._cache[cache_key]
            if now - ts < self._cache_ttl:
                return cached
            del self._cache[cache_key]

        t0 = time.monotonic()
        try:
            async with self._semaphore:
                resp = await self._http.post(
                    "/v1/search",
                    json={
                        "query": query[:400],
                        "limit": max_results,
                        "scrapeOptions": _SCRAPE_OPTIONS,
                    },
                )
                resp.raise_for_status()
                raw = resp.json()
                self._call_count += 1
        except httpx.TimeoutException as exc:
            log.warning("Firecrawl search timeout for query=%.80s", query)
            return SearchResponse(results=(), error=f"Search timeout: {exc}")
        except httpx.HTTPStatusError as exc:
            log.warning("Firecrawl HTTP %d for query=%.80s", exc.response.status_code, query)
            return SearchResponse(results=(), error=f"HTTP {exc.response.status_code}")
        except Exception as exc:
            log.warning("Firecrawl search failed for query=%.80s: %s", query, exc)
            return SearchResponse(results=(), error=str(exc))

        elapsed_ms = (time.monotonic() - t0) * 1000
        items = raw.get("data", []) if isinstance(raw, dict) else []
        if not isinstance(items, list):
            items = []
        results = tuple(
            SearchResult.from_raw(item)
            for item in items
            if isinstance(item, dict) and item.get("url")
        )

        response = SearchResponse(results=results)
        self._cache[cache_key] = (now, response)
        log.info(
            "Firecrawl search: q=%.60s results=%d time=%.0fms",
            query,
            len(results),
            elapsed_ms,
        )
        return response

    async def extract(self, url: str) -> ExtractResult:
        """Scrape a URL to clean markdown via Firecrawl /v1/scrape.

        Renders the page with Playwright and converts to markdown,
        handling JavaScript-heavy sites. Returns empty content on failure.
        """
        try:
            async with self._semaphore:
                resp = await self._http.post(
                    "/v1/scrape",
                    json={"url": url, **_SCRAPE_OPTIONS},
                )
                resp.raise_for_status()
                data = resp.json()
                self._call_count += 1
        except Exception:
            log.warning("Firecrawl scrape failed for url=%.80s", url, exc_info=True)
            return ExtractResult(url=url, content="")

        if not isinstance(data, dict):
            return ExtractResult(url=url, content="")
        inner = data.get("data", {})
        content = str(inner.get("markdown", ""))[:8000] if isinstance(inner, dict) else ""
        return ExtractResult(url=url, content=content)

    async def multi_search(
        self,
        queries: list[str],
        *,
        max_results: int = 5,
    ) -> list[SearchResponse]:
        """Parallel search across multiple queries. Failed queries are omitted."""
        responses = await asyncio.gather(
            *(self.search(q, max_results=max_results) for q in queries),
            return_exceptions=True,
        )
        ok: list[SearchResponse] = []
        for r in responses:
            if isinstance(r, SearchResponse):
                ok.append(r)
            elif isinstance(r, BaseException):
                log.warning("multi_search sub-query failed: %s", r)
        return ok

    @property
    def call_count(self) -> int:
        """Total API calls made by this client (diagnostic)."""
        return self._call_count

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()
