"""Async HTTP + SSE client for the Fathom research service.

Sonality delegates web search, content extraction, and deep research to fathom.
All methods are async (httpx). The agent bridges to sync via ``run_async``.

Three tiers:
  search / extract  — lightweight, low-latency (used by web tools)
  start_research / stream_research — full autonomous sessions with SSE progress
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import httpx
import structlog

log = structlog.get_logger()


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Single web search result from fathom."""

    url: str
    title: str
    snippet: str
    content: str = ""

    @property
    def markdown(self) -> str:
        return self.content or self.snippet

    @property
    def description(self) -> str:
        return self.snippet


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Complete search response."""

    results: tuple[SearchResult, ...]
    query: str = ""
    error: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.error)


@dataclass(frozen=True, slots=True)
class ExtractResult:
    """Content extracted from a single URL."""

    url: str
    content: str
    title: str = ""


@dataclass(frozen=True, slots=True)
class ResearchProgress:
    """SSE progress event from a running research session."""

    event: str
    pages: int = 0
    facts: int = 0
    checklist_answered: int = 0
    checklist_total: int = 0
    document_length: int = 0
    status: str = ""


@dataclass(frozen=True, slots=True)
class ResearchResult:
    """Final result from a completed research session."""

    session_id: str
    status: str
    document: str
    pages_scraped: int = 0


@dataclass(frozen=True, slots=True)
class ResearchSession:
    """Handle for a started research session."""

    session_id: str
    status: str


@dataclass
class ResearchClient:
    """HTTP client for fathom's search, extract, and research APIs.

    All methods gracefully degrade: errors return empty results rather than raising.
    """

    base_url: str
    _http: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            timeout=60.0,
        )

    async def search(self, query: str, *, max_results: int = 8) -> SearchResponse:
        """Search the web via fathom's DuckDuckGo-backed endpoint."""
        try:
            resp = await self._http.post(
                "/search",
                json={"query": query[:400], "max_results": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
        except httpx.TimeoutException:
            log.warning("fathom_search_timeout", query=query[:80])
            return SearchResponse(results=(), query=query, error="Search timeout")
        except httpx.HTTPStatusError as exc:
            log.warning("fathom_http_error", status=exc.response.status_code, query=query[:80])
            return SearchResponse(results=(), query=query, error=f"HTTP {exc.response.status_code}")
        except Exception as exc:
            log.warning("fathom_search_failed", query=query[:80], error=str(exc))
            return SearchResponse(results=(), query=query, error=str(exc))

        raw_results = data.get("results", [])
        results = tuple(
            SearchResult(
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("snippet", ""),
                content=r.get("content", ""),
            )
            for r in raw_results
            if isinstance(r, dict) and r.get("url")
        )
        log.info("fathom_search", query=query[:60], results=len(results))
        return SearchResponse(results=results, query=query)

    async def multi_search(
        self, queries: list[str], *, max_results: int = 5
    ) -> list[SearchResponse]:
        """Parallel search across multiple queries."""
        import asyncio

        responses = await asyncio.gather(
            *(self.search(q, max_results=max_results) for q in queries),
            return_exceptions=True,
        )
        ok: list[SearchResponse] = []
        for r in responses:
            if isinstance(r, SearchResponse):
                ok.append(r)
            elif isinstance(r, BaseException):
                log.warning("multi_search_failed", error=str(r))
        return ok

    async def extract(self, url: str) -> ExtractResult:
        """Extract content from a URL via fathom's Playwright + trafilatura.

        Raises on HTTP or connection failure.
        """
        resp = await self._http.post("/extract", json={"url": url})
        resp.raise_for_status()
        data = resp.json()
        return ExtractResult(
            url=url,
            content=data.get("content", ""),
            title=data.get("title", ""),
        )

    async def start_research(
        self,
        goal: str,
        *,
        max_pages: int | None = None,
        n: int | None = None,
        seeds: list[str] | None = None,
        depth: str = "standard",
    ) -> ResearchSession:
        """Start a full research session. Returns immediately.

        Depth presets control n and max_pages: shallow, standard, deep, exhaustive.
        Only pass max_pages/n to override the depth preset.
        Raises on failure — caller decides how to degrade.
        """
        payload: dict[str, object] = {"goal": goal, "seeds": seeds or [], "depth": depth}
        if max_pages is not None:
            payload["max_pages"] = max_pages
        if n is not None:
            payload["n"] = n
        resp = await self._http.post("/research", json=payload, timeout=30.0)
        resp.raise_for_status()
        data = resp.json()
        return ResearchSession(session_id=data["id"], status=data["status"])

    async def stream_research(self, session_id: str) -> AsyncIterator[ResearchProgress]:
        """Connect to SSE stream for a research session. Yields progress events."""
        async with self._http.stream("GET", f"/research/{session_id}/stream", timeout=None) as resp:
            resp.raise_for_status()
            event_type = ""
            data_buffer = ""
            async for line in resp.aiter_lines():
                if line.startswith("event:"):
                    event_type = line[6:].strip()
                elif line.startswith("data:"):
                    data_buffer = line[5:].strip()
                elif line == "" and data_buffer:
                    try:
                        payload = json.loads(data_buffer)
                    except json.JSONDecodeError:
                        log.warning("sse_json_decode_failed", data=data_buffer[:80])
                        payload = {}
                    is_complete = event_type == "complete"
                    yield ResearchProgress(
                        event=event_type or "progress",
                        pages=payload.get("pages", 0),
                        facts=payload.get("facts", 0),
                        checklist_answered=payload.get("checklist_answered", 0),
                        checklist_total=payload.get("checklist_total", 0),
                        document_length=payload.get("document_length", 0),
                        status=payload.get("status", ""),
                    )
                    data_buffer = ""
                    event_type = ""
                    if is_complete:
                        break

    async def get_research_result(self, session_id: str) -> ResearchResult:
        """Fetch the final research result for a completed session."""
        resp = await self._http.get(f"/research/{session_id}")
        resp.raise_for_status()
        data = resp.json()
        return ResearchResult(
            session_id=session_id,
            status=data.get("status", ""),
            document=data.get("document", ""),
            pages_scraped=data.get("pages_scraped", 0),
        )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        await self._http.aclose()
