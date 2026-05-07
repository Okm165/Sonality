"""Web search, extraction, and deep research tools — delegated to fathom service.

Sonality sends queries directly to fathom. No query expansion on this side —
fathom handles search breadth. Results are passed through as-is from fathom's
search, extract, and research APIs.
"""

from __future__ import annotations

import asyncio
import time
from typing import Final

import structlog

from ..schema import ToolName
from ..web_client import SearchResult
from . import ToolContext

log = structlog.get_logger()

_VALID_DEPTHS: Final = ("shallow", "standard", "deep", "exhaustive")

WEB_SEARCH_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.WEB_SEARCH,
        "description": (
            "Search the live web for current facts, recent events, or source verification. "
            "Best suited when recall_memory returned insufficient or no results. "
            "Specific queries with names, dates, or institutions yield better results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Specific search query — include names, dates, or institutions (under 400 chars)",
                },
            },
            "required": ["query"],
        },
    },
}

WEB_EXTRACT_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.WEB_EXTRACT,
        "description": (
            "Fetch full content from a specific URL. "
            "Useful when a web_search result looks highly relevant but the snippet is too short. "
            "Best results come from exact URLs from prior search results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to extract (from a prior web_search result)",
                },
            },
            "required": ["url"],
        },
    },
}

WEB_RESEARCH_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.WEB_RESEARCH,
        "description": (
            "Launch a deep multi-page research session. "
            "Use for complex questions that need synthesis across many sources — "
            "produces a structured research document with cited evidence. "
            "Match depth to the question's complexity: shallow for quick factual surveys, "
            "standard for typical research questions, deep for comprehensive analysis "
            "requiring multiple perspectives, exhaustive for thorough investigation "
            "of contested or complex topics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "Clear research goal — what you want to learn, with enough context for useful results",
                },
                "depth": {
                    "type": "string",
                    "enum": ["shallow", "standard", "deep", "exhaustive"],
                    "description": "Research depth: shallow for quick facts, standard for typical research, deep/exhaustive for thorough investigation",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Override maximum pages to scrape (1-500). Only set when the depth preset is insufficient.",
                },
                "pages_per_round": {
                    "type": "integer",
                    "description": "Override pages fetched per round (1-20). Higher means broader but shallower per-round.",
                },
                "seeds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional seed URLs to include in the research frontier",
                },
            },
            "required": ["goal", "depth"],
        },
    },
}

DEFINITIONS: Final = [WEB_SEARCH_DEFINITION, WEB_EXTRACT_DEFINITION, WEB_RESEARCH_DEFINITION]


def _format_results(results: list[SearchResult]) -> str:
    """Format search results for the agent. Keeps it simple — fathom does the heavy lifting."""
    lines: list[str] = []
    for i, r in enumerate(results[:10], 1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"    {r.url}")
        content = r.markdown or r.description
        if content:
            lines.append(f"    {content[:300]}")
        lines.append("")
    return "\n".join(lines)


def execute_web_search(args: dict[str, object], ctx: ToolContext) -> str:
    """Search the web via fathom — single query, no expansion."""
    query = str(args.get("query", ""))[:400]
    if not ctx.web_client:
        return "Web search unavailable."

    t0 = time.perf_counter()
    log.info("web_search", query=query[:80])

    resp = ctx.run_async(ctx.web_client.search(query))

    if resp.failed:
        log.warning("web_search_failed", error=resp.error, query=query[:60])
        return f"Search failed: {resp.error}"

    if resp.results:
        formatted = _format_results(list(resp.results))
        log.info("web_search_done", results=len(resp.results), chars=len(formatted), elapsed=f"{time.perf_counter() - t0:.1f}s")
        return f"[{len(resp.results)} results for: {query}]\n\n{formatted}"

    log.info("web_search_empty", elapsed=f"{time.perf_counter() - t0:.1f}s")
    return f"No results for: {query}. Try different keywords or a more specific query."


def execute_web_extract(args: dict[str, object], ctx: ToolContext) -> str:
    """Extract content from a URL via fathom's Playwright scraper."""
    url = str(args.get("url", ""))
    if not url:
        return "Error: no URL provided"
    if not ctx.web_client:
        return "Web extract unavailable."
    log.info("web_extract", url=url[:80])
    try:
        result = ctx.run_async(ctx.web_client.extract(url))
    except Exception:
        log.error("web_extract_failed", url=url[:60], exc_info=True)
        return f"Failed to extract content from: {url}"
    if not result.content:
        return f"Page returned no extractable content: {url}"
    content = result.content[:6000]
    log.info("web_extract_done", chars=len(content))
    return content


def execute_web_research(args: dict[str, object], ctx: ToolContext) -> str:
    """Launch a deep research session via fathom and poll until completion."""
    goal = str(args.get("goal", ""))
    if not goal:
        return "Error: no research goal provided"
    if not ctx.web_client:
        return "Web research unavailable."

    depth = str(args.get("depth", "standard"))
    if depth not in _VALID_DEPTHS:
        depth = "standard"
    raw_seeds = args.get("seeds")
    seeds = [str(s) for s in raw_seeds if s] if isinstance(raw_seeds, list) else []

    max_pages: int | None = None
    n: int | None = None
    raw_max = args.get("max_pages")
    if isinstance(raw_max, (int, float)) and 1 <= int(raw_max) <= 500:
        max_pages = int(raw_max)
    raw_n = args.get("pages_per_round")
    if isinstance(raw_n, (int, float)) and 1 <= int(raw_n) <= 20:
        n = int(raw_n)

    log.info("web_research", goal=goal[:80], depth=depth, max_pages=max_pages, n=n, seeds=len(seeds))
    t0 = time.perf_counter()

    try:
        session = ctx.run_async(ctx.web_client.start_research(
            goal, depth=depth, seeds=seeds, max_pages=max_pages, n=n,
        ))
    except Exception:
        log.error("research_start_failed", goal=goal[:60], exc_info=True)
        return "Failed to start research session."

    log.info("research_started", session_id=session.session_id[:8], depth=depth)

    async def _poll() -> str:
        """Poll fathom for the research document."""
        client = ctx.web_client
        assert client is not None
        async for progress in client.stream_research(session.session_id):
            if progress.status in ("completed", "failed", "stalled"):
                break
        result = await client.get_research_result(session.session_id)
        return result.document

    try:
        document = ctx.run_async(asyncio.wait_for(_poll(), timeout=600))
    except TimeoutError:
        log.warning("research_timeout", session_id=session.session_id[:8], elapsed=f"{time.perf_counter() - t0:.1f}s")
        return f"Research session timed out after {time.perf_counter() - t0:.0f}s. Session {session.session_id} may still be running."

    elapsed = time.perf_counter() - t0
    if not document:
        log.warning("research_empty", session_id=session.session_id[:8], elapsed=f"{elapsed:.1f}s")
        return "Research completed but produced no document."

    log.info("research_done", session_id=session.session_id[:8], chars=len(document), elapsed=f"{elapsed:.1f}s")
    return document


EXECUTORS: Final = {
    ToolName.WEB_SEARCH: execute_web_search,
    ToolName.WEB_EXTRACT: execute_web_extract,
    ToolName.WEB_RESEARCH: execute_web_research,
}
