"""Web search and content extraction tools."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Final

from pydantic import BaseModel, model_validator

from ..schema import ToolName
from ..web.context import format_web_context, sanitize_web_content
from ..web.search import SearchResponse, SearchResult, WebSearchClient
from . import ToolContext

log = logging.getLogger(__name__)


class _QueryVariantsSchema(BaseModel):
    """Multiple search query variants for parallel execution."""

    queries: list[str] = []

    @model_validator(mode="before")
    @classmethod
    def coerce_shape(cls, data: object) -> object:
        if isinstance(data, dict):
            v = data.get("queries")
            if isinstance(v, str):
                data["queries"] = [q.strip() for q in v.split("\n") if q.strip()]
        return data


WEB_SEARCH_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.WEB_SEARCH,
        "description": (
            "Search the live web for current facts, recent events, or source verification. "
            "Best suited when recall_memory returned insufficient or no results. "
            "Specific queries with names, dates, or institutions yield better results. "
            "Multiple focused searches tend to outperform one broad query."
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

DEFINITIONS: Final = [WEB_SEARCH_DEFINITION, WEB_EXTRACT_DEFINITION]


async def _parallel_search(
    queries: list[str], web_client: WebSearchClient
) -> tuple[list[SearchResult], list[str]]:
    """Execute multiple search queries in parallel and merge results.

    Returns (deduplicated_results, queries_used).
    """
    tasks = [web_client.search(q[:400]) for q in queries]
    responses: list[SearchResponse] = await asyncio.gather(*tasks)

    seen_urls: set[str] = set()
    merged: list[SearchResult] = []
    queries_used: list[str] = []

    for query, resp in zip(queries, responses, strict=True):
        if resp.failed:
            log.warning("web_search: query failed (error=%s): %.60s", resp.error, query)
            continue
        if resp.results:
            queries_used.append(query)
        for r in resp.results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                merged.append(r)

    return merged, queries_used


def execute_web_search(args: dict[str, object], ctx: ToolContext) -> str:
    """Search the web with full content scraping via Firecrawl.

    Uses a reasoning model to generate multiple query variants upfront, executes
    all in parallel, then merges and deduplicates results. This approach produces
    better coverage than single-query search with reactive reformulation.
    """
    query = str(args.get("query", ""))[:400]
    if not ctx.web_client:
        return "Web search unavailable."

    from .. import config
    from ..llm.caller import llm_call
    from ..prompts import QUERY_EXPANSION_PROMPT

    t0 = time.perf_counter()
    log.info("web_search: original query=%.80s", query)

    # Generate query variants using structured model (needs reliable JSON output)
    transcript = ctx.build_research_transcript(tool_tail=4, assistant_tail=1)
    expansion = llm_call(
        prompt=QUERY_EXPANSION_PROMPT.format(query=query, context=transcript[:1200]),
        response_model=_QueryVariantsSchema,
        fallback=_QueryVariantsSchema(queries=[query]),
        model=config.STRUCTURED_MODEL,
    )

    # Always include original query, add variants (up to 4 total)
    all_queries = [query]
    for variant in expansion.value.queries:
        v = variant.strip()[:400]
        if v and v.lower() != query.lower() and v not in all_queries:
            all_queries.append(v)
        if len(all_queries) >= 4:
            break

    log.info("web_search: expanded to %d queries: %s", len(all_queries), all_queries)

    # Execute all queries in parallel
    results, queries_used = ctx.run_async(_parallel_search(all_queries, ctx.web_client))

    if results:
        formatted = format_web_context(results[:10], max_chars=8000)
        log.info(
            "web_search: %d unique results from %d queries, %d chars (%.1fs)",
            len(results),
            len(queries_used),
            len(formatted),
            time.perf_counter() - t0,
        )
        header = f"[Searched: {', '.join(queries_used[:3])}{'...' if len(queries_used) > 3 else ''}]\n\n"
        return header + formatted

    log.info("web_search: no results for any variant (%.1fs)", time.perf_counter() - t0)
    return f"No results found for: {query} (also tried {len(all_queries) - 1} variants)"


def execute_web_extract(args: dict[str, object], ctx: ToolContext) -> str:
    """Extract and sanitize content from a URL (markdown via Firecrawl scrape)."""
    url = str(args.get("url", ""))
    if not url:
        return "Error: no URL provided"
    if not ctx.web_client:
        return "Web extract unavailable."
    log.info("web_extract: url=%.80s", url)
    result = ctx.run_async(ctx.web_client.extract(url))
    if not result.content:
        log.warning("web_extract: no content from url=%.60s", url)
        return f"Could not extract content from: {url}"
    sanitized = sanitize_web_content(result.content, max_chars=6000)
    log.info("web_extract: %d raw → %d sanitized chars", len(result.content), len(sanitized))
    return sanitized


EXECUTORS: Final = {
    ToolName.WEB_SEARCH: execute_web_search,
    ToolName.WEB_EXTRACT: execute_web_extract,
}
