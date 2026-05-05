"""Web search and content extraction tools."""

from __future__ import annotations

import logging
import time
from typing import Final

from pydantic import BaseModel, model_validator

from ..schema import ToolName
from ..web.context import format_web_context, sanitize_web_content
from . import ToolContext

log = logging.getLogger(__name__)


class _QueryReformulationSchema(BaseModel):
    """Reformulated search queries."""

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


def execute_web_search(args: dict[str, object], ctx: ToolContext) -> str:
    """Search the web with full content scraping via Firecrawl.

    If the primary query returns no results, attempts a single LLM-reformulated
    fallback query to broaden the search (cf. ReFormeR pattern — reformulation
    only on failure, not preemptively).
    """
    query = str(args.get("query", ""))[:400]
    if not ctx.web_client:
        return "Web search unavailable."
    t0 = time.perf_counter()
    log.info("web_search: q=%.80s", query)
    response = ctx.run_async(ctx.web_client.search(query))
    if response.results:
        formatted = format_web_context(list(response.results[:8]), max_chars=6000)
        log.info(
            "web_search: %d results, %d chars (%.1fs)",
            len(response.results),
            len(formatted),
            time.perf_counter() - t0,
        )
        return formatted
    from .. import config
    from ..llm.caller import llm_call
    from ..prompts import QUERY_REFORMULATION_PROMPT

    transcript = ctx.build_research_transcript(tool_tail=4, assistant_tail=1)
    r = llm_call(
        prompt=QUERY_REFORMULATION_PROMPT.format(query=query, context=transcript[:300]),
        response_model=_QueryReformulationSchema,
        fallback=_QueryReformulationSchema(),
        model=config.FAST_MODEL,
    )
    alt_query = r.value.queries[0][:400] if r.value.queries else ""
    if alt_query and alt_query.lower() != query.lower():
        log.info("web_search: primary returned 0 results, reformulated → %.80s", alt_query)
        response = ctx.run_async(ctx.web_client.search(alt_query))
        if response.results:
            formatted = format_web_context(list(response.results[:8]), max_chars=6000)
            log.info(
                "web_search: reformulated found %d results, %d chars (%.1fs)",
                len(response.results),
                len(formatted),
                time.perf_counter() - t0,
            )
            return f"[Reformulated query: {alt_query}]\n\n{formatted}"
    log.info("web_search: no results for q=%.60s (%.1fs)", query, time.perf_counter() - t0)
    return f"No results found for: {query}"


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
