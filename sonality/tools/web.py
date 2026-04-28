"""Web search and content extraction tools."""

from __future__ import annotations

import logging
from typing import Final

from ..schema import ToolName
from ..web.context import format_web_context, sanitize_web_content
from . import ToolContext

log = logging.getLogger(__name__)

WEB_SEARCH_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.WEB_SEARCH,
        "description": "Investigate claims against external reality.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Focused search query (under 400 chars)",
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
        "description": "Get full content from a specific URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "URL to extract content from",
                },
            },
            "required": ["url"],
        },
    },
}

DEFINITIONS: Final = [WEB_SEARCH_DEFINITION, WEB_EXTRACT_DEFINITION]


def execute_web_search(args: dict[str, object], ctx: ToolContext) -> str:
    """Search the web with full content scraping via Firecrawl."""
    query = str(args.get("query", ""))[:400]
    if not ctx.web_client:
        return "Web search unavailable."
    log.info("web_search: q=%.80s", query)
    response = ctx.run_async(ctx.web_client.search(query))
    if not response.results:
        log.info("web_search: no results for q=%.60s", query)
        return f"No results found for: {query}"
    formatted = format_web_context(list(response.results[:8]), max_chars=6000)
    log.info("web_search: %d results, %d chars", len(response.results), len(formatted))
    return formatted


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
