"""DuckDuckGo search wrapper.

DDGS returns at most ~10 results per call regardless of max_results.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import structlog
from ddgs import DDGS

from .config import settings
from .models import Link

log = structlog.get_logger()

_semaphore: asyncio.Semaphore | None = None


def _get_semaphore() -> asyncio.Semaphore:
    global _semaphore
    if _semaphore is None:
        _semaphore = asyncio.Semaphore(settings.search_concurrency)
    return _semaphore


async def query(q: str, max_results: int = 10) -> list[Link]:
    """Run a single DuckDuckGo search. Returns Links with anchor_text=title, context=snippet."""
    sem = _get_semaphore()
    async with sem:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _sync_search, q, max_results)
    log.info("search_complete", query=q, results=len(results))
    return results


def _sync_search(q: str, max_results: int) -> list[Link]:
    with DDGS() as ddgs:
        raw = list(ddgs.text(q, max_results=max_results))
    return [
        Link(url=r["href"], anchor_text=r.get("title", ""), context=r.get("body", ""))
        for r in raw
        if r.get("href")
    ]


async def search_many(queries: Sequence[str], max_results: int = 10) -> list[Link]:
    """Run multiple searches in parallel, return deduplicated links.

    Individual query failures are logged and skipped — partial results
    are better than none.
    """
    raw = await asyncio.gather(*[query(q, max_results) for q in queries], return_exceptions=True)
    batches: list[list[Link]] = []
    for i, result in enumerate(raw):
        if isinstance(result, BaseException):
            log.warning("search_query_failed", query=queries[i], error=str(result))
        else:
            batches.append(result)
    seen: set[str] = set()
    out: list[Link] = []
    for batch in batches:
        for link in batch:
            if link.url not in seen:
                seen.add(link.url)
                out.append(link)
    return out
