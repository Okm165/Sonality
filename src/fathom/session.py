"""Research loop — fetch, extract, gather facts. Synthesis is Sonality's job."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import structlog
from qdrant_client import AsyncQdrantClient

from neo4j import AsyncDriver

if TYPE_CHECKING:
    from shared.embedder import Embedder

from . import browser, db, llm, search
from .config import settings
from .extract import extract_content, extract_preview
from .models import (
    ChecklistItem,
    ExtractedPage,
    Fact,
    Link,
    PageAnalysisResult,
    SessionMemory,
    SourceEntry,
)
from .ranking import build_ranked_knowledge_context, rank_urls_hybrid
from .source_memory import (
    diverge_probabilistically,
    record_source,
    suggest_sources,
)

log = structlog.get_logger(__name__)


def _emit(
    queue: asyncio.Queue[dict[str, object] | None] | None, event: str, **data: object
) -> None:
    """Push a structured event to the session queue (non-blocking, fire-and-forget)."""
    if queue is None:
        return
    queue.put_nowait({"event": event, **data})


def _rows_to_links(rows: Sequence[Mapping[str, object]]) -> list[Link]:
    """Convert raw Neo4j URL rows to typed Link objects."""
    return [
        Link(
            url=str(r.get("url", "")),
            anchor_text=str(r.get("anchor_text", "")),
            context=str(r.get("context", "")),
        )
        for r in rows
    ]


async def _fetch_previews(urls: list[Link]) -> tuple[dict[str, str], dict[str, str]]:
    """Fetch page previews via Playwright for JS-rendered content.

    Returns both parsed preview strings AND raw HTML cache — pages selected
    after scoring skip the full re-fetch and use cached HTML directly.
    """
    raw_pages = await browser.fetch_preview_batch([lk.url for lk in urls])
    previews: dict[str, str] = {}
    html_cache: dict[str, str] = {}
    for link, result in zip(urls, raw_pages, strict=True):
        if isinstance(result, Exception):
            continue
        html_cache[link.url] = result
        preview = extract_preview(result)
        if preview:
            previews[link.url] = preview
    log.debug("previews_fetched", requested=len(urls), got=len(previews), cached=len(html_cache))
    return previews, html_cache


_DEPTH_PRESETS: dict[str, tuple[int, int, int, int]] = {
    # (batch_size, max_pages, max_checklist, url_pool_size)
    "glance": (1, 1, 3, 10),
    "quick": (2, 3, 5, 50),
    "focused": (3, 8, 10, 200),
    "standard": (6, 30, 25, 1500),
    "thorough": (8, 60, 40, 3000),
    "deep": (10, 100, 60, 5000),
    "exhaustive": (12, 200, 80, 10000),
}


async def _build_knowledge_context(
    embedder: Embedder,
    sources: list[SourceEntry],
    accumulated_facts: list[tuple[str, str, float, float]] | None = None,
    questions: list[str] | None = None,
    goal: str = "",
) -> str:
    """Build knowledge context — uses ranked facts if available, else summaries.

    accumulated_facts: list of (claim, source_url, confidence, source_quality) tuples
    """
    if accumulated_facts and questions:
        return await build_ranked_knowledge_context(
            embedder, accumulated_facts, questions, goal, max_facts=20, char_limit=2000
        )

    claims: list[str] = []
    for src in sources:
        if src.page_quality > 0.3 and src.summary:
            claims.append(src.summary[:200])
    return "\n".join(claims[-20:])[:2000] or "None yet"


async def run(
    driver: AsyncDriver,
    qdrant: AsyncQdrantClient,
    session_id: str,
    goal: str,
    seeds: list[str],
    *,
    n: int | None = None,
    max_pages: int | None = None,
    depth: str = "standard",
    event_queue: asyncio.Queue[dict[str, object] | None] | None = None,
) -> None:
    """Execute the research session — discover URLs, fetch, extract facts.

    Uses embedding-only URL ranking (no LLM scoring).
    Leverages Qdrant + Neo4j source memory for cross-session learning.
    """
    preset_batch, preset_pages, max_checklist, preset_pool = _DEPTH_PRESETS.get(
        depth, _DEPTH_PRESETS["standard"]
    )
    batch_size = n or preset_batch
    max_pages = max_pages or preset_pages
    url_pool_size = preset_pool
    goal = goal[:2000]
    eq = event_queue
    bound_log = log.bind(session_id=session_id[:12])
    t_start = time.monotonic()
    timeout = settings.session_timeout
    embedder = await asyncio.to_thread(settings.make_embedder)

    # ═══ INIT ═══
    bound_log.info("session_init", goal=goal[:120], timeout_s=timeout)
    try:
        checklist_items = await llm.decompose_goal(goal)
    except Exception:
        bound_log.warning("decompose_failed_using_goal", exc_info=True)
        checklist_items = [ChecklistItem(question=goal)]
    bound_log.info("decompose_done", questions=len(checklist_items))
    _emit(eq, "decompose", questions=[it.question for it in checklist_items[:8]])
    await db.save_checklist(driver, session_id, checklist_items)

    questions = [item.question for item in checklist_items]

    # ═══ SOURCE MEMORY: Suggest known productive sources ═══
    suggested = await suggest_sources(qdrant, driver, embedder, goal, questions, limit=15)
    suggested_links = [
        Link(
            url=s.url,
            anchor_text=f"[memory:{s.quality_rate:.0%}]",
            context="from source memory",
        )
        for s in suggested
    ]
    if suggested_links:
        bound_log.info("memory_suggested", count=len(suggested_links))
        _emit(
            eq,
            "memory_sources",
            count=len(suggested_links),
            sources=[{"url": s.url[:60], "score": round(s.score, 2)} for s in suggested[:5]],
        )

    # ═══ SEARCH: Fresh discovery ═══
    # Search with both the goal (contains subject name) and decomposed questions.
    # Questions may use pronouns that search engines can't resolve, so the goal
    # query provides a reliable anchor of relevant results.
    initial_links = await search.search_many([goal] + questions)
    seed_links = [Link(url=u, anchor_text="", context="seed") for u in seeds]
    all_initial = initial_links + seed_links + suggested_links

    frontier_tuples = [(lk.url, lk.anchor_text, lk.context, "search") for lk in all_initial]
    await db.add_urls(driver, session_id, frontier_tuples)

    memory = SessionMemory()
    all_sources: list[SourceEntry] = []
    accumulated_facts: list[
        tuple[str, str, float, float]
    ] = []  # (claim, source_url, confidence, source_quality)
    pages_fetched = 0
    round_num = 0

    # ═══ RESEARCH LOOP ═══
    while pages_fetched < max_pages:
        elapsed = time.monotonic() - t_start
        if elapsed > timeout:
            bound_log.info("session_timeout", elapsed_s=round(elapsed), timeout_s=timeout)
            _emit(eq, "timeout", elapsed_s=round(elapsed), pages=pages_fetched)
            break
        round_num += 1
        pending_links = _rows_to_links(await db.get_pending_urls(driver, session_id))
        bound_log.info("round_start", round=round_num, frontier_size=len(pending_links))

        # --- DISCOVERY (when frontier runs thin) ---
        questions = [item.question for item in checklist_items]
        if len(pending_links) < batch_size * 3 and questions:
            bound_log.debug("discovery_triggered", reason="frontier_low")
            try:
                raw_queries = await llm.generate_queries(
                    goal,
                    questions,
                    memory.productive_urls,
                    memory.unproductive_urls,
                    memory.facts_per_round,
                    "frontier_low",
                )
                new_queries = list(dict.fromkeys(raw_queries))
                _emit(eq, "searching", queries=new_queries[:5], round=round_num)
                new_links = await search.search_many(new_queries)
                new_tuples = [(lk.url, lk.anchor_text, lk.context, "discovery") for lk in new_links]
                await db.add_urls(driver, session_id, new_tuples)
                pending_links = _rows_to_links(await db.get_pending_urls(driver, session_id))
            except Exception:
                bound_log.warning("discovery_failed", exc_info=True)

        # --- PROBABILISTIC DIVERGENCE: Inject novel sources from memory ---
        if round_num > 1:
            known_urls = [lk.url for lk in pending_links[:20]] + memory.productive_urls
            try:
                divergent_urls = await diverge_probabilistically(
                    qdrant,
                    embedder,
                    known_urls,
                    goal,
                    divergence=settings.exploration_divergence,
                    limit=5,
                )
                if divergent_urls:
                    divergent_links = [
                        (url, "[divergent]", "exploration", "diverge") for url in divergent_urls
                    ]
                    await db.add_urls(driver, session_id, divergent_links)
                    bound_log.info("divergence_injected", count=len(divergent_urls))
            except Exception as exc:
                bound_log.warning("divergence_failed", error=str(exc))

        if not pending_links:
            bound_log.debug("frontier_exhausted")
            break

        # --- EMBEDDING-BASED URL SELECTION (no LLM scoring) ---
        remaining = max_pages - pages_fetched

        # Take large pool for embedding ranking (1000+ URLs is fast)
        ranking_pool = pending_links[:url_pool_size]
        bound_log.debug("ranking_pool", size=len(ranking_pool))

        ranked = await rank_urls_hybrid(
            embedder,
            ranking_pool,
            goal=goal,
            questions=questions,
            memory=memory,
            top_k=min(batch_size * 3, remaining * 2),
            temperature=settings.sample_temperature,
        )
        _emit(
            eq, "ranked", pool=len(ranking_pool), top_k=len(ranked), mode="sampled", round=round_num
        )
        bound_log.info(
            "url_selection",
            pool=len(ranking_pool),
            selected=len(ranked),
            mode="sampled",
            top_score=round(ranked[0][2], 4) if ranked else 0,
        )

        # Fetch previews for top candidates to verify content quality
        preview_candidates = [link for _, link, _ in ranked]
        previews, html_cache = await _fetch_previews(preview_candidates)

        # Select batch: first half accepts any URL, second half prefers previews
        batch_n = min(batch_size, remaining)
        min_unconditional = max(1, batch_n // 2)
        batch_links: list[Link] = []
        for _, link, _score in ranked:
            if len(batch_links) >= batch_n:
                break
            if link.url in previews or len(batch_links) < min_unconditional:
                batch_links.append(link)
        batch_urls = [lk.url for lk in batch_links]

        # Build knowledge context for page analysis (not URL scoring)
        knowledge_ctx = await _build_knowledge_context(
            embedder,
            all_sources,
            accumulated_facts=accumulated_facts,
            questions=questions,
            goal=goal,
        )

        _emit(
            eq,
            "fetching",
            urls=[u[:100] for u in batch_urls[:4]],
            count=len(batch_urls),
            round=round_num,
        )

        # --- FETCH (reuse preview HTML when available) ---
        uncached_urls = [u for u in batch_urls if u not in html_cache]
        if uncached_urls:
            fetched = await browser.fetch_batch(uncached_urls)
            fetched_map = dict(zip(uncached_urls, fetched, strict=True))
        else:
            fetched_map = {}
        raw_pages: list[str | Exception] = [
            html_cache[u] if u in html_cache else fetched_map[u] for u in batch_urls
        ]
        bound_log.debug(
            "fetch_complete",
            from_cache=sum(1 for u in batch_urls if u in html_cache),
            fresh=len(uncached_urls),
            success=sum(1 for p in raw_pages if not isinstance(p, Exception)),
        )

        # --- EXTRACT (CPU-bound trafilatura — run off event loop) ---
        def _extract_all(
            pages: list[str | Exception],
            links: list[Link],
        ) -> tuple[list[tuple[Link, ExtractedPage]], list[str]]:
            ok, fail = [], []
            for html, link in zip(pages, links, strict=True):
                if isinstance(html, Exception):
                    fail.append(link.url)
                    continue
                ok.append((link, extract_content(html, link.url)))
            return ok, fail

        extracted, failed_urls = await asyncio.to_thread(
            _extract_all,
            raw_pages,
            batch_links,
        )

        succeeded_urls = [link.url for link, _ in extracted]
        all_attempted = succeeded_urls + failed_urls
        if all_attempted:
            await db.mark_urls_fetched(driver, session_id, all_attempted)
        if failed_urls:
            bound_log.info("fetch_failures", count=len(failed_urls))

        pages_fetched += len(extracted)
        await db.update_session(driver, session_id, pages_scraped=pages_fetched)

        # --- ANALYZE ---
        content_pages = [(link, page) for link, page in extracted if page.has_content]
        analysis_tasks = [
            llm.analyze_page(
                goal,
                questions,
                page.markdown,
                _format_links(page.links),
                knowledge_context=knowledge_ctx,
            )
            for _, page in content_pages
        ]
        raw_analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        analysis_map: dict[str, PageAnalysisResult] = {}
        for (link, _page), result in zip(content_pages, raw_analyses, strict=True):
            if isinstance(result, PageAnalysisResult):
                analysis_map[link.url] = result
                bound_log.info(
                    "page_analyzed",
                    url=link.url[:80],
                    facts=len(result.facts),
                    links=len(result.follow_links),
                )
                if result.facts:
                    _emit(
                        eq,
                        "analyzed",
                        url=link.url[:100],
                        title=_page.title[:100],
                        facts=[f.claim[:120] for f in result.facts[:5]],
                        fact_count=len(result.facts),
                    )
            else:
                bound_log.warning("page_analysis_failed", url=link.url[:80], error=str(result))

        # --- STORE RESULTS ---
        round_fact_count = 0
        for link, page in extracted:
            analysis = analysis_map.get(link.url) if page.has_content else None
            if analysis is None:
                memory.unproductive_urls.append(link.url)
                memory.record_domain(link.url, page_quality=0.0, fact_count=0)
                all_sources.append(
                    SourceEntry(
                        url=link.url,
                        title=page.title,
                        page_quality=0.0,
                        facts_extracted=0,
                        summary="",
                    )
                )
                continue

            page_quality = (
                sum(f.source_quality for f in analysis.facts) / len(analysis.facts)
                if analysis.facts
                else 0.0
            )

            if analysis.facts:
                memory.productive_urls.append(link.url)
                memory.record_domain(
                    link.url, page_quality=page_quality, fact_count=len(analysis.facts)
                )
                await db.insert_facts(driver, session_id, analysis.facts, link.url)
                round_fact_count += len(analysis.facts)
                for f in analysis.facts:
                    accumulated_facts.append((f.claim, link.url, f.confidence, f.source_quality))
                try:
                    await record_source(
                        qdrant,
                        driver,
                        embedder,
                        url=link.url,
                        content=page.markdown[:3000],
                        page_quality=page_quality,
                        facts=[(f.claim, f.topic or "", f.confidence) for f in analysis.facts],
                        query_text=goal,
                    )
                except Exception as exc:
                    bound_log.warning(
                        "source_memory_write_failed", url=link.url[:60], error=str(exc)
                    )
                _emit(
                    eq,
                    "facts",
                    source_url=link.url[:100],
                    source_title=page.title[:120],
                    summary=analysis.summary[:500],
                    items=[
                        {
                            "claim": f.claim,
                            "confidence": f.confidence,
                            "source_quality": f.source_quality,
                            "topic": f.topic,
                        }
                        for f in analysis.facts
                    ],
                )
            else:
                memory.unproductive_urls.append(link.url)
                memory.record_domain(link.url, page_quality=0.0, fact_count=0)
                try:
                    await record_source(
                        qdrant,
                        driver,
                        embedder,
                        url=link.url,
                        content=page.markdown[:1000] if page.has_content else "",
                        page_quality=0.0,
                        facts=[],
                        query_text=goal,
                    )
                except Exception:
                    bound_log.debug("source_memory_write_skipped", url=link.url[:60])

            all_sources.append(
                SourceEntry(
                    url=link.url,
                    title=page.title,
                    page_quality=page_quality,
                    facts_extracted=len(analysis.facts),
                    summary=analysis.summary,
                )
            )

            # Follow links from analyzed pages
            follow_tuples: list[tuple[str, str, str, str]] = []
            for idx in analysis.follow_links:
                if 0 <= idx < len(page.links):
                    fl = page.links[idx]
                    follow_tuples.append((fl.url, fl.anchor_text, fl.context, "link"))
            if follow_tuples:
                await db.add_urls(driver, session_id, follow_tuples)

            # Expand checklist with new questions from analysis
            if len(checklist_items) < max_checklist:
                existing_qs = {item.question for item in checklist_items}
                added = False
                for q in analysis.new_questions:
                    if q not in existing_qs and len(checklist_items) < max_checklist:
                        checklist_items.append(ChecklistItem(question=q))
                        added = True
                if added:
                    await db.save_checklist(driver, session_id, checklist_items)

        memory.facts_per_round.append(round_fact_count)

        total_facts = sum(s.facts_extracted for s in all_sources)
        productive_domains = sum(1 for s in memory.domain_stats.values() if s.quality_rate >= 0.5)
        bound_log.info(
            "round_end",
            round=round_num,
            round_facts=round_fact_count,
            total_facts=total_facts,
            productive_domains=productive_domains,
        )

        productive_sources = [
            {"url": s.url[:100], "title": s.title[:80], "facts": s.facts_extracted}
            for s in all_sources
            if s.page_quality > 0.3
        ][-10:]
        _emit(
            eq,
            "round_end",
            round=round_num,
            round_facts=round_fact_count,
            total_facts=total_facts,
            total_pages=pages_fetched,
            productive_domains=productive_domains,
            sources=productive_sources,
        )

    # ═══ OUTPUT ═══
    elapsed = time.monotonic() - t_start
    timed_out = elapsed > timeout
    final_status = "timed_out" if timed_out else "completed"

    fact_rows = await db.get_facts(driver, session_id)
    persisted_facts = [
        Fact(
            claim=str(r.get("claim", "")),
            confidence=float(r.get("confidence") or 0.5),
            source_quality=float(r.get("source_quality") or 0.5),
            topic=str(r.get("topic") or ""),
        )
        for r in fact_rows
    ]

    productive_count = len(memory.productive_urls)
    total_visited = productive_count + len(memory.unproductive_urls)
    bound_log.info(
        "session_complete",
        status=final_status,
        rounds=round_num,
        pages=pages_fetched,
        facts=len(persisted_facts),
        elapsed_s=round(elapsed),
        productive_rate=round(productive_count / max(1, total_visited), 2),
    )

    await db.update_session(
        driver,
        session_id,
        status=final_status,
        pages_scraped=pages_fetched,
        fact_count=len(persisted_facts),
    )
    _emit(
        eq,
        "complete",
        status=final_status,
        facts=len(persisted_facts),
        pages=pages_fetched,
        rounds=round_num,
    )


def _format_links(links: list[Link]) -> str:
    lines: list[str] = []
    for i, link in enumerate(links[:30]):
        lines.append(f"[{i}] {link.anchor_text} → {link.url}")
    return "\n".join(lines)
