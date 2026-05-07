"""Unified research loop — discovery and research are one activity."""

from __future__ import annotations

import asyncio

import structlog

from neo4j import AsyncDriver

from . import browser, db, llm, search
from .config import settings
from .extract import extract_content
from .models import (
    ChecklistItem,
    Contradiction,
    Fact,
    Link,
    PageAnalysisResult,
    ResearchOutput,
    SessionMemory,
    SourceEntry,
    URLScoring,
)
from .observe import DashboardState, ResearchDashboard
from .selection import format_urls_for_scoring, probabilistic_sample

log = structlog.get_logger()


_DEPTH_PRESETS: dict[str, tuple[int, int, int]] = {
    "shallow": (3, 20, 15),
    "standard": (4, 80, 30),
    "deep": (6, 150, 50),
    "exhaustive": (8, 300, 80),
}


async def run(
    driver: AsyncDriver,
    session_id: str,
    goal: str,
    seeds: list[str],
    *,
    n: int | None = None,
    max_pages: int | None = None,
    depth: str = "standard",
    dashboard: ResearchDashboard | None = None,
) -> ResearchOutput:
    """Execute the full research session. Core of the system."""
    preset_n, preset_pages, max_checklist = _DEPTH_PRESETS.get(depth, _DEPTH_PRESETS["standard"])
    n = n or preset_n
    max_pages = max_pages or preset_pages
    goal = goal[:2000]
    bound_log = log.bind(session_id=session_id[:12])

    # ═══ INIT ═══
    bound_log.info("session_init", goal=goal[:120])
    checklist_items = await llm.decompose_goal(goal)
    await db.save_checklist(driver, session_id, checklist_items)

    unanswered = [item.question for item in checklist_items if not item.answered]
    queries = await llm.generate_queries(goal, unanswered, [], [], [], "initial_discovery")

    # Seed frontier from search + user seeds
    initial_links = await search.search_many(queries)
    seed_links = [Link(url=u, anchor_text="", context="seed") for u in seeds]
    all_initial = initial_links + seed_links

    frontier_tuples = [(lk.url, lk.anchor_text, lk.context, "search") for lk in all_initial]
    await db.add_urls(driver, session_id, frontier_tuples)

    memory = SessionMemory()
    all_facts: list[SourceEntry] = []
    knowledge_summary = ""
    document_sections: dict[str, str] = {}
    written_sections: set[str] = set()
    pages_fetched = 0
    round_num = 0
    stalled = False

    if dashboard:
        dashboard.update(DashboardState(
            checklist=checklist_items,
            memory=memory,
        ))

    # ═══ UNIFIED LOOP ═══
    while pages_fetched < max_pages:
        round_num += 1
        pending_rows = await db.get_pending_urls(driver, session_id)
        pending_links = [
            Link(url=r["url"], anchor_text=r["anchor_text"], context=r["context"])
            for r in pending_rows
        ]
        bound_log.info("round_start", round=round_num, frontier_size=len(pending_links))

        # --- INTEGRATED DISCOVERY ---
        unanswered = [item.question for item in checklist_items if not item.answered]
        frontier_thin = len(pending_links) < n * 3
        has_unresolved_contradictions = bool(memory.contradictions)
        needs_discovery = (
            frontier_thin
            or memory.stall_rounds >= (settings.stall_limit - 1)
            or has_unresolved_contradictions
        )
        if needs_discovery and unanswered:
            trigger = (
                "contradiction_resolution"
                if has_unresolved_contradictions
                else "frontier_low"
                if frontier_thin
                else "stall"
            )
            bound_log.info("discovery_triggered", reason=trigger)
            if dashboard:
                dashboard.set_action(f"discovery triggered: {trigger}")

            new_queries = await llm.generate_queries(
                goal,
                unanswered,
                memory.productive_urls,
                memory.unproductive_urls,
                memory.facts_per_round,
                trigger,
            )
            new_links = await search.search_many(new_queries)
            new_tuples = [(lk.url, lk.anchor_text, lk.context, "discovery") for lk in new_links]
            await db.add_urls(driver, session_id, new_tuples)

            # Refresh pending
            pending_rows = await db.get_pending_urls(driver, session_id)
            pending_links = [
                Link(url=r["url"], anchor_text=r["anchor_text"], context=r["context"])
                for r in pending_rows
            ]
            memory.stall_rounds = 0

        if not pending_links:
            bound_log.info("frontier_exhausted")
            break

        # --- SCORE + PROBABILISTIC SAMPLE ---
        remaining = max_pages - pages_fetched
        batch_n = min(n, remaining)
        scorable = pending_links[:settings.max_urls_for_scoring]
        urls_text = format_urls_for_scoring(scorable)
        try:
            scoring = await llm.score_urls(
                urls_text,
                unanswered,
                knowledge_summary,
                memory.productive_urls,
                memory.unproductive_urls,
                memory.facts_per_round,
            )
        except Exception:
            bound_log.error("score_urls_failed_random_fallback", batch_size=len(scorable), exc_info=True)
            scoring = URLScoring(scores=[0.5] * len(scorable), concentration=1.0)
        selected_indices = probabilistic_sample(scorable, scoring, batch_n)
        batch_links = [scorable[i] for i in selected_indices]
        batch_urls = [lk.url for lk in batch_links]

        bound_log.info(
            "batch_selected",
            urls=batch_urls,
            concentration=scoring.concentration,
        )
        if dashboard:
            dashboard.set_action(f"fetching {len(batch_urls)} pages...")
            dashboard.update(DashboardState(
                round=round_num,
                pages_fetched=pages_fetched,
                total_facts=sum(s.facts_extracted for s in all_facts),
                checklist=checklist_items,
                memory=memory,
                concentration=scoring.concentration,
            ))

        # --- PARALLEL FETCH ---
        raw_pages = await browser.fetch_batch(batch_urls)
        bound_log.info(
            "fetch_complete",
            success=sum(1 for p in raw_pages if not isinstance(p, Exception)),
            failed=sum(1 for p in raw_pages if isinstance(p, Exception)),
        )

        # Mark all attempted URLs so they don't get retried
        await db.mark_urls_fetched(driver, session_id, batch_urls)

        # --- EXTRACT ---
        extracted = []
        for html, link in zip(raw_pages, batch_links, strict=True):
            if isinstance(html, Exception):
                continue
            page = extract_content(html, link.url)
            extracted.append((link, page))

        pages_fetched += len(extracted)
        await db.update_session(driver, session_id, pages_scraped=pages_fetched)

        # --- PARALLEL ANALYZE ---
        if dashboard:
            dashboard.set_action("analyzing pages...")

        content_pages = [(link, page) for link, page in extracted if page.has_content]
        no_content = len(extracted) - len(content_pages)
        if no_content:
            bound_log.info("pages_without_content", count=no_content)
        bound_log.info("analyzing_pages", count=len(content_pages))

        analysis_tasks = [
            llm.analyze_page(
                goal,
                unanswered,
                page.markdown,
                _format_links(page.links),
                knowledge_context=knowledge_summary,
            )
            for _, page in content_pages
        ]
        raw_analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)

        analysis_map: dict[str, PageAnalysisResult] = {}
        for (link, _page), result in zip(content_pages, raw_analyses, strict=True):
            if isinstance(result, PageAnalysisResult):
                analysis_map[link.url] = result
                bound_log.info(
                    "page_analysis",
                    url=link.url,
                    worth=result.worth_extracting,
                    facts=len(result.facts),
                    links=len(result.follow_links),
                )
            else:
                bound_log.warning("page_analysis_failed", url=link.url, error=str(result))

        # --- UPDATE EVERYTHING ---
        round_fact_count = 0
        for link, page in extracted:
            if not page.has_content:
                memory.unproductive_urls.append(link.url)
                all_facts.append(
                    SourceEntry(
                        url=link.url,
                        title=page.title,
                        productive=False,
                        facts_extracted=0,
                        summary="",
                    )
                )
                continue

            analysis = analysis_map.get(link.url)
            if analysis is None:
                memory.unproductive_urls.append(link.url)
                continue

            if analysis.facts:
                memory.productive_urls.append(link.url)
                await db.insert_facts(driver, session_id, analysis.facts, link.url)
                round_fact_count += len(analysis.facts)
                bound_log.info(
                    "productive",
                    url=link.url,
                    facts=len(analysis.facts),
                )
            else:
                memory.unproductive_urls.append(link.url)
                bound_log.info(
                    "unproductive",
                    url=link.url,
                )

            all_facts.append(
                SourceEntry(
                    url=link.url,
                    title=page.title,
                    productive=len(analysis.facts) > 0,
                    facts_extracted=len(analysis.facts),
                    summary=analysis.summary,
                )
            )

            # Organic frontier growth — links from pages
            follow_tuples: list[tuple[str, str, str, str]] = []
            for idx in analysis.follow_links:
                if 0 <= idx < len(page.links):
                    fl = page.links[idx]
                    follow_tuples.append((fl.url, fl.anchor_text, fl.context, "link"))
            if follow_tuples:
                await db.add_urls(driver, session_id, follow_tuples)

            # New questions → expand checklist (capped by depth)
            if len(checklist_items) < max_checklist:
                existing_qs = {item.question for item in checklist_items}
                for q in analysis.new_questions:
                    if q not in existing_qs and len(checklist_items) < max_checklist:
                        checklist_items.append(ChecklistItem(question=q))

        memory.facts_per_round.append(round_fact_count)

        # --- CONTRADICTION DETECTION ---
        round_facts_with_source: list[tuple[Fact, str]] = []
        for link, _page in extracted:
            analysis = analysis_map.get(link.url)
            if analysis and analysis.facts:
                for f in analysis.facts:
                    round_facts_with_source.append((f, link.url))
        new_contradictions = _detect_contradictions(
            round_facts_with_source, memory.contradictions
        )
        if new_contradictions:
            memory.contradictions.extend(new_contradictions)
            bound_log.info("contradictions_found", count=len(new_contradictions))

        if round_fact_count == 0:
            memory.stall_rounds += 1
            if memory.stall_rounds >= settings.stall_limit:
                bound_log.warning("stall_limit_reached", rounds=memory.stall_rounds)
                stalled = True
                break
        else:
            memory.stall_rounds = 0

        # --- CHECKLIST UPDATE (only when we have new evidence) ---
        # Collect actual extracted facts from this round's analyses
        round_fact_claims: list[str] = []
        for link, _page in extracted:
            analysis = analysis_map.get(link.url)
            if analysis and analysis.facts:
                for f in analysis.facts:
                    round_fact_claims.append(f"- {f.claim[:200]}")
        new_facts_text = "\n".join(round_fact_claims[:40])
        if new_facts_text:
            checklist_state = "\n".join(
                f"{'[x]' if item.answered else '[ ]'} {item.question[:200]}"
                for item in checklist_items[:30]
            )
            contradictions_text = (
                "\n".join(
                    f"- {c.claim_a} (from {c.source_a}) vs {c.claim_b} (from {c.source_b})"
                    for c in memory.contradictions[-5:]
                )
                if memory.contradictions
                else "None"
            )
            try:
                llm_checklist = await llm.update_checklist(
                    checklist_state,
                    new_facts_text[:3000],
                    contradictions_text,
                )
                checklist_items = _merge_checklists(checklist_items, llm_checklist)
                checklist_items = checklist_items[:max_checklist]
                await db.save_checklist(driver, session_id, checklist_items)
            except Exception:
                bound_log.error("checklist_update_failed", exc_info=True)
        else:
            bound_log.info("skipping_checklist_update_no_new_facts")

        # --- KNOWLEDGE SUMMARY ---
        if round_fact_count > 0:
            fact_rows = await db.get_facts(driver, session_id)
            budget = settings.llm_max_tokens * 2
            summary_budget = budget // 2
            facts_budget = budget // 2
            recent_claims = "\n".join(r["claim"][:300] for r in fact_rows[-30:])
            knowledge_summary = await llm.compress_knowledge(
                knowledge_summary[:summary_budget], recent_claims[:facts_budget]
            )
            knowledge_summary = knowledge_summary[:summary_budget]

        # --- PROGRESSIVE COMPOSITION ---
        newly_answered = [
            item
            for item in checklist_items
            if item.answered and item.question not in written_sections
        ]
        if newly_answered:
            section_tasks = [
                llm.write_section(
                    item.question[:200],
                    (
                        "\n".join(e[:500] for e in item.evidence[:10])
                        if item.evidence
                        else knowledge_summary[:1000]
                    ),
                )
                for item in newly_answered
            ]
            sections = await asyncio.gather(*section_tasks)
            for item, text in zip(newly_answered, sections, strict=True):
                document_sections[item.question] = text
                written_sections.add(item.question)
                bound_log.info("section_written", topic=item.question)

        # Update DB
        total_fact_count = sum(s.facts_extracted for s in all_facts)
        doc_text = "\n\n".join(document_sections.values())
        await db.update_session(driver, session_id, document=doc_text)

        answered_count = sum(1 for i in checklist_items if i.answered)
        bound_log.info(
            "round_end",
            round=round_num,
            total_facts=total_fact_count,
            checklist_progress=f"{answered_count}/{len(checklist_items)}",
        )
        bound_log.info(
            "health_snapshot",
            round=round_num,
            facts_per_round=memory.facts_per_round,
            productive_rate=(
                len(memory.productive_urls)
                / max(1, len(memory.productive_urls) + len(memory.unproductive_urls))
            ),
            frontier_pending=max(0, len(pending_links) - len(batch_urls)),
            checklist_answered=answered_count,
            checklist_total=len(checklist_items),
            contradictions_open=len(memory.contradictions),
            stall_rounds=memory.stall_rounds,
        )
        if dashboard:
            dashboard.update(DashboardState(
                round=round_num,
                pages_fetched=pages_fetched,
                total_facts=total_fact_count,
                checklist=checklist_items,
                memory=memory,
                concentration=scoring.concentration,
            ))

        # Check completion
        if all(item.answered for item in checklist_items):
            bound_log.info("all_items_answered")
            break

    # ═══ FINAL COMPOSITION ═══
    if document_sections:
        section_summaries = "\n".join(
            f"## {q}\n{text[:200]}..." for q, text in document_sections.items()
        )
        unanswered_final = [i.question for i in checklist_items if not i.answered]
        open_contradictions = (
            "\n".join(
                f"- {c.claim_a} vs {c.claim_b} (topic: {c.topic})"
                for c in memory.contradictions[-10:]
            )
            if memory.contradictions
            else "None"
        )
        intro_conclusion = await llm.write_intro_conclusion(
            goal[:500],
            section_summaries[:4000],
            "\n".join(q[:200] for q in unanswered_final[:15]) if unanswered_final else "None",
            open_contradictions[:2000],
        )
        final_doc = intro_conclusion + "\n\n" + "\n\n".join(document_sections.values())
    else:
        final_doc = knowledge_summary or "Research produced no document sections."

    final_status = "stalled" if stalled and not all(i.answered for i in checklist_items) else "completed"
    await db.update_session(driver, session_id, status=final_status, document=final_doc)

    # Retrieve persisted facts so the output carries the full evidence chain
    fact_rows = await db.get_facts(driver, session_id)
    persisted_facts = [
        Fact(
            claim=r["claim"],
            confidence=r["confidence"],
            has_evidence=r["has_evidence"],
            topic=r.get("topic", ""),
        )
        for r in fact_rows
    ]

    bound_log.info("session_complete", status=final_status, pages=pages_fetched, facts=len(persisted_facts))

    return ResearchOutput(
        document=final_doc,
        sources=all_facts,
        facts=persisted_facts,
        checklist=checklist_items,
        contradictions=memory.contradictions,
    )


def _merge_checklists(
    existing: list[ChecklistItem],
    updated: list[ChecklistItem],
) -> list[ChecklistItem]:
    """Merge LLM-updated checklist with existing, never un-answering items.

    Previously answered items stay answered even if the LLM forgets to mark them.
    New items from the LLM are appended. Evidence is accumulated.
    """
    answered_qs = {item.question for item in existing if item.answered}

    merged: list[ChecklistItem] = []
    seen: set[str] = set()

    for item in updated:
        if item.question in seen or not item.question.strip():
            continue
        seen.add(item.question)
        was_answered = item.question in answered_qs
        merged.append(
            ChecklistItem(
                question=item.question,
                answered=item.answered or was_answered,
                evidence=item.evidence,
            )
        )

    # Keep existing items that the LLM dropped
    for item in existing:
        if item.question not in seen:
            seen.add(item.question)
            merged.append(item)

    return merged


def _format_links(links: list[Link]) -> str:
    lines: list[str] = []
    for i, link in enumerate(links[:30]):
        lines.append(f"[{i}] {link.anchor_text} → {link.url}")
    return "\n".join(lines)


def _detect_contradictions(
    round_facts: list[tuple[Fact, str]],
    existing: list[Contradiction],
) -> list[Contradiction]:
    """Detect contradictions flagged by the LLM via the `contradicts` field on Fact."""
    new: list[Contradiction] = []
    existing_pairs = {(c.claim_a, c.claim_b) for c in existing}

    for fact, source_url in round_facts:
        if not fact.contradicts:
            continue
        topic = fact.topic or ""
        c = Contradiction(
            claim_a=fact.claim[:200],
            source_a=source_url,
            claim_b=f"contradicts fact IDs: {fact.contradicts}",
            source_b="",
            topic=topic,
        )
        if (c.claim_a, c.claim_b) not in existing_pairs:
            new.append(c)
    return new
