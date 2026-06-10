"""Web research tool — unified interface to fathom's research pipeline.

All web access goes through web_research with configurable depth presets.
The agent chooses the appropriate depth based on the question complexity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Final

import structlog

from ..schema import ToolName
from ..web_client import ResearchFact
from . import ToolContext

log = structlog.get_logger(__name__)

VALID_DEPTHS: Final = ("glance", "quick", "focused", "standard", "thorough", "deep", "exhaustive")


@dataclass(frozen=True, slots=True)
class ResearchArgs:
    """Normalized arguments for a web research call."""

    goal: str
    depth: str
    seeds: list[str]
    max_pages: int | None
    pages_per_round: int | None


def parse_research_args(args: dict[str, object]) -> ResearchArgs:
    """Normalize raw tool args for web research — single source of truth."""
    depth = str(args.get("depth", "standard"))
    if depth not in VALID_DEPTHS:
        depth = "standard"
    raw_seeds = args.get("seeds")
    seeds = [str(s) for s in raw_seeds if s] if isinstance(raw_seeds, list) else []
    max_pages: int | None = None
    raw_max = args.get("max_pages")
    if isinstance(raw_max, (int, float)) and 1 <= int(raw_max) <= 500:
        max_pages = int(raw_max)
    pages_per_round: int | None = None
    raw_n = args.get("pages_per_round")
    if isinstance(raw_n, (int, float)) and 1 <= int(raw_n) <= 20:
        pages_per_round = int(raw_n)
    return ResearchArgs(
        goal=str(args.get("goal", "")),
        depth=depth,
        seeds=seeds,
        max_pages=max_pages,
        pages_per_round=pages_per_round,
    )


WEB_RESEARCH_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.WEB_RESEARCH,
        "description": (
            "Research the live internet — your primary tool for building knowledge. "
            "Searches, scrapes, and analyzes web pages, returning extracted facts with "
            "sources and confidence. Use this freely to deepen your understanding, verify "
            "claims, discover current data, and broaden your knowledge on any topic. "
            "Depth scales from glance (single lookup) through exhaustive (comprehensive). "
            "Match depth to the question — use glance/quick for simple facts, focused/standard "
            "for analysis, thorough/deep for comprehensive research requests."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "What you want to learn — be specific for better results",
                },
                "depth": {
                    "type": "string",
                    "enum": [
                        "glance",
                        "quick",
                        "focused",
                        "standard",
                        "thorough",
                        "deep",
                        "exhaustive",
                    ],
                    "description": "Research depth — most questions need glance, quick, or focused",
                },
                "max_pages": {
                    "type": "integer",
                    "description": "Override maximum pages (1-500). Only set when the preset doesn't match.",
                },
                "pages_per_round": {
                    "type": "integer",
                    "description": "Pages analyzed per round (1-20). Higher = broader per-round.",
                },
                "seeds": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Seed URLs to prioritize in the research frontier",
                },
            },
            "required": ["goal", "depth"],
        },
    },
}

DEFINITIONS: Final = [WEB_RESEARCH_DEFINITION]


def merge_facts(
    streamed: list[ResearchFact],
    final: tuple[ResearchFact, ...],
) -> tuple[ResearchFact, ...]:
    """Merge streamed facts (rich metadata) with API facts, deduplicating by claim.

    Streamed facts are preferred because they carry source_title and summary
    from the page analysis context. API facts fill any gaps.
    """
    seen_claims: set[str] = set()
    merged: list[ResearchFact] = []
    for f in streamed:
        key = f.claim.strip().lower()
        if key not in seen_claims:
            seen_claims.add(key)
            merged.append(f)
    for f in final:
        key = f.claim.strip().lower()
        if key not in seen_claims:
            seen_claims.add(key)
            merged.append(f)
    return tuple(merged)


def format_facts(facts: tuple[ResearchFact, ...]) -> str:
    """Format research facts grouped by source with summaries for LLM reasoning."""
    by_source: dict[str, list[ResearchFact]] = {}
    for f in facts:
        by_source.setdefault(f.source_url or "unknown", []).append(f)

    lines: list[str] = []
    for source, source_facts in by_source.items():
        title = source_facts[0].source_title
        header = f"[{title}]({source})" if title else f"[{source}]"
        lines.append(header)
        summary = source_facts[0].summary
        if summary:
            lines.append(f"  Context: {summary}")
        for f in source_facts:
            tag = f"[{f.confidence:.1f}]"
            topic = f" ({f.topic})" if f.topic else ""
            lines.append(f"  {tag} {f.claim}{topic}")
        lines.append("")
    return "\n".join(lines)


def execute_web_research(args: dict[str, object], ctx: ToolContext) -> str:
    """Launch a research session via fathom and poll until completion."""
    ra = parse_research_args(args)
    if not ra.goal:
        return "Error: no research goal provided"
    if not ctx.web_client:
        return "Web research unavailable."

    goal, depth, seeds = ra.goal, ra.depth, ra.seeds
    max_pages, n = ra.max_pages, ra.pages_per_round

    enriched_goal = goal
    if ctx.short_term_memory:
        enriched_goal = f"{goal} [Context: {ctx.short_term_memory[:200]}]"

    log.info(
        "web_research",
        goal=enriched_goal[:80],
        depth=depth,
        max_pages=max_pages,
        n=n,
        seeds=len(seeds),
    )
    t0 = time.perf_counter()
    ctx.progress(f"Researching ({depth}): {goal[:50]}")

    try:
        session = ctx.run_async(
            ctx.web_client.start_research(
                enriched_goal,
                depth=depth,
                seeds=seeds,
                max_pages=max_pages,
                n=n,
            )
        )
    except Exception as exc:
        import httpx

        if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 429:
            log.warning("research_rate_limited", goal=enriched_goal[:60])
            return "Research busy — too many concurrent sessions. Answer from memory or try again shortly."
        log.error("research_start_failed", goal=enriched_goal[:60], exc_info=True)
        return "Failed to start research session."

    log.info("research_started", session_id=session.session_id[:12], depth=depth)
    ctx.progress(f"Research session started ({depth})")

    from ..web_client import ResearchResult

    async def _poll() -> ResearchResult:
        client = ctx.web_client
        assert client is not None
        async for progress in client.stream_research(session.session_id):
            if progress.detail:
                ctx.progress(progress.detail[:60])
            if progress.event in ("complete", "error"):
                break
        return await client.get_research_result(session.session_id)

    result = ctx.run_async(_poll())

    elapsed = time.perf_counter() - t0
    if result.status in ("error", "failed"):
        log.warning(
            "research_failed",
            session_id=session.session_id[:12],
            status=result.status,
            elapsed_s=round(elapsed, 1),
        )
        if not result.facts:
            return f"Research session failed ({result.pages_scraped} pages scraped). Try again with a different query or shallower depth."
        formatted = format_facts(result.facts)
        return f"[Research failed but recovered {len(result.facts)} partial facts]\n\n{formatted}"

    if not result.facts:
        log.warning(
            "research_empty", session_id=session.session_id[:12], elapsed_s=round(elapsed, 1)
        )
        return f"Research completed but found no facts ({result.pages_scraped} pages scraped)."

    formatted = format_facts(result.facts)
    log.info(
        "research_done",
        session_id=session.session_id[:12],
        facts=len(result.facts),
        pages=result.pages_scraped,
        elapsed_s=round(elapsed, 1),
    )
    return f"[Research: {len(result.facts)} facts from {result.pages_scraped} pages]\n\n{formatted}"


EXECUTORS: Final = {
    ToolName.WEB_RESEARCH: execute_web_research,
}

LABELS: Final = {
    ToolName.WEB_RESEARCH: lambda a: str(a.get("goal", "")),
}
