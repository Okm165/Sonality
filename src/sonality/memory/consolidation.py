"""LLM-based segment consolidation.

When a conversation segment closes, the LLM assesses readiness and generates
a summary. Summaries supplement raw episodes — they do not replace them
(Hierarchical Episodic Memory Architecture, HEMA).
"""

from __future__ import annotations

import asyncio
from enum import StrEnum

import structlog
from pydantic import BaseModel, field_validator

from shared.types import new_id

from .. import config
from ..caller import llm_call
from ..prompts import CONSOLIDATION_READINESS_PROMPT, CONSOLIDATION_SUMMARY_PROMPT
from .graph import EpisodeNode, MemoryGraph, format_episode_block, format_episode_line

log = structlog.get_logger()


class _ReadinessDecision(StrEnum):
    """Whether a segment has enough coherent material to summarize."""

    READY = "READY"
    NOT_READY = "NOT_READY"


class _SummarySchema(BaseModel):
    """Structured consolidation summary."""

    summary: str = ""


class _ReadinessResponse(BaseModel):
    """LLM assessment of whether a segment is ready for consolidation."""

    readiness_decision: _ReadinessDecision = _ReadinessDecision.NOT_READY
    confidence: float = 0.0
    reasoning: str = ""
    suggested_summary_focus: str = ""

    @field_validator("suggested_summary_focus", mode="before")
    @classmethod
    def coerce_null_focus(cls, v: object) -> str:
        return "" if v is None else str(v)


async def maybe_consolidate_segment(graph: MemoryGraph, segment_id: str) -> str:
    """Check if a segment is ready for consolidation and summarize if so.

    Returns the summary UID if consolidated, empty string otherwise.
    """
    episodes = await graph.get_segment_episodes(segment_id)
    if len(episodes) < 2:
        return ""

    readiness = await asyncio.to_thread(_check_readiness, segment_id, episodes)
    if readiness.readiness_decision is not _ReadinessDecision.READY:
        log.debug(
            "segment_not_ready",
            segment=segment_id[:8],
            reason=readiness.reasoning[:80],
            confidence=f"{readiness.confidence:.2f}",
        )
        return ""

    summary_text = await asyncio.to_thread(
        _generate_summary, episodes, readiness.suggested_summary_focus
    )
    if not summary_text:
        return ""

    summary_uid = new_id()
    source_uids = [ep.uid for ep in episodes]
    topics = list({t for ep in episodes for t in ep.topics})

    await graph.create_summary(
        uid=summary_uid,
        level=2,
        content=summary_text,
        source_uids=source_uids,
        topics=topics,
    )
    await graph.mark_segment_consolidated(segment_id)

    log.info(
        "segment_consolidated",
        segment_id=segment_id,
        summary_uid=summary_uid[:8],
        episodes=len(episodes),
        summary_chars=len(summary_text),
    )
    return summary_uid


def _check_readiness(segment_id: str, episodes: list[EpisodeNode]) -> _ReadinessResponse:
    """Ask the LLM whether a segment's episodes form a coherent unit worth summarizing."""
    episode_summaries = "\n".join(
        f"- {format_episode_line(created_at=ep.created_at, summary=ep.summary, content=ep.content, content_limit=150)}"
        for ep in episodes[:30]
    )
    prompt = CONSOLIDATION_READINESS_PROMPT.format(
        segment_id=segment_id,
        episode_count=len(episodes),
        start_time=episodes[0].created_at if episodes else "unknown",
        end_time=episodes[-1].created_at if episodes else "unknown",
        episode_summaries=episode_summaries,
    )
    result = llm_call(
        prompt=prompt,
        response_model=_ReadinessResponse,
        fallback=_ReadinessResponse(),
    )
    if not result.success:
        log.warning(
            "consolidation_readiness_failed",
            segment_id=segment_id,
            error=(result.error or "")[:80],
        )
    return result.value


def _generate_summary(episodes: list[EpisodeNode], focus: str) -> str:
    """Generate a consolidation summary from episode content via LLM."""
    content = "\n\n".join(
        format_episode_block(created_at=ep.created_at, content=ep.content, content_limit=500)
        for ep in episodes[:30]
    )
    content = content[:12000]
    prompt = CONSOLIDATION_SUMMARY_PROMPT.format(
        episodes=content,
        focus_instruction=f"\nFocus on: {focus}" if focus else "",
    )
    result = llm_call(
        prompt=prompt,
        response_model=_SummarySchema,
        fallback=_SummarySchema(),
        model=config.settings.fast_model,
    )
    if not result.success:
        log.warning(
            "consolidation_summary_failed",
            error=(result.error or "")[:80],
        )
    return result.value.summary.strip()
