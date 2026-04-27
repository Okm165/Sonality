"""LLM-based segment consolidation.

When a conversation segment closes, the LLM assesses readiness and generates
a summary. Summaries supplement raw episodes (HEMA principle).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from enum import StrEnum

from pydantic import BaseModel, field_validator

from .. import config
from ..llm.caller import llm_call
from ..prompts import CONSOLIDATION_READINESS_PROMPT
from ..provider import default_provider
from ..schema import ChatRole
from .graph import EpisodeNode, MemoryGraph, format_episode_block, format_episode_line

log = logging.getLogger(__name__)


class _ReadinessDecision(StrEnum):
    READY = "READY"
    NOT_READY = "NOT_READY"


class _ReadinessResponse(BaseModel):
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
            "Segment %s not ready: %s (conf=%.2f)",
            segment_id,
            readiness.reasoning,
            readiness.confidence,
        )
        return ""

    summary_text = await asyncio.to_thread(
        _generate_summary, episodes, readiness.suggested_summary_focus
    )
    if not summary_text:
        return ""

    summary_uid = str(uuid.uuid4())
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
        "Consolidated segment %s into summary %s (%d episodes -> %d chars)",
        segment_id,
        summary_uid[:8],
        len(episodes),
        len(summary_text),
    )
    return summary_uid


def _check_readiness(segment_id: str, episodes: list[EpisodeNode]) -> _ReadinessResponse:
    episode_summaries = "\n".join(
        f"- {format_episode_line(created_at=ep.created_at, summary=ep.summary, content=ep.content, content_limit=150)}"
        for ep in episodes
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
        max_tokens=config.STRUCTURED_JSON_MAX_TOKENS,
        assistant_prefix='{"readiness_decision": "',
    )
    if not result.success:
        log.warning("Consolidation readiness failed for segment=%s: %s", segment_id, result.error)
    return result.value


def _generate_summary(episodes: list[EpisodeNode], focus: str) -> str:
    content = "\n\n".join(
        format_episode_block(created_at=ep.created_at, content=ep.content, content_limit=500)
        for ep in episodes
    )
    focus_instruction = f"\n\nFocus on: {focus}" if focus else ""
    prompt = (
        f"Summarize these conversation episodes into a concise, comprehensive summary.\n"
        f"Preserve key facts, decisions, opinions, and important context.\n\n"
        f"Episodes:\n{content}{focus_instruction}\n\nWrite the summary:"
    )
    try:
        completion = default_provider.chat_completion(
            model=config.FAST_LLM_MODEL,
            max_tokens=config.EXTRACTION_MAX_TOKENS,
            messages=({"role": ChatRole.USER, "content": prompt},),
            enable_thinking=False,
        )
        return completion.text.strip()
    except Exception:
        log.exception("Consolidation summary generation failed")
        return ""
