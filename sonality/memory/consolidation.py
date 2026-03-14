"""LLM-based consolidation engine with three-level hierarchy.

Level 1: Raw episodes (full fidelity)
Level 2: Segment summaries (consolidated by LLM readiness check)
Level 3: Topic clusters (cross-segment patterns)

Summaries supplement, not replace, raw episodes (HEMA principle).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from enum import StrEnum

from pydantic import BaseModel, field_validator

from .. import config
from ..llm.caller import llm_call
from ..llm.prompts import CONSOLIDATION_READINESS_PROMPT
from ..provider import chat_completion
from .context_format import format_episode_block, format_episode_line
from .graph import EpisodeNode, MemoryGraph
from .health_trace import trace_consolidation

log = logging.getLogger(__name__)


class ConsolidationReadinessDecision(StrEnum):
    READY = "READY"
    NOT_READY = "NOT_READY"


class ConsolidationReadinessResponse(BaseModel):
    readiness_decision: ConsolidationReadinessDecision = ConsolidationReadinessDecision.NOT_READY
    confidence: float = 0.0
    reasoning: str = ""
    suggested_summary_focus: str = ""

    @field_validator("suggested_summary_focus", mode="before")
    @classmethod
    def coerce_null_focus(cls, v: object) -> str:
        """Accept null from model (common when NOT_READY) and coerce to empty string."""
        return "" if v is None else str(v)


class ConsolidationEngine:
    """Three-level hierarchical consolidation with LLM readiness assessment."""

    def __init__(self, graph: MemoryGraph) -> None:
        self._graph = graph

    async def maybe_consolidate_segment(self, segment_id: str) -> str:
        """Check if a segment is ready for consolidation and summarize if so.

        Returns the summary UID if consolidated, empty string otherwise.
        """
        episodes = await self._graph.get_segment_episodes(segment_id)
        if len(episodes) < 2:
            return ""

        # LLM readiness check (run in thread to avoid blocking the event loop)
        readiness = await asyncio.to_thread(self._check_readiness, segment_id, episodes)
        if readiness.readiness_decision is not ConsolidationReadinessDecision.READY:
            log.debug(
                "Segment %s not ready: %s (conf=%.2f)",
                segment_id,
                readiness.reasoning,
                readiness.confidence,
            )
            return ""

        # Generate summary (run in thread to avoid blocking the event loop)
        summary_text = await asyncio.to_thread(
            self._generate_summary, episodes, readiness.suggested_summary_focus
        )
        if not summary_text:
            return ""

        # Create summary node in graph
        summary_uid = str(uuid.uuid4())
        source_uids = [ep.uid for ep in episodes]
        topics = list({t for ep in episodes for t in ep.topics})

        await self._graph.create_summary(
            uid=summary_uid,
            level=2,  # Segment summary
            content=summary_text,
            source_uids=source_uids,
            topics=topics,
        )
        await self._graph.mark_segment_consolidated(segment_id)

        # Trace consolidation for semantic drift analysis (SSGM framework alignment)
        source_content_len = sum(len(ep.content or "") for ep in episodes)
        trace_consolidation(
            segment_id=segment_id,
            episode_count=len(episodes),
            source_content_len=source_content_len,
            summary_len=len(summary_text),
            topics=topics,
            readiness_confidence=readiness.confidence,
            summary_focus=readiness.suggested_summary_focus,
        )

        log.info(
            "Consolidated segment %s into summary %s (%d episodes -> %d chars)",
            segment_id,
            summary_uid[:8],
            len(episodes),
            len(summary_text),
        )
        return summary_uid

    def _check_readiness(
        self, segment_id: str, episodes: list[EpisodeNode]
    ) -> ConsolidationReadinessResponse:
        """Use LLM to assess if segment is ready for consolidation."""
        episode_summaries = "\n".join(
            f"- {format_episode_line(created_at=ep.created_at, summary=ep.summary, content=ep.content, content_limit=150)}"
            for ep in episodes
        )
        start_time = episodes[0].created_at if episodes else "unknown"
        end_time = episodes[-1].created_at if episodes else "unknown"

        prompt = CONSOLIDATION_READINESS_PROMPT.format(
            segment_id=segment_id,
            episode_count=len(episodes),
            start_time=start_time,
            end_time=end_time,
            episode_summaries=episode_summaries,
        )
        result = llm_call(
            prompt=prompt,
            response_model=ConsolidationReadinessResponse,
            fallback=ConsolidationReadinessResponse(),
        )
        if not result.success:
            log.warning(
                "Consolidation readiness parse failed for segment=%s (returning NOT_READY fallback): %s",
                segment_id,
                result.error,
            )
        return result.value

    def _generate_summary(self, episodes: list[EpisodeNode], focus: str) -> str:
        """Generate a consolidation summary using LLM."""
        content = "\n\n".join(
            format_episode_block(
                created_at=ep.created_at,
                content=ep.content,
                content_limit=500,
            )
            for ep in episodes
        )
        focus_instruction = f"\n\nFocus on: {focus}" if focus else ""

        prompt = (
            f"Summarize these conversation episodes into a concise, comprehensive summary.\n"
            f"Preserve key facts, decisions, opinions, and important context.\n\n"
            f"Episodes:\n{content}{focus_instruction}\n\n"
            f"Write the summary:"
        )
        try:
            completion = chat_completion(
                model=config.FAST_LLM_MODEL,
                max_tokens=config.FAST_LLM_MAX_TOKENS,
                messages=({"role": "user", "content": prompt},),
            )
            return completion.text.strip()
        except Exception:
            log.exception("Consolidation summary generation failed")
            return ""
