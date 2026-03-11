"""LLM-based forgetting engine with importance assessment and soft archival.

Replaces formula-based importance scoring with LLM holistic assessment.
Uses soft deletion: archive episode, remove from pgvector, keep graph node
for provenance. Recovery possible by re-embedding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel

from ..llm.caller import llm_call
from ..llm.prompts import BATCH_FORGETTING_PROMPT
from .dual_store import DualEpisodeStore
from .graph import EpisodeNode, MemoryGraph

log = logging.getLogger(__name__)


class ImportanceResponse(BaseModel):
    importance: float = 0.5
    should_retain: bool = True
    reasoning: str = ""
    is_foundational: bool = False
    redundant_with: list[str] | None = None


class ForgettingDecision(BaseModel):
    uid: str
    action: str  # "KEEP" | "ARCHIVE" | "FORGET"
    reason: str = ""


class BatchForgettingResponse(BaseModel):
    decisions: list[ForgettingDecision]


@dataclass(frozen=True, slots=True)
class ForgettingResult:
    """Result of a forgetting cycle."""

    kept: int
    archived: int
    total_assessed: int


class ForgettingEngine:
    """LLM-based importance assessment with soft archival."""

    def __init__(self, graph: MemoryGraph, store: DualEpisodeStore) -> None:
        self._graph = graph
        self._store = store

    async def assess_and_forget(
        self,
        candidates: list[EpisodeNode],
        snapshot_excerpt: str = "",
    ) -> ForgettingResult:
        """Assess a batch of episode candidates and archive low-importance ones.

        Uses batch LLM assessment for efficiency. Foundational episodes are
        always retained regardless of other signals.
        """
        if not candidates:
            return ForgettingResult(kept=0, archived=0, total_assessed=0)

        decisions = self._batch_assess(candidates, snapshot_excerpt)

        archived = 0
        kept = 0
        for decision in decisions:
            if decision.action == "ARCHIVE" or decision.action == "FORGET":
                try:
                    await self._graph.archive_episode(decision.uid)
                    await self._store.archive_derivatives(decision.uid)
                    archived += 1
                    log.info("Archived episode %s: %s", decision.uid[:8], decision.reason)
                except Exception:
                    log.exception("Failed to archive episode %s", decision.uid[:8])
                    kept += 1
            else:
                kept += 1

        log.info(
            "Forgetting cycle: %d assessed, %d kept, %d archived",
            len(candidates), kept, archived,
        )
        return ForgettingResult(
            kept=kept,
            archived=archived,
            total_assessed=len(candidates),
        )

    def _batch_assess(
        self,
        candidates: list[EpisodeNode],
        snapshot_excerpt: str,
    ) -> list[ForgettingDecision]:
        """Use LLM to batch-assess episode importance."""
        candidates_summary = "\n\n".join(
            f"UID: {ep.uid[:8]}\n"
            f"Content: {ep.content[:200]}\n"
            f"Topics: {', '.join(ep.topics)}\n"
            f"ESS: {ep.ess_score:.2f} | Access count: {ep.access_count} | "
            f"Consolidation: L{ep.consolidation_level}"
            for ep in candidates
        )

        prompt = BATCH_FORGETTING_PROMPT.format(
            candidates_summary=candidates_summary,
            snapshot_excerpt=snapshot_excerpt or "No snapshot available",
        )
        result = llm_call(
            prompt=prompt,
            response_model=BatchForgettingResponse,
            fallback=BatchForgettingResponse(
                decisions=[
                    ForgettingDecision(uid=ep.uid, action="KEEP", reason="Fallback: retain all")
                    for ep in candidates
                ]
            ),
        )
        if result.success and result.value:
            assert isinstance(result.value, BatchForgettingResponse)
            return result.value.decisions
        # Fallback: keep everything
        return [
            ForgettingDecision(uid=ep.uid, action="KEEP", reason="Assessment failed")
            for ep in candidates
        ]
