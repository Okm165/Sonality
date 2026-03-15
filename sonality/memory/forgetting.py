"""LLM-based forgetting engine with archival and hard-forget decisions.

Replaces formula-based importance scoring with LLM holistic assessment.
Supports both ARCHIVE (soft delete) and FORGET (hard delete) actions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, model_validator

from ..llm.caller import llm_call
from ..llm.prompts import BATCH_FORGETTING_PROMPT
from .dual_store import DualEpisodeStore
from .graph import EpisodeNode, MemoryGraph

log = logging.getLogger(__name__)


class ForgettingAction(StrEnum):
    KEEP = "KEEP"
    ARCHIVE = "ARCHIVE"
    FORGET = "FORGET"


class ForgettingDecision(BaseModel):
    uid: str
    action: ForgettingAction = ForgettingAction.KEEP
    reason: str = ""


class BatchForgettingResponse(BaseModel):
    decisions: list[ForgettingDecision]

    @model_validator(mode="before")
    @classmethod
    def normalize_decisions(cls, data: object) -> object:
        """Handle LLM responses that omit the outer decisions wrapper."""
        if isinstance(data, list):
            return {"decisions": data}
        if isinstance(data, dict) and "uid" in data and "decisions" not in data:
            return {"decisions": [data]}
        return data


@dataclass(frozen=True, slots=True)
class ForgettingResult:
    """Result of a forgetting cycle."""

    kept: int
    archived: int
    total_assessed: int


class ForgettingEngine:
    """LLM-based importance assessment with archive/forget actions."""

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

        # Batch LLM assessment
        candidates_summary = "\n\n".join(
            f"UID: {ep.uid}\n"
            f"Content: {ep.content[:200]}\n"
            f"Topics: {', '.join(ep.topics)}\n"
            f"ESS: {ep.ess_score:.2f} | Access count: {ep.access_count} | "
            f"Last accessed: {ep.last_accessed or 'never'} | "
            f"Consolidation: L{ep.consolidation_level}"
            for ep in candidates
        )
        result = llm_call(
            prompt=BATCH_FORGETTING_PROMPT.format(
                candidates_summary=candidates_summary,
                snapshot_excerpt=snapshot_excerpt or "No snapshot available",
            ),
            response_model=BatchForgettingResponse,
            fallback=BatchForgettingResponse(
                decisions=[
                    ForgettingDecision(uid=ep.uid, action=ForgettingAction.KEEP, reason="Fallback: retain all")
                    for ep in candidates
                ]
            ),
        )
        raw_decisions = result.value.decisions if result.success else [
            ForgettingDecision(uid=ep.uid, action=ForgettingAction.KEEP, reason="Assessment failed")
            for ep in candidates
        ]

        # Validate and fill missing decisions
        candidate_uids = {ep.uid for ep in candidates}
        seen: set[str] = set()
        decisions: list[ForgettingDecision] = []
        for d in raw_decisions:
            uid = d.uid.strip()
            if uid not in candidate_uids:
                log.warning("Ignoring forgetting decision for unknown UID: %s", uid)
                continue
            decisions.append(ForgettingDecision(uid=uid, action=d.action, reason=d.reason.strip()))
            seen.add(uid)
        for ep in candidates:
            if ep.uid not in seen:
                decisions.append(ForgettingDecision(uid=ep.uid, action=ForgettingAction.KEEP, reason="Missing decision; default keep"))

        # Execute actions
        archived = 0
        kept = 0
        for decision in decisions:
            if decision.action not in {ForgettingAction.ARCHIVE, ForgettingAction.FORGET}:
                kept += 1
                continue
            try:
                if decision.action is ForgettingAction.ARCHIVE:
                    await self._graph.archive_episode(decision.uid)
                    await self._store.archive_derivatives(decision.uid)
                    label = "Archived"
                else:
                    await self._graph.delete_episode(decision.uid)
                    await self._store.delete_derivatives(decision.uid)
                    label = "Forgot"
                archived += 1
                log.info("%s episode %s: %s", label, decision.uid[:8], decision.reason)
            except Exception:
                log.exception("Failed to %s episode %s", decision.action.value.lower(), decision.uid[:8])
                kept += 1

        log.info("Forgetting cycle: %d assessed, %d kept, %d archived", len(candidates), kept, archived)
        return ForgettingResult(kept=kept, archived=archived, total_assessed=len(candidates))
