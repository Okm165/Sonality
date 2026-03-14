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

        decisions = self._normalize_decisions(
            candidates,
            self._batch_assess(candidates, snapshot_excerpt),
        )

        archived = 0
        kept = 0
        for decision in decisions:
            if decision.action not in {ForgettingAction.ARCHIVE, ForgettingAction.FORGET}:
                kept += 1
                continue
            try:
                action_label = await self._apply_action(decision.uid, decision.action)
                archived += 1
                log.info("%s episode %s: %s", action_label, decision.uid[:8], decision.reason)
            except Exception:
                log.exception(
                    "Failed to %s episode %s", decision.action.value.lower(), decision.uid[:8]
                )
                kept += 1

        log.info(
            "Forgetting cycle: %d assessed, %d kept, %d archived",
            len(candidates),
            kept,
            archived,
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
            f"UID: {ep.uid}\n"
            f"Content: {ep.content[:200]}\n"
            f"Topics: {', '.join(ep.topics)}\n"
            f"ESS: {ep.ess_score:.2f} | Access count: {ep.access_count} | "
            f"Last accessed: {ep.last_accessed or 'never'} | "
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
                    ForgettingDecision(
                        uid=ep.uid,
                        action=ForgettingAction.KEEP,
                        reason="Fallback: retain all",
                    )
                    for ep in candidates
                ]
            ),
        )
        if result.success:
            return result.value.decisions
        # Fallback: keep everything
        return [
            ForgettingDecision(
                uid=ep.uid,
                action=ForgettingAction.KEEP,
                reason="Assessment failed",
            )
            for ep in candidates
        ]

    async def _apply_action(self, uid: str, action: ForgettingAction) -> str:
        """Execute one archive/forget action and return action label."""
        if action is ForgettingAction.ARCHIVE:
            await self._graph.archive_episode(uid)
            await self._store.archive_derivatives(uid)
            return "Archived"
        await self._graph.delete_episode(uid)
        await self._store.delete_derivatives(uid)
        return "Forgot"

    def _normalize_decisions(
        self,
        candidates: list[EpisodeNode],
        decisions: list[ForgettingDecision],
    ) -> list[ForgettingDecision]:
        """Validate decisions against candidate set and fill missing rows."""
        candidate_uids = {candidate.uid for candidate in candidates}
        normalized: list[ForgettingDecision] = []
        seen_uids: set[str] = set()
        for decision in decisions:
            uid = decision.uid.strip()
            if uid not in candidate_uids:
                log.warning("Ignoring forgetting decision for unknown UID: %s", uid)
                continue
            normalized.append(
                ForgettingDecision(
                    uid=uid,
                    action=decision.action,
                    reason=decision.reason.strip(),
                )
            )
            seen_uids.add(uid)
        for candidate in candidates:
            if candidate.uid not in seen_uids:
                normalized.append(
                    ForgettingDecision(
                        uid=candidate.uid,
                        action=ForgettingAction.KEEP,
                        reason="Missing decision; default keep",
                    )
                )
        return normalized
