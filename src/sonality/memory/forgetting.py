"""LLM-based forgetting with archival and hard-forget decisions.

Uses batch LLM assessment to decide which low-utility episodes to archive
or permanently forget. Supports both ARCHIVE (soft) and FORGET (hard) actions.
"""

from __future__ import annotations

import asyncio
from enum import StrEnum

import structlog
from pydantic import BaseModel, model_validator

from shared.llm.parse import normalize_llm_list_response

from .. import config
from ..caller import llm_call
from ..prompts import BATCH_FORGETTING_PROMPT
from .dual_store import DualEpisodeStore
from .graph import EpisodeNode, MemoryGraph

log = structlog.get_logger()


class _Action(StrEnum):
    """Forgetting disposition: KEEP retains, ARCHIVE soft-deletes, FORGET hard-deletes."""

    KEEP = "KEEP"
    ARCHIVE = "ARCHIVE"
    FORGET = "FORGET"


class _Decision(BaseModel):
    """LLM decision for a single forgetting candidate."""

    uid: str
    action: _Action = _Action.KEEP
    reason: str = ""


class _BatchResponse(BaseModel):
    """Batch of forgetting decisions from a single LLM call."""

    decisions: list[_Decision]

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: object) -> object:
        return normalize_llm_list_response(data, list_key="decisions", item_required_key="uid")


async def assess_and_forget(
    candidates: list[EpisodeNode],
    graph: MemoryGraph,
    store: DualEpisodeStore,
    snapshot_excerpt: str = "",
) -> None:
    """Assess episode candidates and archive/forget low-importance ones."""
    if not candidates:
        return

    candidates_summary = "\n\n".join(
        f"UID: {ep.uid[:12]}\n"
        f"Content: {ep.content[:200]}\n"
        f"Topics: {', '.join(ep.topics)}\n"
        f"ESS: {ep.ess_score:.2f} | Access count: {ep.access_count} | "
        f"Last accessed: {ep.last_accessed or 'never'} | "
        f"Consolidation: L{ep.consolidation_level}"
        for ep in candidates
    )
    uid_prefix_map = {ep.uid[:12]: ep.uid for ep in candidates}
    fallback = _BatchResponse(
        decisions=[_Decision(uid=ep.uid, reason="Fallback: retain all") for ep in candidates]
    )
    result = await asyncio.to_thread(
        lambda: llm_call(
            prompt=BATCH_FORGETTING_PROMPT.format(
                candidates_summary=candidates_summary,
                snapshot_excerpt=snapshot_excerpt or "No snapshot available",
            ),
            response_model=_BatchResponse,
            model=config.settings.reasoning_model,
            fallback=fallback,
        )
    )
    raw_decisions = (
        result.value.decisions
        if result.success
        else [_Decision(uid=ep.uid, reason="Assessment failed") for ep in candidates]
    )

    candidate_uids = {ep.uid for ep in candidates}
    seen: set[str] = set()
    decisions: list[_Decision] = []
    for d in raw_decisions:
        uid = d.uid.strip()
        # Resolve truncated UIDs from LLM back to full UIDs
        if uid not in candidate_uids and uid in uid_prefix_map:
            uid = uid_prefix_map[uid]
        if uid not in candidate_uids:
            log.warning(
                "forgetting_unknown_uid_ignored",
                uid=uid,
            )
            continue
        if uid in seen:
            log.debug(
                "forgetting_duplicate_decision_ignored",
                uid=uid,
            )
            continue
        seen.add(uid)
        decisions.append(_Decision(uid=uid, action=d.action, reason=d.reason.strip()))
    for ep in candidates:
        if ep.uid not in seen:
            decisions.append(_Decision(uid=ep.uid, reason="Missing decision; default keep"))

    removed = 0
    kept = 0
    for decision in decisions:
        if decision.action not in {_Action.ARCHIVE, _Action.FORGET}:
            kept += 1
            continue
        try:
            if decision.action is _Action.ARCHIVE:
                await graph.archive_episode(decision.uid)
                try:
                    await store.archive_derivatives(decision.uid)
                except Exception:
                    log.error("qdrant_archive_failed_reverting", episode_uid=decision.uid[:8], exc_info=True)
                    await graph.unarchive_episode(decision.uid)
                    raise
            else:
                await graph.delete_episode(decision.uid)
                try:
                    await store.delete_derivatives(decision.uid)
                except Exception:
                    log.error("qdrant_delete_failed_after_graph", episode_uid=decision.uid[:8], exc_info=True)
                    raise
            removed += 1
            log.info(
                "forgetting_episode_action",
                action=decision.action.value,
                episode_uid=decision.uid[:8],
                reason=decision.reason[:80],
            )
        except Exception:
            log.error(
                "forgetting_action_failed",
                action=decision.action.value.lower(),
                episode_uid=decision.uid[:8],
                exc_info=True,
            )
            kept += 1

    log.info(
        "forgetting_cycle_complete",
        assessed_count=len(candidates),
        kept_count=kept,
        removed_count=removed,
    )
