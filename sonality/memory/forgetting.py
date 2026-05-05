"""LLM-based forgetting with archival and hard-forget decisions.

Uses batch LLM assessment to decide which low-utility episodes to archive
or permanently forget. Supports both ARCHIVE (soft) and FORGET (hard) actions.
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum

from pydantic import BaseModel, model_validator

from .. import config
from ..llm.caller import llm_call
from ..llm.parse import normalize_llm_list_response
from ..prompts import BATCH_FORGETTING_PROMPT
from .dual_store import DualEpisodeStore
from .graph import EpisodeNode, MemoryGraph

log = logging.getLogger(__name__)


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
        f"UID: {ep.uid}\n"
        f"Content: {ep.content[:200]}\n"
        f"Topics: {', '.join(ep.topics)}\n"
        f"ESS: {ep.ess_score:.2f} | Access count: {ep.access_count} | "
        f"Last accessed: {ep.last_accessed or 'never'} | "
        f"Consolidation: L{ep.consolidation_level}"
        for ep in candidates
    )
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
            model=config.REASONING_MODEL,
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
        if uid not in candidate_uids:
            log.warning("Ignoring forgetting decision for unknown UID: %s", uid)
            continue
        decisions.append(_Decision(uid=uid, action=d.action, reason=d.reason.strip()))
        seen.add(uid)
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
                await store.archive_derivatives(decision.uid)
            else:
                await graph.delete_episode(decision.uid)
                await store.delete_derivatives(decision.uid)
            removed += 1
            log.info("%s episode %s: %s", decision.action.value, decision.uid[:8], decision.reason)
        except Exception:
            log.exception(
                "Failed to %s episode %s", decision.action.value.lower(), decision.uid[:8]
            )
            kept += 1

    log.info("Forgetting cycle: %d assessed, %d kept, %d removed", len(candidates), kept, removed)
