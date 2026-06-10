"""LLM-based forgetting with archival and hard-forget decisions.

Uses batch LLM assessment to decide which low-utility episodes to archive
or permanently forget. Supports both ARCHIVE (soft) and FORGET (hard) actions.
"""

from __future__ import annotations

from enum import StrEnum

import structlog
from pydantic import BaseModel, Field, model_validator

from shared.llm.parse import normalize_llm_list_response

from .. import config
from ..caller import async_llm_call, format_prompt
from ..prompts import BATCH_FORGETTING_PROMPT
from .dual_store import DualEpisodeStore
from .graph import EpisodeNode, MemoryGraph

log = structlog.get_logger(__name__)


class _Action(StrEnum):
    """Forgetting disposition: KEEP retains, ARCHIVE soft-deletes, FORGET hard-deletes."""

    KEEP = "KEEP"
    ARCHIVE = "ARCHIVE"
    FORGET = "FORGET"


class _Decision(BaseModel):
    """LLM decision for a single forgetting candidate."""

    uid: str = Field(max_length=100)
    action: _Action = _Action.KEEP
    reason: str = Field(default="", max_length=2000)

    @model_validator(mode="before")
    @classmethod
    def normalize_action(cls, data: object) -> object:
        if isinstance(data, dict) and "action" in data:
            raw = str(data["action"]).strip().upper()
            data["action"] = raw if raw in ("KEEP", "ARCHIVE", "FORGET") else "KEEP"
        return data


class _BatchResponse(BaseModel):
    """Batch of forgetting decisions from a single LLM call."""

    decisions: list[_Decision] = Field(max_length=50)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data: object) -> object:
        data = normalize_llm_list_response(data, list_key="decisions", item_required_key="uid")
        if isinstance(data, dict) and isinstance(data.get("decisions"), list):
            data["decisions"] = data["decisions"][:50]
        return data


async def assess_and_forget(
    candidates: list[EpisodeNode],
    graph: MemoryGraph,
    store: DualEpisodeStore,
    snapshot_excerpt: str = "",
) -> None:
    """Assess episode candidates and archive/forget low-importance ones."""
    if not candidates:
        return

    sole_evidence = await graph.get_belief_connections([ep.uid for ep in candidates])

    def _candidate_line(ep: EpisodeNode) -> str:
        lines = [
            f"UID: {ep.uid[:12]}",
            f"Content: {ep.content[:200]}",
            f"Topics: {', '.join(ep.topics)}",
            f"ESS: {ep.ess_score:.2f} | Access count: {ep.access_count} | "
            f"Last accessed: {ep.last_accessed or 'never'} | "
            f"Consolidation: L{ep.consolidation_level}",
        ]
        beliefs = sole_evidence.get(ep.uid)
        if beliefs:
            lines.append(f"WARNING — sole evidence for beliefs: {', '.join(beliefs)}")
        return "\n".join(lines)

    candidates_summary = "\n\n".join(_candidate_line(ep) for ep in candidates)
    uid_prefix_map = {ep.uid[:12]: ep.uid for ep in candidates}
    fallback = _BatchResponse(
        decisions=[_Decision(uid=ep.uid, reason="Fallback: retain all") for ep in candidates]
    )
    result = await async_llm_call(
        instructions=format_prompt(
            BATCH_FORGETTING_PROMPT,
            candidates_summary=candidates_summary,
            snapshot_excerpt=snapshot_excerpt or "No snapshot available",
        ),
        response_model=_BatchResponse,
        model=config.settings.reasoning_model,
        fallback=fallback,
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
        if decision.action is _Action.FORGET and sole_evidence.get(decision.uid):
            log.info(
                "forgetting_sole_evidence_veto",
                episode_uid=decision.uid[:8],
                beliefs=sole_evidence[decision.uid],
            )
            kept += 1
            continue
        try:
            if decision.action is _Action.ARCHIVE:
                await graph.archive_episode(decision.uid)
                try:
                    await store.remove_knowledge_citations(decision.uid)
                    await store.archive_derivatives(decision.uid)
                except Exception:
                    log.error(
                        "qdrant_archive_failed_reverting",
                        episode_uid=decision.uid[:8],
                        exc_info=True,
                    )
                    await graph.unarchive_episode(decision.uid)
                    raise
            else:
                await graph.delete_episode(decision.uid)
                try:
                    await store.remove_knowledge_citations(decision.uid)
                    await store.delete_derivatives(decision.uid)
                except Exception:
                    log.error(
                        "qdrant_forget_cleanup_failed",
                        episode_uid=decision.uid[:8],
                        exc_info=True,
                    )
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
