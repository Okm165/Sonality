"""LLM-based belief evidence assessment and provenance tracking.

Replaces formula-based belief updating (log2 confidence, fixed contraction ratios)
with LLM semantic assessment of how new evidence affects existing beliefs.
Links beliefs to supporting/contradicting episodes via graph edges.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, field_validator, model_validator

from .. import config
from ..llm.caller import llm_call
from ..llm.prompts import BATCH_BELIEF_UPDATE_PROMPT, BELIEF_UPDATE_PROMPT
from .graph import EdgeType, MemoryGraph
from .health_trace import trace_belief_provenance
from .sponge import BeliefMeta, SpongeState

log = logging.getLogger(__name__)


class UpdateMagnitude(StrEnum):
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    NONE = "NONE"  # Model may return NONE when evidence has no impact


class ContractionAction(StrEnum):
    CONTRACT = "CONTRACT"
    NONE = "NONE"


class BeliefUpdateResponse(BaseModel):
    topic: str = ""
    direction: float = 0.0
    evidence_strength: float = 0.5
    new_uncertainty: float = 0.5
    reasoning: str = ""
    update_magnitude: UpdateMagnitude = UpdateMagnitude.MINOR
    contraction_action: ContractionAction = ContractionAction.NONE

    @field_validator("update_magnitude", mode="before")
    @classmethod
    def coerce_magnitude(cls, v: object) -> object:
        """Map MODERATE → MINOR and similar variants not in the enum."""
        if isinstance(v, str) and v.upper() not in ("MAJOR", "MINOR", "NONE"):
            return "MINOR"
        return v


class BatchBeliefUpdateResponse(BaseModel):
    assessments: list[BeliefUpdateResponse]

    @model_validator(mode="before")
    @classmethod
    def normalize_list(cls, data: object) -> object:
        """Handle bare list or single-assessment dict instead of {assessments: [...]}."""
        if isinstance(data, list):
            return {"assessments": data}
        if isinstance(data, dict) and "assessments" not in data and "topic" in data:
            return {"assessments": [data]}
        return data


@dataclass(frozen=True, slots=True)
class ProvenanceUpdate:
    """Result of a belief evidence assessment."""

    topic: str
    direction: float
    evidence_strength: float
    new_uncertainty: float
    update_magnitude: UpdateMagnitude
    contraction_action: ContractionAction
    reasoning: str


async def _record_provenance(
    topic: str,
    response: BeliefUpdateResponse,
    episode_uid: str,
    sponge: SpongeState,
    graph: MemoryGraph,
) -> ProvenanceUpdate:
    """Update BeliefMeta, create a graph edge, emit a trace, and return ProvenanceUpdate.

    Shared by both the single-topic and batch belief assessment paths.
    Uncertainty is intentionally NOT applied here — it is applied during
    update_opinion (via staged updates) to keep Bayesian floor logic in one place.
    """
    meta = sponge.belief_meta.get(topic)
    if meta is None:
        meta = BeliefMeta(formed_at=sponge.interaction_count)
        sponge.belief_meta[topic] = meta

    if response.direction > 0:
        edge_type = EdgeType.SUPPORTS_BELIEF
        if episode_uid not in meta.supporting_episode_uids:
            meta.supporting_episode_uids.append(episode_uid)
    elif response.direction < 0:
        edge_type = EdgeType.CONTRADICTS_BELIEF
        if episode_uid not in meta.contradicting_episode_uids:
            meta.contradicting_episode_uids.append(episode_uid)
        meta.last_challenged_at = sponge.interaction_count
    else:
        # direction == 0.0: neutral evidence — no graph edge, no metadata update
        log.debug("belief_provenance topic=%s: neutral direction, skipping edge", topic)
        return ProvenanceUpdate(
            topic=topic,
            direction=0.0,
            evidence_strength=response.evidence_strength,
            new_uncertainty=response.new_uncertainty,
            update_magnitude=response.update_magnitude,
            contraction_action=response.contraction_action,
            reasoning=response.reasoning,
        )

    try:
        await graph.link_belief(
            episode_uid,
            topic,
            edge_type=edge_type,
            strength=response.evidence_strength,
            reasoning=response.reasoning[:200],
        )
    except Exception:
        log.exception("Failed to create belief provenance edge for %s", topic)

    trace_belief_provenance(
        interaction_num=sponge.interaction_count,
        topic=topic,
        episode_uid=episode_uid,
        edge_type=edge_type.value,
        strength=response.evidence_strength,
        direction=response.direction,
        update_magnitude=response.update_magnitude.value,
        contraction=response.contraction_action.value,
        reasoning=response.reasoning,
    )
    return ProvenanceUpdate(
        topic=topic,
        direction=response.direction,
        evidence_strength=response.evidence_strength,
        new_uncertainty=response.new_uncertainty,
        update_magnitude=response.update_magnitude,
        contraction_action=response.contraction_action,
        reasoning=response.reasoning,
    )


async def assess_belief_evidence(
    *,
    topic: str,
    episode_uid: str,
    episode_content: str,
    ess_score: float,
    reasoning_type: str,
    source_reliability: str,
    sponge: SpongeState,
    graph: MemoryGraph,
) -> ProvenanceUpdate:
    """Use LLM to assess how new evidence affects a belief, then update provenance.

    Updates BeliefMeta with episode UIDs and uncertainty. Creates graph edges
    (SUPPORTS_BELIEF/CONTRADICTS_BELIEF) for provenance tracking.
    """
    belief = sponge.get_belief(topic)

    log.debug(
        "BELIEF_BEFORE_ASSESS topic=%s | pos=%.3f | conf=%.2f | uncert=%.2f | "
        "support=%d | contra=%d | ess=%.3f | type=%s | ep=%s",
        topic,
        belief.position,
        belief.confidence,
        belief.uncertainty,
        belief.supporting_count,
        belief.contradicting_count,
        ess_score,
        reasoning_type,
        episode_uid[:12],
    )

    prompt = BELIEF_UPDATE_PROMPT.format(
        topic=topic,
        current_value=f"{belief.position:+.2f}",
        confidence=f"{belief.confidence:.2f}",
        supporting_count=belief.supporting_count,
        contradicting_count=belief.contradicting_count,
        uncertainty=f"{belief.uncertainty:.2f}",
        episode_content=episode_content[:1000],
        ess_score=f"{ess_score:.2f}",
        reasoning_type=reasoning_type,
        source_reliability=source_reliability,
    )
    result = await asyncio.to_thread(
        llm_call,
        prompt=prompt,
        response_model=BeliefUpdateResponse,
        fallback=BeliefUpdateResponse(direction=0.0, evidence_strength=0.0),
    )
    if not result.success:
        log.warning(
            "Belief evidence assessment failed for topic=%s (skipping graph edge): %s",
            topic,
            result.error,
        )
        return ProvenanceUpdate(
            topic=topic,
            direction=0.0,
            evidence_strength=0.0,
            new_uncertainty=0.5,
            update_magnitude=UpdateMagnitude.NONE,
            contraction_action=ContractionAction.NONE,
            reasoning="",
        )
    response = result.value
    log.debug(
        "belief_provenance topic=%s: direction=%.2f strength=%.2f uncertainty=%.2f mag=%s contract=%s | %s",
        topic,
        response.direction,
        response.evidence_strength,
        response.new_uncertainty,
        response.update_magnitude.value,
        response.contraction_action.value,
        response.reasoning[:100],
    )

    return await _record_provenance(topic, response, episode_uid, sponge, graph)


async def assess_belief_evidence_batch(
    *,
    topics: list[str],
    episode_uid: str,
    episode_content: str,
    ess_score: float,
    reasoning_type: str,
    source_reliability: str,
    sponge: SpongeState,
    graph: MemoryGraph,
) -> list[ProvenanceUpdate]:
    """Batch version of assess_belief_evidence: one LLM call for all topics.

    Falls back to sequential per-topic calls if batch parsing fails. The batch
    approach reduces N LLM round-trips to 1 when multiple topics need assessment.
    """
    if not topics:
        return []
    if len(topics) == 1:
        return [
            await assess_belief_evidence(
                topic=topics[0],
                episode_uid=episode_uid,
                episode_content=episode_content,
                ess_score=ess_score,
                reasoning_type=reasoning_type,
                source_reliability=source_reliability,
                sponge=sponge,
                graph=graph,
            )
        ]

    topics_data = [
        {
            "topic": t,
            "current_value": f"{sponge.opinion_vectors.get(t, 0.0):+.2f}",
            "confidence": f"{sponge.belief_meta[t].confidence:.2f}"
            if t in sponge.belief_meta
            else "0.00",
            "supporting_count": len(sponge.belief_meta[t].supporting_episode_uids)
            if t in sponge.belief_meta
            else 0,
            "contradicting_count": len(sponge.belief_meta[t].contradicting_episode_uids)
            if t in sponge.belief_meta
            else 0,
            "uncertainty": f"{sponge.belief_meta[t].uncertainty:.2f}"
            if t in sponge.belief_meta
            else "1.00",
        }
        for t in topics
    ]
    prompt = BATCH_BELIEF_UPDATE_PROMPT.format(
        episode_content=episode_content[:1000],
        ess_score=f"{ess_score:.2f}",
        reasoning_type=reasoning_type,
        source_reliability=source_reliability,
        topics_json=json.dumps(topics_data, indent=2),
    )
    fallback = BatchBeliefUpdateResponse(
        assessments=[
            BeliefUpdateResponse(topic=t, direction=0.0, evidence_strength=0.0) for t in topics
        ]
    )
    # Each topic assessment needs ~300 output tokens; scale to avoid truncation.
    # max_retries=2 caps worst-case at 2x LLM_REQUEST_TIMEOUT, keeping this call
    # well within ASYNC_TIMEOUT even when a background worker holds the semaphore.
    batch_max_tokens = max(config.FAST_LLM_MAX_TOKENS, len(topics) * 300)
    result = await asyncio.to_thread(
        llm_call,
        prompt=prompt,
        response_model=BatchBeliefUpdateResponse,
        fallback=fallback,
        max_tokens=batch_max_tokens,
        max_retries=2,
    )
    if not result.success:
        # Server-side errors (transport timeout, network failure) mean the server is busy.
        # Do NOT cascade to N sequential calls — they will all fail the same way.
        error_lower = result.error.lower()
        is_server_error = any(
            kw in error_lower
            for kw in ("transport error", "network", "name resolution", "timed out")
        )
        if is_server_error:
            log.warning(
                "Batch belief assessment server error (%s); skipping graph edges for %d topics",
                result.error,
                len(topics),
            )
            return [
                ProvenanceUpdate(
                    topic=t,
                    direction=0.0,
                    evidence_strength=0.0,
                    new_uncertainty=0.5,
                    update_magnitude=UpdateMagnitude.NONE,
                    contraction_action=ContractionAction.NONE,
                    reasoning="",
                )
                for t in topics
            ]
        log.warning(
            "Batch belief assessment failed (%s); falling back to sequential for %d topics",
            result.error,
            len(topics),
        )
        return [
            await assess_belief_evidence(
                topic=t,
                episode_uid=episode_uid,
                episode_content=episode_content,
                ess_score=ess_score,
                reasoning_type=reasoning_type,
                source_reliability=source_reliability,
                sponge=sponge,
                graph=graph,
            )
            for t in topics
        ]

    assessments_by_topic = {a.topic: a for a in result.value.assessments}
    updates: list[ProvenanceUpdate] = []
    for t in topics:
        response = assessments_by_topic.get(t)
        if response is None:
            log.debug("Batch belief assessment missing result for topic=%s; falling back", t)
            updates.append(
                await assess_belief_evidence(
                    topic=t,
                    episode_uid=episode_uid,
                    episode_content=episode_content,
                    ess_score=ess_score,
                    reasoning_type=reasoning_type,
                    source_reliability=source_reliability,
                    sponge=sponge,
                    graph=graph,
                )
            )
            continue
        log.debug(
            "batch_belief_provenance topic=%s: direction=%.2f strength=%.2f uncertainty=%.2f mag=%s",
            t,
            response.direction,
            response.evidence_strength,
            response.new_uncertainty,
            response.update_magnitude.value,
        )
        updates.append(await _record_provenance(t, response, episode_uid, sponge, graph))
    return updates
