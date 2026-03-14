"""LLM-based belief evidence assessment and provenance tracking.

Replaces formula-based belief updating (log2 confidence, fixed contraction ratios)
with LLM semantic assessment of how new evidence affects existing beliefs.
Links beliefs to supporting/contradicting episodes via graph edges.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, field_validator

from ..llm.caller import llm_call
from ..llm.prompts import BELIEF_UPDATE_PROMPT
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
    meta = sponge.belief_meta.get(topic)
    current_value = sponge.opinion_vectors.get(topic, 0.0)

    # Trace belief state before provenance assessment
    log.debug(
        "BELIEF_BEFORE_ASSESS topic=%s | pos=%.3f | conf=%.2f | uncert=%.2f | "
        "support=%d | contra=%d | ess=%.3f | type=%s | ep=%s",
        topic,
        current_value,
        meta.confidence if meta else 0.0,
        meta.uncertainty if meta else 1.0,
        len(meta.supporting_episode_uids) if meta else 0,
        len(meta.contradicting_episode_uids) if meta else 0,
        ess_score,
        reasoning_type,
        episode_uid[:12],
    )

    prompt = BELIEF_UPDATE_PROMPT.format(
        topic=topic,
        current_value=f"{current_value:+.2f}",
        confidence=f"{meta.confidence:.2f}" if meta else "0.00",
        supporting_count=len(meta.supporting_episode_uids) if meta else 0,
        contradicting_count=len(meta.contradicting_episode_uids) if meta else 0,
        uncertainty=f"{meta.uncertainty:.2f}" if meta else "1.00",
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
            "Belief evidence assessment parse failed for topic=%s (using neutral fallback): %s",
            topic,
            result.error,
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

    # Update provenance in BeliefMeta
    if meta is None:
        meta = BeliefMeta(formed_at=sponge.interaction_count)
        sponge.belief_meta[topic] = meta

    edge_type = EdgeType.SUPPORTS_BELIEF
    if response.direction > 0:
        if episode_uid not in meta.supporting_episode_uids:
            meta.supporting_episode_uids.append(episode_uid)
    else:
        edge_type = EdgeType.CONTRADICTS_BELIEF
        if episode_uid not in meta.contradicting_episode_uids:
            meta.contradicting_episode_uids.append(episode_uid)
        meta.last_challenged_at = sponge.interaction_count
    # Uncertainty is applied during update_opinion (via staged updates) to keep
    # the Bayesian floor logic in a single place.

    # Create graph edge for provenance
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

    # Emit detailed provenance trace for health analysis
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
