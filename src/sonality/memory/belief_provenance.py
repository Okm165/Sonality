"""LLM-based belief evidence assessment and provenance tracking.

Assesses how new evidence affects existing beliefs using LLM semantic judgment.
Links beliefs to supporting/contradicting episodes via graph edges.
Belief value updates happen in the reflection step, not here.
"""

from __future__ import annotations

import asyncio
import json

import structlog
from pydantic import BaseModel, model_validator

from shared.errors import BeliefUpdateError, LLMError
from shared.llm.caller import LLMErrorCategory
from shared.llm.parse import normalize_llm_list_response

from .. import config
from ..caller import llm_call
from ..ess import ReasoningType, SourceReliability
from ..prompts import BATCH_BELIEF_UPDATE_PROMPT, BELIEF_UPDATE_PROMPT, WEB_VERIFICATION_SECTION
from .graph import BeliefNode, EdgeType, MemoryGraph

log = structlog.get_logger()


class _Response(BaseModel):
    """LLM assessment of how evidence affects a single belief.

    direction: positive = supports, negative = contradicts, 0 = neutral.
    evidence_strength: 0-1 how strong the evidence is.
    """

    topic: str = ""
    direction: float = 0.0
    evidence_strength: float = 0.5
    reasoning: str = ""


class _BatchResponse(BaseModel):
    """Batch of belief assessments from a single LLM call."""

    assessments: list[_Response]

    @model_validator(mode="before")
    @classmethod
    def normalize_list(cls, data: object) -> object:
        return normalize_llm_list_response(data, list_key="assessments", item_required_key="topic")


async def _record_provenance(
    topic: str,
    response: _Response,
    episode_uid: str,
    graph: MemoryGraph,
) -> None:
    """Create a graph edge linking an episode to a belief."""
    if response.direction > 0:
        edge_type = EdgeType.SUPPORTS_BELIEF
    elif response.direction < 0:
        edge_type = EdgeType.CONTRADICTS_BELIEF
    else:
        return

    await graph.link_belief(
        episode_uid,
        topic,
        edge_type=edge_type,
        strength=response.evidence_strength,
        reasoning=response.reasoning[:200],
    )
    log.debug(
        "belief_provenance_recorded",
        topic=topic,
        edge_type=edge_type.value,
        evidence_strength=response.evidence_strength,
        direction=response.direction,
    )


async def _assess_single(
    *,
    topic: str,
    episode_uid: str,
    episode_content: str,
    ess_score: float,
    reasoning_type: ReasoningType,
    source_reliability: SourceReliability,
    belief: BeliefNode | None,
    graph: MemoryGraph,
    web_context: str = "",
) -> None:
    """Assess how new evidence affects a single belief and create provenance edge."""
    b = belief or BeliefNode(topic=topic)
    prompt = BELIEF_UPDATE_PROMPT.format(
        topic=topic,
        current_value=f"{b.valence:+.2f}",
        confidence=f"{b.confidence:.2f}",
        supporting_count=b.evidence_count,
        uncertainty=f"{b.uncertainty:.2f}",
        episode_content=episode_content[:1000],
        ess_score=f"{ess_score:.2f}",
        reasoning_type=reasoning_type,
        source_reliability=source_reliability,
    )
    if web_context:
        prompt += "\n\n" + WEB_VERIFICATION_SECTION.format(web_verification_context=web_context)
    result = await asyncio.to_thread(
        llm_call,
        prompt=prompt,
        response_model=_Response,
        fallback=_Response(direction=0.0, evidence_strength=0.0),
        model=config.settings.structured_model,
    )
    if not result.success:
        log.warning(
            "belief_assessment_failed",
            topic=topic,
            error=(result.error or "")[:80],
        )
        return
    response = result.value
    log.info(
        "belief_assessed",
        topic=topic,
        direction=response.direction,
        evidence_strength=response.evidence_strength,
        prior_valence=b.valence,
        prior_confidence=b.confidence,
        reasoning_preview=response.reasoning[:80],
    )
    await _record_provenance(topic, response, episode_uid, graph)


async def assess_belief_evidence_batch(
    *,
    topics: list[str],
    episode_uid: str,
    episode_content: str,
    ess_score: float,
    reasoning_type: ReasoningType,
    source_reliability: SourceReliability,
    beliefs: dict[str, BeliefNode],
    graph: MemoryGraph,
    web_context: str = "",
) -> None:
    """Batch: one LLM call for all topics, falls back to sequential.

    Assesses how new episode evidence affects each listed belief. Creates
    SUPPORTS/CONTRADICTS provenance edges. Single-topic input skips the
    batch call and goes straight to _assess_single.
    """
    if not topics:
        return

    async def _fallback_single(topic: str) -> None:
        await _assess_single(
            topic=topic,
            belief=beliefs.get(topic),
            episode_uid=episode_uid,
            episode_content=episode_content,
            ess_score=ess_score,
            reasoning_type=reasoning_type,
            source_reliability=source_reliability,
            graph=graph,
            web_context=web_context,
        )

    if len(topics) == 1:
        await _fallback_single(topics[0])
        return

    topics_data = [
        {
            "topic": t,
            "current_value": f"{beliefs[t].valence:+.2f}" if t in beliefs else "+0.00",
            "confidence": f"{beliefs[t].confidence:.2f}" if t in beliefs else "0.00",
            "supporting_count": beliefs[t].evidence_count if t in beliefs else 0,
            "uncertainty": f"{beliefs[t].uncertainty:.2f}" if t in beliefs else "1.00",
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
    if web_context:
        prompt += "\n\n" + WEB_VERIFICATION_SECTION.format(web_verification_context=web_context)

    result = await asyncio.to_thread(
        llm_call,
        prompt=prompt,
        response_model=_BatchResponse,
        fallback=_BatchResponse(
            assessments=[_Response(topic=t, direction=0.0, evidence_strength=0.0) for t in topics]
        ),
        model=config.settings.structured_model,
    )
    if not result.success:
        if result.error_category in (LLMErrorCategory.TRANSPORT, LLMErrorCategory.HTTP):
            raise LLMError(f"Belief assessment batch unavailable: {result.error}")
        log.warning("batch_belief_assessment_fallback", topic_count=len(topics))
        for t in topics:
            await _fallback_single(t)
        return

    assessments_by_topic = {a.topic: a for a in result.value.assessments}
    errors: list[str] = []
    for t in topics:
        response = assessments_by_topic.get(t)
        if response is None:
            log.warning("batch_belief_topic_missing", topic=t)
            await _fallback_single(t)
            continue
        log.info(
            "belief_assessed",
            topic=t,
            direction=response.direction,
            evidence_strength=response.evidence_strength,
            reasoning_preview=response.reasoning[:80],
        )
        try:
            await _record_provenance(t, response, episode_uid, graph)
        except Exception:
            log.error("belief_provenance_edge_failed", topic=t, exc_info=True)
            errors.append(t)
    if errors:
        raise BeliefUpdateError(f"Provenance recording failed for topics: {errors}")
