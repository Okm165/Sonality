"""LLM-based belief evidence assessment and provenance tracking.

Assesses how new evidence affects existing beliefs using LLM semantic judgment.
Links beliefs to supporting/contradicting episodes via graph edges.
Belief value updates happen in the reflection step, not here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field, model_validator

from shared.errors import BeliefUpdateError, LLMError
from shared.llm.caller import LLMErrorCategory
from shared.llm.parse import normalize_llm_list_response

from .. import config
from ..caller import async_llm_call, format_prompt
from ..ess import SIGNALS_FALLBACK, CredibilitySignals
from ..prompts import BATCH_BELIEF_UPDATE_PROMPT, BELIEF_UPDATE_PROMPT
from ..schema import normalize_topic
from .graph import BeliefNode, EdgeType, MemoryGraph

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EpisodeEvidence:
    """Context for assessing how a new episode affects beliefs."""

    episode_uid: str
    episode_content: str
    ess_score: float
    signals: CredibilitySignals = SIGNALS_FALLBACK


def _belief_snapshot(topic: str, belief: BeliefNode | None) -> dict[str, str | int]:
    """Serialize a belief's current state for prompt injection."""
    b = belief or BeliefNode(topic=topic)
    return {
        "topic": topic,
        "current_value": f"{b.valence:+.2f}",
        "confidence": f"{b.confidence:.2f}",
        "evidence_count": b.evidence_count,
        "support_count": b.support_count,
        "contradict_count": b.contradict_count,
        "uncertainty": f"{b.uncertainty:.2f}",
    }


class _Response(BaseModel):
    """LLM assessment of how evidence affects a single belief.

    direction: positive = supports, negative = contradicts, 0 = neutral.
    evidence_strength: 0-1 how strong the evidence is.
    bears_on_belief: whether the evidence meaningfully bears on this belief
    (vs. merely sharing a topic area). Only record provenance when True.
    """

    topic: str = Field(default="", max_length=200)
    direction: float = Field(default=0.0, ge=-1.0, le=1.0)
    evidence_strength: float = Field(default=0.5, ge=0.0, le=1.0)
    bears_on_belief: bool = False
    reasoning: str = Field(default="", max_length=1000)

    @model_validator(mode="before")
    @classmethod
    def clamp_floats(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        for key, lo, hi in (("direction", -1.0, 1.0), ("evidence_strength", 0.0, 1.0)):
            v = data.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                data[key] = max(lo, min(hi, float(v)))
        raw_bears = data.get("bears_on_belief")
        if raw_bears is not None and not isinstance(raw_bears, bool):
            data["bears_on_belief"] = str(raw_bears).lower() in ("true", "yes", "1")
        return data


class _BatchResponse(BaseModel):
    """Batch of belief assessments from a single LLM call."""

    assessments: list[_Response] = Field(max_length=30)

    @model_validator(mode="before")
    @classmethod
    def normalize_list(cls, data: object) -> object:
        data = normalize_llm_list_response(data, list_key="assessments", item_required_key="topic")
        if isinstance(data, dict) and isinstance(data.get("assessments"), list):
            data["assessments"] = data["assessments"][:30]
        return data


async def _record_provenance(
    topic: str,
    response: _Response,
    episode_uid: str,
    graph: MemoryGraph,
) -> None:
    """Create a graph edge linking an episode to a belief.

    The LLM decides via ``bears_on_belief`` whether the evidence meaningfully
    addresses the belief.  Direction sign determines edge type.
    """
    if not response.bears_on_belief or response.evidence_strength <= 0.0:
        return
    edge_type = (
        EdgeType.SUPPORTS_BELIEF if response.direction >= 0.0 else EdgeType.CONTRADICTS_BELIEF
    )

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
    evidence: EpisodeEvidence,
    belief: BeliefNode | None,
    graph: MemoryGraph,
) -> None:
    """Assess how new evidence affects a single belief and create provenance edge."""
    snap = _belief_snapshot(topic, belief)
    result = await async_llm_call(
        instructions=format_prompt(
            BELIEF_UPDATE_PROMPT,
            **snap,
            episode_content=evidence.episode_content,
            ess_score=f"{evidence.ess_score:.2f}",
            signals_str=evidence.signals.summary_str(),
        ),
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
        bears_on_belief=response.bears_on_belief,
        prior_valence=snap["current_value"],
        prior_confidence=snap["confidence"],
        reasoning_preview=response.reasoning[:80],
    )
    await _record_provenance(topic, response, evidence.episode_uid, graph)


async def assess_belief_evidence_batch(
    *,
    topics: list[str],
    evidence: EpisodeEvidence,
    beliefs: dict[str, BeliefNode],
    graph: MemoryGraph,
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
            evidence=evidence,
            graph=graph,
        )

    if len(topics) == 1:
        await _fallback_single(topics[0])
        return

    topics_data = [_belief_snapshot(t, beliefs.get(t)) for t in topics]
    result = await async_llm_call(
        instructions=format_prompt(
            BATCH_BELIEF_UPDATE_PROMPT,
            episode_content=evidence.episode_content,
            ess_score=f"{evidence.ess_score:.2f}",
            signals_str=evidence.signals.summary_str(),
            topics_json=json.dumps(topics_data, indent=2),
        ),
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

    assessments_by_topic = {normalize_topic(a.topic): a for a in result.value.assessments}
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
            bears_on_belief=response.bears_on_belief,
            reasoning_preview=response.reasoning[:80],
        )
        try:
            await _record_provenance(t, response, evidence.episode_uid, graph)
        except Exception:
            log.error("belief_provenance_edge_failed", topic=t, exc_info=True)
            errors.append(t)
    if errors:
        raise BeliefUpdateError(f"Provenance recording failed for topics: {errors}")
