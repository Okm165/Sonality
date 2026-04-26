"""LLM-based belief evidence assessment and provenance tracking.

Assesses how new evidence affects existing beliefs using LLM semantic judgment.
Links beliefs to supporting/contradicting episodes via graph edges.
Belief value updates happen in the reflection step, not here.
"""

from __future__ import annotations

import asyncio
import json
import logging

from pydantic import BaseModel, model_validator

from .. import config
from ..ess import ReasoningType, SourceReliability
from ..llm.caller import llm_call
from ..prompts import BATCH_BELIEF_UPDATE_PROMPT, BELIEF_UPDATE_PROMPT
from .graph import BeliefNode, EdgeType, MemoryGraph

log = logging.getLogger(__name__)


class _Response(BaseModel):
    topic: str = ""
    direction: float = 0.0
    evidence_strength: float = 0.5
    new_uncertainty: float = 0.5
    reasoning: str = ""


class _BatchResponse(BaseModel):
    assessments: list[_Response]

    @model_validator(mode="before")
    @classmethod
    def normalize_list(cls, data: object) -> object:
        if isinstance(data, list):
            return {"assessments": data}
        if isinstance(data, dict) and "assessments" not in data and "topic" in data:
            return {"assessments": [data]}
        return data


async def _record_provenance(
    topic: str, response: _Response, episode_uid: str, graph: MemoryGraph,
) -> None:
    """Create a graph edge linking an episode to a belief."""
    if response.direction > 0:
        edge_type = EdgeType.SUPPORTS_BELIEF
    elif response.direction < 0:
        edge_type = EdgeType.CONTRADICTS_BELIEF
    else:
        return

    try:
        await graph.link_belief(
            episode_uid, topic, edge_type=edge_type,
            strength=response.evidence_strength, reasoning=response.reasoning[:200],
        )
    except Exception:
        log.exception("Failed to create belief provenance edge for %s", topic)

    log.debug("Provenance: %s edge=%s str=%.2f dir=%+.2f",
              topic, edge_type.value, response.evidence_strength, response.direction)


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
    result = await asyncio.to_thread(
        llm_call, prompt=prompt, response_model=_Response,
        fallback=_Response(direction=0.0, evidence_strength=0.0),
        assistant_prefix='{"direction": ', max_retries=1,
    )
    if not result.success:
        log.warning("Belief assessment failed for topic=%s: %s", topic, result.error)
        return
    response = result.value
    log.info("BELIEF_ASSESS topic=%s dir=%+.2f str=%.2f | prior=%+.3f conf=%.2f | %s",
             topic, response.direction, response.evidence_strength,
             b.valence, b.confidence, response.reasoning[:140])
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
) -> None:
    """Batch: one LLM call for all topics, falls back to sequential."""
    if not topics:
        return
    if len(topics) == 1:
        await _assess_single(
            topic=topics[0], belief=beliefs.get(topics[0]),
            episode_uid=episode_uid, episode_content=episode_content,
            ess_score=ess_score, reasoning_type=reasoning_type,
            source_reliability=source_reliability, graph=graph,
        )
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
    fallback = _BatchResponse(
        assessments=[_Response(topic=t, direction=0.0, evidence_strength=0.0) for t in topics]
    )
    batch_max_tokens = min(config.LLM_MAX_TOKENS, max(128, len(topics) * 160))
    result = await asyncio.to_thread(
        llm_call, prompt=prompt, response_model=_BatchResponse,
        fallback=fallback, max_tokens=batch_max_tokens, max_retries=1,
        assistant_prefix='{"assessments": [',
    )
    if not result.success:
        error_lower = result.error.lower()
        is_server_error = any(kw in error_lower for kw in ("transport error", "network", "timed out"))
        if is_server_error:
            log.warning("Batch belief assessment server error; skipping %d topics", len(topics))
            return
        log.warning("Batch failed; falling back to sequential for %d topics", len(topics))
        for t in topics:
            await _assess_single(
                topic=t, belief=beliefs.get(t),
                episode_uid=episode_uid, episode_content=episode_content,
                ess_score=ess_score, reasoning_type=reasoning_type,
                source_reliability=source_reliability, graph=graph,
            )
        return

    assessments_by_topic = {a.topic: a for a in result.value.assessments}
    for t in topics:
        response = assessments_by_topic.get(t)
        if response is None:
            log.warning("Batch missing result for topic=%s; falling back", t)
            await _assess_single(
                topic=t, belief=beliefs.get(t),
                episode_uid=episode_uid, episode_content=episode_content,
                ess_score=ess_score, reasoning_type=reasoning_type,
                source_reliability=source_reliability, graph=graph,
            )
            continue
        log.info("BELIEF_ASSESS topic=%s dir=%+.2f str=%.2f | %s",
                 t, response.direction, response.evidence_strength, response.reasoning[:140])
        await _record_provenance(t, response, episode_uid, graph)
