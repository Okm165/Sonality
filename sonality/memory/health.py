"""LLM-based personality health monitoring.

Replaces threshold-based checks (disagreement < 0.15, word count < 15, etc.)
with holistic LLM assessment of personality coherence, consistency, and growth.
Based on PERSIST framework (AAAI 2026) findings on personality instability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel, Field

from ..llm.caller import llm_call
from ..llm.prompts import HEALTH_ASSESSMENT_PROMPT
from .sponge import SpongeState

log = logging.getLogger(__name__)


class HealthMetrics(BaseModel):
    coherence_score: float = 0.5
    consistency_score: float = 0.5
    growth_health_score: float = 0.5


class HealthResponse(BaseModel):
    overall_health: str = "healthy"
    concerns: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    reasoning: str = ""
    metrics: HealthMetrics = Field(default_factory=HealthMetrics)


@dataclass(frozen=True, slots=True)
class HealthReport:
    overall_health: str
    concerns: list[str]
    recommendations: list[str]
    coherence_score: float
    consistency_score: float
    growth_health_score: float
    reasoning: str


def assess_health(sponge: SpongeState) -> HealthReport:
    """Perform LLM-based health assessment of the agent's personality state."""
    high_conf_count = sum(
        1 for meta in sponge.belief_meta.values() if meta.confidence > 0.7
    )
    beliefs_summary = ", ".join(
        f"{topic}={sponge.opinion_vectors.get(topic, 0):.2f} (conf={meta.confidence:.2f})"
        for topic, meta in sorted(
            sponge.belief_meta.items(),
            key=lambda kv: kv[1].confidence,
            reverse=True,
        )[:10]
    ) or "No beliefs yet"

    recent_shifts = "\n".join(
        f"- [{s.interaction}] {s.description} (mag={s.magnitude:.3f})"
        for s in sponge.recent_shifts[-5:]
    ) or "No recent shifts"

    prompt = HEALTH_ASSESSMENT_PROMPT.format(
        snapshot=sponge.snapshot[:500],
        beliefs_summary=beliefs_summary,
        recent_shifts=recent_shifts,
        interaction_count=sponge.interaction_count,
        disagreement_rate=f"{sponge.behavioral_signature.disagreement_rate:.2f}",
        belief_count=len(sponge.belief_meta),
        high_conf_count=high_conf_count,
    )
    result = llm_call(
        prompt=prompt,
        response_model=HealthResponse,
        fallback=HealthResponse(),
    )

    if result.success and result.value:
        response = result.value
        assert isinstance(response, HealthResponse)
        report = HealthReport(
            overall_health=response.overall_health,
            concerns=response.concerns,
            recommendations=response.recommendations,
            coherence_score=response.metrics.coherence_score,
            consistency_score=response.metrics.consistency_score,
            growth_health_score=response.metrics.growth_health_score,
            reasoning=response.reasoning,
        )
        if response.concerns:
            log.warning("Health concerns: %s", response.concerns)
        return report

    return HealthReport(
        overall_health="unknown",
        concerns=["Health assessment failed"],
        recommendations=[],
        coherence_score=0.5,
        consistency_score=0.5,
        growth_health_score=0.5,
        reasoning="Assessment unavailable",
    )
