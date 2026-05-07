"""Epistemic Significance Score (ESS) classifier.

LLM-only evaluator that rates incoming content on a 0-1 salience scale,
classifies reasoning type, urgency, knowledge density, and whether the
agent's beliefs should be updated. All decisions delegated to the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field, model_validator

from . import config
from .caller import llm_call
from .prompts import ESS_CLASSIFICATION_PROMPT

log = structlog.get_logger()


class ReasoningType(StrEnum):
    LOGICAL_ARGUMENT = "logical_argument"
    EMPIRICAL_DATA = "empirical_data"
    EXPERT_OPINION = "expert_opinion"
    ANECDOTAL = "anecdotal"
    DEBUNKED_CLAIM = "debunked_claim"
    SOCIAL_PRESSURE = "social_pressure"
    EMOTIONAL_APPEAL = "emotional_appeal"
    NO_ARGUMENT = "no_argument"
    NEWS_REPORT = "news_report"
    AGGREGATED_SENTIMENT = "aggregated_sentiment"


class UrgencyLevel(StrEnum):
    IMMEDIATE = "immediate"
    STANDARD = "standard"
    LOW = "low"


class OpinionDirection(StrEnum):
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    NEUTRAL = "neutral"


class SourceReliability(StrEnum):
    PEER_REVIEWED = "peer_reviewed"
    ESTABLISHED_EXPERT = "established_expert"
    INFORMED_OPINION = "informed_opinion"
    CASUAL_OBSERVATION = "casual_observation"
    UNVERIFIED_CLAIM = "unverified_claim"
    NOT_APPLICABLE = "not_applicable"


class KnowledgeDensity(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NONE = "none"


def _safe_enum[E: StrEnum](raw: str, cls: type[E], default: E) -> E:
    """Parse enum value from LLM output. Falls back to default on mismatch."""
    if not raw:
        return default
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return cls(normalized)
    except ValueError:
        return default


class _ESSSchema(BaseModel):
    """Raw ESS classification output — LLM is responsible for valid values."""

    score: float = 0.0
    reasoning_type: str = ""
    source_reliability: str = ""
    topics: list[str] = Field(default_factory=list)
    summary: str = ""
    opinion_direction: str = ""
    knowledge_density: str = ""
    belief_update_recommended: bool = False
    urgency: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_types(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw = data.get("score")
        if isinstance(raw, bool):
            data["score"] = 0.0
        elif isinstance(raw, str):
            try:
                data["score"] = float(raw)
            except ValueError:
                data["score"] = 0.0
        raw = data.get("topics")
        if isinstance(raw, str):
            data["topics"] = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
        raw = data.get("belief_update_recommended")
        if raw is not None and not isinstance(raw, bool):
            data["belief_update_recommended"] = str(raw).lower() in ("true", "yes", "1")
        return data

    @model_validator(mode="after")
    def clamp_score(self) -> _ESSSchema:
        self.score = max(0.0, min(1.0, self.score))
        return self


@dataclass(frozen=True, slots=True)
class ESSResult:
    """Structured evidence-strength classification used by update logic."""

    score: float
    reasoning_type: ReasoningType
    source_reliability: SourceReliability
    topics: tuple[str, ...]
    summary: str
    opinion_direction: OpinionDirection = OpinionDirection.NEUTRAL
    knowledge_density: KnowledgeDensity = KnowledgeDensity.NONE
    belief_update_recommended: bool = False
    urgency: UrgencyLevel = UrgencyLevel.STANDARD


ESS_FALLBACK = ESSResult(
    score=0.0,
    reasoning_type=ReasoningType.NO_ARGUMENT,
    source_reliability=SourceReliability.NOT_APPLICABLE,
    topics=(),
    summary="",
    opinion_direction=OpinionDirection.NEUTRAL,
    knowledge_density=KnowledgeDensity.NONE,
    urgency=UrgencyLevel.STANDARD,
)


def classify(user_message: str) -> ESSResult:
    """Classify evidence strength of the user's message via structured LLM call."""
    safe_message = user_message[:4000].replace("{", "{{").replace("}", "}}")
    prompt = ESS_CLASSIFICATION_PROMPT.format(user_message=safe_message)
    log.info("ess_classify", chars=len(user_message))

    result = llm_call(
        prompt=prompt,
        response_model=_ESSSchema,
        fallback=_ESSSchema(),
        model=config.settings.structured_model,
        max_tokens=1024,
    )

    if not result.success:
        log.warning("ess_failed")
        return ESSResult(
            score=0.0,
            reasoning_type=ReasoningType.NO_ARGUMENT,
            source_reliability=SourceReliability.NOT_APPLICABLE,
            topics=(),
            summary=user_message[:120],
        )

    s = result.value
    ess = ESSResult(
        score=s.score,
        reasoning_type=_safe_enum(s.reasoning_type, ReasoningType, ReasoningType.NO_ARGUMENT),
        source_reliability=_safe_enum(s.source_reliability, SourceReliability, SourceReliability.NOT_APPLICABLE),
        topics=tuple(dict.fromkeys(s.topics)),
        summary=s.summary,
        opinion_direction=_safe_enum(s.opinion_direction, OpinionDirection, OpinionDirection.NEUTRAL),
        knowledge_density=_safe_enum(s.knowledge_density, KnowledgeDensity, KnowledgeDensity.NONE),
        belief_update_recommended=s.belief_update_recommended,
        urgency=_safe_enum(s.urgency, UrgencyLevel, UrgencyLevel.STANDARD),
    )
    log.info(
        "ess_result",
        score=ess.score,
        reasoning_type=ess.reasoning_type,
        direction=ess.opinion_direction,
        update=ess.belief_update_recommended,
        urgency=ess.urgency,
        topics=ess.topics,
    )
    return ess
