"""Epistemic Significance Score (ESS) classifier.

LLM-only evaluator that rates incoming content on a 0-1 salience scale
with 5 continuous credibility signal dimensions, urgency, and whether
the agent's beliefs should be updated. All decisions delegated to the LLM.

Credibility signals replace the former ReasoningType and SourceReliability
enums with orthogonal continuous dimensions validated against Meyer & Knobe
(2025), PASTEL (2025), CRACQ, GRADE, and Intersignal Ontology frameworks.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field, model_validator

from . import config
from .caller import format_prompt, llm_call
from .prompts import ESS_CLASSIFICATION_PROMPT

log = structlog.get_logger(__name__)

SIGNAL_NAMES: frozenset[str] = frozenset(
    ("specificity", "grounding", "rigor", "source_quality", "objectivity")
)


@dataclass(frozen=True, slots=True)
class CredibilitySignals:
    """Five orthogonal continuous credibility dimensions, each 0.0-1.0.

    specificity:    Claim precision and falsifiability.
    grounding:      Degree of verifiable supporting evidence.
    rigor:          Logical soundness from premises to conclusions.
    source_quality: Credibility of cited/implied sources.
    objectivity:    Absence of bias, manipulation, emotional framing.
    """

    specificity: float = 0.0
    grounding: float = 0.0
    rigor: float = 0.0
    source_quality: float = 0.0
    objectivity: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in SIGNAL_NAMES}

    def summary_str(self) -> str:
        """Compact string for prompt injection: sq=0.8 gr=0.7 ri=0.6 ob=0.9 sp=0.5."""
        return (
            f"sq={self.source_quality:.1f} gr={self.grounding:.1f} "
            f"ri={self.rigor:.1f} ob={self.objectivity:.1f} sp={self.specificity:.1f}"
        )


SIGNALS_FALLBACK = CredibilitySignals()


class _ESSSchema(BaseModel):
    """Structured ESS classification for LLM JSON output."""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    specificity: float = Field(default=0.0, ge=0.0, le=1.0)
    grounding: float = Field(default=0.0, ge=0.0, le=1.0)
    rigor: float = Field(default=0.0, ge=0.0, le=1.0)
    source_quality: float = Field(default=0.0, ge=0.0, le=1.0)
    objectivity: float = Field(default=0.0, ge=0.0, le=1.0)
    topics: list[str] = Field(default_factory=list, max_length=10)
    summary: str = Field(default="", max_length=2000)
    belief_update_recommended: bool = False
    urgency: float = Field(default=0.5, ge=0.0, le=1.0)

    @model_validator(mode="before")
    @classmethod
    def coerce_types(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        for float_field in ("score", "urgency", *SIGNAL_NAMES):
            raw = data.get(float_field)
            if isinstance(raw, bool):
                data[float_field] = 0.0
            elif isinstance(raw, str):
                try:
                    data[float_field] = max(0.0, min(1.0, float(raw)))
                except ValueError:
                    data[float_field] = 0.5 if float_field == "urgency" else 0.0
            elif isinstance(raw, (int, float)):
                data[float_field] = max(0.0, min(1.0, float(raw)))
        raw = data.get("topics")
        if isinstance(raw, str):
            data["topics"] = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
        elif isinstance(raw, list):
            data["topics"] = [str(t) for t in raw[:10] if t]
        raw = data.get("belief_update_recommended")
        if raw is not None and not isinstance(raw, bool):
            data["belief_update_recommended"] = str(raw).lower() in ("true", "yes", "1")
        return data


@dataclass(frozen=True, slots=True)
class ESSResult:
    """Structured evidence-strength classification used by update logic."""

    score: float
    signals: CredibilitySignals
    topics: tuple[str, ...]
    summary: str
    belief_update_recommended: bool = False
    urgency: float = 0.5


ESS_FALLBACK = ESSResult(
    score=0.0,
    signals=SIGNALS_FALLBACK,
    topics=(),
    summary="",
    urgency=0.5,
)


def classify(user_message: str, existing_topics: str = "") -> ESSResult:
    """Classify evidence strength of the user's message via structured LLM call."""
    log.info("ess_classify", chars=len(user_message))
    result = llm_call(
        instructions=format_prompt(
            ESS_CLASSIFICATION_PROMPT,
            user_message=user_message,
            existing_topics=existing_topics or "(none yet)",
        ),
        response_model=_ESSSchema,
        fallback=_ESSSchema(),
        model=config.settings.structured_model,
    )

    if not result.success:
        log.warning("ess_failed", error=result.error, category=result.error_category)
        return ESSResult(
            score=0.0,
            signals=SIGNALS_FALLBACK,
            topics=(),
            summary=user_message[:120],
        )

    s = result.value
    signals = CredibilitySignals(
        specificity=s.specificity,
        grounding=s.grounding,
        rigor=s.rigor,
        source_quality=s.source_quality,
        objectivity=s.objectivity,
    )
    ess = ESSResult(
        score=s.score,
        signals=signals,
        topics=tuple(dict.fromkeys(s.topics))[:10],
        summary=s.summary,
        belief_update_recommended=s.belief_update_recommended,
        urgency=s.urgency,
    )
    log.info(
        "ess_result",
        score=ess.score,
        signals=signals.summary_str(),
        update=ess.belief_update_recommended,
        urgency=ess.urgency,
        topics=ess.topics,
    )
    return ess
