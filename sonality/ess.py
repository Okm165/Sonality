from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from anthropic import Anthropic
from anthropic.types import Message

from . import config
from .prompts import ESS_CLASSIFICATION_PROMPT

log = logging.getLogger(__name__)

REQUIRED_FIELDS: Final = frozenset({"score", "reasoning_type", "opinion_direction"})
MAX_ESS_RETRIES: Final = 2


class ReasoningType(StrEnum):
    LOGICAL_ARGUMENT = "logical_argument"
    EMPIRICAL_DATA = "empirical_data"
    EXPERT_OPINION = "expert_opinion"
    ANECDOTAL = "anecdotal"
    SOCIAL_PRESSURE = "social_pressure"
    EMOTIONAL_APPEAL = "emotional_appeal"
    NO_ARGUMENT = "no_argument"


class OpinionDirection(StrEnum):
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    NEUTRAL = "neutral"

    @property
    def sign(self) -> float:
        return {self.SUPPORTS: 1.0, self.OPPOSES: -1.0, self.NEUTRAL: 0.0}[self]


class SourceReliability(StrEnum):
    PEER_REVIEWED = "peer_reviewed"
    ESTABLISHED_EXPERT = "established_expert"
    INFORMED_OPINION = "informed_opinion"
    CASUAL_OBSERVATION = "casual_observation"
    UNVERIFIED_CLAIM = "unverified_claim"
    NOT_APPLICABLE = "not_applicable"


def _enum_values(cls: type[StrEnum]) -> list[str]:
    return [v.value for v in cls]


ESS_TOOL: Final = {
    "name": "classify_evidence",
    "description": "Classify the evidence strength and extract metadata from this interaction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "number",
                "description": "Overall argument strength 0.0-1.0.",
            },
            "reasoning_type": {
                "type": "string",
                "enum": _enum_values(ReasoningType),
                "description": "Primary reasoning type used.",
            },
            "source_reliability": {
                "type": "string",
                "enum": _enum_values(SourceReliability),
            },
            "internal_consistency": {
                "type": "boolean",
                "description": "Whether the argument is internally consistent.",
            },
            "novelty": {
                "type": "number",
                "description": "Novelty relative to agent's existing views. 0=known, 1=entirely new.",
            },
            "topics": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key topics discussed (1-3 short labels).",
            },
            "summary": {
                "type": "string",
                "description": "One-sentence summary of the interaction.",
            },
            "opinion_direction": {
                "type": "string",
                "enum": _enum_values(OpinionDirection),
                "description": "Whether the user supports, opposes, or is neutral toward the primary topic.",
            },
        },
        "required": [
            "score",
            "reasoning_type",
            "source_reliability",
            "internal_consistency",
            "novelty",
            "topics",
            "summary",
            "opinion_direction",
        ],
    },
}


@dataclass(frozen=True, slots=True)
class ESSResult:
    score: float
    reasoning_type: ReasoningType
    source_reliability: SourceReliability
    internal_consistency: bool
    novelty: float
    topics: tuple[str, ...]
    summary: str
    opinion_direction: OpinionDirection = OpinionDirection.NEUTRAL
    used_defaults: bool = False


def _extract_tool_data(response: Message) -> dict[str, Any]:
    for block in response.content:
        if block.type == "tool_use":
            return block.input  # type: ignore[return-value]
    return {}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _parse_enum[E: StrEnum](cls: type[E], raw: object, default: E) -> E:
    try:
        return cls(raw) if raw is not None else default
    except ValueError:
        return default


def classify(
    client: Anthropic,
    user_message: str,
    sponge_snapshot: str,
) -> ESSResult:
    """Classify evidence strength of the user's message.

    Uses a separate LLM call with tool_use to extract structured ESS metadata.
    The agent_response is deliberately excluded to avoid self-judge bias
    (up to 50pp shift from attribution labels — ESS should evaluate user input only).
    """
    prompt = ESS_CLASSIFICATION_PROMPT.format(
        user_message=user_message,
        sponge_snapshot=sponge_snapshot,
    )
    log.info("ESS classifying message (%d chars)", len(user_message))

    data: dict[str, Any] = {}
    for attempt in range(MAX_ESS_RETRIES):
        response = client.messages.create(
            model=config.ESS_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
            tools=[ESS_TOOL],
            tool_choice={"type": "tool", "name": "classify_evidence"},
        )
        data = _extract_tool_data(response)

        missing = REQUIRED_FIELDS - set(data.keys())
        if not missing:
            break
        log.warning("ESS attempt %d/%d missing fields %s", attempt + 1, MAX_ESS_RETRIES, missing)

    defaults_used = [f for f in REQUIRED_FIELDS if f not in data]
    used_defaults = bool(defaults_used)
    if used_defaults:
        log.warning("ESS fell back to defaults for: %s", defaults_used)

    score = _clamp(float(data.get("score", 0.0)))
    novelty = _clamp(float(data.get("novelty", 0.0)))

    direction = _parse_enum(
        OpinionDirection, data.get("opinion_direction"), OpinionDirection.NEUTRAL
    )
    reasoning = _parse_enum(ReasoningType, data.get("reasoning_type"), ReasoningType.NO_ARGUMENT)
    reliability = _parse_enum(
        SourceReliability, data.get("source_reliability"), SourceReliability.NOT_APPLICABLE
    )

    result = ESSResult(
        score=score,
        reasoning_type=reasoning,
        source_reliability=reliability,
        internal_consistency=data.get("internal_consistency", True),
        novelty=novelty,
        topics=tuple(data.get("topics", ())),
        summary=data.get("summary", ""),
        opinion_direction=direction,
        used_defaults=used_defaults,
    )
    log.info(
        "ESS: score=%.2f type=%s dir=%s novelty=%.2f topics=%s",
        result.score,
        result.reasoning_type,
        result.opinion_direction,
        result.novelty,
        result.topics,
    )
    return result
