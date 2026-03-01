from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final, Literal, cast

from anthropic import Anthropic
from anthropic.types import Message

from . import config
from .prompts import ESS_CLASSIFICATION_PROMPT

log = logging.getLogger(__name__)

REQUIRED_FIELDS: Final = frozenset({"score", "reasoning_type", "opinion_direction"})
MAX_ESS_RETRIES: Final = 2
ENUM_NORMALIZE_RE: Final = re.compile(r"[^a-z0-9_]+")
BOOL_TRUE_TOKENS: Final[frozenset[str]] = frozenset({"1", "true", "yes", "y"})
BOOL_FALSE_TOKENS: Final[frozenset[str]] = frozenset({"0", "false", "no", "n"})
RETRY_REQUIRED_FIELD_NOTE: Final = (
    "Repair required fields only: score must be numeric, and reasoning_type and "
    "opinion_direction must be exact enum values."
)
DefaultSeverity = Literal["none", "coercion", "missing", "exception"]
MISSING_FIELD_PREFIX: Final = "missing:"
COERCED_FIELD_PREFIX: Final = "coerced:"
CLASSIFIER_EXCEPTION_FIELD: Final = f"{MISSING_FIELD_PREFIX}classifier_exception"


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


REASONING_TYPE_ALIASES: Final[dict[str, ReasoningType]] = {
    "logical": ReasoningType.LOGICAL_ARGUMENT,
    "argument": ReasoningType.LOGICAL_ARGUMENT,
    "empirical": ReasoningType.EMPIRICAL_DATA,
    "data": ReasoningType.EMPIRICAL_DATA,
    "expert": ReasoningType.EXPERT_OPINION,
    "social": ReasoningType.SOCIAL_PRESSURE,
    "pressure": ReasoningType.SOCIAL_PRESSURE,
    "emotional": ReasoningType.EMOTIONAL_APPEAL,
    "none": ReasoningType.NO_ARGUMENT,
}
OPINION_DIRECTION_ALIASES: Final[dict[str, OpinionDirection]] = {
    "support": OpinionDirection.SUPPORTS,
    "pro": OpinionDirection.SUPPORTS,
    "oppose": OpinionDirection.OPPOSES,
    "against": OpinionDirection.OPPOSES,
    "con": OpinionDirection.OPPOSES,
    "mixed": OpinionDirection.NEUTRAL,
    "uncertain": OpinionDirection.NEUTRAL,
}
SOURCE_RELIABILITY_ALIASES: Final[dict[str, SourceReliability]] = {
    "peerreviewed": SourceReliability.PEER_REVIEWED,
    "notapplicable": SourceReliability.NOT_APPLICABLE,
    "na": SourceReliability.NOT_APPLICABLE,
    "n_a": SourceReliability.NOT_APPLICABLE,
}


def _enum_values(cls: type[StrEnum]) -> list[str]:
    return [v.value for v in cls]


REASONING_TYPE_VALUES: Final[tuple[str, ...]] = tuple(_enum_values(ReasoningType))
SOURCE_RELIABILITY_VALUES: Final[tuple[str, ...]] = tuple(_enum_values(SourceReliability))
OPINION_DIRECTION_VALUES: Final[tuple[str, ...]] = tuple(_enum_values(OpinionDirection))
RETRY_ALLOWED_VALUES_NOTE: Final = (
    f"{RETRY_REQUIRED_FIELD_NOTE} Allowed reasoning_type values: "
    f"{', '.join(REASONING_TYPE_VALUES)}. Allowed opinion_direction values: "
    f"{', '.join(OPINION_DIRECTION_VALUES)}."
)


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
                "enum": list(REASONING_TYPE_VALUES),
                "description": "Primary reasoning type used.",
            },
            "source_reliability": {
                "type": "string",
                "enum": list(SOURCE_RELIABILITY_VALUES),
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
                "enum": list(OPINION_DIRECTION_VALUES),
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
    defaulted_fields: tuple[str, ...] = ()
    default_severity: DefaultSeverity = "none"
    attempt_count: int = 1
    input_tokens: int = 0
    output_tokens: int = 0


def classifier_exception_fallback(user_message: str) -> ESSResult:
    return ESSResult(
        score=0.0,
        reasoning_type=ReasoningType.NO_ARGUMENT,
        source_reliability=SourceReliability.NOT_APPLICABLE,
        internal_consistency=True,
        novelty=0.0,
        topics=(),
        summary=user_message[:120],
        used_defaults=True,
        defaulted_fields=(CLASSIFIER_EXCEPTION_FIELD,),
        default_severity="exception",
        attempt_count=0,
    )


def _extract_tool_data(response: Message) -> dict[str, object]:
    for block in response.content:
        if getattr(block, "type", None) != "tool_use":
            continue
        raw = getattr(block, "input", None)
        if isinstance(raw, dict):
            return cast(dict[str, object], raw)
    return {}


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_label(raw: str) -> str:
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = ENUM_NORMALIZE_RE.sub("", normalized)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _parse_enum[E: StrEnum](
    cls: type[E],
    raw: object,
    default: E,
    aliases: Mapping[str, E] | None = None,
) -> tuple[E, bool]:
    if not isinstance(raw, str):
        return default, True
    normalized = _normalize_label(raw)
    if aliases and normalized in aliases:
        return aliases[normalized], False
    try:
        return cls(normalized), False
    except ValueError:
        return default, True


def _to_float(value: object, default: float = 0.0) -> tuple[float, bool]:
    if isinstance(value, bool):
        return default, True
    if isinstance(value, (int, float)):
        return float(value), False
    if isinstance(value, str):
        try:
            return float(value), False
        except ValueError:
            return default, True
    return default, True


def _to_topics(value: object) -> tuple[tuple[str, ...], bool]:
    if not isinstance(value, (list, tuple)):
        if isinstance(value, str):
            parsed = tuple(
                token.strip() for token in value.replace("\n", ",").split(",") if token.strip()
            )
            return parsed, False
        return (), True
    topics = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return topics, False


def _to_bool(value: object, default: bool = True) -> tuple[bool, bool]:
    if isinstance(value, bool):
        return value, False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in BOOL_TRUE_TOKENS:
            return True, False
        if lowered in BOOL_FALSE_TOKENS:
            return False, False
        return default, True
    if isinstance(value, (int, float)):
        if value == 1:
            return True, False
        if value == 0:
            return False, False
        return default, True
    return default, True


def _to_nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


def _extract_usage_tokens(response: Message) -> tuple[int, int]:
    usage = getattr(response, "usage", None)
    input_tokens = _to_nonnegative_int(getattr(usage, "input_tokens", 0))
    output_tokens = _to_nonnegative_int(getattr(usage, "output_tokens", 0))
    return input_tokens, output_tokens


def _default_severity(defaulted_fields: tuple[str, ...]) -> DefaultSeverity:
    if CLASSIFIER_EXCEPTION_FIELD in defaulted_fields:
        return "exception"
    if any(field.startswith(MISSING_FIELD_PREFIX) for field in defaulted_fields):
        return "missing"
    if any(field.startswith(COERCED_FIELD_PREFIX) for field in defaulted_fields):
        return "coercion"
    return "none"


def _record_coercion(
    coerced_fields: list[str],
    field: str,
    defaulted: bool,
    data: Mapping[str, object],
) -> None:
    if defaulted and field in data:
        coerced_fields.append(field)


def _build_defaulted_fields(
    missing_fields: tuple[str, ...], coerced_fields: list[str]
) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                *(f"{MISSING_FIELD_PREFIX}{field}" for field in missing_fields),
                *(f"{COERCED_FIELD_PREFIX}{field}" for field in coerced_fields),
            }
        )
    )


def _required_field_coercions(data: Mapping[str, object]) -> tuple[str, ...]:
    """Return required fields that still parse as defaults."""
    coercions: list[str] = []
    if _to_float(data.get("score", 0.0), 0.0)[1]:
        coercions.append("score")
    if _parse_enum(
        ReasoningType,
        data.get("reasoning_type"),
        ReasoningType.NO_ARGUMENT,
        REASONING_TYPE_ALIASES,
    )[1]:
        coercions.append("reasoning_type")
    if _parse_enum(
        OpinionDirection,
        data.get("opinion_direction"),
        OpinionDirection.NEUTRAL,
        OPINION_DIRECTION_ALIASES,
    )[1]:
        coercions.append("opinion_direction")
    return tuple(coercions)


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

    data: dict[str, object] = {}
    attempts_executed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    for attempt in range(MAX_ESS_RETRIES):
        attempts_executed = attempt + 1
        prompt_with_retry_guidance = (
            prompt if attempt == 0 else f"{prompt}\n\n{RETRY_ALLOWED_VALUES_NOTE}"
        )
        response = cast(
            Message,
            cast(Any, client.messages.create)(
                model=config.ESS_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt_with_retry_guidance}],
                tools=[cast(Any, ESS_TOOL)],
                tool_choice=cast(Any, {"type": "tool", "name": "classify_evidence"}),
            ),
        )
        input_tokens, output_tokens = _extract_usage_tokens(response)
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        data = _extract_tool_data(response)

        missing = REQUIRED_FIELDS - set(data.keys())
        required_coercions = _required_field_coercions(data) if not missing else ()
        if not missing and not required_coercions:
            break
        log.warning(
            "ESS attempt %d/%d missing fields %s malformed_required %s",
            attempt + 1,
            MAX_ESS_RETRIES,
            missing,
            required_coercions,
        )

    missing_fields = tuple(sorted(field for field in REQUIRED_FIELDS if field not in data))
    coerced_fields: list[str] = []

    score_raw = data.get("score", 0.0)
    score_value, score_defaulted = _to_float(score_raw, 0.0)
    _record_coercion(coerced_fields, "score", score_defaulted, data)

    novelty_raw = data.get("novelty", 0.0)
    novelty_value, novelty_defaulted = _to_float(novelty_raw, 0.0)
    _record_coercion(coerced_fields, "novelty", novelty_defaulted, data)

    direction, direction_defaulted = _parse_enum(
        OpinionDirection,
        data.get("opinion_direction"),
        OpinionDirection.NEUTRAL,
        OPINION_DIRECTION_ALIASES,
    )
    _record_coercion(coerced_fields, "opinion_direction", direction_defaulted, data)

    reasoning, reasoning_defaulted = _parse_enum(
        ReasoningType,
        data.get("reasoning_type"),
        ReasoningType.NO_ARGUMENT,
        REASONING_TYPE_ALIASES,
    )
    _record_coercion(coerced_fields, "reasoning_type", reasoning_defaulted, data)

    reliability, reliability_defaulted = _parse_enum(
        SourceReliability,
        data.get("source_reliability"),
        SourceReliability.NOT_APPLICABLE,
        SOURCE_RELIABILITY_ALIASES,
    )
    _record_coercion(coerced_fields, "source_reliability", reliability_defaulted, data)

    internal_consistency, consistency_defaulted = _to_bool(
        data.get("internal_consistency", True), True
    )
    _record_coercion(coerced_fields, "internal_consistency", consistency_defaulted, data)

    topics, topics_defaulted = _to_topics(data.get("topics", ()))
    _record_coercion(coerced_fields, "topics", topics_defaulted, data)

    summary_raw = data.get("summary", "")
    summary = summary_raw if isinstance(summary_raw, str) else str(summary_raw)
    _record_coercion(coerced_fields, "summary", not isinstance(summary_raw, str), data)

    defaulted_fields = _build_defaulted_fields(missing_fields, coerced_fields)
    used_defaults = bool(defaulted_fields)
    default_severity = _default_severity(defaulted_fields)
    if used_defaults:
        log.warning(
            "ESS fell back/coerced fields %s",
            defaulted_fields,
        )

    result = ESSResult(
        score=_clamp(score_value),
        reasoning_type=reasoning,
        source_reliability=reliability,
        internal_consistency=internal_consistency,
        novelty=_clamp(novelty_value),
        topics=topics,
        summary=summary,
        opinion_direction=direction,
        used_defaults=used_defaults,
        defaulted_fields=defaulted_fields,
        default_severity=default_severity,
        attempt_count=max(attempts_executed, 1),
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
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
