from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, Protocol, cast

from . import config
from .prompts import ESS_CLASSIFICATION_PROMPT
from .provider import chat_completion, extract_tool_call_arguments, parse_json_object

log = logging.getLogger(__name__)

REQUIRED_FIELDS: Final = frozenset({"score", "reasoning_type", "opinion_direction"})
MAX_ESS_RETRIES: Final = 2
ENUM_NORMALIZE_RE: Final = re.compile(r"[^a-z0-9_]+")
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
        """Map symbolic direction to signed numeric update direction."""
        return {self.SUPPORTS: 1.0, self.OPPOSES: -1.0, self.NEUTRAL: 0.0}[self]


class SourceReliability(StrEnum):
    PEER_REVIEWED = "peer_reviewed"
    ESTABLISHED_EXPERT = "established_expert"
    INFORMED_OPINION = "informed_opinion"
    CASUAL_OBSERVATION = "casual_observation"
    UNVERIFIED_CLAIM = "unverified_claim"
    NOT_APPLICABLE = "not_applicable"


class InternalConsistencyStatus(StrEnum):
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"


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
INTERNAL_CONSISTENCY_ALIASES: Final[dict[str, InternalConsistencyStatus]] = {
    # _normalize_label() lowercases everything, so "CONSISTENT" → "consistent"
    "consistent": InternalConsistencyStatus.CONSISTENT,
    "inconsistent": InternalConsistencyStatus.INCONSISTENT,
    "true": InternalConsistencyStatus.CONSISTENT,
    "false": InternalConsistencyStatus.INCONSISTENT,
    "yes": InternalConsistencyStatus.CONSISTENT,
    "no": InternalConsistencyStatus.INCONSISTENT,
    "y": InternalConsistencyStatus.CONSISTENT,
    "n": InternalConsistencyStatus.INCONSISTENT,
    "1": InternalConsistencyStatus.CONSISTENT,
    "0": InternalConsistencyStatus.INCONSISTENT,
}


def _enum_values(cls: type[StrEnum]) -> list[str]:
    """Return enum values as plain strings for JSON schema fields."""
    return [v.value for v in cls]


REASONING_TYPE_VALUES: Final[tuple[str, ...]] = tuple(_enum_values(ReasoningType))
SOURCE_RELIABILITY_VALUES: Final[tuple[str, ...]] = tuple(_enum_values(SourceReliability))
OPINION_DIRECTION_VALUES: Final[tuple[str, ...]] = tuple(_enum_values(OpinionDirection))
INTERNAL_CONSISTENCY_VALUES: Final[tuple[str, ...]] = tuple(_enum_values(InternalConsistencyStatus))
RETRY_ALLOWED_VALUES_NOTE: Final = (
    f"{RETRY_REQUIRED_FIELD_NOTE} Allowed reasoning_type values: "
    f"{', '.join(REASONING_TYPE_VALUES)}. Allowed opinion_direction values: "
    f"{', '.join(OPINION_DIRECTION_VALUES)}."
)
PROVIDER_JSON_ONLY_NOTE: Final = (
    "Return ONLY a valid JSON object with keys: score, reasoning_type, source_reliability, "
    "internal_consistency, novelty, topics, summary, opinion_direction."
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
                "type": "string",
                "enum": list(INTERNAL_CONSISTENCY_VALUES),
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
PROVIDER_ESS_TOOL: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": "classify_evidence",
        "description": ESS_TOOL["description"],
        "parameters": ESS_TOOL["input_schema"],
    },
}
PROVIDER_ESS_TOOL_CHOICE: Final[dict[str, object]] = {
    "type": "function",
    "function": {"name": "classify_evidence"},
}


class _MessagesClientProtocol(Protocol):
    """Minimal protocol for mocked `client.messages` implementations in tests."""

    def create(self, **kwargs: object) -> _ToolUseResponseProtocol: ...


class _ClientProtocol(Protocol):
    """Minimal protocol for optional test-client injection."""

    messages: _MessagesClientProtocol


PROVIDER_CLIENT: Final[_ClientProtocol] = cast(_ClientProtocol, object())


class _ToolUseBlockProtocol(Protocol):
    """Single tool-use block in mocked classifier responses."""

    type: str
    input: Mapping[str, object] | object


class _UsageProtocol(Protocol):
    """Token usage fields exposed by mocked classifier responses."""

    input_tokens: int | float | bool
    output_tokens: int | float | bool


class _ToolUseResponseProtocol(Protocol):
    """Minimal response contract used by ESS extraction helpers."""

    content: Sequence[_ToolUseBlockProtocol]
    usage: _UsageProtocol


@dataclass(frozen=True, slots=True)
class ESSResult:
    """Structured evidence-strength classification used by update logic."""

    score: float
    reasoning_type: ReasoningType
    source_reliability: SourceReliability
    internal_consistency: InternalConsistencyStatus
    novelty: float
    topics: tuple[str, ...]
    summary: str
    opinion_direction: OpinionDirection = OpinionDirection.NEUTRAL
    defaulted_fields: tuple[str, ...] = ()
    default_severity: DefaultSeverity = "none"
    attempt_count: int = 1
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def used_defaults(self) -> bool:
        """Return whether classifier defaults/coercions were applied."""
        return bool(self.defaulted_fields)


@dataclass(frozen=True, slots=True)
class ClassificationAttempts:
    """Aggregate retry-loop outputs from one ESS classification request."""

    data: dict[str, object]
    attempts_executed: int
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class CoercedEssPayload:
    """Coerced classifier payload fields before final ESSResult construction."""

    score: float
    novelty: float
    reasoning_type: ReasoningType
    source_reliability: SourceReliability
    internal_consistency: InternalConsistencyStatus
    topics: tuple[str, ...]
    summary: str
    opinion_direction: OpinionDirection
    defaulted_fields: tuple[str, ...]
    default_severity: DefaultSeverity


def classifier_exception_fallback(user_message: str) -> ESSResult:
    """Return a safe, explicit fallback result when classification crashes."""
    return ESSResult(
        score=0.0,
        reasoning_type=ReasoningType.NO_ARGUMENT,
        source_reliability=SourceReliability.NOT_APPLICABLE,
        internal_consistency=InternalConsistencyStatus.CONSISTENT,
        novelty=0.0,
        topics=(),
        summary=user_message[:120],
        defaulted_fields=(CLASSIFIER_EXCEPTION_FIELD,),
        default_severity="exception",
        attempt_count=0,
    )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a score-like value into the configured inclusive range."""
    return max(low, min(high, value))


def _normalize_label(raw: str) -> str:
    """Normalize enum-like free text into snake_case tokens."""
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = ENUM_NORMALIZE_RE.sub("", normalized)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_")


def _parse_enum[E: StrEnum](
    cls: type[E],
    raw: object,
    default: E,
    aliases: Mapping[str, E],
) -> tuple[E, bool]:
    """Parse untrusted enum text with alias support and coercion signal."""
    if not isinstance(raw, str):
        return default, True
    normalized = _normalize_label(raw)
    if normalized in aliases:
        return aliases[normalized], False
    try:
        return cls(normalized), False
    except ValueError:
        return default, True


def _to_float(value: object, default: float = 0.0) -> tuple[float, bool]:
    """Parse float-like values and report whether coercion was required."""
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
    """Parse topic labels from list-like or comma/newline-delimited strings."""
    if not isinstance(value, (list, tuple)):
        if isinstance(value, str):
            parsed = tuple(
                token.strip() for token in value.replace("\n", ",").split(",") if token.strip()
            )
            return parsed, False
        return (), True
    topics = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return topics, False


def _to_internal_consistency(value: object) -> tuple[InternalConsistencyStatus, bool]:
    """Parse internal-consistency status from untrusted LLM output."""
    if isinstance(value, bool):
        return (
            InternalConsistencyStatus.CONSISTENT
            if value
            else InternalConsistencyStatus.INCONSISTENT,
            False,
        )
    if isinstance(value, str):
        parsed, defaulted = _parse_enum(
            InternalConsistencyStatus,
            value,
            InternalConsistencyStatus.CONSISTENT,
            INTERNAL_CONSISTENCY_ALIASES,
        )
        return parsed, defaulted
    if isinstance(value, (int, float)):
        if value == 1:
            return InternalConsistencyStatus.CONSISTENT, False
        if value == 0:
            return InternalConsistencyStatus.INCONSISTENT, False
        return InternalConsistencyStatus.CONSISTENT, True
    return InternalConsistencyStatus.CONSISTENT, True


def _to_nonnegative_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


def _extract_tool_data(response: _ToolUseResponseProtocol) -> dict[str, object]:
    """Extract tool-use payload from injected test-client responses."""
    for block in response.content:
        if block.type != "tool_use":
            continue
        payload = block.input
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _extract_usage_tokens(response: _ToolUseResponseProtocol) -> tuple[int, int]:
    usage = response.usage
    return (
        _to_nonnegative_int(usage.input_tokens),
        _to_nonnegative_int(usage.output_tokens),
    )


def _default_severity(defaulted_fields: tuple[str, ...]) -> DefaultSeverity:
    """Collapse field-level defaults into a single severity bucket."""
    if CLASSIFIER_EXCEPTION_FIELD in defaulted_fields:
        return "exception"
    if any(field.startswith(MISSING_FIELD_PREFIX) for field in defaulted_fields):
        return "missing"
    if any(field.startswith(COERCED_FIELD_PREFIX) for field in defaulted_fields):
        return "coercion"
    return "none"


def _build_defaulted_fields(
    missing_fields: tuple[str, ...], coerced_fields: list[str]
) -> tuple[str, ...]:
    """Build stable prefixed identifiers for missing/coerced fields."""
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


def _run_classification_attempts(
    client: _ClientProtocol,
    prompt: str,
    model: str,
) -> ClassificationAttempts:
    """Run classifier retries and return final payload with token totals."""
    data: dict[str, object] = {}
    attempts_executed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    for attempt in range(MAX_ESS_RETRIES):
        attempts_executed = attempt + 1
        prompt_with_retry_guidance = (
            prompt if attempt == 0 else f"{prompt}\n\n{RETRY_ALLOWED_VALUES_NOTE}"
        )
        if client is not PROVIDER_CLIENT:
            response = client.messages.create(
                model=model,
                max_tokens=config.FAST_LLM_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt_with_retry_guidance}],
                tools=[ESS_TOOL],
                tool_choice={"type": "tool", "name": "classify_evidence"},
            )
            input_tokens, output_tokens = _extract_usage_tokens(response)
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            data = _extract_tool_data(response)
        else:
            completion = chat_completion(
                model=model,
                max_tokens=config.FAST_LLM_MAX_TOKENS,
                temperature=0.0,
                messages=(
                    {
                        "role": "user",
                        "content": f"{prompt_with_retry_guidance}\n\n{PROVIDER_JSON_ONLY_NOTE}",
                    },
                ),
                tools=(PROVIDER_ESS_TOOL,),
                tool_choice=PROVIDER_ESS_TOOL_CHOICE,
                disable_thinking=True,
            )
            total_input_tokens += completion.input_tokens
            total_output_tokens += completion.output_tokens
            data = extract_tool_call_arguments(completion.raw, "classify_evidence")
            if not data:
                data = parse_json_object(completion.text)

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
    return ClassificationAttempts(
        data=data,
        attempts_executed=attempts_executed,
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
    )


def _coerce_float_field(
    data: Mapping[str, object],
    field: str,
    default: float,
    coerced_fields: list[str],
) -> float:
    """Parse one float field and append coercion marker when needed."""
    value, defaulted = _to_float(data.get(field, default), default)
    if defaulted and field in data:
        coerced_fields.append(field)
    return value


def _coerce_enum_field[E: StrEnum](
    *,
    cls: type[E],
    data: Mapping[str, object],
    field: str,
    default: E,
    aliases: Mapping[str, E],
    coerced_fields: list[str],
) -> E:
    """Parse one enum field and append coercion marker when needed."""
    value, defaulted = _parse_enum(
        cls=cls,
        raw=data.get(field),
        default=default,
        aliases=aliases,
    )
    if defaulted and field in data:
        coerced_fields.append(field)
    return value


def _coerce_payload(data: Mapping[str, object]) -> CoercedEssPayload:
    """Coerce untrusted classifier output into typed ESS payload fields."""
    missing_fields = tuple(sorted(field for field in REQUIRED_FIELDS if field not in data))
    coerced_fields: list[str] = []

    score_value = _coerce_float_field(data, "score", 0.0, coerced_fields)
    novelty_value = _coerce_float_field(data, "novelty", 0.0, coerced_fields)
    direction = _coerce_enum_field(
        cls=OpinionDirection,
        data=data,
        field="opinion_direction",
        default=OpinionDirection.NEUTRAL,
        aliases=OPINION_DIRECTION_ALIASES,
        coerced_fields=coerced_fields,
    )
    reasoning = _coerce_enum_field(
        cls=ReasoningType,
        data=data,
        field="reasoning_type",
        default=ReasoningType.NO_ARGUMENT,
        aliases=REASONING_TYPE_ALIASES,
        coerced_fields=coerced_fields,
    )
    reliability = _coerce_enum_field(
        cls=SourceReliability,
        data=data,
        field="source_reliability",
        default=SourceReliability.NOT_APPLICABLE,
        aliases=SOURCE_RELIABILITY_ALIASES,
        coerced_fields=coerced_fields,
    )

    internal_consistency, consistency_defaulted = _to_internal_consistency(
        data.get("internal_consistency", InternalConsistencyStatus.CONSISTENT)
    )
    if consistency_defaulted and "internal_consistency" in data:
        coerced_fields.append("internal_consistency")

    topics, topics_defaulted = _to_topics(data.get("topics", ()))
    if topics_defaulted and "topics" in data:
        coerced_fields.append("topics")

    summary_raw = data.get("summary", "")
    summary = summary_raw if isinstance(summary_raw, str) else str(summary_raw)
    if not isinstance(summary_raw, str) and "summary" in data:
        coerced_fields.append("summary")

    defaulted_fields = _build_defaulted_fields(missing_fields, coerced_fields)
    return CoercedEssPayload(
        score=score_value,
        novelty=novelty_value,
        reasoning_type=reasoning,
        source_reliability=reliability,
        internal_consistency=internal_consistency,
        topics=topics,
        summary=summary,
        opinion_direction=direction,
        defaulted_fields=defaulted_fields,
        default_severity=_default_severity(defaulted_fields),
    )


def classify(
    client: _ClientProtocol,
    user_message: str,
    sponge_snapshot: str,
    model: str = config.ESS_MODEL,
) -> ESSResult:
    """Classify evidence strength of the user's message.

    Uses a separate LLM call with tool_use to extract structured ESS metadata.
    The agent_response is deliberately excluded to avoid self-judge bias
    (up to 50pp shift from attribution labels — ESS should evaluate user input only).
    Assumes classifier outputs may be malformed; coercion/default tracking is
    preserved in the result for downstream safety gating and auditing.
    """
    prompt = ESS_CLASSIFICATION_PROMPT.format(
        user_message=user_message,
        sponge_snapshot=sponge_snapshot,
    )
    log.info("ESS classifying message (%d chars)", len(user_message))
    attempts = _run_classification_attempts(client, prompt, model)
    payload = _coerce_payload(attempts.data)

    if payload.defaulted_fields:
        log.warning(
            "ESS fell back/coerced fields %s",
            payload.defaulted_fields,
        )

    result = ESSResult(
        score=_clamp(payload.score),
        reasoning_type=payload.reasoning_type,
        source_reliability=payload.source_reliability,
        internal_consistency=payload.internal_consistency,
        novelty=_clamp(payload.novelty),
        topics=payload.topics,
        summary=payload.summary,
        opinion_direction=payload.opinion_direction,
        defaulted_fields=payload.defaulted_fields,
        default_severity=payload.default_severity,
        attempt_count=max(attempts.attempts_executed, 1),
        input_tokens=attempts.input_tokens,
        output_tokens=attempts.output_tokens,
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
