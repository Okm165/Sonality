from __future__ import annotations

import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal, Protocol, cast

from . import config
from .prompts import ESS_CLASSIFICATION_PROMPT
from .provider import (
    _to_nonnegative_int,
    chat_completion,
    extract_tool_call_arguments,
    parse_json_object,
)

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


class KnowledgeDensity(StrEnum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    NONE = "none"


class InternalConsistencyStatus(StrEnum):
    CONSISTENT = "CONSISTENT"
    INCONSISTENT = "INCONSISTENT"


REASONING_TYPE_ALIASES: Final[dict[str, ReasoningType]] = {
    "logical": ReasoningType.LOGICAL_ARGUMENT,
    "argument": ReasoningType.LOGICAL_ARGUMENT,
    "empirical": ReasoningType.EMPIRICAL_DATA,
    "data": ReasoningType.EMPIRICAL_DATA,
    "expert": ReasoningType.EXPERT_OPINION,
    "debunked": ReasoningType.DEBUNKED_CLAIM,
    "misinformation": ReasoningType.DEBUNKED_CLAIM,
    "conspiracy": ReasoningType.DEBUNKED_CLAIM,
    "social": ReasoningType.SOCIAL_PRESSURE,
    "pressure": ReasoningType.SOCIAL_PRESSURE,
    "emotional": ReasoningType.EMOTIONAL_APPEAL,
    "none": ReasoningType.NO_ARGUMENT,
    "news": ReasoningType.NEWS_REPORT,
    "report": ReasoningType.NEWS_REPORT,
    "newsreport": ReasoningType.NEWS_REPORT,
    "sentiment": ReasoningType.AGGREGATED_SENTIMENT,
    "aggregated": ReasoningType.AGGREGATED_SENTIMENT,
    "aggregatedsentiment": ReasoningType.AGGREGATED_SENTIMENT,
}
URGENCY_LEVEL_ALIASES: Final[dict[str, UrgencyLevel]] = {
    "immediate": UrgencyLevel.IMMEDIATE,
    "urgent": UrgencyLevel.IMMEDIATE,
    "breaking": UrgencyLevel.IMMEDIATE,
    "standard": UrgencyLevel.STANDARD,
    "normal": UrgencyLevel.STANDARD,
    "low": UrgencyLevel.LOW,
    "background": UrgencyLevel.LOW,
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
    # Legacy prompt values (model used to output these before prompt was fixed)
    "low": SourceReliability.UNVERIFIED_CLAIM,
    "medium": SourceReliability.INFORMED_OPINION,
    "high": SourceReliability.ESTABLISHED_EXPERT,
}
KNOWLEDGE_DENSITY_ALIASES: Final[dict[str, KnowledgeDensity]] = {
    "high": KnowledgeDensity.HIGH,
    "moderate": KnowledgeDensity.MODERATE,
    "medium": KnowledgeDensity.MODERATE,
    "low": KnowledgeDensity.LOW,
    "none": KnowledgeDensity.NONE,
    "n_a": KnowledgeDensity.NONE,
    "na": KnowledgeDensity.NONE,
}
INTERNAL_CONSISTENCY_ALIASES: Final[dict[str, InternalConsistencyStatus]] = {
    # _parse_enum lowercases everything, so "CONSISTENT" → "consistent"
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
KNOWLEDGE_DENSITY_VALUES: Final[tuple[str, ...]] = tuple(_enum_values(KnowledgeDensity))
URGENCY_LEVEL_VALUES: Final[tuple[str, ...]] = tuple(_enum_values(UrgencyLevel))
RETRY_ALLOWED_VALUES_NOTE: Final = (
    f"{RETRY_REQUIRED_FIELD_NOTE} Allowed reasoning_type values: "
    f"{', '.join(REASONING_TYPE_VALUES)}. Allowed opinion_direction values: "
    f"{', '.join(OPINION_DIRECTION_VALUES)}."
)
PROVIDER_JSON_ONLY_NOTE: Final = (
    "Return ONLY a valid JSON object with keys: score, reasoning_type, source_reliability, "
    "internal_consistency, novelty, topics, summary, opinion_direction, knowledge_density."
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
                "description": (
                    "Subject-matter domain or concept labels the message is substantively about "
                    "(1-3 short lowercase labels). Derive labels ONLY from concepts explicitly "
                    "named or directly asserted in the message. "
                    "Use the precise, standard name for the subject (e.g. 'universal basic income', "
                    "not 'ubiquitous basic income'). "
                    "NEVER use conversational meta-labels as topics: "
                    "banned topics include 'social pressure', 'pressure', 'peer pressure', "
                    "'consensus', 'industry consensus', 'scientific consensus', 'expert consensus', 'group consensus', "
                    "'disagreement', 'argument', 'evidence', 'manipulation', 'survey method', "
                    "'consistency', 'reliability', 'memory', 'credibility', "
                    "'emotion', 'emotional appeal', 'opinion', 'reasoning'. "
                    "If the message is purely social/meta (pressure, questioning, probing) "
                    "with no substantive subject, use an empty list []."
                ),
            },
            "summary": {
                "type": "string",
                "description": (
                    "One-sentence third-person summary of what the USER asserts or asks, in concrete "
                    "specific terms. State the actual subject matter directly: name the claim, study, "
                    "or data point. Do NOT use vague meta-descriptions like 'User discusses X' or "
                    "'User asks about X' — instead write what they assert: "
                    "'User presents Stanford (2022) data showing hybrid work lowers productivity.'"
                ),
            },
            "opinion_direction": {
                "type": "string",
                "enum": list(OPINION_DIRECTION_VALUES),
                "description": "Directional stance of the user's message: supports = positive argument or evidence FOR a claim; opposes = counter-evidence, rebuttal, or argument AGAINST a prior claim; neutral = no directional stance (questions, recaps, social).",
            },
            "knowledge_density": {
                "type": "string",
                "enum": list(KNOWLEDGE_DENSITY_VALUES),
                "description": (
                    "Density of learnable factual/conceptual content in the USER's message: "
                    "high = multiple specific numbers, percentages, named sources, or detailed factual exposition; "
                    "moderate = at least one verifiable fact, statistic, or citation; "
                    "low = mostly user opinion, social commentary, or questions without data; "
                    "none = greetings, chitchat, bare questions without data, or requests with no factual content. "
                    "ANY message presenting specific data points, statistics, named sources, or technical facts MUST be at least 'moderate'."
                ),
            },
            "belief_update_recommended": {
                "type": "boolean",
                "description": (
                    "Whether this evidence warrants updating the agent's beliefs. Set true when "
                    "the input contains substantive claims backed by identifiable evidence, named "
                    "sources, or verifiable data. Set false for greetings, bare opinions without "
                    "reasoning, social pressure, emotional appeals, or debunked content."
                ),
            },
            "urgency": {
                "type": "string",
                "enum": list(URGENCY_LEVEL_VALUES),
                "description": (
                    "Time-sensitivity of this information. 'immediate' for breaking events with "
                    "time-sensitive consequences (surprise policy decisions, natural disasters, "
                    "military actions). 'standard' for routine reports and analysis. 'low' for "
                    "historical analysis or background context."
                ),
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
            "knowledge_density",
            "belief_update_recommended",
            "urgency",
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
    knowledge_density: KnowledgeDensity = KnowledgeDensity.NONE
    belief_update_recommended: bool = False
    urgency: UrgencyLevel = UrgencyLevel.STANDARD
    defaulted_fields: tuple[str, ...] = ()
    default_severity: DefaultSeverity = "none"
    attempt_count: int = 1
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def used_defaults(self) -> bool:
        """Return whether classifier defaults/coercions were applied."""
        return bool(self.defaulted_fields)

    @property
    def cooling_period(self) -> int:
        """Map urgency level to belief update cooling period."""
        return {UrgencyLevel.IMMEDIATE: 1, UrgencyLevel.STANDARD: 2, UrgencyLevel.LOW: 3}[
            self.urgency
        ]


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
    knowledge_density: KnowledgeDensity
    belief_update_recommended: bool
    urgency: UrgencyLevel
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
        belief_update_recommended=False,
        urgency=UrgencyLevel.STANDARD,
        defaulted_fields=(CLASSIFIER_EXCEPTION_FIELD,),
        default_severity="exception",
        attempt_count=0,
    )


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a score-like value into the configured inclusive range."""
    return max(low, min(high, value))


def _parse_enum[E: StrEnum](
    cls: type[E],
    raw: object,
    default: E,
    aliases: Mapping[str, E],
) -> tuple[E, bool]:
    """Parse untrusted enum text with alias support and coercion signal."""
    if not isinstance(raw, str):
        return default, True
    normalized = raw.strip().lower().replace("-", "_").replace(" ", "_")
    normalized = ENUM_NORMALIZE_RE.sub("", normalized)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
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
    """Parse topic labels from list-like or comma/newline-delimited strings, deduplicating."""
    if not isinstance(value, (list, tuple)):
        if isinstance(value, str):
            parsed = tuple(
                dict.fromkeys(
                    token.strip() for token in value.replace("\n", ",").split(",") if token.strip()
                )
            )
            return parsed, False
        return (), True
    topics = tuple(
        dict.fromkeys(item.strip() for item in value if isinstance(item, str) and item.strip())
    )
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
    bad_fields: frozenset[str] = frozenset()
    for attempt in range(MAX_ESS_RETRIES):
        attempts_executed = attempt + 1
        if attempt == 0:
            prompt_with_retry_guidance = prompt
        else:
            bad_list = ", ".join(sorted(bad_fields)) if bad_fields else "required fields"
            prompt_with_retry_guidance = (
                f"{prompt}\n\nPrevious response was missing or invalid for: {bad_list}. "
                f"{RETRY_ALLOWED_VALUES_NOTE}"
            )
        if client is not PROVIDER_CLIENT:
            response = client.messages.create(
                model=model,
                max_tokens=config.FAST_LLM_MAX_TOKENS,
                messages=[{"role": "user", "content": prompt_with_retry_guidance}],
                tools=[ESS_TOOL],
                tool_choice={"type": "tool", "name": "classify_evidence"},
            )
            usage = response.usage
            total_input_tokens += _to_nonnegative_int(usage.input_tokens)
            total_output_tokens += _to_nonnegative_int(usage.output_tokens)
            data = {}
            for block in response.content:
                if block.type == "tool_use" and isinstance(block.input, Mapping):
                    data = dict(block.input)
                    break
        else:
            completion = chat_completion(
                model=model,
                max_tokens=384,  # ESS tool call: 11 fields including one-sentence summary
                temperature=0.0,
                messages=(
                    {
                        "role": "user",
                        "content": f"{prompt_with_retry_guidance}\n\n{PROVIDER_JSON_ONLY_NOTE}",
                    },
                ),
                tools=(PROVIDER_ESS_TOOL,),
                tool_choice=PROVIDER_ESS_TOOL_CHOICE,
                enable_thinking=False,
            )
            total_input_tokens += completion.input_tokens
            total_output_tokens += completion.output_tokens
            data = extract_tool_call_arguments(completion.raw, "classify_evidence")
            if not data:
                data = parse_json_object(completion.text)

        if not data:
            data = {}
        missing = REQUIRED_FIELDS - set(data.keys())
        required_coercions = _required_field_coercions(data) if not missing else ()
        if not missing and not required_coercions:
            break
        bad_fields = missing | frozenset(required_coercions)
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

    knowledge_density = _coerce_enum_field(
        cls=KnowledgeDensity,
        data=data,
        field="knowledge_density",
        default=KnowledgeDensity.NONE,
        aliases=KNOWLEDGE_DENSITY_ALIASES,
        coerced_fields=coerced_fields,
    )

    belief_update_raw = data.get("belief_update_recommended", False)
    if isinstance(belief_update_raw, bool):
        belief_update_recommended = belief_update_raw
    elif isinstance(belief_update_raw, str):
        belief_update_recommended = belief_update_raw.lower() in ("true", "yes", "1")
    else:
        belief_update_recommended = False
        if "belief_update_recommended" in data:
            coerced_fields.append("belief_update_recommended")

    urgency = _coerce_enum_field(
        cls=UrgencyLevel,
        data=data,
        field="urgency",
        default=UrgencyLevel.STANDARD,
        aliases=URGENCY_LEVEL_ALIASES,
        coerced_fields=coerced_fields,
    )

    defaulted_fields = tuple(
        sorted(
            {
                *(f"{MISSING_FIELD_PREFIX}{f}" for f in missing_fields),
                *(f"{COERCED_FIELD_PREFIX}{f}" for f in coerced_fields),
            }
        )
    )
    if CLASSIFIER_EXCEPTION_FIELD in defaulted_fields:
        severity: DefaultSeverity = "exception"
    elif any(f.startswith(MISSING_FIELD_PREFIX) for f in defaulted_fields):
        severity = "missing"
    elif any(f.startswith(COERCED_FIELD_PREFIX) for f in defaulted_fields):
        severity = "coercion"
    else:
        severity = "none"
    return CoercedEssPayload(
        score=score_value,
        novelty=novelty_value,
        reasoning_type=reasoning,
        source_reliability=reliability,
        internal_consistency=internal_consistency,
        topics=topics,
        summary=summary,
        opinion_direction=direction,
        knowledge_density=knowledge_density,
        belief_update_recommended=belief_update_recommended,
        urgency=urgency,
        defaulted_fields=defaulted_fields,
        default_severity=severity,
    )


def classify(
    client: _ClientProtocol,
    user_message: str,
    sponge_snapshot: str,
    model: str = config.ESS_MODEL,
    recent_topics: tuple[str, ...] = (),
) -> ESSResult:
    """Classify evidence strength of the user's message.

    Uses a separate LLM call with tool_use to extract structured ESS metadata.
    The agent_response is deliberately excluded to avoid self-judge bias
    (up to 50pp shift from attribution labels — ESS should evaluate user input only).
    Assumes classifier outputs may be malformed; coercion/default tracking is
    preserved in the result for downstream safety gating and auditing.

    recent_topics: top tracked topic labels from sponge — passed so the model
    can reuse existing labels and avoid generating synonyms that fragment beliefs.
    """
    tracked = ", ".join(recent_topics) if recent_topics else "none yet"
    prompt = ESS_CLASSIFICATION_PROMPT.format(
        user_message=user_message,
        sponge_snapshot=sponge_snapshot,
        tracked_topics=tracked,
    )
    log.info("ESS classifying message (%d chars)", len(user_message))
    attempts = _run_classification_attempts(client, prompt, model)
    payload = _coerce_payload(attempts.data)

    if payload.defaulted_fields:
        log.warning(
            "ESS fell back/coerced fields %s",
            payload.defaulted_fields,
        )

    # Consistency guard: substantive reasoning types inherently carry learnable
    # content, so knowledge_density=NONE is contradictory for them.
    # logical_argument with score >= 0.30 also qualifies — a real argument with
    # named concepts and structured reasoning is worth extracting propositions from.
    kd = payload.knowledge_density
    if kd == KnowledgeDensity.NONE and (
        payload.reasoning_type
        in (
            ReasoningType.EMPIRICAL_DATA,
            ReasoningType.EXPERT_OPINION,
            ReasoningType.NEWS_REPORT,
        )
        or (payload.reasoning_type is ReasoningType.LOGICAL_ARGUMENT and payload.score >= 0.30)
    ):
        kd = KnowledgeDensity.LOW
        log.debug(
            "ESS: upgraded knowledge_density NONE→LOW for type=%s score=%.2f",
            payload.reasoning_type,
            payload.score,
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
        knowledge_density=kd,
        belief_update_recommended=payload.belief_update_recommended,
        urgency=payload.urgency,
        defaulted_fields=payload.defaulted_fields,
        default_severity=payload.default_severity,
        attempt_count=max(attempts.attempts_executed, 1),
        input_tokens=attempts.input_tokens,
        output_tokens=attempts.output_tokens,
    )
    log.info(
        "ESS: score=%.2f type=%s dir=%s novelty=%.2f update=%s urgency=%s topics=%s",
        result.score,
        result.reasoning_type,
        result.opinion_direction,
        result.novelty,
        result.belief_update_recommended,
        result.urgency,
        result.topics,
    )
    return result
