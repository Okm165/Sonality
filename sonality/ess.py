"""Epistemic Significance Score (ESS) classifier.

LLM-only evaluator that rates incoming content on a 0-1 salience scale,
classifies reasoning type, urgency, knowledge density, and whether the
agent's beliefs should be updated. No code-based heuristics — all
decisions are delegated to the language model.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from . import config
from .prompts import ESS_CLASSIFICATION_PROMPT
from .provider import default_provider, extract_tool_call_arguments, parse_json_object
from .schema import ChatRole

log = logging.getLogger(__name__)

REQUIRED_FIELDS: Final = frozenset({"score", "reasoning_type", "opinion_direction"})
MAX_ESS_RETRIES: Final = 2
ESS_TIMEOUT_SECONDS: Final = 120  # 2 minute timeout for ESS classification
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


def _build_aliases[E: StrEnum](cls: type[E], extras: dict[str, E] | None = None) -> dict[str, E]:
    """Auto-generate alias table from enum values + optional extras."""
    aliases: dict[str, E] = {}
    for member in cls:
        key = member.value.lower().replace("_", "")
        aliases[key] = member
        aliases[member.value.lower()] = member
    if extras:
        aliases.update(extras)
    return aliases


REASONING_TYPE_ALIASES: Final = _build_aliases(ReasoningType, {
    "logical": ReasoningType.LOGICAL_ARGUMENT,
    "empirical": ReasoningType.EMPIRICAL_DATA,
    "debunked": ReasoningType.DEBUNKED_CLAIM,
    "none": ReasoningType.NO_ARGUMENT,
})
URGENCY_LEVEL_ALIASES: Final = _build_aliases(UrgencyLevel, {
    "urgent": UrgencyLevel.IMMEDIATE,
    "normal": UrgencyLevel.STANDARD,
})
OPINION_DIRECTION_ALIASES: Final = _build_aliases(OpinionDirection, {
    "support": OpinionDirection.SUPPORTS,
    "oppose": OpinionDirection.OPPOSES,
    "mixed": OpinionDirection.NEUTRAL,
})
SOURCE_RELIABILITY_ALIASES: Final = _build_aliases(SourceReliability, {
    "na": SourceReliability.NOT_APPLICABLE,
})
KNOWLEDGE_DENSITY_ALIASES: Final = _build_aliases(KnowledgeDensity, {
    "medium": KnowledgeDensity.MODERATE,
    "na": KnowledgeDensity.NONE,
})
INTERNAL_CONSISTENCY_ALIASES: Final = _build_aliases(InternalConsistencyStatus, {
    "true": InternalConsistencyStatus.CONSISTENT,
    "false": InternalConsistencyStatus.INCONSISTENT,
    "yes": InternalConsistencyStatus.CONSISTENT,
    "no": InternalConsistencyStatus.INCONSISTENT,
})


def _vals(cls: type[StrEnum]) -> list[str]:
    return [v.value for v in cls]


RETRY_ALLOWED_VALUES_NOTE: Final = (
    f"{RETRY_REQUIRED_FIELD_NOTE} Allowed reasoning_type values: "
    f"{', '.join(_vals(ReasoningType))}. Allowed opinion_direction values: "
    f"{', '.join(_vals(OpinionDirection))}."
)
PROVIDER_JSON_ONLY_NOTE: Final = (
    "Return ONLY a valid JSON object with keys: score, reasoning_type, source_reliability, "
    "internal_consistency, novelty, topics, summary, opinion_direction, knowledge_density."
)


ESS_TOOL_SCHEMA: Final[dict[str, object]] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "Overall argument strength 0.0-1.0."},
        "reasoning_type": {
            "type": "string",
            "enum": _vals(ReasoningType),
            "description": "Primary reasoning type used.",
        },
        "source_reliability": {"type": "string", "enum": _vals(SourceReliability)},
        "internal_consistency": {
            "type": "string",
            "enum": _vals(InternalConsistencyStatus),
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
                "Subject-matter domain or concept labels (1-3 short lowercase). "
                "Derive ONLY from explicitly named concepts. Use precise standard names. "
                "NEVER use meta-labels: social pressure, consensus, disagreement, argument, evidence, manipulation, etc. "
                "Empty list [] if purely social/meta with no substantive subject."
            ),
        },
        "summary": {
            "type": "string",
            "description": "One-sentence third-person summary of USER's assertion in concrete terms. State the claim directly, not 'User discusses X'.",
        },
        "opinion_direction": {
            "type": "string",
            "enum": _vals(OpinionDirection),
            "description": "Directional stance: supports/opposes/neutral.",
        },
        "knowledge_density": {
            "type": "string",
            "enum": _vals(KnowledgeDensity),
            "description": "Density of learnable content: high=multiple facts/stats, moderate=some data, low=opinion, none=chitchat.",
        },
        "belief_update_recommended": {
            "type": "boolean",
            "description": "True if evidence warrants belief update (substantive claims with sources).",
        },
        "urgency": {
            "type": "string",
            "enum": _vals(UrgencyLevel),
            "description": "Time-sensitivity: immediate/standard/low.",
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
}
PROVIDER_ESS_TOOL: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": "classify_evidence",
        "description": "Classify evidence strength and extract metadata.",
        "parameters": ESS_TOOL_SCHEMA,
    },
}
PROVIDER_ESS_TOOL_CHOICE: Final[dict[str, object]] = {
    "type": "function",
    "function": {"name": "classify_evidence"},
}


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
        return (InternalConsistencyStatus.CONSISTENT if value
                else InternalConsistencyStatus.INCONSISTENT, False)
    return _parse_enum(InternalConsistencyStatus, str(value) if not isinstance(value, str) else value,
                       InternalConsistencyStatus.CONSISTENT, INTERNAL_CONSISTENCY_ALIASES)


def _run_classification_attempts(
    prompt: str, model: str,
) -> tuple[dict[str, object], int, int, int]:
    """Run classifier retries. Returns (data, attempts, input_tokens, output_tokens)."""
    data: dict[str, object] = {}
    attempts_executed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    for attempt in range(MAX_ESS_RETRIES):
        attempts_executed = attempt + 1
        full_prompt = (prompt if attempt == 0
                       else f"{prompt}\n\nPrevious response had bad fields. {RETRY_ALLOWED_VALUES_NOTE}")
        completion = default_provider.chat_completion(
            model=model,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=0.0,
            messages=({"role": ChatRole.USER, "content": f"{full_prompt}\n\n{PROVIDER_JSON_ONLY_NOTE}"},),
            tools=(PROVIDER_ESS_TOOL,),
            tool_choice=PROVIDER_ESS_TOOL_CHOICE,
            enable_thinking=False,
        )
        total_input_tokens += completion.input_tokens
        total_output_tokens += completion.output_tokens
        data = extract_tool_call_arguments(completion.raw, "classify_evidence") or parse_json_object(completion.text) or {}
        if not (REQUIRED_FIELDS - set(data.keys())):
            break
        log.warning("ESS attempt %d/%d missing=%s", attempt + 1, MAX_ESS_RETRIES,
                    REQUIRED_FIELDS - set(data.keys()))
    return data, attempts_executed, total_input_tokens, total_output_tokens


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


def _classify_inner(prompt: str, model: str) -> ESSResult:
    """Core classification: LLM call with retries, then coerce into ESSResult."""
    data, attempts_executed, input_tokens, output_tokens = _run_classification_attempts(prompt, model)

    missing_fields = tuple(sorted(field for field in REQUIRED_FIELDS if field not in data))
    coerced_fields: list[str] = []

    score = _coerce_float_field(data, "score", 0.0, coerced_fields)
    novelty = _coerce_float_field(data, "novelty", 0.0, coerced_fields)
    direction = _coerce_enum_field(
        cls=OpinionDirection, data=data, field="opinion_direction",
        default=OpinionDirection.NEUTRAL, aliases=OPINION_DIRECTION_ALIASES,
        coerced_fields=coerced_fields,
    )
    reasoning = _coerce_enum_field(
        cls=ReasoningType, data=data, field="reasoning_type",
        default=ReasoningType.NO_ARGUMENT, aliases=REASONING_TYPE_ALIASES,
        coerced_fields=coerced_fields,
    )
    reliability = _coerce_enum_field(
        cls=SourceReliability, data=data, field="source_reliability",
        default=SourceReliability.NOT_APPLICABLE, aliases=SOURCE_RELIABILITY_ALIASES,
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
        cls=KnowledgeDensity, data=data, field="knowledge_density",
        default=KnowledgeDensity.NONE, aliases=KNOWLEDGE_DENSITY_ALIASES,
        coerced_fields=coerced_fields,
    )

    belief_update_raw = data.get("belief_update_recommended", False)
    belief_update_recommended = (
        belief_update_raw if isinstance(belief_update_raw, bool)
        else str(belief_update_raw).lower() in ("true", "yes", "1")
    )

    urgency = _coerce_enum_field(
        cls=UrgencyLevel, data=data, field="urgency",
        default=UrgencyLevel.STANDARD, aliases=URGENCY_LEVEL_ALIASES,
        coerced_fields=coerced_fields,
    )

    defaulted_fields = tuple(sorted(
        {*(f"{MISSING_FIELD_PREFIX}{f}" for f in missing_fields),
         *(f"{COERCED_FIELD_PREFIX}{f}" for f in coerced_fields)}
    ))
    severity: DefaultSeverity = (
        "missing" if missing_fields else "coercion" if coerced_fields else "none"
    )
    if defaulted_fields:
        log.warning("ESS fell back/coerced fields %s", defaulted_fields)

    return ESSResult(
        score=score, novelty=novelty,
        reasoning_type=reasoning, source_reliability=reliability,
        internal_consistency=internal_consistency, topics=topics,
        summary=summary, opinion_direction=direction,
        knowledge_density=knowledge_density,
        belief_update_recommended=belief_update_recommended,
        urgency=urgency, defaulted_fields=defaulted_fields,
        default_severity=severity,
        attempt_count=max(attempts_executed, 1),
        input_tokens=input_tokens, output_tokens=output_tokens,
    )


def classify(
    user_message: str,
    snapshot_text: str,
    model: str = config.ESS_MODEL,
    tracked_topics: str = "",
) -> ESSResult:
    """Classify evidence strength of the user's message.

    Uses a separate LLM call with tool_use to extract structured ESS metadata.
    The agent_response is deliberately excluded to avoid self-judge bias.
    Assumes classifier outputs may be malformed; coercion/default tracking is
    preserved in the result for downstream safety gating and auditing.
    """
    prompt = ESS_CLASSIFICATION_PROMPT.format(
        user_message=user_message,
        snapshot_text=snapshot_text,
        tracked_topics=tracked_topics or "none yet",
    )
    log.info("ESS classifying message (%d chars)", len(user_message))

    # Run classification with timeout protection
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_classify_inner, prompt, model)
        try:
            result = future.result(timeout=ESS_TIMEOUT_SECONDS)
        except FuturesTimeoutError:
            log.warning(
                "ESS_TIMEOUT: classification exceeded %ds, using fallback",
                ESS_TIMEOUT_SECONDS,
            )
            return classifier_exception_fallback(user_message)
        except Exception:
            log.exception("ESS classification error, using fallback")
            return classifier_exception_fallback(user_message)

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
