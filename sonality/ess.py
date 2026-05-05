"""Epistemic Significance Score (ESS) classifier.

LLM-only evaluator that rates incoming content on a 0-1 salience scale,
classifies reasoning type, urgency, knowledge density, and whether the
agent's beliefs should be updated. No code-based heuristics — all
decisions are delegated to the language model.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from . import config
from .llm.parse import extract_tool_call_arguments, parse_json_object
from .prompts import ESS_CLASSIFICATION_PROMPT
from .provider import default_provider
from .schema import ChatRole

log = logging.getLogger(__name__)

REQUIRED_FIELDS: Final = frozenset({"score", "reasoning_type", "opinion_direction"})
ENUM_NORMALIZE_RE: Final = re.compile(r"[^a-z0-9_]+")
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


REASONING_TYPE_ALIASES: Final = _build_aliases(
    ReasoningType,
    {
        "logical": ReasoningType.LOGICAL_ARGUMENT,
        "empirical": ReasoningType.EMPIRICAL_DATA,
        "debunked": ReasoningType.DEBUNKED_CLAIM,
        "none": ReasoningType.NO_ARGUMENT,
    },
)
URGENCY_LEVEL_ALIASES: Final = _build_aliases(
    UrgencyLevel,
    {
        "urgent": UrgencyLevel.IMMEDIATE,
        "normal": UrgencyLevel.STANDARD,
    },
)
OPINION_DIRECTION_ALIASES: Final = _build_aliases(
    OpinionDirection,
    {
        "support": OpinionDirection.SUPPORTS,
        "oppose": OpinionDirection.OPPOSES,
        "mixed": OpinionDirection.NEUTRAL,
    },
)
SOURCE_RELIABILITY_ALIASES: Final = _build_aliases(
    SourceReliability,
    {
        "na": SourceReliability.NOT_APPLICABLE,
    },
)
KNOWLEDGE_DENSITY_ALIASES: Final = _build_aliases(
    KnowledgeDensity,
    {
        "medium": KnowledgeDensity.MODERATE,
        "na": KnowledgeDensity.NONE,
    },
)


def _vals(cls: type[StrEnum]) -> list[str]:
    return [v.value for v in cls]


PROVIDER_JSON_ONLY_NOTE: Final = (
    "The response is a single valid JSON object with keys: score, reasoning_type, source_reliability, "
    "topics, summary, opinion_direction, knowledge_density, belief_update_recommended, urgency. "
    "Compact numeric literals for floats (e.g. 0.55 not 0.5500001); round to at most 4 decimal places."
)


class _ESSClassificationSchema(BaseModel):
    """Pydantic schema for raw ESS classification output.

    Handles common LLM type variations (string scores, string booleans,
    comma-separated topics). Enum resolution via alias tables happens after
    validation in _classify_inner.
    """

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
        elif isinstance(raw, tuple):
            data["topics"] = list(raw)
        raw = data.get("summary")
        if raw is not None and not isinstance(raw, str):
            data["summary"] = str(raw)
        raw = data.get("belief_update_recommended")
        if raw is not None and not isinstance(raw, bool):
            data["belief_update_recommended"] = str(raw).lower() in ("true", "yes", "1")
        return data

    @model_validator(mode="after")
    def clamp_score(self) -> _ESSClassificationSchema:
        self.score = max(0.0, min(1.0, self.score))
        return self


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
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "Subject-matter domain or concept labels (1-3 short lowercase). "
                "The concrete subjects discussed regardless of reasoning_type. "
                "Even debunked claims, questions, and emotional appeals have subjects. "
                "Precise standard names work best (e.g. 'climate_change', 'elon_musk', 'ukraine'). "
                "Meta-labels like 'argument', 'evidence', 'consensus' are not useful here. "
                "Empty list for pure greetings or chitchat with no factual subject matter."
            ),
        },
        "summary": {
            "type": "string",
            "description": (
                "One-sentence third-person summary of the user's assertion in concrete terms. "
                "State the claim directly rather than 'User discusses X'. "
                "Covers only what the user said, not agent responses or system context."
            ),
        },
        "opinion_direction": {
            "type": "string",
            "enum": _vals(OpinionDirection),
            "description": "Directional stance: supports=content affirms a view, opposes=content challenges a view, neutral=no clear stance.",
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
    aliases: dict[str, E],
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


def _classify_inner(prompt: str, model: str) -> ESSResult:
    """Core classification: provider call with tool_choice, Pydantic validation, enum resolution."""
    completion = default_provider.chat_completion(
        model=model,
        max_tokens=config.LLM_MAX_TOKENS,
        temperature=0.0,
        messages=({"role": ChatRole.USER, "content": f"{prompt}\n\n{PROVIDER_JSON_ONLY_NOTE}"},),
        tools=(PROVIDER_ESS_TOOL,),
        tool_choice=PROVIDER_ESS_TOOL_CHOICE,
    )

    raw = (
        extract_tool_call_arguments(completion.raw, "classify_evidence")
        or parse_json_object(completion.text)
        or {}
    )

    missing_fields = sorted(f for f in REQUIRED_FIELDS if f not in raw)

    try:
        s = _ESSClassificationSchema.model_validate(dict(raw))
    except ValidationError as exc:
        log.warning("ESS Pydantic validation failed, using defaults: %s", exc)
        s = _ESSClassificationSchema()

    coerced: list[str] = []

    def _resolve[E: StrEnum](field: str, cls: type[E], default: E, aliases: dict[str, E]) -> E:
        val, bad = _parse_enum(cls, getattr(s, field), default, aliases)
        if bad and field in raw:
            coerced.append(field)
        return val

    reasoning = _resolve(
        "reasoning_type", ReasoningType, ReasoningType.NO_ARGUMENT, REASONING_TYPE_ALIASES
    )
    direction = _resolve(
        "opinion_direction", OpinionDirection, OpinionDirection.NEUTRAL, OPINION_DIRECTION_ALIASES
    )
    reliability = _resolve(
        "source_reliability",
        SourceReliability,
        SourceReliability.NOT_APPLICABLE,
        SOURCE_RELIABILITY_ALIASES,
    )
    density = _resolve(
        "knowledge_density", KnowledgeDensity, KnowledgeDensity.NONE, KNOWLEDGE_DENSITY_ALIASES
    )
    urgency = _resolve("urgency", UrgencyLevel, UrgencyLevel.STANDARD, URGENCY_LEVEL_ALIASES)

    raw_score = raw.get("score")
    if isinstance(raw_score, bool) or (
        isinstance(raw_score, str) and not _is_float_like(raw_score)
    ):
        coerced.append("score")
    if "summary" in raw and not isinstance(raw.get("summary"), str):
        coerced.append("summary")
    if "topics" in raw and not isinstance(raw.get("topics"), (list, tuple, str)):
        coerced.append("topics")

    defaulted_fields = tuple(
        sorted(
            {
                *(f"{MISSING_FIELD_PREFIX}{f}" for f in missing_fields),
                *(f"{COERCED_FIELD_PREFIX}{f}" for f in coerced),
            }
        )
    )
    severity: DefaultSeverity = "missing" if missing_fields else "coercion" if coerced else "none"
    if defaulted_fields:
        log.warning("ESS fell back/coerced fields %s", defaulted_fields)

    return ESSResult(
        score=s.score,
        reasoning_type=reasoning,
        source_reliability=reliability,
        topics=tuple(dict.fromkeys(s.topics)),
        summary=s.summary,
        opinion_direction=direction,
        knowledge_density=density,
        belief_update_recommended=s.belief_update_recommended,
        urgency=urgency,
        defaulted_fields=defaulted_fields,
        default_severity=severity,
        attempt_count=1,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
    )


def _is_float_like(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def classify(
    user_message: str,
    model: str = config.STRUCTURED_MODEL,
) -> ESSResult:
    """Classify evidence strength of the user's message.

    Uses a separate LLM call with tool_use to extract structured ESS metadata.
    ESS is a pure text signal classifier — only the message content matters.
    """
    # Escape braces in user message to prevent .format() crashes on {}/{{}} in content
    safe_message = user_message.replace("{", "{{").replace("}", "}}")
    prompt = ESS_CLASSIFICATION_PROMPT.format(user_message=safe_message)
    log.info("ESS classifying message (%d chars)", len(user_message))

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_classify_inner, prompt, model)
        try:
            result = future.result(timeout=config.ESS_TIMEOUT)
        except FuturesTimeoutError:
            log.warning(
                "ESS_TIMEOUT: classification exceeded %ds, using fallback",
                config.ESS_TIMEOUT,
            )
            return classifier_exception_fallback(user_message)
        except Exception:
            log.exception("ESS classification error, using fallback")
            return classifier_exception_fallback(user_message)

    log.info(
        "ESS: score=%.2f type=%s dir=%s update=%s urgency=%s topics=%s",
        result.score,
        result.reasoning_type,
        result.opinion_direction,
        result.belief_update_recommended,
        result.urgency,
        result.topics,
    )
    return result
