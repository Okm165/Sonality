"""LLM-based query router for memory retrieval strategy selection.

The LLM determines category, result count, signal weights for quality-boosted
retrieval, and optionally multiple search passes with different criteria.
Falls back to a deterministic default when the LLM call fails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field, model_validator

from ... import config
from ...caller import async_llm_call, format_prompt
from ...ess import SIGNAL_NAMES
from ...prompts import QUERY_ROUTING_PROMPT

log = structlog.get_logger(__name__)
_MAX_WEIGHT = 0.3
_MAX_PASSES = 3


class QueryCategory(StrEnum):
    NONE = "NONE"
    SIMPLE = "SIMPLE"
    TEMPORAL = "TEMPORAL"
    AGGREGATION = "AGGREGATION"
    BELIEF_QUERY = "BELIEF_QUERY"


class TemporalExpansionDecision(StrEnum):
    EXPAND = "EXPAND"
    NO_EXPAND = "NO_EXPAND"


class SemanticMemoryDecision(StrEnum):
    SEARCH = "SEARCH"
    SKIP = "SKIP"


def _sanitize_weights(raw: object) -> dict[str, float]:
    """Clamp and filter signal weights from LLM output."""
    if not isinstance(raw, dict):
        return {}
    return {
        k: max(0.0, min(_MAX_WEIGHT, float(v)))
        for k, v in raw.items()
        if k in SIGNAL_NAMES and isinstance(v, (int, float)) and not isinstance(v, bool)
    }


class _SearchPass(BaseModel):
    """A single retrieval pass with optional query rewrite and signal weights."""

    query: str = Field(default="", max_length=500)
    signal_weights: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def normalize_pass(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        data["signal_weights"] = _sanitize_weights(data.get("signal_weights", {}))
        return data


_SKIP_SYNONYMS: frozenset[str] = frozenset(
    {"NO", "FALSE", "NONE", "EXCLUDE", "DISABLE", "OFF", "UNNECESSARY", "OMIT"}
)


class _RoutingResponse(BaseModel):
    category: QueryCategory = QueryCategory.SIMPLE
    n_results: int = Field(default=7, ge=1, le=20)
    temporal_expansion: TemporalExpansionDecision = TemporalExpansionDecision.NO_EXPAND
    semantic_memory: SemanticMemoryDecision = SemanticMemoryDecision.SKIP
    passes: list[_SearchPass] = Field(default_factory=list, max_length=_MAX_PASSES)
    reasoning: str = Field(default="", max_length=3000)

    @model_validator(mode="before")
    @classmethod
    def normalize_enums(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        for f in ("category", "temporal_expansion"):
            raw = data.get(f)
            if isinstance(raw, str):
                data[f] = raw.strip().upper().replace(" ", "_").replace("-", "_")
        raw_sm = data.get("semantic_memory")
        if isinstance(raw_sm, str):
            normalized = raw_sm.strip().upper().replace(" ", "_").replace("-", "_")
            data["semantic_memory"] = "SKIP" if normalized in _SKIP_SYNONYMS else "SEARCH"
        n = data.get("n_results")
        if isinstance(n, (int, float)) and not isinstance(n, bool):
            data["n_results"] = max(1, min(20, int(n)))
        passes = data.get("passes")
        if not isinstance(passes, list) or not passes:
            sw = _sanitize_weights(data.pop("signal_weights", {}))
            data["passes"] = [{"query": "", "signal_weights": sw}]
        else:
            data["passes"] = passes[:_MAX_PASSES]
        return data


@dataclass(frozen=True, slots=True)
class SearchPass:
    """A single search pass with optional query rewrite and per-signal boosting weights."""

    query: str = ""
    signal_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Resolved retrieval strategy returned by route_query.

    passes: one or more search passes, each with optional query rewrite
            and signal weights for Qdrant FormulaQuery rescoring.
    """

    category: QueryCategory
    n_results: int
    temporal_expansion: TemporalExpansionDecision
    semantic_memory: SemanticMemoryDecision
    passes: tuple[SearchPass, ...] = (SearchPass(),)


_FALLBACK = RoutingDecision(
    category=QueryCategory.SIMPLE,
    n_results=7,
    temporal_expansion=TemporalExpansionDecision.NO_EXPAND,
    semantic_memory=SemanticMemoryDecision.SKIP,
)


async def route_query(query: str) -> RoutingDecision:
    """Classify a query and determine retrieval strategy."""
    result = await async_llm_call(
        instructions=format_prompt(QUERY_ROUTING_PROMPT, query=query),
        response_model=_RoutingResponse,
        fallback=_RoutingResponse(),
        model=config.settings.structured_model,
    )

    if not result.success:
        log.warning("route_fallback", error=result.error)
        return _FALLBACK

    r = result.value
    passes = tuple(
        SearchPass(query=p.query.strip(), signal_weights=p.signal_weights) for p in r.passes
    ) or (SearchPass(),)

    decision = RoutingDecision(
        category=r.category,
        n_results=r.n_results,
        temporal_expansion=r.temporal_expansion,
        semantic_memory=r.semantic_memory,
        passes=passes,
    )
    log.info(
        "query_routed",
        category=decision.category,
        n=decision.n_results,
        temporal=decision.temporal_expansion,
        semantic=decision.semantic_memory,
        passes=len(decision.passes),
        reason=r.reasoning[:200],
    )
    return decision
