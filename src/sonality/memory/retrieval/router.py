"""LLM-based query router for memory retrieval strategy selection.

Every query goes through the LLM router — no heuristic fast-paths.
The LLM determines category, result count, and flags for the retrieval pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field

from ...caller import llm_call
from ...prompts import QUERY_ROUTING_PROMPT

log = structlog.get_logger()


class QueryCategory(StrEnum):
    NONE = "NONE"
    SIMPLE = "SIMPLE"
    TEMPORAL = "TEMPORAL"
    MULTI_ENTITY = "MULTI_ENTITY"
    AGGREGATION = "AGGREGATION"
    BELIEF_QUERY = "BELIEF_QUERY"


class TemporalExpansionDecision(StrEnum):
    EXPAND = "EXPAND"
    NO_EXPAND = "NO_EXPAND"


class SemanticMemoryDecision(StrEnum):
    SEARCH = "SEARCH"
    SKIP = "SKIP"


class _RoutingResponse(BaseModel):
    category: QueryCategory = QueryCategory.SIMPLE
    n_results: int = Field(default=7, ge=1, le=20)
    temporal_expansion: TemporalExpansionDecision = TemporalExpansionDecision.NO_EXPAND
    semantic_memory: SemanticMemoryDecision = SemanticMemoryDecision.SKIP
    should_decompose: bool = False
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Resolved retrieval strategy returned by route_query.

    n_results: episode count chosen by the LLM (1–20).
    """

    category: QueryCategory
    n_results: int
    temporal_expansion: TemporalExpansionDecision
    semantic_memory: SemanticMemoryDecision
    should_decompose: bool = False


_FALLBACK = RoutingDecision(
    category=QueryCategory.SIMPLE,
    n_results=7,
    temporal_expansion=TemporalExpansionDecision.NO_EXPAND,
    semantic_memory=SemanticMemoryDecision.SKIP,
)


def route_query(query: str) -> RoutingDecision:
    """Classify a query and determine retrieval strategy."""
    result = llm_call(
        prompt=QUERY_ROUTING_PROMPT.format(query=query),
        response_model=_RoutingResponse,
        fallback=_RoutingResponse(),
    )

    if not result.success:
        log.warning("route_fallback", error=result.error)
        return _FALLBACK

    r = result.value
    decision = RoutingDecision(
        category=r.category,
        n_results=r.n_results,
        temporal_expansion=r.temporal_expansion,
        semantic_memory=r.semantic_memory,
        should_decompose=r.should_decompose,
    )
    log.info(
        "query_routed",
        category=decision.category,
        n=decision.n_results,
        temporal=decision.temporal_expansion,
        semantic=decision.semantic_memory,
        reason=r.reasoning[:200],
    )
    return decision
