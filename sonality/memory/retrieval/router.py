"""LLM-based query router for memory retrieval strategy selection.

Every query goes through the LLM router - no heuristic fast-paths.
Returns category, depth, and flags that determine the retrieval pipeline.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from ...llm.caller import llm_call
from ...prompts import QUERY_ROUTING_PROMPT

log = logging.getLogger(__name__)


class QueryCategory(StrEnum):
    """Semantic category of a user query, determines retrieval strategy.

    NONE: no memory retrieval needed (greetings, meta).
    SIMPLE: single-topic lookup.
    TEMPORAL: time-ordered retrieval (e.g. "what happened last week").
    MULTI_ENTITY: comparison across entities, triggers split_retrieve.
    AGGREGATION: summarize/count across episodes.
    BELIEF_QUERY: queries about the agent's own beliefs.
    """

    NONE = "NONE"
    SIMPLE = "SIMPLE"
    TEMPORAL = "TEMPORAL"
    MULTI_ENTITY = "MULTI_ENTITY"
    AGGREGATION = "AGGREGATION"
    BELIEF_QUERY = "BELIEF_QUERY"


class RetrievalDepth(StrEnum):
    """How many episodes to retrieve — maps to concrete counts via DEPTH_TO_COUNT."""

    MINIMAL = "MINIMAL"
    MODERATE = "MODERATE"
    DEEP = "DEEP"


class TemporalExpansionDecision(StrEnum):
    """Whether to expand results with temporally adjacent episodes."""

    EXPAND = "EXPAND"
    NO_EXPAND = "NO_EXPAND"


class SemanticMemoryDecision(StrEnum):
    """Whether to include semantic feature memory in retrieval context."""

    SEARCH = "SEARCH"
    SKIP = "SKIP"


DEPTH_TO_COUNT: dict[RetrievalDepth, int] = {
    RetrievalDepth.MINIMAL: 2,
    RetrievalDepth.MODERATE: 7,
    RetrievalDepth.DEEP: 15,
}


class _RoutingResponse(BaseModel):
    category: QueryCategory = QueryCategory.SIMPLE
    depth: RetrievalDepth = RetrievalDepth.MODERATE
    temporal_expansion: TemporalExpansionDecision = TemporalExpansionDecision.NO_EXPAND
    semantic_memory: SemanticMemoryDecision = SemanticMemoryDecision.SKIP
    should_decompose: bool = False
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Resolved retrieval strategy returned by route_query.

    n_results: concrete episode count derived from depth via DEPTH_TO_COUNT.
    should_decompose: if True, the query should be split into sub-queries.
    """

    category: QueryCategory
    depth: RetrievalDepth
    n_results: int
    temporal_expansion: TemporalExpansionDecision
    semantic_memory: SemanticMemoryDecision
    should_decompose: bool = False


_FALLBACK = RoutingDecision(
    category=QueryCategory.SIMPLE,
    depth=RetrievalDepth.MODERATE,
    n_results=DEPTH_TO_COUNT[RetrievalDepth.MODERATE],
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
        log.warning("Query routing fallback: error=%s", result.error)
        return _FALLBACK

    r = result.value
    decision = RoutingDecision(
        category=r.category,
        depth=r.depth,
        n_results=DEPTH_TO_COUNT[r.depth],
        temporal_expansion=r.temporal_expansion,
        semantic_memory=r.semantic_memory,
        should_decompose=r.should_decompose,
    )
    log.info(
        "Query routed: category=%s depth=%s n=%d temporal=%s semantic=%s reason=%.200s",
        decision.category,
        decision.depth,
        decision.n_results,
        decision.temporal_expansion,
        decision.semantic_memory,
        r.reasoning,
    )
    return decision
