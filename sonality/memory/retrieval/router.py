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
from ...llm.prompts import QUERY_ROUTING_PROMPT

log = logging.getLogger(__name__)


class QueryCategory(StrEnum):
    NONE = "NONE"
    SIMPLE = "SIMPLE"
    TEMPORAL = "TEMPORAL"
    MULTI_ENTITY = "MULTI_ENTITY"
    AGGREGATION = "AGGREGATION"
    BELIEF_QUERY = "BELIEF_QUERY"


class RetrievalDepth(StrEnum):
    MINIMAL = "MINIMAL"
    MODERATE = "MODERATE"
    DEEP = "DEEP"


class TemporalExpansionDecision(StrEnum):
    EXPAND = "EXPAND"
    NO_EXPAND = "NO_EXPAND"


class SemanticMemoryDecision(StrEnum):
    SEARCH = "SEARCH"
    SKIP = "SKIP"


DEPTH_TO_COUNT: dict[RetrievalDepth, int] = {
    RetrievalDepth.MINIMAL: 2,
    RetrievalDepth.MODERATE: 7,
    RetrievalDepth.DEEP: 15,
}


class RoutingResponse(BaseModel):
    category: QueryCategory = QueryCategory.SIMPLE
    depth: RetrievalDepth = RetrievalDepth.MODERATE
    temporal_expansion: TemporalExpansionDecision = TemporalExpansionDecision.NO_EXPAND
    semantic_memory: SemanticMemoryDecision = SemanticMemoryDecision.SKIP
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    category: QueryCategory
    depth: RetrievalDepth
    n_results: int
    temporal_expansion: TemporalExpansionDecision
    semantic_memory: SemanticMemoryDecision
    reasoning: str


class QueryRouter:
    """LLM-first query router. No heuristic bypass."""

    def route(self, query: str, context: str = "") -> RoutingDecision:
        """Classify a query and determine retrieval strategy."""
        prompt = QUERY_ROUTING_PROMPT.format(
            query=query,
            context=context or "No recent context",
        )
        result = llm_call(
            prompt=prompt,
            response_model=RoutingResponse,
            fallback=RoutingResponse(),
        )

        if result.success:
            response = result.value
            category = response.category
            depth = response.depth

            decision = RoutingDecision(
                category=category,
                depth=depth,
                n_results=DEPTH_TO_COUNT[depth],
                temporal_expansion=response.temporal_expansion,
                semantic_memory=response.semantic_memory,
                reasoning=response.reasoning,
            )
            log.info(
                "Query routed: category=%s depth=%s n=%d temporal=%s semantic=%s | %s",
                decision.category,
                decision.depth,
                decision.n_results,
                decision.temporal_expansion,
                decision.semantic_memory,
                response.reasoning[:100] if response.reasoning else "no reasoning",
            )
            return decision

        log.warning(
            "Query routing fallback: result.success=%s error=%s", result.success, result.error
        )
        return RoutingDecision(
            category=QueryCategory.SIMPLE,
            depth=RetrievalDepth.MODERATE,
            n_results=DEPTH_TO_COUNT[RetrievalDepth.MODERATE],
            temporal_expansion=TemporalExpansionDecision.NO_EXPAND,
            semantic_memory=SemanticMemoryDecision.SKIP,
            reasoning="Fallback routing",
        )
