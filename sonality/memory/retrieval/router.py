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


DEPTH_TO_COUNT: dict[RetrievalDepth, int] = {
    RetrievalDepth.MINIMAL: 2,
    RetrievalDepth.MODERATE: 7,
    RetrievalDepth.DEEP: 15,
}


class RoutingResponse(BaseModel):
    category: str
    depth: str = "MODERATE"
    needs_temporal_expansion: bool = False
    search_semantic_memory: bool = False
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    category: QueryCategory
    depth: RetrievalDepth
    n_results: int
    needs_temporal_expansion: bool
    search_semantic_memory: bool
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
            fallback=RoutingResponse(category="SIMPLE", depth="MODERATE"),
        )

        if result.success and result.value:
            response = result.value
            assert isinstance(response, RoutingResponse)
            try:
                category = QueryCategory(response.category)
            except ValueError:
                category = QueryCategory.SIMPLE
            try:
                depth = RetrievalDepth(response.depth)
            except ValueError:
                depth = RetrievalDepth.MODERATE

            decision = RoutingDecision(
                category=category,
                depth=depth,
                n_results=DEPTH_TO_COUNT[depth],
                needs_temporal_expansion=response.needs_temporal_expansion,
                search_semantic_memory=response.search_semantic_memory,
                reasoning=response.reasoning,
            )
            log.info(
                "Query routed: category=%s depth=%s temporal=%s",
                decision.category, decision.depth, decision.needs_temporal_expansion,
            )
            return decision

        # Fallback: moderate simple search
        return RoutingDecision(
            category=QueryCategory.SIMPLE,
            depth=RetrievalDepth.MODERATE,
            n_results=DEPTH_TO_COUNT[RetrievalDepth.MODERATE],
            needs_temporal_expansion=False,
            search_semantic_memory=False,
            reasoning="Fallback routing",
        )
