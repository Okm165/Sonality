from __future__ import annotations

from collections.abc import Callable

from sonality.memory.retrieval.router import (
    QueryCategory,
    QueryRouter,
    RetrievalDepth,
    SemanticMemoryDecision,
    TemporalExpansionDecision,
)


def test_router_uses_canned_llm_decision(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Classify this query": {
                "category": "TEMPORAL",
                "depth": "DEEP",
                "temporal_expansion": "EXPAND",
                "semantic_memory": "SKIP",
                "reasoning": "Needs chronology",
            }
        }
    )
    decision = QueryRouter().route("What changed over time?")
    assert decision.category is QueryCategory.TEMPORAL
    assert decision.depth is RetrievalDepth.DEEP
    assert decision.temporal_expansion is TemporalExpansionDecision.EXPAND
    assert decision.semantic_memory is SemanticMemoryDecision.SKIP
    assert decision.n_results == 15
