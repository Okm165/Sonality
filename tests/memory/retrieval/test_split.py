from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast
from unittest.mock import AsyncMock

from sonality.memory.dual_store import DualEpisodeStore
from sonality.memory.graph import EpisodeNode, MemoryGraph
from sonality.memory.retrieval.split import AggregationStrategy, SplitQueryAgent, SplitResult


def _store_mock() -> DualEpisodeStore:
    store = AsyncMock(spec=DualEpisodeStore)

    async def _vector_search(query: str, top_k: int = 10) -> list[tuple[str, str, float]]:
        _ = top_k
        return [("d1", f"{query}-ep", 0.1)]

    store.vector_search = AsyncMock(side_effect=_vector_search)
    return cast(DualEpisodeStore, store)


def _graph_mock() -> MemoryGraph:
    graph = AsyncMock(spec=MemoryGraph)

    async def _get_episodes(uids: list[str]) -> list[EpisodeNode]:
        return [
            EpisodeNode(
                uid=uid,
                content=f"content {uid}",
                summary=f"summary {uid}",
                topics=[],
                ess_score=0.5,
                created_at=(
                    "2026-02-01T00:00:00Z" if uid.startswith("later") else "2026-01-01T00:00:00Z"
                ),
                valid_at="2026-01-01T00:00:00Z",
            )
            for uid in uids
        ]

    graph.get_episodes = AsyncMock(side_effect=_get_episodes)
    return cast(MemoryGraph, graph)


def _retrieve(query: str) -> SplitResult:
    return asyncio.run(
        SplitQueryAgent(
            _store_mock(),
            _graph_mock(),
        ).retrieve(query)
    )


def test_split_query_parallel_subqueries(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Decompose this query into independent sub-queries": {
                "sub_queries": ["one", "two"],
                "aggregation_strategy": "compare",
            }
        }
    )
    result = _retrieve("compare A vs B")
    assert result.sub_query_count == 2
    assert len(result.episodes) == 2
    assert result.aggregation_strategy is AggregationStrategy.COMPARE


def test_split_query_timeline_strategy_orders_by_created_at(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Decompose this query into independent sub-queries": {
                "sub_queries": ["later", "earlier"],
                "aggregation_strategy": "timeline",
            }
        }
    )
    result = _retrieve("timeline request")
    assert [episode.uid for episode in result.episodes] == ["earlier-ep", "later-ep"]
    assert result.aggregation_strategy is AggregationStrategy.TIMELINE
