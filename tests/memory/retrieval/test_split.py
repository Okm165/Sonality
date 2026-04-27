from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast
from unittest.mock import AsyncMock

from sonality.memory.dual_store import DualEpisodeStore, SearchHit
from sonality.memory.graph import EpisodeNode, MemoryGraph
from sonality.memory.retrieval.split import split_retrieve


def _store_mock() -> DualEpisodeStore:
    store = AsyncMock(spec=DualEpisodeStore)

    async def _vector_search(query: str, top_k: int = 10) -> list[SearchHit]:
        _ = top_k
        return [SearchHit("d1", f"{query}-ep", 0.1)]

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


def test_split_query_parallel_subqueries(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Decompose complex query into independent sub-queries": {
                "sub_queries": ["one", "two"],
                "aggregation_strategy": "compare",
            }
        }
    )
    episodes = asyncio.run(split_retrieve(_store_mock(), _graph_mock(), "compare A vs B"))
    assert len(episodes) == 2


def test_split_query_timeline_strategy_orders_by_created_at(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Decompose complex query into independent sub-queries": {
                "sub_queries": ["later", "earlier"],
                "aggregation_strategy": "timeline",
            }
        }
    )
    episodes = asyncio.run(split_retrieve(_store_mock(), _graph_mock(), "timeline request"))
    assert [ep.uid for ep in episodes] == ["earlier-ep", "later-ep"]
