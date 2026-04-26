from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast
from unittest.mock import AsyncMock

from sonality.memory.dual_store import DualEpisodeStore, SearchHit
from sonality.memory.graph import EpisodeNode, MemoryGraph
from sonality.memory.retrieval.chain import chain_retrieve


def _store_mock() -> DualEpisodeStore:
    store = AsyncMock(spec=DualEpisodeStore)
    store.vector_search = AsyncMock(return_value=[SearchHit("d1", "ep-1", 0.1)])
    return cast(DualEpisodeStore, store)


def _graph_mock() -> MemoryGraph:
    graph = AsyncMock(spec=MemoryGraph)
    graph.get_episodes = AsyncMock(
        return_value=[
            EpisodeNode(
                uid="ep-1",
                content="content",
                summary="summary",
                topics=[],
                ess_score=0.5,
                created_at="2026-01-01T00:00:00Z",
                valid_at="2026-01-01T00:00:00Z",
            )
        ]
    )
    return cast(MemoryGraph, graph)


def test_chain_stops_when_sufficient(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Given this query and retrieved context": {
                "sufficiency_decision": "SUFFICIENT",
                "confidence": 0.95,
                "reasoning": "Enough context",
                "suggested_refinement": None,
            }
        }
    )
    episodes = asyncio.run(
        chain_retrieve(_store_mock(), _graph_mock(), "query", base_n=3)
    )
    assert len(episodes) == 1
