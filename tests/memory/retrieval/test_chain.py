from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast

from sonality.memory.dual_store import DualEpisodeStore
from sonality.memory.graph import EpisodeNode, MemoryGraph
from sonality.memory.retrieval.chain import ChainOfQueryAgent


class _FakeStore:
    async def hybrid_search(
        self, query: str, *, top_k: int = 10, rrf_k: int = 60
    ) -> list[tuple[str, str, float]]:
        _ = (query, top_k, rrf_k)
        return [("d1", "ep-1", 0.1)]


class _FakeGraph:
    async def get_episodes(self, uids: list[str]) -> list[EpisodeNode]:
        _ = uids
        return [
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
    result = asyncio.run(
        ChainOfQueryAgent(
            cast(DualEpisodeStore, _FakeStore()),
            cast(MemoryGraph, _FakeGraph()),
        ).retrieve("query", base_n=3)
    )
    assert not result.exhausted
    assert result.iterations_used == 1
    assert result.confidence >= 0.9
    assert len(result.episodes) == 1
