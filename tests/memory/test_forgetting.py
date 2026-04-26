from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast
from unittest.mock import AsyncMock

from sonality.memory.dual_store import DualEpisodeStore
from sonality.memory.forgetting import assess_and_forget
from sonality.memory.graph import EpisodeNode, MemoryGraph


def _candidate(uid: str) -> EpisodeNode:
    return EpisodeNode(
        uid=uid,
        content=f"content {uid}",
        summary=f"summary {uid}",
        topics=["topic"],
        ess_score=0.5,
        created_at="2026-01-01T00:00:00Z",
        valid_at="2026-01-01T00:00:00Z",
    )


def _assess(
    graph: MemoryGraph,
    store: DualEpisodeStore,
    candidates: list[EpisodeNode],
) -> None:
    asyncio.run(assess_and_forget(
        candidates,
        graph=cast(MemoryGraph, graph),
        store=cast(DualEpisodeStore, store),
        snapshot_excerpt="snapshot",
    ))


def _graph_mock() -> MemoryGraph:
    graph = AsyncMock(spec=MemoryGraph)
    graph.archive_episode = AsyncMock()
    graph.delete_episode = AsyncMock()
    return cast(MemoryGraph, graph)


def _store_mock() -> DualEpisodeStore:
    store = AsyncMock(spec=DualEpisodeStore)
    store.archive_derivatives = AsyncMock()
    store.delete_derivatives = AsyncMock()
    return cast(DualEpisodeStore, store)


def test_forgetting_uses_full_uid_and_hard_delete_path(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Review these memory candidates for potential archival": {
                "decisions": [
                    {
                        "uid": "episode-aaa",
                        "action": "FORGET",
                        "reason": "Superseded by newer evidence",
                    },
                    {
                        "uid": "unknown-short-id",
                        "action": "ARCHIVE",
                        "reason": "Should be ignored",
                    },
                ]
            }
        }
    )

    graph = _graph_mock()
    store = _store_mock()
    _assess(graph, store, [_candidate("episode-aaa"), _candidate("episode-bbb")])

    cast(AsyncMock, graph.delete_episode).assert_awaited_once_with("episode-aaa")
    cast(AsyncMock, store.delete_derivatives).assert_awaited_once_with("episode-aaa")
    cast(AsyncMock, graph.archive_episode).assert_not_awaited()
    cast(AsyncMock, store.archive_derivatives).assert_not_awaited()


def test_forgetting_does_not_use_foundational_substring_heuristic(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Review these memory candidates for potential archival": {
                "decisions": [
                    {
                        "uid": "episode-aaa",
                        "action": "ARCHIVE",
                        "reason": "Looks foundational but should still archive",
                    }
                ]
            }
        }
    )
    graph = _graph_mock()
    store = _store_mock()
    _assess(graph, store, [_candidate("episode-aaa")])
    cast(AsyncMock, graph.archive_episode).assert_awaited_once_with("episode-aaa")
    cast(AsyncMock, store.archive_derivatives).assert_awaited_once_with("episode-aaa")
