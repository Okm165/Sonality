from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import cast

from sonality.memory.dual_store import DualEpisodeStore
from sonality.memory.forgetting import ForgettingEngine, ForgettingResult
from sonality.memory.graph import EpisodeNode, MemoryGraph


class _FakeGraph:
    def __init__(self) -> None:
        self.archived: list[str] = []
        self.deleted: list[str] = []

    async def archive_episode(self, episode_uid: str) -> None:
        self.archived.append(episode_uid)

    async def delete_episode(self, episode_uid: str) -> None:
        self.deleted.append(episode_uid)


class _FakeStore:
    def __init__(self) -> None:
        self.archived: list[str] = []
        self.deleted: list[str] = []

    async def archive_derivatives(self, episode_uid: str) -> None:
        self.archived.append(episode_uid)

    async def delete_derivatives(self, episode_uid: str) -> None:
        self.deleted.append(episode_uid)


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
    graph: _FakeGraph,
    store: _FakeStore,
    candidates: list[EpisodeNode],
) -> ForgettingResult:
    return asyncio.run(
        ForgettingEngine(
            graph=cast(MemoryGraph, graph),
            store=cast(DualEpisodeStore, store),
        ).assess_and_forget(
            candidates,
            snapshot_excerpt="snapshot",
        )
    )


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

    graph = _FakeGraph()
    store = _FakeStore()
    result = _assess(graph, store, [_candidate("episode-aaa"), _candidate("episode-bbb")])

    assert graph.deleted == ["episode-aaa"]
    assert store.deleted == ["episode-aaa"]
    assert graph.archived == []
    assert store.archived == []
    assert result.archived == 1
    assert result.kept == 1


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
    graph = _FakeGraph()
    store = _FakeStore()
    result = _assess(graph, store, [_candidate("episode-aaa")])
    assert graph.archived == ["episode-aaa"]
    assert store.archived == ["episode-aaa"]
    assert result.archived == 1
    assert result.kept == 0
