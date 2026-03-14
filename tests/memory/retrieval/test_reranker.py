from __future__ import annotations

from collections.abc import Callable

from sonality.memory.graph import EpisodeNode
from sonality.memory.retrieval.reranker import rerank_episodes


def _episode(uid: str, summary: str) -> EpisodeNode:
    return EpisodeNode(
        uid=uid,
        content=summary,
        summary=summary,
        topics=[],
        ess_score=0.5,
        created_at="2026-01-01T00:00:00Z",
        valid_at="2026-01-01T00:00:00Z",
    )


def test_reranker_applies_listwise_order(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Given this query and candidate episodes": {
                "ranking": [2, 1, 3],
                "reasoning": "Second episode is most specific",
            }
        }
    )
    candidates = [
        _episode("a", "first"),
        _episode("b", "second"),
        _episode("c", "third"),
    ]
    ranked = rerank_episodes("test query", candidates)
    assert [episode.uid for episode in ranked] == ["b", "a", "c"]
