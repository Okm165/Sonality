"""Reciprocal Rank Fusion (RRF) primitives — Cormack et al., 2009.

Shared by both ``sonality.tools.reflect`` (belief ranking) and
``fathom.ranking`` (URL / fact ranking).  One canonical location.
"""

from __future__ import annotations

from typing import Final

RRF_K: Final = 60


def rrf_score(ranks: list[int], k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion: sum of 1/(k + rank) across all ranking lists."""
    return sum(1.0 / (k + r) for r in ranks)


def scores_to_ranks(scores: list[float]) -> list[int]:
    """Convert scores to ranks (1-indexed, lower is better)."""
    indexed = sorted(enumerate(scores), key=lambda x: -x[1])
    ranks = [0] * len(scores)
    for rank, (idx, _) in enumerate(indexed, start=1):
        ranks[idx] = rank
    return ranks
