"""Retrieval pipeline: routing, search orchestration, and reranking.

Public entry point is ``retrieve()`` — a single async function that hides
query routing, multi-pass vector/graph search, temporal expansion,
reranking, and semantic feature search.
"""

from __future__ import annotations

from .pipeline import retrieve
from .reranker import rerank_episodes
from .router import (
    QueryCategory,
    RoutingDecision,
    SemanticMemoryDecision,
    TemporalExpansionDecision,
    route_query,
)

__all__ = [
    "QueryCategory",
    "RoutingDecision",
    "SemanticMemoryDecision",
    "TemporalExpansionDecision",
    "rerank_episodes",
    "retrieve",
    "route_query",
]
