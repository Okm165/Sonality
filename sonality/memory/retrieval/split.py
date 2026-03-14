"""SplitQueryAgent: LLM-based query decomposition with parallel sub-query execution.

Decomposes multi-entity or comparison queries into independent sub-queries,
executes them in parallel, and aggregates results.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from ...llm.caller import llm_call
from ...llm.prompts import DECOMPOSITION_PROMPT
from ..dual_store import DualEpisodeStore
from ..graph import EpisodeNode, MemoryGraph

log = logging.getLogger(__name__)


class AggregationStrategy(StrEnum):
    MERGE = "merge"
    COMPARE = "compare"
    TIMELINE = "timeline"


class DecompositionResponse(BaseModel):
    sub_queries: list[str]
    aggregation_strategy: AggregationStrategy = AggregationStrategy.MERGE


@dataclass(frozen=True, slots=True)
class SplitResult:
    episodes: list[EpisodeNode]
    sub_query_count: int
    aggregation_strategy: AggregationStrategy


class SplitQueryAgent:
    """Decomposes multi-entity queries into parallel sub-queries."""

    def __init__(self, store: DualEpisodeStore, graph: MemoryGraph) -> None:
        self._store = store
        self._graph = graph

    async def retrieve(self, query: str, n_per_sub: int = 10) -> SplitResult:
        """Decompose query, execute sub-queries in parallel, aggregate."""
        decomposition = self._decompose(query)
        sub_queries = decomposition.sub_queries
        strategy = decomposition.aggregation_strategy

        if len(sub_queries) <= 1:
            # Fall back to simple search
            results = await self._store.vector_search(query, top_k=n_per_sub)
            episode_uids = list({r[1] for r in results})
            episodes = await self._graph.get_episodes(episode_uids)
            return SplitResult(
                episodes=episodes,
                sub_query_count=1,
                aggregation_strategy=AggregationStrategy.MERGE,
            )

        # Execute sub-queries in parallel (bounded concurrency)
        sem = asyncio.Semaphore(4)

        async def search_one(sq: str) -> list[EpisodeNode]:
            async with sem:
                try:
                    results = await self._store.vector_search(sq, top_k=n_per_sub)
                    uids = list({r[1] for r in results})
                    return await self._graph.get_episodes(uids)
                except Exception:
                    log.exception("Sub-query failed: %s", sq[:60])
                    return []

        tasks = [search_one(sq) for sq in sub_queries]
        sub_results = await asyncio.gather(*tasks)

        all_episodes = self._aggregate(sub_results, strategy)

        return SplitResult(
            episodes=all_episodes,
            sub_query_count=len(sub_queries),
            aggregation_strategy=strategy,
        )

    def _decompose(self, query: str) -> DecompositionResponse:
        """Use LLM to decompose query into sub-queries."""
        prompt = DECOMPOSITION_PROMPT.format(query=query)
        result = llm_call(
            prompt=prompt,
            response_model=DecompositionResponse,
            fallback=DecompositionResponse(sub_queries=[query]),
        )
        if not result.success:
            return DecompositionResponse(
                sub_queries=[query],
                aggregation_strategy=AggregationStrategy.MERGE,
            )
        strategy = result.value.aggregation_strategy
        sub_queries = [part.strip() for part in result.value.sub_queries if part.strip()][:4]
        if not sub_queries:
            return DecompositionResponse(
                sub_queries=[query],
                aggregation_strategy=AggregationStrategy.MERGE,
            )
        log.info("Query decomposed into %d sub-queries (%s)", len(sub_queries), strategy)
        return DecompositionResponse(sub_queries=sub_queries, aggregation_strategy=strategy)

    def _aggregate(
        self, sub_results: list[list[EpisodeNode]], strategy: AggregationStrategy
    ) -> list[EpisodeNode]:
        """Aggregate per-sub-query episodes using the requested strategy."""
        if strategy is AggregationStrategy.COMPARE:
            # Interleave batches round-robin to preserve cross-entity balance
            interleaved: list[EpisodeNode] = []
            max_len = max((len(batch) for batch in sub_results), default=0)
            for index in range(max_len):
                for batch in sub_results:
                    if index < len(batch):
                        interleaved.append(batch[index])
            return self._dedupe_by_uid(interleaved)
        if strategy is AggregationStrategy.TIMELINE:
            merged = self._dedupe_by_uid([ep for batch in sub_results for ep in batch])
            return sorted(merged, key=lambda episode: episode.created_at)
        return self._dedupe_by_uid([ep for batch in sub_results for ep in batch])

    def _dedupe_by_uid(self, episodes: list[EpisodeNode]) -> list[EpisodeNode]:
        """Keep first occurrence per UID while preserving order."""
        seen: dict[str, EpisodeNode] = {}
        for ep in episodes:
            seen.setdefault(ep.uid, ep)
        return list(seen.values())
