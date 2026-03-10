"""SplitQueryAgent: LLM-based query decomposition with parallel sub-query execution.

Decomposes multi-entity or comparison queries into independent sub-queries,
executes them in parallel, and aggregates results.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from pydantic import BaseModel

from ...llm.caller import llm_call
from ...llm.prompts import DECOMPOSITION_PROMPT
from ..dual_store import DualEpisodeStore
from ..graph import EpisodeNode, MemoryGraph

log = logging.getLogger(__name__)


class DecompositionResponse(BaseModel):
    sub_queries: list[str]
    aggregation_strategy: str = "merge"


@dataclass(frozen=True, slots=True)
class SplitResult:
    episodes: list[EpisodeNode]
    sub_query_count: int
    aggregation_strategy: str


class SplitQueryAgent:
    """Decomposes multi-entity queries into parallel sub-queries."""

    def __init__(self, store: DualEpisodeStore, graph: MemoryGraph) -> None:
        self._store = store
        self._graph = graph

    async def retrieve(self, query: str, n_per_sub: int = 10) -> SplitResult:
        """Decompose query, execute sub-queries in parallel, aggregate."""
        # LLM decomposition
        sub_queries = self._decompose(query)

        if len(sub_queries) <= 1:
            # Fall back to simple search
            results = await self._store.vector_search(query, top_k=n_per_sub)
            episode_uids = list({r[1] for r in results})
            episodes = await self._graph.get_episodes(episode_uids)
            return SplitResult(episodes=episodes, sub_query_count=1, aggregation_strategy="merge")

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

        # Aggregate: deduplicate by UID
        seen_uids: set[str] = set()
        all_episodes: list[EpisodeNode] = []
        for episodes in sub_results:
            for ep in episodes:
                if ep.uid not in seen_uids:
                    seen_uids.add(ep.uid)
                    all_episodes.append(ep)

        return SplitResult(
            episodes=all_episodes,
            sub_query_count=len(sub_queries),
            aggregation_strategy="merge",
        )

    def _decompose(self, query: str) -> list[str]:
        """Use LLM to decompose query into sub-queries."""
        prompt = DECOMPOSITION_PROMPT.format(query=query)
        result = llm_call(
            prompt=prompt,
            response_model=DecompositionResponse,
            fallback=DecompositionResponse(sub_queries=[query]),
        )
        if result.success and result.value:
            assert isinstance(result.value, DecompositionResponse)
            subs = result.value.sub_queries[:4]  # Cap at 4
            if subs:
                log.info("Query decomposed into %d sub-queries", len(subs))
                return subs
        return [query]
