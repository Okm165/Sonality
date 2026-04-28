"""LLM-based query decomposition with parallel sub-query execution.

Decomposes multi-entity or comparison queries into independent sub-queries,
executes them in parallel, and aggregates results.
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum

from pydantic import BaseModel, model_validator

from ... import config
from ...llm.caller import llm_call
from ...prompts import DECOMPOSITION_PROMPT
from ..dual_store import DualEpisodeStore
from ..graph import EpisodeNode, MemoryGraph

log = logging.getLogger(__name__)


class _AggregationStrategy(StrEnum):
    MERGE = "merge"
    COMPARE = "compare"
    TIMELINE = "timeline"


class _DecompositionResponse(BaseModel):
    sub_queries: list[str]
    aggregation_strategy: _AggregationStrategy = _AggregationStrategy.MERGE

    @model_validator(mode="before")
    @classmethod
    def normalize_response(cls, data: object) -> object:
        if isinstance(data, list):
            data = {"sub_queries": data}
        if isinstance(data, dict) and "sub_queries" in data:
            raw = data["sub_queries"]
            data["sub_queries"] = [
                str(x)
                if isinstance(x, str)
                else " ".join(v for v in x.values() if isinstance(v, str))
                for x in raw
                if isinstance(x, (str, dict))
            ]
        return data


def _decompose(query: str) -> _DecompositionResponse:
    result = llm_call(
        prompt=DECOMPOSITION_PROMPT.format(query=query),
        response_model=_DecompositionResponse,
        fallback=_DecompositionResponse(sub_queries=[query]),
        max_tokens=config.STRUCTURED_JSON_MAX_TOKENS,
        max_retries=1,
        assistant_prefix='{"sub_queries": [',
    )
    if not result.success:
        log.warning("Query decomposition failed: %s; using raw query", result.error)
        return _DecompositionResponse(sub_queries=[query])
    sub_queries = [part.strip() for part in result.value.sub_queries if part.strip()][:4]
    if not sub_queries:
        return _DecompositionResponse(sub_queries=[query])
    log.info(
        "Query decomposed into %d sub-queries (%s)",
        len(sub_queries),
        result.value.aggregation_strategy,
    )
    return _DecompositionResponse(
        sub_queries=sub_queries, aggregation_strategy=result.value.aggregation_strategy
    )


def _dedupe(episodes: list[EpisodeNode]) -> list[EpisodeNode]:
    seen: dict[str, EpisodeNode] = {}
    for ep in episodes:
        seen.setdefault(ep.uid, ep)
    return list(seen.values())


def _aggregate(
    sub_results: list[list[EpisodeNode]], strategy: _AggregationStrategy
) -> list[EpisodeNode]:
    if strategy is _AggregationStrategy.COMPARE:
        interleaved: list[EpisodeNode] = []
        max_len = max((len(batch) for batch in sub_results), default=0)
        for index in range(max_len):
            for batch in sub_results:
                if index < len(batch):
                    interleaved.append(batch[index])
        return _dedupe(interleaved)
    if strategy is _AggregationStrategy.TIMELINE:
        return sorted(
            _dedupe([ep for batch in sub_results for ep in batch]),
            key=lambda episode: episode.created_at,
        )
    return _dedupe([ep for batch in sub_results for ep in batch])


async def split_retrieve(
    store: DualEpisodeStore,
    graph: MemoryGraph,
    query: str,
    n_per_sub: int = 10,
) -> list[EpisodeNode]:
    """Decompose query, execute sub-queries in parallel, aggregate."""
    decomposition = await asyncio.to_thread(_decompose, query)
    sub_queries = decomposition.sub_queries
    strategy = decomposition.aggregation_strategy

    if len(sub_queries) <= 1:
        results = await store.vector_search(query, top_k=n_per_sub)
        episode_uids = list({h.episode_uid for h in results})
        return await graph.get_episodes(episode_uids)

    sem = asyncio.Semaphore(4)

    async def search_one(sq: str) -> list[EpisodeNode]:
        async with sem:
            try:
                results = await store.vector_search(sq, top_k=n_per_sub)
                uids = list({h.episode_uid for h in results})
                return await graph.get_episodes(uids)
            except Exception:
                log.exception("Sub-query failed: %s", sq[:60])
                return []

    sub_results = await asyncio.gather(*(search_one(sq) for sq in sub_queries))
    return _aggregate(sub_results, strategy)
