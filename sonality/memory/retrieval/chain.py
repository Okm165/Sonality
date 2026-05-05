"""Iterative retrieval with LLM sufficiency checking.

Executes vector search, checks if results are sufficient via LLM, and
iteratively refines the query until satisfied or max iterations reached.
"""

from __future__ import annotations

import asyncio
import logging
from enum import StrEnum

from pydantic import BaseModel, field_validator

from ... import config
from ...llm.caller import llm_call
from ...prompts import SUFFICIENCY_PROMPT
from ..dual_store import DualEpisodeStore
from ..graph import EpisodeNode, MemoryGraph, format_episode_line

log = logging.getLogger(__name__)


class _SufficiencyDecision(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class _SufficiencyResponse(BaseModel):
    sufficiency_decision: _SufficiencyDecision = _SufficiencyDecision.INSUFFICIENT
    confidence: float = 0.0
    reasoning: str = ""
    suggested_refinement: str = ""

    @field_validator("suggested_refinement", mode="before")
    @classmethod
    def _coerce_none(cls, value: object) -> str:
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)


async def chain_retrieve(
    store: DualEpisodeStore,
    graph: MemoryGraph,
    query: str,
    base_n: int = 10,
) -> list[EpisodeNode]:
    """Iteratively search and refine until sufficient results found.

    Each iteration: vector search → LLM sufficiency check → optional query
    refinement. Stops when sufficient, no new results, or RETRIEVAL_MAX_ITERATIONS
    reached. Accumulates unique episodes across all iterations.
    """
    all_uids: set[str] = set()
    all_episodes: list[EpisodeNode] = []
    current_query = query
    max_iter = config.RETRIEVAL_MAX_ITERATIONS

    if max_iter < 1:
        return []

    for iteration in range(1, max_iter + 1):
        results = await store.vector_search(current_query, top_k=base_n)
        new_uids = [h.episode_uid for h in results if h.episode_uid not in all_uids]

        if new_uids:
            episodes = await graph.get_episodes(new_uids)
            for ep in episodes:
                if ep.uid not in all_uids:
                    all_uids.add(ep.uid)
                    all_episodes.append(ep)
        elif iteration > 1:
            break

        if not all_episodes:
            break

        context = "\n\n".join(
            format_episode_line(
                created_at=ep.created_at,
                summary=ep.summary,
                content=ep.content,
                content_limit=200,
            )
            for ep in all_episodes
        )
        _suf = await asyncio.to_thread(
            llm_call,
            prompt=SUFFICIENCY_PROMPT.format(query=query, context=context),
            response_model=_SufficiencyResponse,
            fallback=_SufficiencyResponse(),
        )
        sufficiency = _suf.value

        if sufficiency.sufficiency_decision is _SufficiencyDecision.SUFFICIENT:
            log.info(
                "Chain retrieval sufficient after %d iterations (conf=%.2f)",
                iteration,
                sufficiency.confidence,
            )
            return all_episodes

        if sufficiency.suggested_refinement:
            current_query = sufficiency.suggested_refinement
        else:
            break

    log.info("Chain retrieval exhausted: %d episodes", len(all_episodes))
    return all_episodes
