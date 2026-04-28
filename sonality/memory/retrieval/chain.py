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
from ..graph import EpisodeNode, MemoryGraph

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
    """Iteratively search and refine until sufficient results found."""
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
            f"[{ep.created_at}] {ep.summary or ep.content[:200]}" for ep in all_episodes
        )
        _suf = await asyncio.to_thread(
            llm_call,
            prompt=SUFFICIENCY_PROMPT.format(query=query, context=context),
            response_model=_SufficiencyResponse,
            fallback=_SufficiencyResponse(),
            max_tokens=config.STRUCTURED_JSON_MAX_TOKENS,
            max_retries=1,
            assistant_prefix='{"sufficiency_decision": "',
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

    log.info("Chain retrieval done: %d iterations, %d episodes", iteration, len(all_episodes))
    return all_episodes
