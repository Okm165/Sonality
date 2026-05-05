"""LLM Listwise Reranker for episode relevance ranking.

Takes a query and candidate episodes, uses LLM to rank them by relevance
with cross-document reasoning.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, field_validator

from ... import config
from ...llm.caller import llm_call
from ...prompts import RERANK_PROMPT
from ..graph import EpisodeNode, format_episode_line

log = logging.getLogger(__name__)


class _RerankResponse(BaseModel):
    ranking: list[int]

    @field_validator("ranking", mode="before")
    @classmethod
    def flatten_ranking(cls, v: object) -> list[int]:
        """Flatten nested arrays and filter to integers — handles model quirks."""
        if isinstance(v, list):
            flat: list[int] = []
            for item in v:
                if isinstance(item, int):
                    flat.append(item)
                elif isinstance(item, list):
                    flat.extend(x for x in item if isinstance(x, int))
            return flat
        return []


def rerank_episodes(
    query: str,
    candidates: list[EpisodeNode],
) -> list[EpisodeNode]:
    """Rerank candidate episodes using LLM Listwise approach.

    Only the first MAX_RERANK_CANDIDATES are ranked — extras are dropped
    to avoid context overflow. Callers should slice the result to their
    desired count.
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates

    # Limit candidates to avoid context overflow
    max_candidates = config.MAX_RERANK_CANDIDATES
    to_rank = candidates[:max_candidates]

    # Format numbered candidates
    numbered = "\n\n".join(
        f"[{i + 1}] {format_episode_line(created_at=ep.created_at, summary=ep.summary, content=ep.content)}"
        for i, ep in enumerate(to_rank)
    )

    prompt = RERANK_PROMPT.format(query=query, numbered_candidates=numbered)
    result = llm_call(
        prompt=prompt,
        response_model=_RerankResponse,
        fallback=_RerankResponse(ranking=list(range(1, len(to_rank) + 1))),
    )

    if result.success:
        ranking = result.value.ranking
        log.info("Reranked %d candidates. Top=%s", len(to_rank), ranking[:5])

        # Map 1-indexed ranking to 0-indexed episodes
        reranked: list[EpisodeNode] = []
        seen: set[int] = set()
        for idx in ranking:
            zero_idx = idx - 1
            if 0 <= zero_idx < len(to_rank) and zero_idx not in seen:
                reranked.append(to_rank[zero_idx])
                seen.add(zero_idx)

        # Add any candidates not in the ranking (LLM might skip some)
        for i, ep in enumerate(to_rank):
            if i not in seen:
                reranked.append(ep)

        if reranked:
            top = reranked[0]
            log.debug(
                "Top episode after rerank: %s | summary=%.80s",
                top.uid[:8],
                top.summary or top.content[:80],
            )
        return reranked

    log.warning("Rerank fallback: %d candidates unchanged", len(to_rank))
    return to_rank
