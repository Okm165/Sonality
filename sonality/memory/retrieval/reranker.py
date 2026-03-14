"""LLM Listwise Reranker for episode relevance ranking.

Takes a query and candidate episodes, uses LLM to rank them by relevance
with cross-document reasoning. Replaces formula-based utility scoring.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from ... import config
from ...llm.caller import llm_call
from ...llm.prompts import RERANK_PROMPT
from ..graph import EpisodeNode

log = logging.getLogger(__name__)


class RerankResponse(BaseModel):
    ranking: list[int]
    reasoning: str = ""


def rerank_episodes(
    query: str,
    candidates: list[EpisodeNode],
    *,
    top_k: int = 0,
) -> list[EpisodeNode]:
    """Rerank candidate episodes using LLM Listwise approach.

    Parameters
    ----------
    query:
        The original search query.
    candidates:
        Episodes to rank (max ~25 for context efficiency).
    top_k:
        Number of top results to return. 0 means return all.

    Returns
    -------
    Episodes in LLM-determined relevance order.
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
        f"[{i + 1}] ({ep.created_at[:10] if ep.created_at else 'unknown'}) "
        f"{ep.summary or ep.content[:300]}"
        for i, ep in enumerate(to_rank)
    )

    prompt = RERANK_PROMPT.format(query=query, numbered_candidates=numbered)
    result = llm_call(
        prompt=prompt,
        response_model=RerankResponse,
        fallback=RerankResponse(ranking=list(range(1, len(to_rank) + 1))),
    )

    if result.success:
        ranking = result.value.ranking
        log.info(
            "Reranked %d→%d candidates. Top=%s | %s",
            len(to_rank),
            top_k or len(to_rank),
            ranking[:5],
            result.value.reasoning[:80] if result.value.reasoning else "no reasoning",
        )

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

        final = reranked[:top_k] if top_k > 0 else reranked
        if final:
            top = final[0]
            log.debug(
                "Top episode after rerank: %s | summary=%.80s",
                top.uid[:8],
                top.summary or top.content[:80],
            )
        return final

    log.warning("Rerank fallback: %d candidates unchanged", len(to_rank))
    return to_rank[:top_k] if top_k > 0 else to_rank
