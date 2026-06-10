"""LLM Listwise Reranker for episode relevance ranking.

Takes a query and candidate episodes, uses LLM to rank them by relevance
with cross-document reasoning.
"""

from __future__ import annotations

import structlog
from pydantic import BaseModel, Field, field_validator

from ... import config
from ...caller import async_llm_call, format_prompt
from ...prompts import RERANK_PROMPT
from ..graph import EpisodeNode, format_episode_line

log = structlog.get_logger(__name__)


class _RerankResponse(BaseModel):
    ranking: list[int] = Field(max_length=50)
    reasoning: str = Field(default="", max_length=3000)

    @field_validator("ranking", mode="before")
    @classmethod
    def flatten_ranking(cls, v: object) -> list[int]:
        """Flatten nested arrays, coerce string digits to int — handles model quirks."""

        def _coerce(x: object) -> int | None:
            if isinstance(x, int):
                return x
            if isinstance(x, (str, float)):
                try:
                    return int(x)
                except (ValueError, OverflowError):
                    return None
            return None

        if isinstance(v, list):
            flat: list[int] = []
            for item in v:
                if isinstance(item, list):
                    flat.extend(c for x in item if (c := _coerce(x)) is not None)
                elif (c := _coerce(item)) is not None:
                    flat.append(c)
            return flat[:50]
        return []


async def rerank_episodes(
    query: str,
    candidates: list[EpisodeNode],
) -> list[EpisodeNode]:
    """Rerank candidate episodes using LLM Listwise approach.

    Only the first MAX_RERANK_CANDIDATES are ranked to fit the LLM context
    window; overflow candidates are appended after the ranked portion.
    """
    if not candidates:
        return []
    if len(candidates) == 1:
        return candidates

    max_candidates = config.settings.max_rerank_candidates
    to_rank = candidates[:max_candidates]
    overflow = candidates[max_candidates:]

    numbered = "\n\n".join(
        f"[{i + 1}] {format_episode_line(created_at=ep.created_at, summary=ep.summary, content=ep.content, ess_score=ep.ess_score, source_quality=ep.source_quality, grounding=ep.grounding)}"
        for i, ep in enumerate(to_rank)
    )

    result = await async_llm_call(
        instructions=format_prompt(RERANK_PROMPT, query=query, numbered_candidates=numbered),
        response_model=_RerankResponse,
        fallback=_RerankResponse(ranking=list(range(1, len(to_rank) + 1))),
        model=config.settings.structured_model,
    )

    if result.success:
        ranking = result.value.ranking
        log.info(
            "rerank_complete",
            candidate_count=len(to_rank),
            overflow_appended=len(overflow),
            top_ranking=ranking[:5],
            reasoning=result.value.reasoning[:120] if result.value.reasoning else "",
        )

        reranked: list[EpisodeNode] = []
        seen: set[int] = set()
        for idx in ranking:
            zero_idx = idx - 1
            if 0 <= zero_idx < len(to_rank) and zero_idx not in seen:
                reranked.append(to_rank[zero_idx])
                seen.add(zero_idx)

        for i, ep in enumerate(to_rank):
            if i not in seen:
                reranked.append(ep)

        if reranked:
            top = reranked[0]
            log.debug(
                "rerank_top_episode",
                episode_uid=top.uid[:8],
                summary_or_content_preview=(top.summary or top.content)[:80],
            )
        return reranked + overflow

    log.warning(
        "rerank_fallback_unchanged",
        candidate_count=len(to_rank),
        overflow_appended=len(overflow),
    )
    return to_rank + overflow
