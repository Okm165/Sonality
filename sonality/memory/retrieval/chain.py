"""ChainOfQueryAgent: iterative retrieval with LLM sufficiency checking.

Executes vector search, checks if results are sufficient via LLM, and
iteratively refines the query until satisfied or max iterations reached.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, field_validator

from ... import config
from ...llm.caller import llm_call
from ...llm.prompts import SUFFICIENCY_PROMPT
from ..dual_store import DualEpisodeStore
from ..graph import EpisodeNode, MemoryGraph

log = logging.getLogger(__name__)


class SufficiencyDecision(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"


class SufficiencyResponse(BaseModel):
    sufficiency_decision: SufficiencyDecision = SufficiencyDecision.INSUFFICIENT
    confidence: float = 0.0
    reasoning: str = ""
    suggested_refinement: str = ""

    @field_validator("suggested_refinement", mode="before")
    @classmethod
    def _coerce_suggested_refinement(cls, value: object) -> str:
        if value is None:
            return ""
        return value if isinstance(value, str) else str(value)


@dataclass(frozen=True, slots=True)
class ChainResult:
    episodes: list[EpisodeNode]
    confidence: float
    iterations_used: int
    exhausted: bool


class ChainOfQueryAgent:
    """Iterative retrieval with LLM-based sufficiency checking."""

    def __init__(self, store: DualEpisodeStore, graph: MemoryGraph) -> None:
        self._store = store
        self._graph = graph
        self._max_iterations = config.RETRIEVAL_MAX_ITERATIONS
        self._confidence_threshold = config.RETRIEVAL_CONFIDENCE_THRESHOLD

    async def retrieve(self, query: str, base_n: int = 10) -> ChainResult:
        """Iteratively search and refine until sufficient results found."""
        all_episode_uids: set[str] = set()
        all_episodes: list[EpisodeNode] = []
        current_query = query
        best_confidence = 0.0

        for iteration in range(1, self._max_iterations + 1):
            # Hybrid search: RRF(vector, full-text) for better recall on exact terms
            results = await self._store.hybrid_search(current_query, top_k=base_n)
            new_episode_uids = [r[1] for r in results if r[1] not in all_episode_uids]

            if new_episode_uids:
                episodes = await self._graph.get_episodes(new_episode_uids)
                for ep in episodes:
                    if ep.uid not in all_episode_uids:
                        all_episode_uids.add(ep.uid)
                        all_episodes.append(ep)

            # LLM sufficiency check
            context = "\n\n".join(
                f"[{ep.created_at}] {ep.summary or ep.content[:200]}" for ep in all_episodes
            )
            sufficiency = self._check_sufficiency(query, context)

            if sufficiency.confidence > best_confidence:
                best_confidence = sufficiency.confidence

            if (
                sufficiency.sufficiency_decision is SufficiencyDecision.SUFFICIENT
                and sufficiency.confidence >= self._confidence_threshold
            ):
                log.info(
                    "Chain retrieval sufficient after %d iterations (conf=%.2f)",
                    iteration,
                    sufficiency.confidence,
                )
                return ChainResult(
                    episodes=all_episodes,
                    confidence=sufficiency.confidence,
                    iterations_used=iteration,
                    exhausted=False,
                )

            if sufficiency.suggested_refinement:
                current_query = sufficiency.suggested_refinement
                log.debug("Refining query to: %s", current_query[:80])
            else:
                break

        log.info(
            "Chain retrieval exhausted %d iterations (best conf=%.2f)",
            self._max_iterations,
            best_confidence,
        )
        return ChainResult(
            episodes=all_episodes,
            confidence=best_confidence,
            iterations_used=self._max_iterations,
            exhausted=True,
        )

    def _check_sufficiency(self, query: str, context: str) -> SufficiencyResponse:
        prompt = SUFFICIENCY_PROMPT.format(query=query, context=context or "No results found")
        result = llm_call(
            prompt=prompt,
            response_model=SufficiencyResponse,
            fallback=SufficiencyResponse(),
        )
        if result.success:
            return result.value
        return SufficiencyResponse()
