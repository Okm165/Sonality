"""Hybrid URL ranking for Fathom research sessions.

Uses Reciprocal Rank Fusion (RRF) to combine:
1. Embedding similarity (dense vectors) — semantic match to goal/questions
2. Domain productivity (from persistent source memory) — learned from history

Selection uses softmax-temperature sampling from the score distribution,
balancing exploitation (high-scoring URLs) with exploration (discovering
new valuable sources).

All embedding calls go through ``fathom.caller`` async gated wrappers.
"""

from __future__ import annotations

import asyncio
import math
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from shared.embedder import Embedder, cosine_similarity
from shared.ranking import rrf_score, scores_to_ranks

from .caller import async_embed_documents, async_embed_query
from .models import extract_domain

if TYPE_CHECKING:
    from .models import Link, SessionMemory

log = structlog.get_logger(__name__)


def _softmax(scores: list[float], temperature: float = 1.0) -> list[float]:
    """Convert scores to probability distribution via softmax with temperature.

    Higher temperature = more uniform (exploration)
    Lower temperature = sharper (exploitation)
    """
    if not scores:
        return []
    # Scale scores and apply temperature
    scaled = [s / temperature for s in scores]
    max_s = max(scaled)
    exp_scores = [math.exp(s - max_s) for s in scaled]  # Subtract max for numerical stability
    total = sum(exp_scores)
    return [e / total for e in exp_scores]


def _sample_from_distribution[T](
    items: list[T],
    probabilities: list[float],
    n: int,
) -> list[T]:
    """Sample n items from distribution without replacement."""
    if n >= len(items):
        return items

    indices = list(range(len(items)))
    probs = list(probabilities)
    selected: list[T] = []

    for _ in range(n):
        if not indices or sum(probs) <= 0:
            break
        # Normalize remaining probabilities
        total = sum(probs)
        normalized = [p / total for p in probs]

        # Sample
        r = random.random()
        cumsum = 0.0
        chosen_idx = 0
        for i, p in enumerate(normalized):
            cumsum += p
            if r <= cumsum:
                chosen_idx = i
                break

        selected.append(items[indices[chosen_idx]])
        # Remove selected item
        del indices[chosen_idx]
        del probs[chosen_idx]

    return selected


# --- URL Ranking ---


@dataclass(slots=True)
class ScoredURL:
    """URL with RRF-fused relevance score."""

    link: Link
    index: int
    rrf_score: float = 0.0


async def rank_urls_hybrid(
    embedder: Embedder,
    urls: list[Link],
    goal: str,
    questions: list[str],
    memory: SessionMemory | None = None,
    *,
    top_k: int = 50,
    temperature: float = 2.0,
) -> list[tuple[int, Link, float]]:
    """Rank URLs using embedding similarity + domain reputation via RRF.

    Selection uses softmax-temperature sampling: higher temperature explores
    more broadly, lower temperature exploits top-scoring URLs.
    """
    if not urls:
        return []

    n = len(urls)

    query_text = goal + " " + " ".join(questions)

    url_texts = [" ".join([link.url, link.anchor_text, link.context]) for link in urls]

    # --- Signal 1: Embedding similarity (query + documents in parallel) ---
    query_emb, url_embs = await asyncio.gather(
        async_embed_query(embedder, query_text[:500]),
        async_embed_documents(embedder, [t[:500] for t in url_texts]),
    )

    embedding_scores = [max(0.0, cosine_similarity(query_emb, emb)) for emb in url_embs]

    # --- Signal 2: Domain quality from session memory ---
    domain_scores: list[float] = []
    if memory:
        for link in urls:
            domain = extract_domain(link.url)
            stats = memory.domain_stats.get(domain)
            domain_scores.append(stats.quality_rate if stats else 0.5)
    else:
        domain_scores = [0.5] * n

    # --- RRF fusion ---
    emb_ranks = scores_to_ranks(embedding_scores)
    dom_ranks = scores_to_ranks(domain_scores)

    scored: list[ScoredURL] = []
    for i, link in enumerate(urls):
        rrf = rrf_score([emb_ranks[i], dom_ranks[i]])
        scored.append(ScoredURL(link=link, index=i, rrf_score=rrf))

    # --- Softmax-temperature sampling ---
    if len(scored) <= top_k:
        scored.sort(key=lambda s: s.rrf_score, reverse=True)
        return [(s.index, s.link, s.rrf_score) for s in scored]

    scores_for_softmax = [s.rrf_score for s in scored]
    probabilities = _softmax(scores_for_softmax, temperature)
    items = [(s.index, s.link, s.rrf_score) for s in scored]
    selected = _sample_from_distribution(items, probabilities, top_k)

    log.debug(
        "url_sampling",
        pool=n,
        selected=len(selected),
        temperature=temperature,
        top_prob=round(max(probabilities), 4) if probabilities else 0,
    )
    return selected


# --- Fact Ranking ---


async def _rank_facts_for_context(
    embedder: Embedder,
    facts: list[tuple[str, str, float, float]],  # (claim, source_url, confidence, source_quality)
    questions: list[str],
    goal: str = "",
    *,
    top_k: int = 20,
) -> list[str]:
    """Rank facts by relevance using embeddings + confidence + source_quality via RRF."""
    if not facts:
        return []
    claims = [claim for claim, _, _, _ in facts]
    if len(facts) <= top_k:
        return claims

    query_text = goal + " " + " ".join(questions)

    query_emb, fact_embs = await asyncio.gather(
        async_embed_query(embedder, query_text[:500]),
        async_embed_documents(embedder, [c[:500] for c in claims]),
    )

    embedding_scores = [max(0.0, cosine_similarity(query_emb, emb)) for emb in fact_embs]
    confidence_scores = [conf for _, _, conf, _ in facts]
    quality_scores = [sq for _, _, _, sq in facts]

    emb_ranks = scores_to_ranks(embedding_scores)
    conf_ranks = scores_to_ranks(confidence_scores)
    qual_ranks = scores_to_ranks(quality_scores)

    scored = [
        (i, rrf_score([emb_ranks[i], conf_ranks[i], qual_ranks[i]])) for i in range(len(facts))
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    return [claims[i] for i, _ in scored[:top_k]]


async def build_ranked_knowledge_context(
    embedder: Embedder,
    facts: list[tuple[str, str, float, float]],
    questions: list[str],
    goal: str = "",
    *,
    max_facts: int = 20,
    char_limit: int = 2000,
) -> str:
    """Build knowledge context with relevance-ranked facts."""
    ranked = await _rank_facts_for_context(embedder, facts, questions, goal, top_k=max_facts)
    if not ranked:
        return "None yet"

    lines: list[str] = []
    total_chars = 0
    for claim in ranked:
        line = f"- {claim}"
        if total_chars + len(line) > char_limit:
            break
        lines.append(line)
        total_chars += len(line) + 1

    return "\n".join(lines) or "None yet"
