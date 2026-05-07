"""Probabilistic URL scoring + power-law sampling.

LLM controls all value judgments via scores + concentration.
The sampling is purely mechanical: power-law transform of LLM scores,
then weighted random draw without replacement.
"""

from __future__ import annotations

import random

import structlog

from .config import settings
from .models import Link, URLScoring

log = structlog.get_logger()


def probabilistic_sample(
    urls: list[Link],
    scoring: URLScoring,
    n: int,
) -> list[int]:
    """Select N URL indices via power-law weighted sampling.

    LLM-produced concentration controls the distribution shape:
      low concentration → exploratory (more uniform)
      high concentration → exploitative (top-heavy)
    """
    if not urls:
        return []

    scores = scoring.scores[: len(urls)]
    while len(scores) < len(urls):
        scores.append(0.01)

    floored = [max(s, 0.01) for s in scores]
    weights = [s**scoring.concentration for s in floored]
    total = sum(weights)
    if total == 0:
        return list(range(min(n, len(urls))))
    probs = [w / total for w in weights]

    n = min(n, len(urls))
    selected: list[int] = []
    remaining_indices = list(range(len(urls)))
    remaining_probs = list(probs)

    for _ in range(n):
        if not remaining_indices:
            break
        total = sum(remaining_probs)
        if total == 0:
            break
        normalized = [p / total for p in remaining_probs]
        idx_pos = random.choices(range(len(remaining_indices)), weights=normalized, k=1)[0]
        selected.append(remaining_indices[idx_pos])
        remaining_indices.pop(idx_pos)
        remaining_probs.pop(idx_pos)

    log.info(
        "sample_selected",
        n=len(selected),
        concentration=scoring.concentration,
        top_prob=max(probs) if probs else 0,
    )
    return selected


def format_urls_for_scoring(urls: list[Link]) -> str:
    """Format URL list for the scoring prompt."""
    capped = urls[:settings.max_urls_for_scoring]
    lines: list[str] = []
    for i, link in enumerate(capped):
        parts = [f"[{i}] {link.url}"]
        if link.anchor_text:
            parts.append(f"  anchor: {link.anchor_text[:100]}")
        if link.context:
            parts.append(f"  context: {link.context[:120]}")
        lines.append("\n".join(parts))
    return "\n".join(lines)
