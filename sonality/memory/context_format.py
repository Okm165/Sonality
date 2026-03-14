"""Shared helpers for rendering episode context strings."""

from __future__ import annotations


def format_episode_line(
    *,
    created_at: str,
    summary: str,
    content: str,
    content_limit: int = 300,
) -> str:
    """Render one compact context line for retrieval/reflection."""
    date_text = created_at[:10] if created_at else "?"
    summary_or_excerpt = summary or content[:content_limit]
    return f"[{date_text}] {summary_or_excerpt}"


def format_episode_block(
    *,
    created_at: str,
    content: str,
    content_limit: int = 500,
) -> str:
    """Render one dated episode content block for summarization prompts."""
    date_text = created_at[:10] if created_at else "?"
    return f"[{date_text}]\n{content[:content_limit]}"
