"""Memory recall and knowledge integration (store + reflect) tools."""

from __future__ import annotations

import time
from typing import Final

import structlog

from shared.errors import StorageError
from shared.types import deterministic_id

from ..memory.knowledge_extract import extract_and_store_knowledge, retrieve_relevant_knowledge
from ..schema import ToolName
from . import ToolContext

log = structlog.get_logger(__name__)

RECALL_MEMORY_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.RECALL_MEMORY,
        "description": (
            "Search your memory — past conversations, research findings, and learned "
            "beliefs. This is your first source of knowledge: check here before "
            "going to the web. You have accumulated real research across many sessions. "
            "When someone asks what you know or what you've learned, this is where "
            "those answers live."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Topic, question, entity, or concept to retrieve (be specific for better results)",
                },
            },
            "required": ["query"],
        },
    },
}

INTEGRATE_KNOWLEDGE_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.INTEGRATE_KNOWLEDGE,
        "description": (
            "Persists facts to permanent storage and "
            "updates your belief system. Unlike accumulated session knowledge, "
            "these facts survive across conversations. Use when you "
            "have findings worth remembering permanently."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Facts with sources (e.g. 'Per BBC 2026-05-01: Germany plans to withdraw 5,000 troops by 2026.')",
                },
                "topic": {
                    "type": "string",
                    "description": "The belief topic to reflect on (use existing topic names when updating)",
                },
            },
            "required": ["text", "topic"],
        },
    },
}

DEFINITIONS: Final = [RECALL_MEMORY_DEFINITION, INTEGRATE_KNOWLEDGE_DEFINITION]


def execute_recall_memory(args: dict[str, object], ctx: ToolContext) -> str:
    """Retrieve relevant episodes and knowledge from memory stores."""
    query = str(args.get("query", ""))
    if not query:
        return "Error: no query provided for memory recall."
    t0 = time.perf_counter()
    ctx.progress(f"Searching memories: {query[:50]}")
    ep_error = kn_error = None
    try:
        episodes = ctx.retrieve(query)
    except Exception as exc:
        log.warning("episode_recall_failed", exc_info=True)
        episodes = []
        ep_error = exc
    if episodes:
        ctx.progress(f"Found {len(episodes)} episodes")
    try:
        knowledge = ctx.run_async(retrieve_relevant_knowledge(query, ctx.qdrant, ctx.embedder))
    except Exception as exc:
        log.warning("knowledge_retrieval_failed", exc_info=True)
        knowledge = []
        kn_error = exc
    if knowledge:
        ctx.progress(f"Found {len(knowledge)} knowledge items")

    elapsed = time.perf_counter() - t0
    if ep_error and kn_error:
        raise StorageError("Both memory stores unavailable") from ep_error
    if not episodes and not knowledge:
        log.info("recall_memory_empty", query=query[:60], elapsed_s=round(elapsed, 1))
        return f"No relevant memories found for: {query}. Try rephrasing or search the web."

    parts: list[str] = []
    if ep_error and not kn_error:
        parts.append("*Note: episode memory unavailable, showing knowledge only.*")
    elif kn_error and not ep_error:
        parts.append("*Note: knowledge store unavailable, showing episodes only.*")
    if ctx.short_term_memory:
        parts.append(f"## Current Focus\n{ctx.short_term_memory}")
    if episodes:
        parts.append(f"## Your Memory ({len(episodes)} relevant episodes)")
        parts.extend(f"- {ep}" for ep in episodes[:8])
    if knowledge:
        parts.append(f"\n## Stored Research ({len(knowledge)} items)")
        parts.extend(f"- {k}" for k in knowledge[:10])
    result = "\n".join(parts)
    log.info(
        "recall_memory",
        query=query[:60],
        episodes=len(episodes),
        knowledge=len(knowledge),
        elapsed_s=round(elapsed, 1),
    )
    return result


_INTEGRATE_TEXT_CAP = 2000


def execute_integrate_knowledge(args: dict[str, object], ctx: ToolContext) -> str:
    """Store knowledge propositions then reflect to update beliefs — atomic pipeline."""
    raw_text = str(args.get("text", ""))
    if len(raw_text) > _INTEGRATE_TEXT_CAP:
        log.warning("integrate_text_truncated", raw_chars=len(raw_text), cap=_INTEGRATE_TEXT_CAP)
    text = raw_text[:_INTEGRATE_TEXT_CAP]
    topic = str(args.get("topic", ""))
    if not text:
        return "Error: text is required"
    if not topic:
        return "Error: topic is required"
    t0 = time.perf_counter()
    episode_uid = deterministic_id(f"tool_knowledge:{topic}:{text[:200]}")
    log.info(
        "integrate_knowledge", topic=topic[:60], text_chars=len(text), episode_uid=episode_uid[:12]
    )
    ctx.progress(f"Storing knowledge: {topic[:40]}")

    stored, boosted = ctx.run_async(
        extract_and_store_knowledge(
            text=text, episode_uid=episode_uid, qdrant=ctx.qdrant, embedder=ctx.embedder
        )
    )
    log.info("integrate_knowledge_store", stored=stored, boosted=boosted)
    if stored or boosted:
        ctx.progress(f"Stored {stored} new, boosted {boosted} existing")

    from .reflect import execute_reflect_inner

    ctx.progress(f"Reflecting on: {topic[:40]}")
    reflection = execute_reflect_inner(
        topic=topic,
        evidence=text,
        ctx=ctx,
        web_context=ctx.research_transcript(),
        episode_uid=episode_uid,
    )
    elapsed = time.perf_counter() - t0

    parts: list[str] = []
    if stored:
        parts.append(f"Stored {stored} new knowledge propositions.")
    if boosted:
        parts.append(f"Boosted confidence for {boosted} existing propositions.")
    if reflection:
        parts.append(reflection)
    if not stored and not boosted and not reflection:
        parts.append("No knowledge extracted and reflection produced no changes.")
    log.info(
        "integrate_knowledge_done", stored=stored, boosted=boosted, elapsed_s=round(elapsed, 1)
    )
    return " ".join(parts)


EXECUTORS: Final = {
    ToolName.RECALL_MEMORY: execute_recall_memory,
    ToolName.INTEGRATE_KNOWLEDGE: execute_integrate_knowledge,
}

LABELS: Final = {
    ToolName.RECALL_MEMORY: lambda a: str(a.get("query", "")),
    ToolName.INTEGRATE_KNOWLEDGE: lambda a: str(a.get("topic", "")),
}
