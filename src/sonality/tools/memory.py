"""Memory recall and knowledge integration (store + reflect) tools."""

from __future__ import annotations

import time
from typing import Final

import structlog

from shared.errors import StorageError

from ..memory.knowledge_extract import extract_and_store_knowledge, retrieve_relevant_knowledge
from ..schema import ToolName
from . import ToolContext

log = structlog.get_logger()

RECALL_MEMORY_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.RECALL_MEMORY,
        "description": (
            "Retrieve what you already know from past conversations, stored knowledge, "
            "and learned beliefs. Broad queries surface topic overviews; specific queries "
            "target exact recall. With prior context on a topic, recall often reduces the need "
            "for web search."
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
            "Stores verified knowledge and updates beliefs in one step. "
            "Persists facts to long-term memory, then reflects on implications "
            "for your worldview. Most effective after synthesis, when confirmed facts "
            "are ready for long-term retention and belief integration."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Verified facts with sources (e.g. 'Per BBC 2026-05-01: Germany plans to withdraw 5,000 troops by 2026.')",
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
    ep_error = kn_error = None
    try:
        episodes = ctx.retrieve(query)
    except Exception as exc:
        log.warning("episode_recall_failed", exc_info=True)
        episodes = []
        ep_error = exc
    try:
        knowledge = ctx.run_async(retrieve_relevant_knowledge(query, ctx.qdrant, ctx.embedder))
    except Exception as exc:
        log.warning("knowledge_retrieval_failed", exc_info=True)
        knowledge = []
        kn_error = exc

    elapsed = time.perf_counter() - t0
    if ep_error and kn_error:
        raise StorageError("Both memory stores unavailable") from ep_error
    if not episodes and not knowledge:
        log.info("recall_memory_empty", query=query[:60], elapsed=f"{elapsed:.1f}s")
        return f"No memories found for: {query}. This is either a new topic or the query needs different wording."

    parts: list[str] = []
    if episodes:
        parts.append("## Past Conversations")
        parts.extend(f"- {ep}" for ep in episodes[:8])
    if knowledge:
        parts.append("\n## Stored Knowledge")
        parts.extend(f"- {k}" for k in knowledge[:10])
    result = "\n".join(parts)
    log.info(
        "recall_memory",
        query=query[:60],
        episodes=len(episodes),
        knowledge=len(knowledge),
        elapsed=f"{elapsed:.1f}s",
    )
    return result


def execute_integrate_knowledge(args: dict[str, object], ctx: ToolContext) -> str:
    """Store knowledge propositions then reflect to update beliefs — atomic pipeline."""
    from shared.types import new_id

    text = str(args.get("text", ""))[:2000]
    topic = str(args.get("topic", ""))
    if not text:
        return "Error: text is required"
    if not topic:
        return "Error: topic is required"
    t0 = time.perf_counter()
    episode_uid = new_id()
    log.info("integrate_knowledge", topic=topic[:60], text_chars=len(text), episode_uid=episode_uid[:8])

    stored, boosted = ctx.run_async(
        extract_and_store_knowledge(
            text=text, episode_uid=episode_uid, qdrant=ctx.qdrant, embedder=ctx.embedder
        )
    )
    log.info("integrate_knowledge_store", stored=stored, boosted=boosted)

    from .reflect import execute_reflect_inner

    transcript = ctx.build_research_transcript(tool_tail=4, assistant_tail=0)
    reflection = execute_reflect_inner(
        topic=topic, evidence=text, ctx=ctx, web_context=transcript, episode_uid=episode_uid,
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
    log.info("integrate_knowledge_done", episode_uid=episode_uid[:8], stored=stored, boosted=boosted, elapsed=f"{elapsed:.1f}s")
    return " ".join(parts)


EXECUTORS: Final = {
    ToolName.RECALL_MEMORY: execute_recall_memory,
    ToolName.INTEGRATE_KNOWLEDGE: execute_integrate_knowledge,
}
