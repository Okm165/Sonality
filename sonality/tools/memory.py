"""Memory recall and knowledge storage tools."""

from __future__ import annotations

import logging
from typing import Final

from ..memory.knowledge_extract import extract_and_store_knowledge, retrieve_relevant_knowledge
from ..schema import ToolName
from . import ToolContext

log = logging.getLogger(__name__)

RECALL_MEMORY_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.RECALL_MEMORY,
        "description": "Access your accumulated knowledge and past reasoning.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to recall — topic, question, or concept to search for",
                },
            },
            "required": ["query"],
        },
    },
}

STORE_KNOWLEDGE_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.STORE_KNOWLEDGE,
        "description": "Commit verified facts to permanent memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Text containing verified knowledge to extract and store",
                },
            },
            "required": ["text"],
        },
    },
}

DEFINITIONS: Final = [RECALL_MEMORY_DEFINITION, STORE_KNOWLEDGE_DEFINITION]


def execute_recall_memory(args: dict[str, object], ctx: ToolContext) -> str:
    """Retrieve relevant episodes and knowledge from memory stores."""
    query = str(args.get("query", ""))
    if not query:
        return "Error: no query provided for memory recall."
    try:
        episodes = ctx.retrieve(query)
    except Exception:
        log.warning("Memory recall failed", exc_info=True)
        episodes = []
    try:
        knowledge = ctx.run_async(retrieve_relevant_knowledge(query, ctx.qdrant, ctx.embedder))
    except Exception:
        log.warning("Knowledge retrieval failed", exc_info=True)
        knowledge = []

    if not episodes and not knowledge:
        return f"No relevant memories found for: {query}"

    parts: list[str] = []
    if episodes:
        parts.append("## Relevant Past Conversations")
        parts.extend(f"- {ep}" for ep in episodes[:8])
    if knowledge:
        parts.append("\n## Learned Knowledge")
        parts.extend(f"- {k}" for k in knowledge[:10])
    result = "\n".join(parts)
    log.info(
        "recall_memory: q=%.60s → %d episodes, %d knowledge", query, len(episodes), len(knowledge)
    )
    return result


def execute_store_knowledge(args: dict[str, object], ctx: ToolContext) -> str:
    """Extract and persist knowledge propositions."""
    text = str(args.get("text", ""))[:2000]
    if not text:
        return "Error: text is required"
    log.info("store_knowledge: %d chars", len(text))
    try:
        stored = ctx.run_async(
            extract_and_store_knowledge(
                text=text, episode_uid="", qdrant=ctx.qdrant, embedder=ctx.embedder
            )
        )
        if stored:
            log.info("store_knowledge: stored %d propositions", stored)
            return f"Stored {stored} knowledge propositions."
        return "No knowledge propositions extracted from the text."
    except Exception:
        log.warning("store_knowledge failed", exc_info=True)
        return "Knowledge storage failed."


EXECUTORS: Final = {
    ToolName.RECALL_MEMORY: execute_recall_memory,
    ToolName.STORE_KNOWLEDGE: execute_store_knowledge,
}
