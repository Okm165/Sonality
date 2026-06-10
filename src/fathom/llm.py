"""LLM wrappers for each research stage — structured JSON only.

Thin typed wrappers over ``fathom.caller.async_llm_call``.  Infrastructure
(provider, concurrency gate, format_prompt) lives in ``fathom.caller`` —
symmetric with ``sonality.caller``.
"""

from __future__ import annotations

import asyncio

import structlog
from pydantic import BaseModel

from shared.errors import LLMError

from .caller import async_llm_call, format_prompt
from .models import (
    Checklist,
    ChecklistItem,
    PageAnalysisResult,
    QueryGeneration,
)
from .prompts import (
    ANALYZE_PAGE_PROMPT,
    DECOMPOSE_GOAL_PROMPT,
    GENERATE_QUERIES_PROMPT,
)

log = structlog.get_logger(__name__)


async def _call[T: BaseModel](instructions: str, *, response_model: type[T]) -> T:
    """Structured LLM call — delegates to ``fathom.caller.async_llm_call``."""
    schema_name = response_model.__name__
    log.debug("llm_call", schema=schema_name, instructions_chars=len(instructions))
    result = await async_llm_call(
        instructions=instructions,
        response_model=response_model,
        fallback=response_model(),
    )
    if not result.success:
        log.warning(
            "llm_call_failed", schema=schema_name, error=result.error, attempts=result.attempts
        )
        raise LLMError(f"LLM call failed for {schema_name}: {result.error}")
    log.debug(
        "llm_call_done",
        schema=schema_name,
        elapsed_s=result.elapsed_s,
        in_tok=result.input_tokens,
        out_tok=result.output_tokens,
        attempts=result.attempts,
    )
    return result.value


# ---------------------------------------------------------------------------
# Typed wrappers (one per research stage)
# ---------------------------------------------------------------------------


def _bullets(items: list[str], limit: int = 10) -> str:
    return "\n".join(f"- {s}" for s in items[:limit]) or "None"


async def _format_off_loop(template: str, **kwargs: str) -> str:
    """Run format_prompt in a thread — it may trigger synchronous LLM
    compression for oversized values, which would block the event loop."""
    return await asyncio.to_thread(format_prompt, template, **kwargs)


async def decompose_goal(goal: str, *, max_questions: int = 8) -> list[ChecklistItem]:
    instructions = await _format_off_loop(DECOMPOSE_GOAL_PROMPT, goal=goal)
    result = await _call(instructions, response_model=Checklist)
    return result.items[:max_questions]


async def generate_queries(
    goal: str,
    questions: list[str],
    productive: list[str],
    unproductive: list[str],
    facts_per_round: list[int],
    trigger_reason: str,
) -> list[str]:
    instructions = await _format_off_loop(
        GENERATE_QUERIES_PROMPT,
        goal=goal,
        unanswered_questions=_bullets(questions),
        productive_urls=_bullets(productive[-8:]),
        unproductive_urls=_bullets(unproductive[-8:]),
        facts_per_round=", ".join(str(x) for x in facts_per_round[-5:]) or "None",
        trigger_reason=trigger_reason,
    )
    result = await _call(instructions, response_model=QueryGeneration)
    return result.queries


async def analyze_page(
    goal: str,
    questions: list[str],
    page_markdown: str,
    numbered_links: str,
    knowledge_context: str = "",
) -> PageAnalysisResult:
    """Analyze page content.  No pre-truncation — format_prompt's overlapping
    window compression handles oversized content, preserving tail details
    that hard truncation would permanently lose."""
    instructions = await _format_off_loop(
        ANALYZE_PAGE_PROMPT,
        goal=goal,
        unanswered_questions=_bullets(questions, limit=10),
        knowledge_context=knowledge_context or "None yet",
        page_markdown=page_markdown,
        numbered_links=numbered_links,
    )
    return await _call(instructions, response_model=PageAnalysisResult)
