"""LLM call infrastructure + typed wrappers for each research stage.

Uses the same OpenAI-compatible HTTP transport as sonality (shared.llm),
with JSON repair and retry. Bridges sync shared calls to async via to_thread.
"""

from __future__ import annotations

import asyncio

import structlog
from pydantic import BaseModel

from shared.errors import LLMError
from shared.llm.caller import raw_call as _shared_raw_call

from .caller import llm_call as _sync_llm_call
from .config import settings
from .models import (
    Checklist,
    ChecklistItem,
    PageAnalysisResult,
    QueryGeneration,
    URLScoring,
)
from .prompts import (
    ANALYZE_PAGE,
    COMPRESS_KNOWLEDGE,
    DECOMPOSE_GOAL,
    GENERATE_QUERIES,
    SCORE_URLS,
    UPDATE_CHECKLIST,
    WRITE_INTRO_CONCLUSION,
    WRITE_SECTION,
)
from .provider import default_provider

log = structlog.get_logger()

# ---------------------------------------------------------------------------
# Generic async call helpers
# ---------------------------------------------------------------------------


async def call[T: BaseModel](prompt: str, *, format: type[T], max_tokens: int = 0) -> T:
    """Structured LLM call with JSON repair and retry (async via to_thread).

    Args:
        max_tokens: Per-call output budget. 0 = use settings.llm_max_tokens.
    """
    schema_name = format.__name__
    budget = max_tokens or settings.llm_max_tokens
    log.debug("llm_call_start", schema=schema_name, prompt_len=len(prompt), max_tokens=budget)
    result = await asyncio.to_thread(
        _sync_llm_call,
        prompt=prompt,
        response_model=format,
        fallback=format.model_construct(),
        max_tokens=budget,
    )
    if not result.success:
        log.warning(
            "llm_call_failed", schema=schema_name, error=result.error, attempts=result.attempts
        )
        raise LLMError(f"LLM call failed for {schema_name}: {result.error}")
    log.debug(
        "llm_call_ok",
        schema=schema_name,
        attempts=result.attempts,
        raw_len=len(result.raw_text) if result.raw_text else 0,
    )
    return result.value


async def call_text(prompt: str, *, max_tokens: int = 0) -> str:
    """Single LLM call returning raw text."""
    budget = max_tokens or settings.llm_max_tokens
    return await asyncio.to_thread(
        _shared_raw_call,
        default_provider,
        prompt=prompt,
        model=settings.model,
        max_tokens=budget,
    )


# ---------------------------------------------------------------------------
# Typed wrappers (thin — one per prompt)
# ---------------------------------------------------------------------------


def _bullets(items: list[str], limit: int = 10) -> str:
    return "\n".join(f"- {s}" for s in items[:limit]) or "None"


async def decompose_goal(goal: str) -> list[ChecklistItem]:
    result = await call(DECOMPOSE_GOAL.format(goal=goal), format=Checklist, max_tokens=2048)
    return result.items


async def generate_queries(
    goal: str,
    unanswered: list[str],
    productive: list[str],
    unproductive: list[str],
    facts_per_round: list[int],
    trigger_reason: str,
) -> list[str]:
    result = await call(
        GENERATE_QUERIES.format(
            goal=goal,
            unanswered_questions=_bullets(unanswered),
            productive_urls=_bullets(productive[-8:]),
            unproductive_urls=_bullets(unproductive[-8:]),
            facts_per_round=", ".join(str(x) for x in facts_per_round[-5:]) or "None",
            trigger_reason=trigger_reason,
        ),
        format=QueryGeneration,
        max_tokens=2048,
    )
    return result.queries


async def score_urls(
    urls_text: str,
    unanswered: list[str],
    knowledge_summary: str,
    productive: list[str],
    unproductive: list[str],
    facts_per_round: list[int],
) -> URLScoring:
    return await call(
        SCORE_URLS.format(
            unanswered_questions=_bullets(unanswered),
            knowledge_summary=knowledge_summary or "None yet",
            productive_urls=_bullets(productive[-8:]),
            unproductive_urls=_bullets(unproductive[-8:]),
            facts_per_round=", ".join(str(x) for x in facts_per_round[-5:]) or "None",
            urls=urls_text,
        ),
        format=URLScoring,
        max_tokens=2048,
    )


async def analyze_page(
    goal: str,
    unanswered: list[str],
    page_markdown: str,
    numbered_links: str,
    knowledge_context: str = "",
) -> PageAnalysisResult:
    content_budget = 6000
    links_budget = 1500
    return await call(
        ANALYZE_PAGE.format(
            goal=goal,
            unanswered_questions=_bullets(unanswered, limit=10),
            knowledge_context=knowledge_context[:1500] or "None yet",
            page_markdown=page_markdown[:content_budget],
            numbered_links=numbered_links[:links_budget],
        ),
        format=PageAnalysisResult,
        max_tokens=2048,
    )


async def update_checklist(
    checklist_state: str,
    new_facts: str,
    contradictions: str,
) -> list[ChecklistItem]:
    result = await call(
        UPDATE_CHECKLIST.format(
            checklist_state=checklist_state,
            new_facts=new_facts,
            contradictions=contradictions,
        ),
        format=Checklist,
        max_tokens=2048,
    )
    return result.items


async def compress_knowledge(existing_summary: str, new_facts: str) -> str:
    return await call_text(
        COMPRESS_KNOWLEDGE.format(existing_summary=existing_summary, new_facts=new_facts),
        max_tokens=2048,
    )


async def write_section(section_question: str, relevant_facts: str) -> str:
    return await call_text(
        WRITE_SECTION.format(section_question=section_question, relevant_facts=relevant_facts),
        max_tokens=2048,
    )


async def write_intro_conclusion(
    goal: str,
    section_summaries: str,
    unanswered_items: str,
    open_contradictions: str,
) -> str:
    return await call_text(
        WRITE_INTRO_CONCLUSION.format(
            goal=goal,
            section_summaries=section_summaries,
            unanswered_items=unanswered_items,
            open_contradictions=open_contradictions,
        ),
        max_tokens=4096,
    )
