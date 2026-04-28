"""Belief reflection — deep reflection, belief graph updates, forgetting.

The reflect tool lets the agent evaluate significant evidence mid-conversation
and update its belief graph. apply_reflection writes changes to Neo4j.
run_forgetting performs periodic memory cleanup.
"""

from __future__ import annotations

import logging
import time
from typing import Final

from pydantic import BaseModel, Field

from .. import config
from ..llm.caller import llm_call
from ..memory.forgetting import assess_and_forget
from ..memory.graph import BELIEF_PROMPT_WINDOW, BeliefNode
from ..prompts import REFLECTION_DEEP_PROMPT, REFLECTION_WEB_SECTION
from ..request_identity import get_request_identity
from ..schema import ToolName
from . import ToolContext

log = logging.getLogger(__name__)


# --- Pydantic models for reflection LLM contracts ---


class BeliefPatch(BaseModel):
    """Single belief create/update from LLM reflection."""

    topic: str = ""
    valence: float = 0.0
    confidence: float = 0.5
    belief_text: str = ""
    reasoning: str = ""


class DeepReflectionResponse(BaseModel):
    belief_updates: list[BeliefPatch] = Field(default_factory=list)
    new_beliefs: list[BeliefPatch] = Field(default_factory=list)
    snapshot_revision: str = ""
    snapshot_changed: bool = False
    followup_queries: list[str] = Field(default_factory=list)


# --- Tool definition ---

REFLECT_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.REFLECT,
        "description": (
            "ESSENTIAL: Update your beliefs when evidence warrants. Use when you've learned "
            "something new, been corrected, discovered contradictions, or formed new opinions. "
            "This is how you grow - conversations without reflection are missed opportunities. "
            "Err toward reflecting when uncertain."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The belief topic to reflect on",
                },
                "evidence": {
                    "type": "string",
                    "description": "Key evidence or reasoning that justifies the belief update (2-5 sentences)",
                },
            },
            "required": ["topic", "evidence"],
        },
    },
}

DEFINITIONS: Final = [REFLECT_DEFINITION]


# --- Shared helpers ---


def _format_beliefs(belief_nodes: list[BeliefNode]) -> str:
    if not belief_nodes:
        return "(no beliefs yet)"
    return "\n".join(
        f"{b.topic} — valence: {b.valence:+.2f}, confidence: {b.confidence:.2f}"
        + (f" | {b.belief_text}" if b.belief_text else "")
        for b in belief_nodes
    )


def _enrich_web(queries: list[str], existing: str, ctx: ToolContext) -> str:
    """Run web searches for reflection enrichment and merge into existing context."""
    if not queries or not ctx.web_client:
        return existing
    from ..web.context import format_web_context
    from ..web.search import SearchResult

    try:
        capped = queries[: config.REFLECTION_WEB_QUERIES]
        responses = ctx.run_async(ctx.web_client.multi_search(capped, max_results=5))
        seen: set[str] = set()
        all_results: list[SearchResult] = []
        for resp in responses:
            for r in resp.results:
                if r.url not in seen:
                    seen.add(r.url)
                    all_results.append(r)
        text = format_web_context(all_results[:8], max_chars=3000)
        if text:
            log.info("Web enrichment: %d sources from %d queries", len(all_results), len(capped))
            return f"{existing}\n\n{text}" if existing else text
    except Exception:
        log.warning("Web enrichment failed", exc_info=True)
    return existing


def apply_reflection(
    reflection: DeepReflectionResponse, episode_uid: str, ctx: ToolContext
) -> None:
    """Write belief updates and snapshot revision to graph, then run forgetting."""
    log.debug(
        "Applying reflection: %d updates, %d new beliefs, snapshot_changed=%s",
        len(reflection.belief_updates),
        len(reflection.new_beliefs),
        reflection.snapshot_changed,
    )
    all_updates = [
        *((b, f"reflection:{episode_uid[:8]}") for b in reflection.belief_updates),
        *((b, f"new_belief:{episode_uid[:8]}") for b in reflection.new_beliefs),
    ]
    for patch, provenance in all_updates:
        if not patch.topic:
            continue
        try:
            ctx.run_async(
                ctx.graph.upsert_belief(
                    patch.topic,
                    valence=patch.valence,
                    confidence=patch.confidence,
                    belief_text=patch.belief_text,
                    provenance=provenance,
                )
            )
            log.info(
                "Belief %s: %s val=%+.2f conf=%.2f reason=%.100s",
                provenance.split(":")[0],
                patch.topic,
                patch.valence,
                patch.confidence,
                patch.reasoning,
            )
        except Exception:
            log.exception("Failed to upsert belief: %s", patch.topic)

    if reflection.snapshot_changed and reflection.snapshot_revision:
        text = reflection.snapshot_revision[:2000]
        try:
            ctx.run_async(ctx.graph.upsert_personality_snapshot(text))
            log.info("Personality snapshot updated (%d chars)", len(text))
        except Exception:
            log.exception("Failed to update personality snapshot")

    run_forgetting(ctx)


def run_forgetting(ctx: ToolContext) -> None:
    """Evaluate and forget low-value memories."""
    try:
        candidates = ctx.run_async(
            ctx.graph.get_forgetting_candidates(limit=config.FORGETTING_CANDIDATE_LIMIT)
        )
        if candidates:
            snapshot = ctx.run_async(ctx.graph.get_personality_snapshot())
            ctx.run_async(
                assess_and_forget(
                    candidates,
                    ctx.graph,
                    ctx.dual_store,
                    snapshot_excerpt=snapshot.text[:500],
                )
            )
    except Exception:
        log.warning("Forgetting cycle skipped", exc_info=True)


# --- Tool executor ---


def execute_reflect(args: dict[str, object], ctx: ToolContext) -> str:
    """Reflect tool: direct deep reflection — no triage gate.

    When the agent decides to reflect, it has already determined the evidence
    is worth integrating. Triage is redundant and adds latency.
    """
    topic = str(args.get("topic", ""))
    evidence = str(args.get("evidence", ""))[:800]
    if not topic or not evidence:
        return "Error: topic and evidence are required"
    log.info("reflect tool: topic=%.60s evidence=%d chars", topic, len(evidence))
    t0 = time.perf_counter()

    identity = ctx.identity or get_request_identity()
    if identity is None:
        return "Cannot reflect: identity not loaded"

    belief_nodes = list(identity.all_beliefs[:BELIEF_PROMPT_WINDOW])
    beliefs_text = _format_beliefs(belief_nodes)

    web_queries = [f"{topic} evidence research"]
    reflection_web = _enrich_web(web_queries, "", ctx)
    episodes = ctx.run_async(ctx.graph.list_recent_episode_context(10))
    web_section = (
        REFLECTION_WEB_SECTION.format(web_content=reflection_web) if reflection_web else ""
    )

    deep_prompt = REFLECTION_DEEP_PROMPT.format(
        snapshot=identity.snapshot_text,
        beliefs=beliefs_text,
        episode_count=len(episodes),
        episodes="\n".join(episodes) or "(no recent episodes)",
        user_message=evidence,
        agent_response="(reflecting mid-conversation — no response yet)",
        web_context_section=web_section,
    )
    deep_result = llm_call(
        prompt=deep_prompt,
        response_model=DeepReflectionResponse,
        fallback=DeepReflectionResponse(),
        max_tokens=config.EXTRACTION_MAX_TOKENS,
        max_retries=1,
    )
    if not deep_result.success:
        log.warning("reflect tool: deep reflection failed: %s", deep_result.error)
        return "Reflection attempted but failed."

    reflection = deep_result.value
    apply_reflection(reflection, episode_uid="", ctx=ctx)
    elapsed = time.perf_counter() - t0
    log.info(
        "reflect tool: updates=%d new=%d elapsed=%.1fs",
        len(reflection.belief_updates),
        len(reflection.new_beliefs),
        elapsed,
    )
    parts = [f"Reflected on '{topic}'."]
    if reflection.belief_updates:
        parts.append(f"Updated {len(reflection.belief_updates)} beliefs.")
    if reflection.new_beliefs:
        parts.append(f"Formed {len(reflection.new_beliefs)} new beliefs.")
    if reflection.snapshot_changed:
        parts.append("Personality snapshot updated.")
    if not reflection.belief_updates and not reflection.new_beliefs:
        parts.append("No belief changes needed.")
    return " ".join(parts)


EXECUTORS: Final = {ToolName.REFLECT: execute_reflect}
