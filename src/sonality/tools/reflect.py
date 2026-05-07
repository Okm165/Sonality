"""Belief reflection — deep reflection, belief graph updates, forgetting.

Called by integrate_knowledge (not exposed as a standalone tool). The pipeline:
1. Select relevant beliefs via LLM reranking
2. Run deep reflection via multi-model consensus to produce belief patches (REASONING + STRUCTURED models)
3. Apply patches to Neo4j (upsert beliefs, update personality snapshot)
4. Run forgetting cycle to prune low-value memories
"""

from __future__ import annotations

import re
import time

import structlog
from pydantic import BaseModel, Field, model_validator

from shared.errors import BeliefUpdateError

from .. import config
from ..caller import consensus_call, llm_call
from ..memory.forgetting import assess_and_forget
from ..memory.graph import BeliefNode, format_beliefs_for_prompt_from_nodes
from ..prompts import BELIEF_RELEVANCE_PROMPT, REFLECTION_DEEP_PROMPT, REFLECTION_WEB_SECTION
from ..request_identity import get_request_identity
from . import ToolContext

log = structlog.get_logger()


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

    @model_validator(mode="before")
    @classmethod
    def coerce_structure(cls, data: object) -> object:
        """Handle bare list from LLM — wraps [{"topic":...}] as {"belief_updates": [...]}."""
        if isinstance(data, list):
            return {"belief_updates": data}
        if isinstance(data, dict):
            data.pop("followup_queries", None)
            if isinstance(data.get("snapshot_changed"), str):
                data["snapshot_changed"] = str(data["snapshot_changed"]).lower() in ("true", "yes", "1")
        return data


def _merge_reflections(
    a: DeepReflectionResponse, b: DeepReflectionResponse
) -> DeepReflectionResponse:
    """Merge two reflections: union belief patches, prefer more conservative confidence."""
    seen_updates: dict[str, BeliefPatch] = {}
    for patch in [*a.belief_updates, *b.belief_updates]:
        if not patch.topic:
            continue
        existing = seen_updates.get(patch.topic)
        if existing is None:
            seen_updates[patch.topic] = patch
        else:
            seen_updates[patch.topic] = BeliefPatch(
                topic=patch.topic,
                valence=(existing.valence + patch.valence) / 2,
                confidence=min(existing.confidence, patch.confidence),
                belief_text=existing.belief_text or patch.belief_text,
                reasoning=f"{existing.reasoning} | {patch.reasoning}"[:400],
            )
    seen_new: dict[str, BeliefPatch] = {}
    for patch in [*a.new_beliefs, *b.new_beliefs]:
        if patch.topic and patch.topic not in seen_updates:
            seen_new.setdefault(patch.topic, patch)
    return DeepReflectionResponse(
        belief_updates=list(seen_updates.values()),
        new_beliefs=list(seen_new.values()),
        snapshot_revision=a.snapshot_revision or b.snapshot_revision,
        snapshot_changed=a.snapshot_changed or b.snapshot_changed,
    )


def _enrich_web(queries: list[str], existing: str, ctx: ToolContext) -> str:
    """Run web searches for reflection enrichment and merge into existing context."""
    if not queries or not ctx.web_client:
        return existing

    try:
        capped = queries[: config.settings.reflection_web_queries]
        responses = ctx.run_async(ctx.web_client.multi_search(capped, max_results=5))
        lines: list[str] = []
        seen: set[str] = set()
        for resp in responses:
            for r in resp.results:
                if r.url not in seen:
                    seen.add(r.url)
                    content = r.markdown or r.description
                    lines.append(f"[{r.title}] ({r.url})\n{content[:300]}")
        text = "\n\n".join(lines[:8])
        if text:
            log.info("web_enrichment", sources=len(seen), queries=len(capped))
            return f"{existing}\n\n{text}" if existing else text
    except Exception:
        log.warning("web_enrichment_failed", exc_info=True)
    return existing


def apply_reflection(
    reflection: DeepReflectionResponse, episode_uid: str, ctx: ToolContext
) -> None:
    """Write belief updates and snapshot revision to graph, then run forgetting."""
    log.debug(
        "apply_reflection",
        updates=len(reflection.belief_updates),
        new_beliefs=len(reflection.new_beliefs),
        snapshot_changed=reflection.snapshot_changed,
    )
    all_updates = [
        *((b, f"reflection:{episode_uid[:8]}") for b in reflection.belief_updates),
        *((b, f"new_belief:{episode_uid[:8]}") for b in reflection.new_beliefs),
    ]
    errors: list[str] = []
    for patch, provenance in all_updates:
        if not patch.topic:
            continue
        topic = re.sub(r"[^a-z0-9]+", "_", patch.topic.lower()).strip("_")
        try:
            ctx.run_async(
                ctx.graph.upsert_belief(
                    topic,
                    valence=patch.valence,
                    confidence=patch.confidence,
                    belief_text=patch.belief_text,
                    provenance=provenance,
                )
            )
            log.info(
                "belief_upserted",
                action=provenance.split(":")[0],
                topic=topic,
                valence=f"{patch.valence:+.2f}",
                confidence=f"{patch.confidence:.2f}",
                reason=patch.reasoning[:100],
            )
        except Exception:
            log.error("belief_upsert_failed", topic=topic, exc_info=True)
            errors.append(topic)

    if reflection.snapshot_changed and reflection.snapshot_revision:
        text = reflection.snapshot_revision[:2000]
        ctx.run_async(ctx.graph.upsert_personality_snapshot(text))
        log.info("snapshot_updated", chars=len(text))

    if errors:
        raise BeliefUpdateError(f"Belief upsert failed for: {errors}")

    run_forgetting(ctx)


def run_forgetting(ctx: ToolContext) -> None:
    """Evaluate and forget low-value memories."""
    try:
        candidates = ctx.run_async(
            ctx.graph.get_forgetting_candidates(limit=config.settings.forgetting_candidate_limit)
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
        log.warning("forgetting_cycle_failed", exc_info=True)


class _BeliefRelevance(BaseModel):
    """Single belief's relevance assessment."""

    index: int = 0
    relevance: float = 0.0


class _BeliefRanking(BaseModel):
    """LLM-judged relevance ranking of beliefs to evidence."""

    ranked: list[_BeliefRelevance] = Field(default_factory=list)


def _select_relevant_beliefs(
    topic: str, evidence: str, all_beliefs: list[BeliefNode]
) -> list[BeliefNode]:
    """LLM-based belief reranker — returns beliefs the LLM deems relevant.

    The LLM scores all beliefs for relevance. We trust its ranking:
    any belief the LLM includes (relevance > 0) is used, sorted by score.
    """
    if not all_beliefs:
        return []

    capped_beliefs = all_beliefs[:50]
    numbered = "\n".join(
        f"{i + 1}. [{b.topic}] {b.belief_text[:80]}" for i, b in enumerate(capped_beliefs)
    )
    prompt = BELIEF_RELEVANCE_PROMPT.format(
        topic=topic,
        evidence=evidence[:500],
        numbered_beliefs=numbered,
    )
    result = llm_call(
        prompt=prompt,
        response_model=_BeliefRanking,
        fallback=_BeliefRanking(),
        model=config.settings.structured_model,
    )
    if result.success and result.value.ranked:
        ranked = sorted(result.value.ranked, key=lambda r: r.relevance, reverse=True)
        selected = [all_beliefs[r.index - 1] for r in ranked if 0 < r.index <= len(capped_beliefs)][
            : config.settings.belief_prompt_window
        ]
        if selected:
            return selected

    return all_beliefs[: config.settings.belief_prompt_window]


def execute_reflect_inner(
    *,
    topic: str,
    evidence: str,
    ctx: ToolContext,
    web_context: str = "",
    episode_uid: str = "",
) -> str:
    """Core reflection logic — called by integrate_knowledge, not directly as a tool.

    web_context: pre-fetched web context (e.g. from prior web_search in the agentic loop).
    When provided, skips redundant web enrichment searches.
    episode_uid: provenance anchor for belief updates.

    Returns a human-readable summary of belief changes, or empty string on failure.
    """
    evidence = evidence[:800]
    if not topic or not evidence:
        return ""
    log.info("reflect", topic=topic[:60], evidence_chars=len(evidence))
    t0 = time.perf_counter()

    identity = ctx.identity or get_request_identity()
    if identity is None:
        log.warning("reflect: identity not loaded")
        return ""

    belief_nodes = _select_relevant_beliefs(topic, evidence, list(identity.all_beliefs))
    beliefs_text = format_beliefs_for_prompt_from_nodes(belief_nodes)
    log.info("reflect_beliefs_selected", selected=len(belief_nodes), total=len(identity.all_beliefs))

    if web_context:
        reflection_web = web_context
        log.info("reflect_web_context_reused", chars=len(web_context))
    else:
        topic_phrase = topic.replace("_", " ")
        reflection_web = _enrich_web([topic_phrase], "", ctx)
    web_section = (
        REFLECTION_WEB_SECTION.format(web_content=reflection_web) if reflection_web else ""
    )

    deep_prompt = REFLECTION_DEEP_PROMPT.format(
        snapshot=identity.snapshot_text[:2000],
        beliefs=beliefs_text[:3000],
        user_message=evidence,
        web_context_section=web_section[:2000],
    )
    deep_result = consensus_call(
        prompt=deep_prompt,
        response_model=DeepReflectionResponse,
        fallback=DeepReflectionResponse(),
        models=(config.settings.reasoning_model, config.settings.structured_model),
        merge=_merge_reflections,
    )
    if not deep_result.success:
        log.warning("reflect_failed", error=deep_result.error)
        return "Reflection attempted but failed."

    reflection = deep_result.value
    if not episode_uid:
        from shared.types import new_id
        episode_uid = new_id()
    apply_reflection(reflection, episode_uid=episode_uid, ctx=ctx)
    elapsed = time.perf_counter() - t0
    log.info(
        "reflect_done",
        updates=len(reflection.belief_updates),
        new_beliefs=len(reflection.new_beliefs),
        elapsed=f"{elapsed:.1f}s",
    )
    parts: list[str] = []
    if reflection.belief_updates:
        parts.append(f"Updated {len(reflection.belief_updates)} beliefs.")
    if reflection.new_beliefs:
        parts.append(f"Formed {len(reflection.new_beliefs)} new beliefs.")
    if reflection.snapshot_changed:
        parts.append("Personality snapshot updated.")
    if not reflection.belief_updates and not reflection.new_beliefs:
        parts.append("No belief changes needed.")
    return " ".join(parts)
