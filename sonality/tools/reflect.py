"""Belief reflection — deep reflection, belief graph updates, forgetting.

Called by integrate_knowledge (not exposed as a standalone tool). The pipeline:
1. Select relevant beliefs via LLM reranking
2. Run deep reflection via multi-model consensus to produce belief patches (REASONING + STRUCTURED models)
3. Apply patches to Neo4j (upsert beliefs, update personality snapshot)
4. Run forgetting cycle to prune low-value memories
"""

from __future__ import annotations

import logging
import re
import time

from pydantic import BaseModel, Field, model_validator

from .. import config
from ..llm.caller import consensus_call, llm_call
from ..memory.forgetting import assess_and_forget
from ..memory.graph import BELIEF_PROMPT_WINDOW, BeliefNode, format_beliefs_for_prompt_from_nodes
from ..prompts import REFLECTION_DEEP_PROMPT, REFLECTION_WEB_SECTION
from ..request_identity import get_request_identity
from . import ToolContext

log = logging.getLogger(__name__)


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

    @model_validator(mode="before")
    @classmethod
    def coerce_structure(cls, data: object) -> object:
        """Handle bare list from LLM — wraps [{"topic":...}] as {"belief_updates": [...]}."""
        if isinstance(data, list):
            return {"belief_updates": data}
        if isinstance(data, dict) and isinstance(data.get("snapshot_changed"), str):
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
        followup_queries=list(dict.fromkeys([*a.followup_queries, *b.followup_queries]))[:5],
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
        if abs(patch.valence) < 0.05 and patch.confidence < 0.35:
            log.debug(
                "Skipping nil-opinion patch topic=%s val=%+.2f conf=%.2f",
                patch.topic,
                patch.valence,
                patch.confidence,
            )
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
                "Belief %s: %s val=%+.2f conf=%.2f reason=%.100s",
                provenance.split(":")[0],
                topic,
                patch.valence,
                patch.confidence,
                patch.reasoning,
            )
        except Exception:
            log.exception("Failed to upsert belief: %s", topic)

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


class _BeliefRelevance(BaseModel):
    """Single belief's relevance assessment."""

    index: int = 0
    relevance: float = 0.0


class _BeliefRanking(BaseModel):
    """LLM-judged relevance ranking of beliefs to evidence."""

    ranked: list[_BeliefRelevance] = Field(default_factory=list)


_RELEVANCE_CUTOFF: float = 0.3


def _select_relevant_beliefs(
    topic: str, evidence: str, all_beliefs: list[BeliefNode]
) -> list[BeliefNode]:
    """LLM-based belief reranker with relevance cutoff.

    Presents all beliefs with their topics and texts to the structured model.
    Each gets a relevance score (0.0–1.0). Only beliefs above the cutoff
    are passed to reflection — ensuring purity of the belief update process.
    """
    if not all_beliefs:
        return []

    numbered = "\n".join(
        f"{i + 1}. [{b.topic}] {b.belief_text[:80]}" for i, b in enumerate(all_beliefs)
    )
    prompt = (
        f"Rate each belief's relevance to this evidence (0.0 = unrelated, 1.0 = directly addressed).\n\n"
        f"Evidence topic: {topic}\n"
        f"Evidence: {evidence[:500]}\n\n"
        f"Beliefs:\n{numbered}\n\n"
        f"A belief is relevant when the evidence could confirm, challenge, or add nuance to it. "
        f"Adjacent themes or shared keywords alone do not make a belief relevant.\n\n"
        f'JSON: {{"ranked": [{{"index": 1, "relevance": 0.9}}, {{"index": 3, "relevance": 0.6}}]}}\n\n'
        f"Include beliefs with relevance > 0; omit unrelated ones."
    )
    result = llm_call(
        prompt=prompt,
        response_model=_BeliefRanking,
        fallback=_BeliefRanking(),
        model=config.STRUCTURED_MODEL,
    )
    if result.success and result.value.ranked:
        above_cutoff = sorted(
            (r for r in result.value.ranked if r.relevance >= _RELEVANCE_CUTOFF),
            key=lambda r: r.relevance,
            reverse=True,
        )
        selected = [
            all_beliefs[r.index - 1] for r in above_cutoff if 0 < r.index <= len(all_beliefs)
        ][:BELIEF_PROMPT_WINDOW]
        if selected:
            return selected

    return all_beliefs[:BELIEF_PROMPT_WINDOW]


def execute_reflect_inner(
    *, topic: str, evidence: str, ctx: ToolContext, web_context: str = ""
) -> str:
    """Core reflection logic — called by integrate_knowledge, not directly as a tool.

    web_context: pre-fetched web context (e.g. from prior web_search in the agentic loop).
    When provided, skips redundant web enrichment searches.

    Returns a human-readable summary of belief changes, or empty string on failure.
    """
    evidence = evidence[:800]
    if not topic or not evidence:
        return ""
    log.info("reflect: topic=%.60s evidence=%d chars", topic, len(evidence))
    t0 = time.perf_counter()

    identity = ctx.identity or get_request_identity()
    if identity is None:
        log.warning("reflect: identity not loaded")
        return ""

    belief_nodes = _select_relevant_beliefs(topic, evidence, list(identity.all_beliefs))
    beliefs_text = format_beliefs_for_prompt_from_nodes(belief_nodes)
    log.info(
        "reflect: showing %d/%d beliefs (LLM-selected)",
        len(belief_nodes),
        len(identity.all_beliefs),
    )

    if web_context:
        reflection_web = web_context
        log.info(
            "reflect: using %d-char web context from caller (skipping enrichment)", len(web_context)
        )
    else:
        evidence_snippet = evidence.split(".")[0].strip()[:120]
        topic_phrase = topic.replace("_", " ")
        web_queries = [q for q in (evidence_snippet, topic_phrase) if len(q) > 10][:2]
        reflection_web = _enrich_web(web_queries, "", ctx)
    web_section = (
        REFLECTION_WEB_SECTION.format(web_content=reflection_web) if reflection_web else ""
    )

    deep_prompt = REFLECTION_DEEP_PROMPT.format(
        snapshot=identity.snapshot_text,
        beliefs=beliefs_text,
        user_message=evidence,
        web_context_section=web_section,
    )
    deep_result = consensus_call(
        prompt=deep_prompt,
        response_model=DeepReflectionResponse,
        fallback=DeepReflectionResponse(),
        models=(config.REASONING_MODEL, config.STRUCTURED_MODEL),
        merge=_merge_reflections,
    )
    if not deep_result.success:
        log.warning("reflect: deep reflection failed: %s", deep_result.error)
        return "Reflection attempted but failed."

    reflection = deep_result.value
    apply_reflection(reflection, episode_uid="", ctx=ctx)
    elapsed = time.perf_counter() - t0
    log.info(
        "reflect: updates=%d new=%d elapsed=%.1fs",
        len(reflection.belief_updates),
        len(reflection.new_beliefs),
        elapsed,
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
