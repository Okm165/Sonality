"""Belief reflection — deep reflection and belief graph updates.

Called by integrate_knowledge (not exposed as a standalone tool). The pipeline:
1. Rank beliefs by relevance (embedding similarity + graph proximity)
2. Run deep reflection via LLM to produce belief patches
3. Apply patches to Neo4j (upsert beliefs, update personality snapshot)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field, model_validator

from shared.embedder import EmbedderProtocol, cosine_similarity
from shared.errors import BeliefUpdateError
from shared.ranking import rrf_score, scores_to_ranks

from .. import config
from ..caller import format_prompt, llm_call
from ..memory.graph import BeliefNode, format_beliefs_for_prompt_from_nodes
from ..prompts import REFLECTION_DEEP_PROMPT, WEB_EVIDENCE_HEADER
from ..request_identity import get_request_identity
from ..schema import normalize_topic
from . import ToolContext

log = structlog.get_logger(__name__)


# --- Belief Ranking (pure embeddings) ---


def rank_beliefs_by_similarity(
    context: str,
    beliefs: list[BeliefNode],
    embedder: EmbedderProtocol,
    *,
    max_results: int = 8,
) -> list[BeliefNode]:
    """Rank beliefs by embedding similarity to context. Used by agent for quick filtering."""
    if not beliefs or not context.strip():
        return beliefs[:max_results]
    if len(beliefs) <= max_results:
        return beliefs

    ctx_emb = embedder.embed_query(context[:1500])
    texts = [b.belief_text or b.topic for b in beliefs]
    embs = embedder.embed_documents(texts)

    if len(embs) != len(beliefs):
        log.warning("embed_count_mismatch", beliefs=len(beliefs), embeddings=len(embs))
        return beliefs[:max_results]
    scored = [
        (belief, cosine_similarity(ctx_emb, emb)) for belief, emb in zip(beliefs, embs, strict=True)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [b for b, _ in scored[:max_results]]


@dataclass(slots=True)
class _ScoredBelief:
    """Belief with RRF-fused relevance score."""

    belief: BeliefNode
    rrf_score: float = 0.0


def rank_beliefs_algorithmically(
    evidence: str,
    all_beliefs: list[BeliefNode],
    ctx: ToolContext,
    *,
    max_results: int = 15,
) -> list[BeliefNode]:
    """Rank beliefs using RRF fusion of embedding similarity + graph signals.

    Two ranking signals fused with RRF:
    1. Embedding similarity: semantic match between evidence and belief text
    2. Graph strength: belief confidence × evidence_count (from Neo4j)
    """
    if not all_beliefs:
        return []
    if len(all_beliefs) <= max_results:
        return all_beliefs

    # --- Signal 1: Embedding similarity (dense vectors) ---
    evidence_embedding = ctx.embedder.embed_query(evidence[:1500])
    belief_texts = [b.belief_text or b.topic for b in all_beliefs]
    belief_embeddings = ctx.embedder.embed_documents(belief_texts)
    if len(belief_embeddings) != len(all_beliefs):
        return all_beliefs[:max_results]

    embedding_scores = [
        max(0.0, cosine_similarity(evidence_embedding, emb)) for emb in belief_embeddings
    ]

    # --- Signal 2: Graph strength (confidence * log evidence, Weber-Fechner) ---
    graph_scores = [
        b.confidence * (1.0 + math.log1p(b.evidence_count) / math.log(21)) for b in all_beliefs
    ]

    # --- RRF fusion ---
    emb_ranks = scores_to_ranks(embedding_scores)
    graph_ranks = scores_to_ranks(graph_scores)

    scored = [
        _ScoredBelief(belief=belief, rrf_score=rrf_score([emb_ranks[i], graph_ranks[i]]))
        for i, belief in enumerate(all_beliefs)
    ]

    scored.sort(key=lambda s: s.rrf_score, reverse=True)

    if scored:
        log.debug(
            "belief_ranking",
            top=[(s.belief.topic[:30], round(s.rrf_score, 4)) for s in scored[:3]],
            total=len(all_beliefs),
        )

    return [s.belief for s in scored[:max_results]]


class BeliefPatch(BaseModel):
    """Single belief create/update from LLM reflection."""

    topic: str = Field(default="", max_length=200)
    valence: float = Field(default=0.0, ge=-1.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    belief_text: str = Field(default="", max_length=2000)
    reasoning: str = Field(default="", max_length=1000)

    @model_validator(mode="before")
    @classmethod
    def clamp_floats(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        for key, lo, hi in (("valence", -1.0, 1.0), ("confidence", 0.0, 1.0)):
            v = data.get(key)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                data[key] = max(lo, min(hi, float(v)))
        return data


class DeepReflectionResponse(BaseModel):
    belief_updates: list[BeliefPatch] = Field(default_factory=list, max_length=20)
    new_beliefs: list[BeliefPatch] = Field(default_factory=list, max_length=10)
    snapshot_revision: str = Field(default="", max_length=5000)
    snapshot_changed: bool = False

    @model_validator(mode="before")
    @classmethod
    def coerce_structure(cls, data: object) -> object:
        """Handle bare list from LLM — wraps [{"topic":...}] as {"belief_updates": [...]}."""
        if isinstance(data, list):
            return {"belief_updates": data}
        if not isinstance(data, dict):
            return data
        result = dict(data)
        if isinstance(result.get("snapshot_changed"), str):
            result["snapshot_changed"] = str(result["snapshot_changed"]).lower() in (
                "true",
                "yes",
                "1",
            )
        if isinstance(result.get("belief_updates"), list):
            result["belief_updates"] = result["belief_updates"][:20]
        if isinstance(result.get("new_beliefs"), list):
            result["new_beliefs"] = result["new_beliefs"][:10]
        return result


def _enrich_web(queries: list[str], existing: str, ctx: ToolContext) -> str:
    """Run lightweight web research for reflection enrichment.

    Uses a glance-depth research session (1 page) through the same pipeline
    as all other web access — symmetric architecture, no separate endpoints.
    """
    if not queries or not ctx.web_client:
        return existing

    try:
        goal = "; ".join(queries[: config.settings.reflection_web_queries])

        async def _research():
            client = ctx.web_client
            assert client is not None
            sess = await client.start_research(goal, depth="glance")
            async for progress in client.stream_research(sess.session_id):
                if progress.event in ("complete", "error"):
                    break
            return await client.get_research_result(sess.session_id)

        result = ctx.run_async(_research())
        if not result.facts:
            return existing

        lines = [
            f"[{f.source_url}] {f.claim} (confidence: {f.confidence:.1f})" for f in result.facts[:8]
        ]
        text = "\n".join(lines)
        if text:
            log.info("web_enrichment", facts=len(result.facts), pages=result.pages_scraped)
            return f"{existing}\n\n{text}" if existing else text
    except Exception:
        log.warning("web_enrichment_failed", exc_info=True)
    return existing


def apply_reflection(
    reflection: DeepReflectionResponse, episode_uid: str, ctx: ToolContext
) -> None:
    """Write belief updates and snapshot revision to graph."""
    log.debug(
        "apply_reflection",
        updates=len(reflection.belief_updates),
        new_beliefs=len(reflection.new_beliefs),
        snapshot_changed=reflection.snapshot_changed,
    )
    all_updates = [
        *((b, f"reflection:{episode_uid[:12]}") for b in reflection.belief_updates),
        *((b, f"new_belief:{episode_uid[:12]}") for b in reflection.new_beliefs),
    ]
    errors: list[str] = []
    for patch, provenance in all_updates:
        if not patch.topic:
            continue
        topic = normalize_topic(patch.topic)
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
                valence=round(patch.valence, 2),
                confidence=round(patch.confidence, 2),
                reason=patch.reasoning[:100],
            )
        except Exception:
            log.error("belief_upsert_failed", topic=topic, exc_info=True)
            errors.append(topic)

    if errors:
        raise BeliefUpdateError(f"Belief upsert failed for: {errors}")

    if reflection.snapshot_changed and reflection.snapshot_revision:
        text = reflection.snapshot_revision[:2000]
        ctx.run_async(ctx.graph.upsert_personality_snapshot(text))
        log.info("snapshot_updated", chars=len(text))


def execute_reflect_inner(
    *,
    topic: str,
    evidence: str,
    ctx: ToolContext,
    web_context: str = "",
    episode_uid: str = "",
) -> str:
    """Core reflection logic — called by integrate_knowledge, not directly as a tool.

    web_context: pre-fetched web context (e.g. from prior web_research in the agentic loop).
    When provided, skips redundant web enrichment searches.
    episode_uid: provenance anchor for belief updates.

    Returns a human-readable summary of belief changes, or empty string on failure.
    """
    if not topic or not evidence:
        return ""
    log.info("reflect", topic=topic[:60], evidence_chars=len(evidence))
    t0 = time.perf_counter()

    identity = ctx.identity or get_request_identity()
    if identity is None:
        log.warning("reflect_identity_not_loaded")
        return ""

    # Rank beliefs via RRF (embedding similarity + graph signals) — no LLM
    belief_nodes = rank_beliefs_algorithmically(
        evidence=evidence,
        all_beliefs=list(identity.all_beliefs),
        ctx=ctx,
        max_results=config.settings.belief_prompt_window,
    )
    beliefs_text = format_beliefs_for_prompt_from_nodes(belief_nodes)
    log.info("reflect_beliefs_ranked", count=len(belief_nodes), total=len(identity.all_beliefs))

    if web_context:
        reflection_web = web_context
        log.info("reflect_web_context_reused", chars=len(web_context))
    else:
        topic_phrase = topic.replace("_", " ")
        reflection_web = _enrich_web([topic_phrase], "", ctx)
    web_section = f"{WEB_EVIDENCE_HEADER}{reflection_web}" if reflection_web else ""

    deep_result = llm_call(
        instructions=format_prompt(
            REFLECTION_DEEP_PROMPT,
            snapshot=identity.snapshot_text,
            beliefs=beliefs_text,
            user_message=evidence,
            web_context_section=web_section,
        ),
        response_model=DeepReflectionResponse,
        fallback=DeepReflectionResponse(),
        model=config.settings.reasoning_model,
    )
    if not deep_result.success:
        log.warning("reflect_failed", error=deep_result.error)
        return "Reflection attempted but failed."

    reflection = deep_result.value
    apply_reflection(reflection, episode_uid=episode_uid or "inline_reflection", ctx=ctx)
    elapsed = time.perf_counter() - t0
    log.info(
        "reflect_done",
        updates=len(reflection.belief_updates),
        new_beliefs=len(reflection.new_beliefs),
        elapsed_s=round(elapsed, 1),
    )
    parts: list[str] = []
    if reflection.belief_updates:
        parts.append(f"Updated {len(reflection.belief_updates)} beliefs.")
    if reflection.new_beliefs:
        parts.append(f"Formed {len(reflection.new_beliefs)} new beliefs.")
    if reflection.snapshot_changed:
        parts.append("Personality snapshot updated.")
    if not parts:
        parts.append("No belief changes needed.")
    return " ".join(parts)
