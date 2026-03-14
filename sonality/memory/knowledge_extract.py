"""Knowledge proposition extraction, deduplication, and storage.

Five-stage pipeline synthesizing recent NLP research:
  1. Selection (Claimify, ACL 2025)
  2. Decontextualization (FactReasoner, EMNLP 2025; Molecular Facts, EMNLP 2024)
  3. Decomposition into molecular propositions (Dense X Retrieval, EMNLP 2024)
  4. Confidence calibration (ConFix, Huawei/Tsinghua 2024)
  5. Quality gate — reject under-decontextualized props

Deduplication uses embedding similarity (intra-batch + against existing store)
with canonicalization-aware merging (EDC, 2025).  Long texts are processed
via overlapping sliding windows (SLIDE, 2025) with LLM-generated context
summaries to mitigate the "lost in the middle" effect (Liu et al., TACL 2024).

Called inline from agent._post_process when ESS knowledge_density >= LOW.
"""

from __future__ import annotations

import asyncio
import logging
import math
import uuid
from enum import StrEnum

from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, Field

from ..llm.caller import llm_call
from ..llm.prompts import (
    KNOWLEDGE_CONSOLIDATION_PROMPT,
    KNOWLEDGE_EXTRACTION_PROMPT,
    WINDOW_CONTEXT_SUMMARY_PROMPT,
)
from .embedder import ExternalEmbedder
from .sponge import SpongeState

log = logging.getLogger(__name__)

WINDOW_SIZE_WORDS = 1500
WINDOW_OVERLAP_RATIO = 0.20
DEDUP_THRESHOLD_EXISTING = 0.85
DEDUP_THRESHOLD_INTRABATCH = 0.90


class PropositionType(StrEnum):
    FACT = "fact"
    OPINION = "opinion"
    SPECULATION = "speculation"
    NOISE = "noise"


class ExtractedProposition(BaseModel):
    text: str
    type: PropositionType = PropositionType.NOISE
    confidence: float = 0.5
    source_entity: str = ""
    key_concepts: list[str] = Field(default_factory=list)
    sentiment: float = 1.0  # -1.0 (unfavorable) to +1.0 (favorable), opinions only


class ExtractionResponse(BaseModel):
    propositions: list[ExtractedProposition] = Field(default_factory=list)


class KnowledgeConsolidation(BaseModel):
    """LLM consolidation output used during reflection."""

    contradictions: list[dict[str, str]] = Field(default_factory=list)
    merges: list[dict[str, object]] = Field(default_factory=list)
    opinion_candidates: list[dict[str, str]] = Field(default_factory=list)
    weak_propositions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 0: Sliding window with LLM context summaries (SLIDE-inspired)
# ---------------------------------------------------------------------------


class _WindowSummary(BaseModel):
    """Plain-text summary for inter-window context propagation."""

    summary: str = ""


def _split_windows(text: str) -> list[tuple[str, str]]:
    """Split text into overlapping windows with LLM-generated context summaries.

    Returns (window_text, preceding_context_summary) tuples.  For the first
    window the summary is empty.  Subsequent windows get a concise LLM
    summary of the previous window's content (SLIDE, 2025) rather than raw
    word overlap — this produces 24-39% better entity/fact extraction than
    naive chunking and avoids the "lost in the middle" problem.
    """
    words = text.split()
    if len(words) <= WINDOW_SIZE_WORDS:
        return [(text, "")]
    overlap = int(WINDOW_SIZE_WORDS * WINDOW_OVERLAP_RATIO)
    stride = WINDOW_SIZE_WORDS - overlap
    windows: list[tuple[str, str]] = []
    prev_summary = ""
    for start in range(0, len(words), stride):
        chunk = words[start : start + WINDOW_SIZE_WORDS]
        window_text = " ".join(chunk)
        windows.append((window_text, prev_summary))
        if start + WINDOW_SIZE_WORDS >= len(words):
            break
        summary_result = llm_call(
            prompt=WINDOW_CONTEXT_SUMMARY_PROMPT.format(text=window_text[:3000]),
            response_model=_WindowSummary,
            fallback=_WindowSummary(),
        )
        prev_summary = summary_result.value.summary if summary_result.success else ""
    return windows


# ---------------------------------------------------------------------------
# Stage 1-5: LLM extraction (fully LLM-driven quality gating)
# ---------------------------------------------------------------------------


def _extract_propositions(text: str, preceding_context: str = "") -> list[ExtractedProposition]:
    """Run five-stage LLM extraction on a single window.

    If preceding_context is provided (LLM summary of previous window),
    it's prepended so the LLM can resolve cross-window references.
    The extraction prompt's Stage 5 quality gate handles decontextualization
    and self-containment checks — no post-hoc heuristic filtering.
    """
    prompt_text = text
    if preceding_context:
        prompt_text = (
            f"[Preceding context for reference — do NOT extract from this section, "
            f"only use it to resolve references:]\n{preceding_context}\n\n"
            f"[Text to extract from:]\n{text}"
        )
    result = llm_call(
        prompt=KNOWLEDGE_EXTRACTION_PROMPT.format(text=prompt_text),
        response_model=ExtractionResponse,
        fallback=ExtractionResponse(),
    )
    if not result.success:
        log.warning("Knowledge extraction parse failed: %s", result.error)
        return []
    return [p for p in result.value.propositions if p.type != PropositionType.NOISE]


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def _load_existing_knowledge_embeddings(
    pg_pool: AsyncConnectionPool,
    limit: int = 200,
) -> list[tuple[str, list[float]]]:
    """Load existing knowledge feature embeddings for dedup comparison."""
    async with pg_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT value, embedding::text
            FROM semantic_features
            WHERE category = 'knowledge' AND embedding IS NOT NULL
            ORDER BY updated_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = await cur.fetchall()
    result: list[tuple[str, list[float]]] = []
    for value, emb_text in rows:
        try:
            emb = [float(x) for x in emb_text.strip("[]").split(",")]
            result.append((str(value), emb))
        except (ValueError, AttributeError):
            continue
    return result


# ---------------------------------------------------------------------------
# Stage 4: Two-pass deduplication (intra-batch + against existing store)
# ---------------------------------------------------------------------------


def _deduplicate_intrabatch(
    propositions: list[ExtractedProposition],
    embeddings: list[list[float]],
) -> list[tuple[ExtractedProposition, list[float]]]:
    """Remove near-duplicate propositions within the same extraction batch.

    When multi-window extraction produces overlapping facts, keep the one
    with higher confidence. Uses a tighter threshold (0.95) since intra-batch
    duplicates are usually near-identical reformulations.
    """
    kept: list[tuple[ExtractedProposition, list[float]]] = []
    for prop, emb in zip(propositions, embeddings, strict=True):
        is_dup = False
        for kept_prop, kept_emb in kept:
            if _cosine_similarity(emb, kept_emb) > DEDUP_THRESHOLD_INTRABATCH:
                if prop.confidence > kept_prop.confidence:
                    kept.remove((kept_prop, kept_emb))
                    kept.append((prop, emb))
                is_dup = True
                break
        if not is_dup:
            kept.append((prop, emb))
    return kept


async def _deduplicate_against_existing(
    batch: list[tuple[ExtractedProposition, list[float]]],
    existing: list[tuple[str, list[float]]],
    pg_pool: AsyncConnectionPool,
    episode_uid: str,
) -> list[tuple[ExtractedProposition, list[float]]]:
    """Deduplicate against existing knowledge; boost confidence for repeated evidence.

    When a proposition is semantically similar to an existing one, instead of
    silently dropping it we boost the existing entry's confidence and add the
    episode citation (MMA 2025: evidence accumulation via repeated mentions).
    """
    kept: list[tuple[ExtractedProposition, list[float]]] = []
    for prop, emb in batch:
        matched_existing_text: str | None = None
        for existing_text, existing_emb in existing:
            if _cosine_similarity(emb, existing_emb) > DEDUP_THRESHOLD_EXISTING:
                matched_existing_text = existing_text
                break
        if matched_existing_text:
            await _boost_existing_confidence(
                pg_pool, matched_existing_text, prop.confidence, episode_uid
            )
            log.debug("Evidence boost for existing: '%s'", matched_existing_text[:60])
        else:
            kept.append((prop, emb))
    return kept


async def _boost_existing_confidence(
    pg_pool: AsyncConnectionPool,
    value_text: str,
    new_confidence: float,
    episode_uid: str,
) -> None:
    """Update confidence of an existing knowledge feature on repeated evidence.

    Uses the new proposition's LLM-assigned confidence: if the new mention has
    higher confidence (e.g. from a better source), the stored entry adopts it.
    Episode citation is always added for provenance tracking.
    """
    try:
        async with pg_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE semantic_features
                SET confidence = GREATEST(confidence, %s),
                    updated_at = NOW(),
                    episode_citations = (
                        SELECT ARRAY(
                            SELECT DISTINCT citation
                            FROM unnest(episode_citations || %s::text[]) AS citation
                        )
                    )
                WHERE category = 'knowledge' AND value = %s
                """,
                (min(0.99, new_confidence), [episode_uid], value_text),
            )
    except Exception:
        log.debug("Failed to boost confidence for '%s'", value_text[:60], exc_info=True)


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

PROPOSITION_TAG_MAP: dict[PropositionType, str] = {
    PropositionType.FACT: "Verified Facts",
    PropositionType.OPINION: "Attributed Opinions",
    PropositionType.SPECULATION: "Speculative Claims",
}


async def _persist_proposition(
    pg_pool: AsyncConnectionPool,
    prop: ExtractedProposition,
    embedding: list[float],
    episode_uid: str,
) -> None:
    """Store a single proposition as a knowledge semantic feature."""
    tag = PROPOSITION_TAG_MAP.get(prop.type, "Verified Facts")
    feature_name = " | ".join(prop.key_concepts[:3]) if prop.key_concepts else prop.text[:60]
    seed = f"semantic:knowledge:{tag.strip().lower()}:{prop.text.strip().lower()[:120]}"
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    async with pg_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            INSERT INTO semantic_features (
                uid, category, tag, feature_name, value,
                episode_citations, confidence, embedding, updated_at
            )
            VALUES (%s, 'knowledge', %s, %s, %s, %s, %s, %s::vector, NOW())
            ON CONFLICT (uid) DO UPDATE
            SET
                confidence = GREATEST(semantic_features.confidence, EXCLUDED.confidence),
                updated_at = NOW(),
                episode_citations = (
                    SELECT ARRAY(
                        SELECT DISTINCT citation
                        FROM unnest(
                            semantic_features.episode_citations || EXCLUDED.episode_citations
                        ) AS citation
                    )
                )
            """,
            (
                uid,
                tag,
                feature_name,
                prop.text,
                [episode_uid],
                max(0.0, min(1.0, prop.confidence)),
                embedding,
            ),
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def extract_and_store_knowledge(
    text: str,
    episode_uid: str,
    pg_pool: AsyncConnectionPool,
    embedder: ExternalEmbedder,
    sponge: SpongeState,
    *,
    stage_opinions: bool = True,
) -> int:
    """Full pipeline: window → extract → dedup → store.

    Pipeline stages:
      0. Split into overlapping windows with context (SLIDE-inspired)
      1-3. LLM extraction per window (Claimify three-stage)
           Confidence calibration is handled entirely by the LLM prompt —
           source credibility, specificity, and verifiability are assessed
           during extraction, not by hardcoded rules.
      4a. Intra-batch deduplication (across windows)
      4b. Dedup against existing + evidence accumulation (MMA 2025)
      5. Persist to semantic_features; route opinions to sponge

    Args:
        stage_opinions: If False, skip staging opinion-type propositions as
            belief updates. Use for manipulative turns (social_pressure,
            emotional_appeal, etc.) where the textual content may contain
            claims that should be stored as knowledge but not influence beliefs.
    """
    windows = _split_windows(text)
    log.debug("Knowledge pipeline: %d windows from %d chars", len(windows), len(text))
    all_propositions: list[ExtractedProposition] = []
    for i, (window_text, preceding_context) in enumerate(windows):
        props = _extract_propositions(window_text, preceding_context)
        log.debug(
            "Window %d/%d: %d propositions extracted (types: %s)",
            i + 1,
            len(windows),
            len(props),
            ", ".join(f"{p.type}:{p.confidence:.2f}" for p in props),
        )
        all_propositions.extend(props)

    if not all_propositions:
        log.debug("No propositions extracted from any window")
        return 0

    texts_to_embed = [p.text for p in all_propositions]
    new_embeddings = await asyncio.to_thread(embedder.embed_documents, texts_to_embed)

    batch = _deduplicate_intrabatch(all_propositions, new_embeddings)
    log.debug("Intra-batch dedup: %d → %d", len(all_propositions), len(batch))
    existing = await _load_existing_knowledge_embeddings(pg_pool)
    log.debug("Existing knowledge entries for dedup: %d", len(existing))
    kept = await _deduplicate_against_existing(batch, existing, pg_pool, episode_uid)
    log.debug(
        "After existing dedup: %d kept, %d evidence-boosted", len(kept), len(batch) - len(kept)
    )

    stored = 0
    for prop, emb in kept:
        try:
            await _persist_proposition(pg_pool, prop, emb, episode_uid)
            stored += 1
        except Exception:
            log.exception("Failed to persist proposition: %s", prop.text[:60])
            continue

        combined = abs(prop.sentiment) * prop.confidence
        if (
            stage_opinions
            and prop.type == PropositionType.OPINION
            and prop.key_concepts
            and combined >= 0.15
        ):
            topic = prop.key_concepts[0]
            direction = 1.0 if prop.sentiment > 0 else -1.0
            sponge.stage_opinion_update(
                topic=topic,
                direction=direction,
                magnitude=combined * 0.3,
                cooling_period=2,
                provenance=f"knowledge_extraction: {prop.text[:80]}",
            )

    intra_dedup = len(all_propositions) - len(batch)
    evidence_boosted = len(batch) - len(kept)
    log.info(
        "Knowledge extraction: %d extracted, %d intra-dedup, %d evidence-boosted, %d new stored",
        len(all_propositions),
        intra_dedup,
        evidence_boosted,
        stored,
    )
    return stored


# ---------------------------------------------------------------------------
# Consolidation (called during reflection)
# ---------------------------------------------------------------------------


async def consolidate_knowledge(
    pg_pool: AsyncConnectionPool,
    snapshot: str,
    limit: int = 50,
) -> KnowledgeConsolidation | None:
    """Review stored propositions for contradictions and merges, then apply.

    Executes merge and prune actions directly in PostgreSQL so the knowledge
    base stays tidy across reflection cycles.
    """
    async with pg_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT uid, tag, value, confidence
            FROM semantic_features
            WHERE category = 'knowledge'
            ORDER BY confidence DESC, updated_at DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = await cur.fetchall()

    if len(rows) < 2:
        return None

    value_to_uid = {str(row[2]): str(row[0]) for row in rows}

    def _fuzzy_uid(text: str) -> str | None:
        """Exact match first, then prefix match as fallback for LLM reformulations."""
        if text in value_to_uid:
            return value_to_uid[text]
        text_lower = text.lower().strip()
        for stored, uid in value_to_uid.items():
            if stored.lower().strip().startswith(text_lower[:80]):
                return uid
        return None

    propositions_text = "\n".join(
        f"- [{row[1]}] {row[2]} (confidence={row[3]:.2f})" for row in rows
    )

    result = llm_call(
        prompt=KNOWLEDGE_CONSOLIDATION_PROMPT.format(
            propositions=propositions_text,
            snapshot=snapshot,
        ),
        response_model=KnowledgeConsolidation,
        fallback=KnowledgeConsolidation(),
    )
    if not result.success:
        log.warning("Knowledge consolidation parse failed: %s", result.error)
        return None
    consolidation = result.value

    # Apply weak proposition deletions
    for weak_text in consolidation.weak_propositions:
        uid = _fuzzy_uid(weak_text)
        if uid:
            try:
                async with pg_pool.connection() as conn, conn.cursor() as cur:
                    await cur.execute("DELETE FROM semantic_features WHERE uid = %s", (uid,))
                log.info("Consolidation: pruned weak proposition '%s'", weak_text[:60])
            except Exception:
                log.debug("Failed to prune weak proposition", exc_info=True)

    # Apply merges: keep the canonical text, delete source duplicates
    for merge in consolidation.merges:
        sources = merge.get("sources", [])
        merged_text = merge.get("merged", "")
        if not sources or not merged_text or not isinstance(sources, list):
            continue
        for source_text in sources:
            if not isinstance(source_text, str):
                continue
            uid = _fuzzy_uid(source_text)
            if uid:
                try:
                    async with pg_pool.connection() as conn, conn.cursor() as cur:
                        await cur.execute(
                            """
                            UPDATE semantic_features
                            SET value = %s, updated_at = NOW()
                            WHERE uid = %s
                            """,
                            (str(merged_text), uid),
                        )
                    log.info("Consolidation: merged '%s' -> canonical", source_text[:50])
                    break
                except Exception:
                    log.debug("Failed to apply merge", exc_info=True)

    # Resolve contradictions: the LLM recommends which to keep via "keep a"
    # or "keep b" in the resolution text.
    for contradiction in consolidation.contradictions:
        a_text = contradiction.get("a", "")
        b_text = contradiction.get("b", "")
        resolution = contradiction.get("resolution", "").lower()
        # Determine which proposition to drop based on LLM recommendation
        if "keep b" in resolution or "drop a" in resolution:
            loser = a_text
        elif "keep a" in resolution or "drop b" in resolution:
            loser = b_text
        else:
            log.debug("Ambiguous contradiction resolution, skipping: %s", resolution[:80])
            continue
        uid = _fuzzy_uid(loser)
        if uid:
            try:
                async with pg_pool.connection() as conn, conn.cursor() as cur:
                    await cur.execute("DELETE FROM semantic_features WHERE uid = %s", (uid,))
                log.info("Contradiction resolved: pruned '%s'", loser[:60])
            except Exception:
                log.debug("Failed to resolve contradiction", exc_info=True)

    log.info(
        "Knowledge consolidation: %d contradictions resolved, %d merges, %d pruned",
        len(consolidation.contradictions),
        len(consolidation.merges),
        len(consolidation.weak_propositions),
    )
    return consolidation


# ---------------------------------------------------------------------------
# Reflection: prune stale/low-quality knowledge from pgvector
# ---------------------------------------------------------------------------


async def prune_stale_knowledge(
    pg_pool: AsyncConnectionPool,
    max_age_interactions: int = 50,
    min_confidence: float = 0.2,
) -> int:
    """Remove low-confidence knowledge entries with no recent evidence.

    Called during reflection to keep the knowledge store lean and accurate.
    Entries below min_confidence that haven't been updated recently are noise
    that wastes retrieval bandwidth and can pollute response context.
    """
    async with pg_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            DELETE FROM semantic_features
            WHERE category = 'knowledge'
              AND confidence < %s
              AND updated_at < NOW() - INTERVAL '1 hour'
            RETURNING uid
            """,
            (min_confidence,),
        )
        deleted = await cur.fetchall()
    pruned = len(deleted)
    if pruned:
        log.info("Knowledge pruning: removed %d low-confidence stale entries", pruned)
    return pruned


# ---------------------------------------------------------------------------
# Knowledge retrieval for response context injection
# ---------------------------------------------------------------------------


async def retrieve_relevant_knowledge(
    query: str,
    pg_pool: AsyncConnectionPool,
    embedder: ExternalEmbedder,
    top_k: int = 8,
    min_confidence: float = 0.3,
) -> list[str]:
    """Retrieve stored knowledge propositions relevant to a query.

    Embeds the query, performs vector similarity search against the knowledge
    store, and returns formatted knowledge lines for system prompt injection.
    High-confidence facts are presented authoritatively; lower-confidence ones
    are hedged. This closes the learn-use loop so the agent can actually
    leverage its accumulated knowledge during response generation.
    """
    query_embedding = await asyncio.to_thread(embedder.embed_query, query)

    async with pg_pool.connection() as conn, conn.cursor() as cur:
        await cur.execute(
            """
            SELECT tag, value, confidence,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM semantic_features
            WHERE category = 'knowledge'
              AND embedding IS NOT NULL
              AND confidence >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_embedding, min_confidence, query_embedding, top_k),
        )
        rows = await cur.fetchall()

    if not rows:
        log.debug("Knowledge retrieval: no matches for query (min_conf=%.2f)", min_confidence)
        return []

    log.debug(
        "Knowledge retrieval: %d results (top similarity=%.3f, conf range=%.2f–%.2f)",
        len(rows),
        float(rows[0][3]),
        min(float(r[2]) for r in rows),
        max(float(r[2]) for r in rows),
    )
    return [
        f"[{tag}] (confidence={confidence:.2f}) {value}"
        for tag, value, confidence, _similarity in rows
    ]
