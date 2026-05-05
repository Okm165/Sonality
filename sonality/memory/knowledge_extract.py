"""Knowledge proposition extraction, deduplication, and storage.

Extracts self-contained propositions from text via LLM, deduplicates against
existing Qdrant store using embedding similarity, and persists new knowledge.
Long texts use overlapping sliding windows with LLM context summaries.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from .. import config
from ..llm.caller import llm_call
from ..llm.parse import normalize_llm_list_response
from ..prompts import (
    KNOWLEDGE_EXTRACTION_PROMPT,
    WINDOW_CONTEXT_SUMMARY_PROMPT,
)
from ..schema import DENSE_VECTOR, Collection, SemanticCategory
from .embedder import Embedder, cosine_similarity

log = logging.getLogger(__name__)

WINDOW_SIZE_WORDS = 1500
WINDOW_OVERLAP_RATIO = 0.20
DEDUP_THRESHOLD_EXISTING = 0.78
DEDUP_THRESHOLD_INTRABATCH = 0.82


class _WindowSummarySchema(BaseModel):
    """Structured window context summary for sliding-window extraction."""

    summary: str = ""


class PropositionType(StrEnum):
    """Classification of an extracted knowledge proposition."""

    FACT = "fact"
    OPINION = "opinion"
    SPECULATION = "speculation"
    NOISE = "noise"


class ExtractedProposition(BaseModel):
    """A single self-contained knowledge proposition extracted by LLM."""

    text: str
    type: PropositionType = PropositionType.FACT
    confidence: float = 0.5
    key_concepts: list[str] = Field(default_factory=list)
    negation: bool = False

    @field_validator("type", mode="before")
    @classmethod
    def strip_type(cls, v: object) -> object:
        if not isinstance(v, str):
            return v
        return v.strip().lower()

    @field_validator("negation", mode="before")
    @classmethod
    def coerce_negation(cls, v: object) -> object:
        if v is None:
            return False
        return v


class ExtractionResponse(BaseModel):
    """LLM response containing extracted propositions from a text window."""

    propositions: list[ExtractedProposition] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_response(cls, data: object) -> object:
        return normalize_llm_list_response(data, list_key="propositions", item_required_key="text")


# ---------------------------------------------------------------------------
# Stage 0: Sliding window with LLM context summaries (SLIDE-inspired)
# ---------------------------------------------------------------------------


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
        r = llm_call(
            prompt=WINDOW_CONTEXT_SUMMARY_PROMPT.format(text=window_text[:3000]),
            response_model=_WindowSummarySchema,
            fallback=_WindowSummarySchema(),
            model=config.FAST_MODEL,
        )
        prev_summary = r.value.summary.strip()
    return windows


# ---------------------------------------------------------------------------
# Stage 1-5: LLM extraction (fully LLM-driven quality gating)
# ---------------------------------------------------------------------------


def _extract_propositions(text: str, preceding_context: str = "") -> list[ExtractedProposition]:
    """Run LLM extraction on a single window.

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
        model=config.STRUCTURED_MODEL,
    )
    if not result.success:
        log.warning("Knowledge extraction parse failed: %s", result.error)
        return []
    return [
        p
        for p in result.value.propositions
        if p.type != PropositionType.NOISE
        # Filter LLM template placeholders ("..." or "[bracket]" text values)
        and p.text.strip() not in {"...", ""}
        and not p.text.startswith("[")
    ]


async def _find_nearest_knowledge(
    qdrant: AsyncQdrantClient,
    embedding: list[float],
) -> tuple[str, float] | None:
    """Find the nearest knowledge feature to the given embedding using ANN.

    Returns (uid, score) of the nearest match above DEDUP_THRESHOLD_EXISTING, or None.
    Replaces full-scan scroll with O(log N) vector search for scalable dedup.
    """
    response = await qdrant.query_points(
        collection_name=Collection.SEMANTIC_FEATURES,
        query=embedding,
        using=DENSE_VECTOR,
        query_filter=Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value=SemanticCategory.KNOWLEDGE))
            ]
        ),
        limit=1,
        with_payload=True,
        score_threshold=DEDUP_THRESHOLD_EXISTING,
    )
    results = response.points
    if results:
        uid = str(results[0].payload.get("uid", "") if results[0].payload else "") or str(
            results[0].id
        )
        return uid, results[0].score
    return None


# ---------------------------------------------------------------------------
# Stage 4: Two-pass deduplication (intra-batch + against existing store)
# ---------------------------------------------------------------------------


def _deduplicate_intrabatch(
    propositions: list[ExtractedProposition],
    embeddings: list[list[float]],
) -> list[tuple[ExtractedProposition, list[float]]]:
    """Remove near-duplicate propositions within the same extraction batch.

    When multi-window extraction produces overlapping facts, keep the one
    with higher confidence. Uses a tighter threshold since intra-batch
    duplicates are usually near-identical reformulations.
    """
    kept: list[tuple[ExtractedProposition, list[float]]] = []
    for prop, emb in zip(propositions, embeddings, strict=True):
        match_idx = next(
            (
                i
                for i, (_, ke) in enumerate(kept)
                if cosine_similarity(emb, ke) > DEDUP_THRESHOLD_INTRABATCH
            ),
            None,
        )
        if match_idx is None:
            kept.append((prop, emb))
        elif prop.confidence > kept[match_idx][0].confidence:
            kept[match_idx] = (prop, emb)
    return kept


async def _deduplicate_against_existing(
    batch: list[tuple[ExtractedProposition, list[float]]],
    qdrant: AsyncQdrantClient,
    episode_uid: str,
) -> list[tuple[ExtractedProposition, list[float]]]:
    """Deduplicate against existing knowledge; boost confidence for repeated evidence.

    When a proposition is semantically similar to an existing one, instead of
    silently dropping it we boost the existing entry's confidence and add the
    episode citation (MMA 2025: evidence accumulation via repeated mentions).
    Uses Qdrant ANN search instead of a full O(N) scroll for scalable dedup.
    """
    kept: list[tuple[ExtractedProposition, list[float]]] = []
    boosts: list[tuple[str, float]] = []  # (existing_uid, new_confidence)
    for prop, emb in batch:
        match = await _find_nearest_knowledge(qdrant, emb)
        if match:
            match_uid, _score = match
            boosts.append((match_uid, prop.confidence))
            log.debug("Evidence boost queued for uid=%s", match_uid[:8])
        else:
            kept.append((prop, emb))

    if boosts:
        try:
            for existing_uid, new_confidence in boosts:
                results, _ = await qdrant.scroll(
                    collection_name=Collection.SEMANTIC_FEATURES,
                    scroll_filter=Filter(
                        must=[
                            FieldCondition(
                                key="category", match=MatchValue(value=SemanticCategory.KNOWLEDGE)
                            ),
                            FieldCondition(key="uid", match=MatchValue(value=existing_uid)),
                        ]
                    ),
                    limit=1,
                    with_payload=True,
                )
                if results and results[0].payload:
                    point = results[0]
                    payload = point.payload
                    assert payload is not None
                    old_citations = payload.get("episode_citations", []) or []
                    new_citations = list(dict.fromkeys([*old_citations, episode_uid]))
                    await qdrant.set_payload(
                        collection_name=Collection.SEMANTIC_FEATURES,
                        payload={
                            "confidence": min(
                                0.99, max(float(payload.get("confidence") or 0), new_confidence)
                            ),
                            "episode_citations": new_citations,
                            "updated_at": datetime.now(UTC).isoformat(),
                        },
                        points=[existing_uid],
                    )
                    log.debug(
                        "Evidence boost applied: uid=%s citations=%d",
                        existing_uid[:8],
                        len(new_citations),
                    )
        except Exception:
            log.debug("Failed to boost confidence for existing entries", exc_info=True)

    return kept


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

_TAG_MAP: dict[PropositionType, str] = {
    PropositionType.FACT: "Verified Facts",
    PropositionType.OPINION: "Attributed Opinions",
    PropositionType.SPECULATION: "Speculative Claims",
}


async def _persist_proposition(
    qdrant: AsyncQdrantClient,
    prop: ExtractedProposition,
    embedding: list[float],
    episode_uid: str,
) -> None:
    """Store a single proposition as a knowledge semantic feature."""
    tag = _TAG_MAP.get(prop.type, "Verified Facts")
    text_to_store = f"[REBUTTAL] {prop.text}" if prop.negation else prop.text
    feature_name = " | ".join(prop.key_concepts[:3]) if prop.key_concepts else text_to_store[:60]
    # Seed excludes tag so the same proposition text always maps to the same UID,
    # preventing duplicate storage when the same fact is classified under different tags.
    seed = f"semantic:knowledge:{prop.text.strip().lower()[:120]}"
    uid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
    now = datetime.now(UTC).isoformat()

    existing, _ = await qdrant.scroll(
        collection_name=Collection.SEMANTIC_FEATURES,
        scroll_filter=Filter(must=[FieldCondition(key="uid", match=MatchValue(value=uid))]),
        limit=1,
        with_payload=True,
    )
    citations = [episode_uid]
    confidence = max(0.0, min(1.0, prop.confidence))
    if existing and existing[0].payload:
        old_citations = existing[0].payload.get("episode_citations", []) or []
        citations = list(dict.fromkeys([*old_citations, episode_uid]))
        confidence = max(confidence, float(existing[0].payload.get("confidence", 0)))

    point = PointStruct(
        id=uid,
        vector={DENSE_VECTOR: embedding},
        payload={
            "uid": uid,
            "category": SemanticCategory.KNOWLEDGE,
            "tag": tag,
            "feature_name": feature_name,
            "value": text_to_store,
            "episode_citations": citations,
            "confidence": confidence,
            "created_at": now,
            "updated_at": now,
        },
    )
    await qdrant.upsert(collection_name=Collection.SEMANTIC_FEATURES, points=[point])
    log.debug(
        "Persisted proposition: uid=%s tag=%s conf=%.2f citations=%d",
        uid[:8],
        tag,
        confidence,
        len(citations),
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def extract_and_store_knowledge(
    text: str,
    episode_uid: str,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
) -> tuple[int, int]:
    """Full pipeline: window → extract → dedup → store. Returns (stored, evidence_boosted).

    Stages:
      0. Split into overlapping windows with LLM context summaries (SLIDE-inspired)
      1. LLM extraction per window (select, extract, classify, score, format)
      2. Intra-batch deduplication (across windows)
      3. Dedup against existing + evidence accumulation (MMA 2025)
      4. Persist to semantic_features collection in Qdrant.
    """
    windows = await asyncio.to_thread(_split_windows, text)
    log.debug("Knowledge pipeline: %d windows from %d chars", len(windows), len(text))
    all_propositions: list[ExtractedProposition] = []
    for i, (window_text, preceding_context) in enumerate(windows):
        props = await asyncio.to_thread(_extract_propositions, window_text, preceding_context)
        log.debug(
            "Window %d/%d: %d propositions extracted (types: %s)",
            i + 1,
            len(windows),
            len(props),
            ", ".join(f"{p.type}:{p.confidence:.2f}" for p in props),
        )
        for p in props:
            log.debug("  prop[%s conf=%.2f] %s", p.type, p.confidence, p.text[:100])
        all_propositions.extend(props)

    if not all_propositions:
        log.debug("No propositions extracted from any window")
        return 0, 0

    texts_to_embed = [p.text for p in all_propositions]
    new_embeddings = await asyncio.to_thread(embedder.embed_documents, texts_to_embed)

    batch = _deduplicate_intrabatch(all_propositions, new_embeddings)
    log.debug("Intra-batch dedup: %d → %d", len(all_propositions), len(batch))
    kept = await _deduplicate_against_existing(batch, qdrant, episode_uid)
    log.debug(
        "After existing dedup: %d kept, %d evidence-boosted", len(kept), len(batch) - len(kept)
    )

    stored = 0
    for prop, emb in kept:
        try:
            await _persist_proposition(qdrant, prop, emb, episode_uid)
            stored += 1
        except Exception:
            log.exception("Failed to persist proposition: %s", prop.text[:60])

    intra_dedup = len(all_propositions) - len(batch)
    evidence_boosted = len(batch) - len(kept)
    log.info(
        "Knowledge extraction: %d extracted, %d intra-dedup, %d evidence-boosted, %d new stored",
        len(all_propositions),
        intra_dedup,
        evidence_boosted,
        stored,
    )
    return stored, evidence_boosted


# ---------------------------------------------------------------------------
# Knowledge retrieval for response context injection
# ---------------------------------------------------------------------------


async def retrieve_relevant_knowledge(
    query: str,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
    top_k: int = 8,
    min_similarity: float = 0.3,
    min_stored_confidence: float = 0.3,
) -> list[str]:
    """Retrieve stored knowledge propositions relevant to a query.

    Embeds the query, performs vector similarity search against the knowledge
    store, and returns formatted knowledge lines for system prompt injection.
    Returns empty list if embedding is unavailable (graceful degradation).

    min_similarity: cosine similarity threshold for the ANN vector search.
    min_stored_confidence: minimum LLM-assigned confidence for stored propositions.
    """
    try:
        query_embedding = await asyncio.to_thread(embedder.embed_query, query)
    except Exception:
        log.debug("Knowledge retrieval: embedding unavailable", exc_info=True)
        return []

    response = await qdrant.query_points(
        collection_name=Collection.SEMANTIC_FEATURES,
        query=query_embedding,
        using=DENSE_VECTOR,
        query_filter=Filter(
            must=[
                FieldCondition(key="category", match=MatchValue(value=SemanticCategory.KNOWLEDGE))
            ]
        ),
        limit=top_k,
        score_threshold=min_similarity,
        with_payload=True,
    )
    results = response.points

    if not results:
        log.debug("Knowledge retrieval: no matches for query (min_sim=%.2f)", min_similarity)
        return []

    rows = [
        (
            str(p.payload.get("tag") or ""),
            str(p.payload.get("value") or ""),
            float(p.payload.get("confidence") or 0),
            float(p.score),
        )
        for p in results
        if p.payload and float(p.payload.get("confidence") or 0) >= min_stored_confidence
    ]

    if not rows:
        return []

    log.debug(
        "Knowledge retrieval: %d results (top similarity=%.3f, conf range=%.2f–%.2f)",
        len(rows),
        rows[0][3],
        min(r[2] for r in rows),
        max(r[2] for r in rows),
    )
    return [
        f"[{tag}] (confidence={confidence:.2f}) {value}"
        for tag, value, confidence, _similarity in rows
    ]
