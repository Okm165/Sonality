"""Knowledge proposition extraction, deduplication, and storage.

Extracts self-contained propositions from text via LLM, deduplicates against
existing Qdrant store using embedding similarity, and persists new knowledge.
Long texts use overlapping sliding windows with LLM context summaries.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from pydantic import BaseModel, Field, model_validator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    QuantizationSearchParams,
    SearchParams,
)

from shared.config import VECTOR_SEARCH_THRESHOLD
from shared.embedder import Embedder, cosine_similarity
from shared.errors import KnowledgeStorageError
from shared.llm.parse import normalize_llm_list_response
from shared.types import deterministic_id

from .. import config
from ..caller import async_embed_documents, async_embed_query, async_llm_call, format_prompt
from ..prompts import KNOWLEDGE_EXTRACTION_PROMPT, WINDOW_CONTEXT_SUMMARY_PROMPT
from ..schema import DENSE_VECTOR, Collection, SemanticCategory

log = structlog.get_logger(__name__)

WINDOW_SIZE_WORDS = 1500
WINDOW_OVERLAP_RATIO = 0.20
DEDUP_THRESHOLD_EXISTING = 0.78
DEDUP_THRESHOLD_INTRABATCH = 0.82
_MAX_CITATIONS = 50


class _WindowSummarySchema(BaseModel):
    """Structured window context summary for sliding-window extraction."""

    summary: str = Field(default="", max_length=2000)


class ExtractedProposition(BaseModel):
    """A single self-contained knowledge proposition extracted by LLM."""

    text: str = Field(max_length=2000)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    key_concepts: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="before")
    @classmethod
    def clamp_confidence(cls, data: object) -> object:
        if isinstance(data, dict):
            v = data.get("confidence")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                data["confidence"] = max(0.0, min(1.0, float(v)))
        return data


class ExtractionResponse(BaseModel):
    """LLM response containing extracted propositions from a text window."""

    propositions: list[ExtractedProposition] = Field(default_factory=list, max_length=50)

    @model_validator(mode="before")
    @classmethod
    def normalize_response(cls, data: object) -> object:
        data = normalize_llm_list_response(data, list_key="propositions", item_required_key="text")
        if isinstance(data, dict) and isinstance(data.get("propositions"), list):
            data["propositions"] = data["propositions"][:50]
        return data


# ---------------------------------------------------------------------------
# Stage 0: Sliding window with LLM context summaries (SLIDE-inspired)
# ---------------------------------------------------------------------------


async def _split_windows(text: str) -> list[tuple[str, str]]:
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
        r = await async_llm_call(
            instructions=format_prompt(WINDOW_CONTEXT_SUMMARY_PROMPT, text=window_text),
            response_model=_WindowSummarySchema,
            fallback=_WindowSummarySchema(),
            model=config.settings.fast_model,
        )
        prev_summary = r.value.summary.strip()
    return windows


# ---------------------------------------------------------------------------
# Stage 1-5: LLM extraction (fully LLM-driven quality gating)
# ---------------------------------------------------------------------------


async def _extract_propositions(
    text: str, preceding_context: str = ""
) -> list[ExtractedProposition]:
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
    result = await async_llm_call(
        instructions=format_prompt(KNOWLEDGE_EXTRACTION_PROMPT, text=prompt_text),
        response_model=ExtractionResponse,
        fallback=ExtractionResponse(),
        model=config.settings.structured_model,
    )
    if not result.success:
        log.warning(
            "knowledge_extraction_parse_failed",
            error=(result.error or "")[:80],
        )
        return []
    return [p for p in result.value.propositions if p.text.strip() not in {"...", ""}]


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
        search_params=SearchParams(
            hnsw_ef=config.settings.qdrant_search_ef,
            quantization=QuantizationSearchParams(rescore=True),
        ),
    )
    results = response.points
    if results:
        payload = results[0].payload or {}
        uid = str(payload.get("uid") or results[0].id)
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
    """Deduplicate against existing knowledge; accumulate citations for repeated evidence.

    When a proposition is semantically similar to an existing one, we add the
    episode citation rather than storing a duplicate (MMA 2025: evidence
    accumulation via repeated mentions). Citation count serves as the evidence
    strength signal — confidence remains as originally assessed by the LLM to
    prevent repetition-based inflation attacks.
    """
    kept: list[tuple[ExtractedProposition, list[float]]] = []
    boost_uids: list[str] = []
    for prop, emb in batch:
        match = await _find_nearest_knowledge(qdrant, emb)
        if match:
            match_uid, _score = match
            boost_uids.append(match_uid)
            log.debug("evidence_boost_queued", match_uid=match_uid[:8])
        else:
            kept.append((prop, emb))

    for existing_uid in boost_uids:
        try:
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
                new_citations = list(dict.fromkeys([*old_citations, episode_uid]))[-_MAX_CITATIONS:]
                await qdrant.set_payload(
                    collection_name=Collection.SEMANTIC_FEATURES,
                    payload={
                        "episode_citations": new_citations,
                        "updated_at": datetime.now(UTC).isoformat(),
                    },
                    points=[existing_uid],
                )
                log.debug(
                    "evidence_boost_applied",
                    existing_uid=existing_uid[:8],
                    citation_count=len(new_citations),
                )
        except Exception:
            log.warning("evidence_boost_failed", existing_uid=existing_uid[:8], exc_info=True)

    return kept


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


async def _persist_proposition(
    qdrant: AsyncQdrantClient,
    prop: ExtractedProposition,
    embedding: list[float],
    episode_uid: str,
) -> None:
    """Store a single proposition as a knowledge semantic feature."""
    tag = "Knowledge"
    feature_name = " | ".join(prop.key_concepts[:3]) if prop.key_concepts else prop.text[:60]
    uid = deterministic_id(f"semantic:knowledge:{prop.text.strip().lower()}")
    now = datetime.now(UTC).isoformat()

    existing, _ = await qdrant.scroll(
        collection_name=Collection.SEMANTIC_FEATURES,
        scroll_filter=Filter(must=[FieldCondition(key="uid", match=MatchValue(value=uid))]),
        limit=1,
        with_payload=True,
    )
    citations = [episode_uid]
    confidence = max(0.0, min(1.0, prop.confidence))
    created_at = now
    if existing and existing[0].payload:
        raw_cit = existing[0].payload.get("episode_citations")
        old_citations = raw_cit if isinstance(raw_cit, list) else []
        citations = list(dict.fromkeys([*old_citations, episode_uid]))[-_MAX_CITATIONS:]
        confidence = float(existing[0].payload.get("confidence") or confidence)
        created_at = existing[0].payload.get("created_at", now)

    point = PointStruct(
        id=uid,
        vector={DENSE_VECTOR: embedding},
        payload={
            "uid": uid,
            "category": SemanticCategory.KNOWLEDGE,
            "tag": tag,
            "feature_name": feature_name,
            "value": prop.text,
            "episode_citations": citations,
            "confidence": confidence,
            "created_at": created_at,
            "updated_at": now,
        },
    )
    await qdrant.upsert(collection_name=Collection.SEMANTIC_FEATURES, points=[point])
    log.debug(
        "proposition_persisted",
        uid=uid[:8],
        tag=tag,
        confidence=confidence,
        citation_count=len(citations),
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
    windows = await _split_windows(text)
    all_propositions: list[ExtractedProposition] = []
    for window_text, preceding_context in windows:
        all_propositions.extend(await _extract_propositions(window_text, preceding_context))

    if not all_propositions:
        return 0, 0

    texts_to_embed = [p.text for p in all_propositions]
    new_embeddings = await async_embed_documents(embedder, texts_to_embed)

    batch = _deduplicate_intrabatch(all_propositions, new_embeddings)
    kept = await _deduplicate_against_existing(batch, qdrant, episode_uid)

    stored = 0
    failed = 0
    for prop, emb in kept:
        try:
            await _persist_proposition(qdrant, prop, emb, episode_uid)
            stored += 1
        except Exception:
            log.error("proposition_persist_failed", text_preview=prop.text[:80], exc_info=True)
            failed += 1

    intra_dedup = len(all_propositions) - len(batch)
    evidence_boosted = len(batch) - len(kept)
    log.info(
        "knowledge_extraction_complete",
        episode_uid=episode_uid[:8],
        extracted=len(all_propositions),
        deduped=intra_dedup,
        boosted=evidence_boosted,
        stored=stored,
        failed=failed,
    )
    if kept and stored == 0:
        raise KnowledgeStorageError(f"All {failed} proposition persists failed")
    return stored, evidence_boosted


# ---------------------------------------------------------------------------
# Knowledge retrieval for response context injection
# ---------------------------------------------------------------------------


async def retrieve_relevant_knowledge(
    query: str,
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
    top_k: int = 8,
    min_similarity: float = VECTOR_SEARCH_THRESHOLD,
    min_stored_confidence: float = VECTOR_SEARCH_THRESHOLD,
) -> list[str]:
    """Retrieve stored knowledge propositions relevant to a query.

    Embeds the query, performs vector similarity search against the knowledge
    store, and returns formatted knowledge lines for system prompt injection.

    min_similarity: cosine similarity threshold for the ANN vector search.
    min_stored_confidence: minimum LLM-assigned confidence for stored propositions.
    """
    query_embedding = await async_embed_query(embedder, query)

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
        search_params=SearchParams(
            hnsw_ef=config.settings.qdrant_search_ef,
            quantization=QuantizationSearchParams(rescore=True),
        ),
    )
    results = response.points

    if not results:
        return []

    rows = [
        (
            str(p.payload.get("tag") or ""),
            str(p.payload.get("value") or ""),
            float(p.payload.get("confidence") or 0),
            len(p.payload.get("episode_citations") or []),
        )
        for p in results
        if p.payload and float(p.payload.get("confidence") or 0) >= min_stored_confidence
    ]
    return [
        f"[{tag}] (confidence={confidence:.2f}, sources={citations}) {value}"
        for tag, value, confidence, citations in rows
    ]
