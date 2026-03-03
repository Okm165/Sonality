"""Episode retrieval and reranking over persistent vector memory.

This module is intentionally small: it keeps ChromaDB usage isolated and applies
lightweight quality gates so low-value memories are less likely to be replayed
into future prompts.
"""

from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

import chromadb

from ..ess import REASONING_QUALITY_WEIGHTS, SOURCE_RELIABILITY_WEIGHTS, ESSResult

log = logging.getLogger(__name__)

SOURCE_QUALITY_MULTIPLIER: Final = {
    reliability.value: weight for reliability, weight in SOURCE_RELIABILITY_WEIGHTS.items()
}
REASONING_QUALITY_MULTIPLIER: Final = {
    reasoning.value: weight for reasoning, weight in REASONING_QUALITY_WEIGHTS.items()
}
CONSISTENCY_PENALTY: Final = 0.75
PROVENANCE_MULTIPLIER: Final = {
    "trusted": 1.00,
    "uncertain": 0.85,
    "low": 0.70,
    "unknown": 1.00,
}
ADMISSION_MULTIPLIER: Final = {
    "semantic_strict": 1.00,
    "episodic_low_ess": 0.85,
    "episodic_quality_demotion": 0.75,
    "unspecified": 1.00,
}
RELATIONAL_TOPIC_BONUS: Final = 1.08
EMPTY_WHERE: Final[Mapping[str, Any]] = MappingProxyType({})
TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]+")
TOKEN_STOPWORDS: Final[frozenset[str]] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "with",
    }
)
CROSS_DOMAIN_GUARD_SIMILARITY: Final = 0.50
SEMANTIC_CROSS_DOMAIN_FLOOR: Final = 0.45
MIN_CROSS_DOMAIN_OVERLAP_TOKENS_LONG_QUERY: Final = 2
LOW_SIMILARITY_BLOCKED_REASONING_TYPES: Final[frozenset[str]] = frozenset(
    {"social_pressure", "emotional_appeal", "no_argument"}
)


class MemoryType(StrEnum):
    """Memory channel used for retrieval routing and guard strictness."""

    EPISODIC = "episodic"
    SEMANTIC = "semantic"


class AdmissionPolicy(StrEnum):
    """Admission policy used to record why an episode was stored."""

    SEMANTIC_STRICT = "semantic_strict"
    EPISODIC_LOW_ESS = "episodic_low_ess"
    EPISODIC_QUALITY_DEMOTION = "episodic_quality_demotion"
    UNSPECIFIED = "unspecified"


class ProvenanceQuality(StrEnum):
    """Provenance quality tag used by retrieval quality multipliers."""

    TRUSTED = "trusted"
    UNCERTAIN = "uncertain"
    LOW = "low"
    UNKNOWN = "unknown"


class CrossDomainGuardMode(StrEnum):
    """Cross-domain retrieval leakage guard mode."""

    ENFORCE = "enforce"
    DISABLED = "disabled"


def _quality_multiplier(meta: Mapping[str, object]) -> float:
    """Scale retrieval score by evidence/source quality metadata.

    This exists to down-rank memories captured from weak argumentation so
    retrieval favors durable, high-signal examples. Assumes metadata values
    follow the enums emitted by `ESSResult`; unknown labels safely fallback.
    """
    source = SOURCE_QUALITY_MULTIPLIER.get(str(meta.get("source_reliability", "")), 0.8)
    reasoning = REASONING_QUALITY_MULTIPLIER.get(str(meta.get("reasoning_type", "")), 0.8)
    quality = (source + reasoning) / 2.0
    if not bool(meta.get("internal_consistency", True)):
        quality *= CONSISTENCY_PENALTY
    provenance = PROVENANCE_MULTIPLIER.get(
        str(meta.get("provenance_quality", ProvenanceQuality.UNKNOWN.value)),
        1.0,
    )
    admission = ADMISSION_MULTIPLIER.get(
        str(meta.get("admission_policy", AdmissionPolicy.UNSPECIFIED.value)),
        1.0,
    )
    return quality * provenance * admission


def _memory_type(meta: Mapping[str, object]) -> MemoryType:
    """Parse memory-type metadata with a safe episodic fallback."""
    raw = str(meta.get("memory_type", MemoryType.EPISODIC.value))
    try:
        return MemoryType(raw)
    except ValueError:
        return MemoryType.EPISODIC


def _tokenize(text: str) -> set[str]:
    """Return lowercase alphanumeric tokens for overlap checks."""
    return set(TOKEN_PATTERN.findall(text.lower()))


def _content_tokens(text: str) -> set[str]:
    """Drop short/common tokens so overlap reflects semantic content."""
    return {token for token in _tokenize(text) if len(token) >= 3 and token not in TOKEN_STOPWORDS}


def _relational_topic_bonus(meta: Mapping[str, object], query: str) -> float:
    """Boost scores when memory topic labels are fully present in query text."""
    query_tokens = _tokenize(query)
    if not query_tokens:
        return 1.0
    topics_raw = str(meta.get("topics", ""))
    if not topics_raw:
        return 1.0
    for topic in (t.strip().lower() for t in topics_raw.split(",") if t.strip()):
        topic_tokens = _tokenize(topic)
        if topic_tokens and topic_tokens.issubset(query_tokens):
            return RELATIONAL_TOPIC_BONUS
    return 1.0


def _metadata_tokens(meta: Mapping[str, object]) -> set[str]:
    """Collect content-bearing tokens from topics and summary metadata."""
    topics_raw = str(meta.get("topics", ""))
    summary_raw = str(meta.get("summary", ""))
    return _content_tokens(topics_raw.replace(",", " ")) | _content_tokens(summary_raw)


def _has_sufficient_cross_domain_overlap(
    query_tokens: set[str],
    metadata_tokens: set[str],
) -> bool:
    """Require stronger lexical overlap for longer queries."""
    overlap_count = len(query_tokens & metadata_tokens)
    required = 1 if len(query_tokens) < 4 else MIN_CROSS_DOMAIN_OVERLAP_TOKENS_LONG_QUERY
    return overlap_count >= required


def _passes_cross_domain_guard(meta: Mapping[str, object], query: str, similarity: float) -> bool:
    """Filter weakly related memories to reduce cross-domain leakage risk.

    PersistBench (2026) reports high cross-domain leakage rates in long-term
    memory systems. We keep this lightweight: low-similarity memories are only
    allowed when query-memory lexical overlap exists, with a small exception
    for semantic memories that pass a modest similarity floor.
    """
    if similarity >= CROSS_DOMAIN_GUARD_SIMILARITY:
        return True
    query_tokens = _content_tokens(query)
    if not query_tokens:
        return (
            _memory_type(meta) == MemoryType.SEMANTIC and similarity >= SEMANTIC_CROSS_DOMAIN_FLOOR
        )
    if _has_sufficient_cross_domain_overlap(query_tokens, _metadata_tokens(meta)):
        reasoning_type = str(meta.get("reasoning_type", "")).lower()
        return reasoning_type not in LOW_SIMILARITY_BLOCKED_REASONING_TYPES
    return _memory_type(meta) == MemoryType.SEMANTIC and similarity >= SEMANTIC_CROSS_DOMAIN_FLOOR


def _to_float(value: object, default: float = 0.0) -> float:
    """Safely coerce untyped metadata values to floats."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


class EpisodeStore:
    """Persistent episodic+semantic memory backed by ChromaDB."""

    def __init__(self, persist_dir: str) -> None:
        """Open (or create) the episode collection in a local directory."""
        log.info("Initializing episode store at %s", persist_dir)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="episodes",
            metadata={"hnsw:space": "cosine"},
        )
        log.info("Episode store ready (%d episodes)", self.collection.count())

    def store(
        self,
        user_message: str,
        agent_response: str,
        ess: ESSResult,
        *,
        interaction_count: int = 0,
        memory_type: MemoryType = MemoryType.EPISODIC,
        admission_policy: AdmissionPolicy = AdmissionPolicy.UNSPECIFIED,
        provenance_quality: ProvenanceQuality = ProvenanceQuality.UNKNOWN,
    ) -> None:
        """Persist one interaction as a retrieval candidate.

        The `ESSResult` is passed directly to avoid prop-drilling many primitive
        arguments and to keep this method aligned with the classifier schema.
        `memory_type`, `admission_policy`, and `provenance_quality` are explicit
        enums so storage metadata stays consistent across runtime and benches.
        """
        summary = ess.summary
        doc = summary if summary else user_message[:200]
        episode_id = uuid.uuid4().hex

        self.collection.add(
            documents=[doc],
            metadatas=[
                {
                    "ess_score": ess.score,
                    "topics": ",".join(ess.topics),
                    "summary": summary,
                    "user_message": user_message[:500],
                    "agent_response": agent_response[:500],
                    "timestamp": datetime.now(UTC).isoformat(),
                    "interaction": interaction_count,
                    "memory_type": memory_type.value,
                    "reasoning_type": ess.reasoning_type,
                    "source_reliability": ess.source_reliability,
                    "internal_consistency": ess.internal_consistency,
                    "admission_policy": admission_policy.value,
                    "provenance_quality": provenance_quality.value,
                }
            ],
            ids=[episode_id],
        )
        log.info(
            "Stored episode %s (type=%s, ESS %.2f, topics=%s, summary=%d chars, total=%d)",
            episode_id[:8],
            memory_type.value,
            ess.score,
            ess.topics,
            len(doc),
            self.collection.count(),
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        min_relevance: float = 0.3,
        cross_domain_guard: CrossDomainGuardMode = CrossDomainGuardMode.ENFORCE,
        where: Mapping[str, Any] = EMPTY_WHERE,
    ) -> list[str]:
        """Return ranked memory summaries relevant to a query string.

        `min_relevance` filters low-similarity candidates before reranking.
        `cross_domain_guard` controls whether low-similarity memories must pass
        lexical-overlap/provenance safeguards.
        """
        count = self.collection.count()
        if count == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, count),
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = dict(where)

        results = self.collection.query(**kwargs)

        if not results or not results["metadatas"] or not results["distances"]:
            return []

        # Rerank by similarity * (1 + ess_score) * quality to avoid replaying
        # weak or unsafe exemplars (Experience-following 2025; MemoryGraft 2025).
        enforce_guard = cross_domain_guard == CrossDomainGuardMode.ENFORCE
        candidates: list[tuple[str, float]] = []
        for meta, distance in zip(results["metadatas"][0], results["distances"][0], strict=True):
            similarity = 1.0 - distance
            if similarity < min_relevance:
                continue
            if enforce_guard and not _passes_cross_domain_guard(meta, query, similarity):
                continue
            summary_raw = meta.get("summary", "")
            summary = summary_raw if isinstance(summary_raw, str) else str(summary_raw)
            if not summary:
                continue
            ess = _to_float(meta.get("ess_score", 0.0))
            quality = _quality_multiplier(meta)
            relational = _relational_topic_bonus(meta, query)
            candidates.append((summary, similarity * (1.0 + ess) * quality * relational))

        best_by_summary: dict[str, float] = {}
        for summary, score in candidates:
            prev = best_by_summary.get(summary)
            if prev is None or score > prev:
                best_by_summary[summary] = score

        ranked = sorted(best_by_summary.items(), key=lambda item: item[1], reverse=True)
        episodes = [summary for summary, _ in ranked]
        log.info("Retrieved %d/%d episodes (min_rel=%.2f)", len(episodes), count, min_relevance)
        return episodes

    def retrieve_typed(
        self,
        query: str,
        episodic_n: int = 3,
        semantic_n: int = 2,
        min_relevance: float = 0.3,
        cross_domain_guard: CrossDomainGuardMode = CrossDomainGuardMode.ENFORCE,
    ) -> list[str]:
        """Retrieve typed memories: semantic first, then episodic.

        ENGRAM (2025) shows memory typing improves long-context behavior without
        requiring graph infrastructure. We keep this lightweight by tagging
        episodes in Chroma metadata and routing through simple filters.
        Returns de-duplicated summaries with semantic memory prioritized.
        """
        semantic = self.retrieve(
            query=query,
            n_results=semantic_n,
            min_relevance=min_relevance,
            cross_domain_guard=cross_domain_guard,
            where={"memory_type": MemoryType.SEMANTIC.value},
        )
        episodic = self.retrieve(
            query=query,
            n_results=episodic_n,
            min_relevance=min_relevance,
            cross_domain_guard=cross_domain_guard,
            where={"memory_type": MemoryType.EPISODIC.value},
        )
        ordered_unique = list(dict.fromkeys([*semantic, *episodic]))
        return ordered_unique[: semantic_n + episodic_n]
