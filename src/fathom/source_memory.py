"""Persistent source memory — Qdrant (vectors) + Neo4j (graph).

Qdrant: Source embeddings for semantic similarity search.
Neo4j: Domain productivity graph (:SourceDomain)-[:COVERS]->(:TopicCluster).

Usage:
  1. suggest_sources() — retrieve relevant known sources for a goal
  2. record_source() — persist embeddings + graph links after analysis
  3. diverge_probabilistically() — find novel sources for exploration
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    QuantizationSearchParams,
    SearchParams,
    VectorParams,
)

from shared.config import VECTOR_SEARCH_THRESHOLD
from shared.embedder import Embedder
from shared.types import deterministic_id

from .caller import async_embed_documents, async_embed_query
from .config import settings
from .models import extract_domain

if TYPE_CHECKING:
    from neo4j import AsyncDriver

log = structlog.get_logger(__name__)

COLLECTION_NAME = "fathom_sources"


@dataclass(slots=True)
class SourceSuggestion:
    """A suggested source from memory."""

    url: str
    domain: str
    score: float
    fact_count: int
    quality_rate: float


async def init_source_collection(qdrant: AsyncQdrantClient, *, dims: int = 0) -> None:
    """Initialize the fathom_sources collection if it doesn't exist.

    Args:
        dims: Embedding dimensions. When 0, reads from config.
    """
    effective_dims = dims or settings.embedding_dimensions
    collections = await qdrant.get_collections()
    existing = {c.name for c in collections.collections}

    if COLLECTION_NAME not in existing:
        await qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=effective_dims, distance=Distance.COSINE),
        )
        await qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="domain",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        log.info("source_collection_created", collection=COLLECTION_NAME)


async def record_source(
    qdrant: AsyncQdrantClient,
    driver: AsyncDriver,
    embedder: Embedder,
    *,
    url: str,
    content: str,
    page_quality: float,
    facts: list[tuple[str, str, float]],  # (claim, topic, confidence)
    query_text: str = "",
) -> None:
    """Record a visited source to both Qdrant (embedding) and Neo4j (graph).

    Args:
        page_quality: Continuous 0-1 quality score (mean of fact source_quality
                      values, or 0.0 for pages yielding no facts).
    """
    domain = extract_domain(url)

    # --- Qdrant: Store/update source embedding ---
    if content:
        content_for_embedding = f"{url} {content[:2000]}"
        vecs = await async_embed_documents(embedder, [content_for_embedding])
        embedding = vecs[0]

        point_id = deterministic_id(url)

        topics = sorted({t.strip().lower() for _, t, _ in facts if t})
        await qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "url": url,
                        "domain": domain,
                        "topic": ", ".join(topics) or "",
                        "page_quality": page_quality,
                        "facts_count": len(facts),
                    },
                )
            ],
        )

    # --- Neo4j: Update graph relationships ---
    async with driver.session(database=settings.neo4j_database) as session:
        await session.run(
            """
            MERGE (d:SourceDomain {domain: $domain})
            ON CREATE SET d.visit_count = 0, d.quality_sum = 0.0,
                          d.fact_count = 0, d.created_at = datetime()
            SET d.visit_count = d.visit_count + 1,
                d.quality_sum = d.quality_sum + $page_quality,
                d.fact_count = d.fact_count + $fact_count,
                d.last_seen = datetime()
            WITH d
            SET d.quality_rate = (d.quality_sum + 1.0) / (d.visit_count + 2.0)
            """,
            domain=domain,
            page_quality=page_quality,
            fact_count=len(facts),
        )

        if query_text:
            query_normalized = query_text.strip().lower()[:500]
            await session.run(
                """
                MATCH (d:SourceDomain {domain: $domain})
                MERGE (q:ResearchQuery {text: $qtext})
                ON CREATE SET q.created_at = datetime(), q.use_count = 0
                SET q.use_count = q.use_count + 1, q.last_used = datetime()
                MERGE (q)-[r:FOUND]->(d)
                ON CREATE SET r.count = 0, r.quality_sum = 0.0
                SET r.count = r.count + 1,
                    r.quality_sum = r.quality_sum + $page_quality
                """,
                domain=domain,
                qtext=query_normalized,
                page_quality=page_quality,
            )

        topic_rows = [
            {"tname": t.strip().lower().replace("_", " ")[:100]} for _, t, _ in facts if t
        ]
        if topic_rows:
            await session.run(
                """
                MATCH (d:SourceDomain {domain: $domain})
                UNWIND $rows AS row
                MERGE (t:TopicCluster {name: row.tname})
                ON CREATE SET t.fact_count = 0, t.created_at = datetime()
                SET t.fact_count = t.fact_count + 1, t.updated_at = datetime()
                MERGE (d)-[r:COVERS]->(t)
                ON CREATE SET r.fact_count = 0
                SET r.fact_count = r.fact_count + 1
                """,
                domain=domain,
                rows=topic_rows,
            )

    log.debug("source_recorded", domain=domain, quality=round(page_quality, 2), facts=len(facts))


async def suggest_sources(
    qdrant: AsyncQdrantClient,
    driver: AsyncDriver,
    embedder: Embedder,
    goal: str,
    questions: list[str],
    *,
    limit: int = 20,
    min_quality: float = 0.2,
) -> list[SourceSuggestion]:
    """Suggest relevant known sources for a research goal.

    Combines Qdrant vector similarity with Neo4j domain quality stats.
    Score formula: similarity * quality_rate^0.5 — power-law modulator
    that downweights low-quality domains sub-linearly while keeping
    semantic relevance as the dominant signal.
    """
    query_text = f"{goal} {' '.join(questions)}"
    query_embedding = await async_embed_query(embedder, query_text[:1000])

    try:
        response = await qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            limit=limit * 2,
            score_threshold=VECTOR_SEARCH_THRESHOLD,
            with_payload=True,
            search_params=SearchParams(
                hnsw_ef=settings.qdrant_search_ef,
                quantization=QuantizationSearchParams(rescore=True),
            ),
        )
        results = response.points
    except Exception as exc:
        log.warning("qdrant_search_failed", error=str(exc))
        return []

    if not results:
        return []

    domains = list({r.payload.get("domain", "") for r in results if r.payload})
    domain_stats: dict[str, tuple[float, int]] = {}

    if domains:
        async with driver.session(database=settings.neo4j_database) as session:
            result = await session.run(
                """
                MATCH (d:SourceDomain)
                WHERE d.domain IN $domains
                RETURN d.domain AS domain, d.quality_rate AS rate, d.fact_count AS facts
                """,
                domains=domains,
            )
            async for record in result:
                domain_stats[record["domain"]] = (
                    record["rate"] if record["rate"] is not None else 0.5,
                    record["facts"] or 0,
                )

    suggestions: list[SourceSuggestion] = []
    for r in results:
        payload = r.payload or {}
        domain = payload.get("domain", "")
        url = payload.get("url", "")

        if not url:
            continue

        quality_rate, fact_count = domain_stats.get(domain, (0.5, 0))

        if quality_rate < min_quality:
            continue

        combined_score = r.score * quality_rate**0.5

        suggestions.append(
            SourceSuggestion(
                url=url,
                domain=domain,
                score=combined_score,
                fact_count=fact_count,
                quality_rate=quality_rate,
            )
        )

    suggestions.sort(key=lambda s: s.score, reverse=True)

    log.info("sources_suggested", query_len=len(query_text), found=len(suggestions))
    return suggestions[:limit]


async def diverge_probabilistically(
    qdrant: AsyncQdrantClient,
    embedder: Embedder,
    known_urls: list[str],
    goal: str,
    *,
    divergence: float = 0.3,
    limit: int = 10,
) -> list[str]:
    """Find sources that are semantically related but different from known ones.

    This enables exploration of new territory while staying relevant.

    Args:
        known_urls: URLs we've already visited or are considering
        goal: Research goal for relevance
        divergence: 0=stay close to known, 1=maximize novelty
        limit: Max sources to return

    Returns:
        List of novel but relevant URLs
    """
    if not known_urls:
        return []

    # All vectors in document space (consistent with stored embeddings)
    known_texts = [url[:500] for url in known_urls[:20]]
    all_texts = [*known_texts, goal[:500]]
    all_embeddings = await async_embed_documents(embedder, all_texts)
    known_embeddings = all_embeddings[:-1]
    goal_embedding = all_embeddings[-1]

    ndims = len(goal_embedding)
    centroid = [sum(e[i] for e in known_embeddings) / len(known_embeddings) for i in range(ndims)]

    # Blend: low divergence = closer to goal, high = explore around known centroid
    blended = [
        (1 - divergence) * goal_embedding[i] + divergence * centroid[i] for i in range(ndims)
    ]

    try:
        response = await qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=blended,
            limit=limit * 3,
            score_threshold=VECTOR_SEARCH_THRESHOLD,
            with_payload=True,
            search_params=SearchParams(
                hnsw_ef=settings.qdrant_search_ef,
                quantization=QuantizationSearchParams(rescore=True),
            ),
        )
        results = response.points
    except Exception as exc:
        log.warning("diverge_search_failed", error=str(exc))
        return []

    # Filter out known URLs and return novel ones
    known_set = set(known_urls)
    novel_urls = [
        r.payload.get("url", "")
        for r in results
        if r.payload and r.payload.get("url", "") not in known_set
    ]

    # Add some randomness for exploration
    if len(novel_urls) > limit:
        random.shuffle(novel_urls)
        novel_urls = novel_urls[:limit]

    log.debug("divergence_search", known=len(known_urls), novel_found=len(novel_urls))
    return novel_urls
