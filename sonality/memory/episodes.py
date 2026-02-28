from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import chromadb

log = logging.getLogger(__name__)


class EpisodeStore:
    def __init__(self, persist_dir: str) -> None:
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
        ess_score: float,
        topics: Sequence[str],
        summary: str,
        interaction_count: int = 0,
    ) -> None:
        doc = summary if summary else user_message[:200]
        episode_id = uuid.uuid4().hex

        self.collection.add(
            documents=[doc],
            metadatas=[
                {
                    "ess_score": ess_score,
                    "topics": ",".join(topics),
                    "summary": summary,
                    "user_message": user_message[:500],
                    "agent_response": agent_response[:500],
                    "timestamp": datetime.now(UTC).isoformat(),
                    "interaction": interaction_count,
                }
            ],
            ids=[episode_id],
        )
        log.info(
            "Stored episode %s (ESS %.2f, topics=%s, summary=%d chars, total=%d)",
            episode_id[:8],
            ess_score,
            topics,
            len(doc),
            self.collection.count(),
        )

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        min_relevance: float = 0.3,
        where: dict[str, Any] | None = None,
    ) -> list[str]:
        count = self.collection.count()
        if count == 0:
            return []

        kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": min(n_results, count),
            "include": ["metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self.collection.query(**kwargs)

        if not results or not results["metadatas"] or not results["distances"]:
            return []

        # Rerank by similarity * (1 + ess_score): prefer high-quality memories.
        # (experience-following property 2025: memory quality directly affects output;
        # MemoryGraft 2025: 47.9% poisoned retrievals without quality gating)
        candidates: list[tuple[str, float]] = []
        for meta, distance in zip(results["metadatas"][0], results["distances"][0], strict=True):
            similarity = 1.0 - distance
            if similarity < min_relevance:
                continue
            summary = meta.get("summary", "")
            if not summary:
                continue
            ess = float(meta.get("ess_score", 0.0))
            candidates.append((summary, similarity * (1.0 + ess)))

        candidates.sort(key=lambda x: x[1], reverse=True)
        episodes = [c[0] for c in candidates]
        log.info("Retrieved %d/%d episodes (min_rel=%.2f)", len(episodes), count, min_relevance)
        return episodes
