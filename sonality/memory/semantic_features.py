"""Semantic feature extraction and storage (personality, preferences, knowledge, relationships).

Background ingestion worker extracts features from episodes using category-specific
LLM prompts. Features stored in PostgreSQL with pgvector embeddings.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar

_T = TypeVar("_T")

from psycopg_pool import AsyncConnectionPool
import re

from pydantic import BaseModel, Field, field_validator, model_validator

from .. import config
from ..llm.caller import llm_call
from ..llm.prompts import FEATURE_CONSOLIDATION_PROMPT, FEATURE_EXTRACTION_PROMPT, FEATURE_TAGS
from .embedder import ExternalEmbedder

log = logging.getLogger(__name__)

SEMANTIC_CATEGORIES: list[str] = ["personality", "preferences", "knowledge", "relationships"]


class FeatureCommandType(StrEnum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


class FeatureConsolidationDecision(StrEnum):
    CONSOLIDATE = "CONSOLIDATE"
    SKIP = "SKIP"


class FeatureCommand(BaseModel):
    command: FeatureCommandType
    tag: str
    feature: str
    value: str = ""
    confidence: float = 0.5
    reason: str = ""

    @field_validator("value", mode="before")
    @classmethod
    def strip_inline_conf(cls, v: object) -> object:
        """Strip trailing '(conf=X.XX)' that LLMs sometimes append to value strings."""
        if isinstance(v, str):
            return re.sub(r"\s*\(conf=[\d.]+\)\s*$", "", v).strip()
        return v


class FeatureExtractionResponse(BaseModel):
    commands: list[FeatureCommand] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_commands(cls, data: object) -> object:
        """Handle model responses that omit the outer commands wrapper.

        LLMs sometimes return a bare FeatureCommand object or a bare list
        instead of {"commands": [...]}. Normalise both into the expected shape.
        """
        if isinstance(data, list):
            return {"commands": data}
        if isinstance(data, dict) and "command" in data and "commands" not in data:
            return {"commands": [data]}
        return data


class FeatureConsolidationAction(BaseModel):
    source_uid: str
    target_uid: str
    canonical_tag: str = ""
    canonical_feature: str = ""
    canonical_value: str = ""
    reason: str = ""


class FeatureConsolidationResponse(BaseModel):
    consolidation_decision: FeatureConsolidationDecision = FeatureConsolidationDecision.SKIP
    reasoning: str = ""
    actions: list[FeatureConsolidationAction] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class SemanticFeatureRow:
    uid: str
    tag: str
    feature_name: str
    value: str
    confidence: float
    citations: list[str]


class SemanticIngestionWorker:
    """Background daemon thread that extracts semantic features from episodes.

    Receives episode UIDs via thread-safe queue. Processes in adaptive batches.
    Uses LLM for category-specific feature extraction.

    Runs its own dedicated event loop to decouple async DB writes from the
    main agent loop, preventing contention during high-load interactions.
    """

    def __init__(
        self,
        pg_pool: AsyncConnectionPool,
        embedder: ExternalEmbedder,
    ) -> None:
        self._pg_pool = pg_pool
        self._embedder = embedder
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()  # (episode_uid, content)
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run, name="semantic-ingestion", daemon=True)
        self._stop_event = threading.Event()

    def _run_async(self, coro: Coroutine[object, object, _T]) -> _T:
        """Submit a coroutine to the worker's own dedicated event loop."""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=config.ASYNC_TIMEOUT)

    def start(self) -> None:
        """Start the background processing thread and its dedicated event loop."""
        if not self._thread.is_alive():
            # Spin the dedicated event loop in a separate daemon thread
            loop_thread = threading.Thread(
                target=self._loop.run_forever, name="semantic-ingestion-loop", daemon=True
            )
            loop_thread.start()
            self._thread.start()
        log.info("Semantic ingestion worker started")

    def stop(self) -> None:
        """Signal the worker to stop and shut down the dedicated event loop."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self._loop.call_soon_threadsafe(self._loop.stop)

    def enqueue(self, episode_uid: str, content: str) -> None:
        """Queue an episode for semantic feature extraction."""
        self._queue.put((episode_uid, content))

    def _run(self) -> None:
        """Main worker loop: wait for episodes, process in batches."""
        while not self._stop_event.is_set():
            try:
                first = self._queue.get(timeout=60.0)
                batch = [first]
                while len(batch) < 5:
                    try:
                        batch.append(self._queue.get_nowait())
                    except queue.Empty:
                        break

                for episode_uid, content in batch:
                    self._process_episode(episode_uid, content)

            except queue.Empty:
                continue
            except Exception:
                log.exception("Semantic ingestion error; continuing")

    def _process_episode(self, episode_uid: str, content: str) -> None:
        """Extract features for all categories from a single episode."""
        for category in SEMANTIC_CATEGORIES:
            try:
                self._extract_features(episode_uid, content, category)
            except Exception:
                log.exception(
                    "Feature extraction failed for episode=%s category=%s",
                    episode_uid[:8],
                    category,
                )

    def _extract_features(self, episode_uid: str, content: str, category: str) -> None:
        """Use LLM to extract features for one category, then apply commands."""
        existing_features = self._load_existing_features(category)

        tags = FEATURE_TAGS.get(category, "Domain, Traits, Style, Stance")
        prompt = FEATURE_EXTRACTION_PROMPT.format(
            episode_content=content,
            category=category,
            existing_features=existing_features,
            tags=tags,
        )
        result = llm_call(
            prompt=prompt,
            response_model=FeatureExtractionResponse,
            fallback=FeatureExtractionResponse(commands=[]),
        )
        if not result.success:
            log.warning(
                "Feature extraction parse failed for category=%s (skipping): %s",
                category,
                result.error,
            )
            return
        response = result.value

        for cmd in response.commands:
            if cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE}:
                log.info(
                    "Feature UPSERT: %s/%s/%s = %s (conf=%.2f)",
                    category,
                    cmd.tag,
                    cmd.feature,
                    cmd.value,
                    cmd.confidence,
                )
            elif cmd.command is FeatureCommandType.DELETE:
                # Require a non-empty reason (explicit contradiction evidence).
                # Topic shifts produce reason="" which we skip to prevent personality erosion.
                if not cmd.reason.strip():
                    log.debug(
                        "Feature DELETE skipped (no contradiction evidence): %s/%s/%s",
                        category,
                        cmd.tag,
                        cmd.feature,
                    )
                    continue
                log.info(
                    "Feature DELETE: %s/%s/%s reason=%s",
                    category,
                    cmd.tag,
                    cmd.feature,
                    cmd.reason,
                )
            else:
                continue
            try:
                self._run_async(self._persist_command_async(episode_uid, category, cmd))
            except Exception:
                log.exception(
                    "Feature persistence failed for episode=%s %s/%s/%s",
                    episode_uid[:8],
                    category,
                    cmd.tag,
                    cmd.feature,
                )
        try:
            self._consolidate_features(category)
        except Exception:
            log.exception("Feature consolidation failed for category=%s", category)

    @staticmethod
    def _feature_uid(category: str, tag: str, feature_name: str) -> str:
        """Build stable UID so repeated writes upsert the same semantic feature."""
        seed = f"semantic:{category}:{tag.strip().lower()}:{feature_name.strip().lower()}"
        return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))

    def _load_existing_features(self, category: str) -> str:
        """Load current category features to give extractor update/delete context."""
        try:
            rows = self._run_async(self._load_feature_rows_async(category, limit=30))
        except Exception:
            log.debug("Failed to load existing semantic features", exc_info=True)
            return "None yet"
        if not rows:
            return "None yet"
        return "\n".join(
            f"- [{row.tag}] {row.feature_name}: {row.value} (conf={row.confidence:.2f})"
            for row in rows
        )

    def _consolidate_features(self, category: str) -> None:
        """Use LLM to consolidate overlapping semantic features in a category."""
        features = self._run_async(self._load_feature_rows_async(category, limit=40))
        if len(features) < 2:
            return
        features_text = "\n".join(
            f"- uid={row.uid} | [{row.tag}] {row.feature_name}: {row.value}"
            f" (conf={row.confidence:.2f})"
            for row in features
        )
        result = llm_call(
            prompt=FEATURE_CONSOLIDATION_PROMPT.format(
                category=category,
                features=features_text,
            ),
            response_model=FeatureConsolidationResponse,
            fallback=FeatureConsolidationResponse(),
        )
        if not result.success:
            log.warning(
                "Feature consolidation parse failed for category=%s (skipping): %s",
                category,
                result.error,
            )
            return
        response = result.value
        if response.consolidation_decision is not FeatureConsolidationDecision.CONSOLIDATE:
            return
        valid_uids = {row.uid for row in features}
        for action in response.actions:
            source_uid = action.source_uid.strip()
            target_uid = action.target_uid.strip()
            if (
                source_uid == target_uid
                or source_uid not in valid_uids
                or target_uid not in valid_uids
            ):
                continue
            self._run_async(
                self._merge_features_async(
                    category=category,
                    source_uid=source_uid,
                    target_uid=target_uid,
                    action=action,
                )
            )

    @staticmethod
    def _normalize_citations(citations_obj: object) -> list[str]:
        if isinstance(citations_obj, list):
            return [str(citation) for citation in citations_obj]
        return []

    @classmethod
    def _row_to_feature(
        cls, row: tuple[object, object, object, object, object, object]
    ) -> SemanticFeatureRow:
        confidence_obj = row[4]
        confidence = float(confidence_obj) if isinstance(confidence_obj, (int, float, str)) else 0.0
        return SemanticFeatureRow(
            uid=str(row[0]),
            tag=str(row[1]),
            feature_name=str(row[2]),
            value=str(row[3]),
            confidence=confidence,
            citations=cls._normalize_citations(row[5]),
        )

    async def _load_feature_rows_async(
        self,
        category: str,
        *,
        limit: int,
    ) -> list[SemanticFeatureRow]:
        """Load semantic feature rows for one category."""
        async with self._pg_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT uid, tag, feature_name, value, confidence, episode_citations
                FROM semantic_features
                WHERE category = %s
                ORDER BY confidence DESC, updated_at DESC
                LIMIT %s
                """,
                (category, limit),
            )
            rows: list[tuple[object, object, object, object, object, object]] = await cur.fetchall()
        return [self._row_to_feature(row) for row in rows]

    async def _load_feature_pair_async(
        self,
        *,
        category: str,
        source_uid: str,
        target_uid: str,
    ) -> dict[str, SemanticFeatureRow]:
        """Load exactly two candidate feature rows for a merge action."""
        async with self._pg_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                SELECT uid, tag, feature_name, value, confidence, episode_citations
                FROM semantic_features
                WHERE category = %s AND uid IN (%s, %s)
                """,
                (category, source_uid, target_uid),
            )
            rows: list[tuple[object, object, object, object, object, object]] = await cur.fetchall()
        return {feature.uid: feature for feature in (self._row_to_feature(row) for row in rows)}

    async def _merge_features_async(
        self,
        *,
        category: str,
        source_uid: str,
        target_uid: str,
        action: FeatureConsolidationAction,
    ) -> None:
        """Merge one redundant source feature into the target feature row."""
        by_uid = await self._load_feature_pair_async(
            category=category,
            source_uid=source_uid,
            target_uid=target_uid,
        )
        if source_uid not in by_uid or target_uid not in by_uid:
            return
        source = by_uid[source_uid]
        target = by_uid[target_uid]
        canonical_tag = action.canonical_tag.strip() or target.tag
        canonical_feature = action.canonical_feature.strip() or target.feature_name
        canonical_value = action.canonical_value.strip() or target.value
        confidence = max(source.confidence, target.confidence)
        citations = list(dict.fromkeys(target.citations + source.citations))
        async with self._pg_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(
                """
                UPDATE semantic_features
                SET
                    tag = %s,
                    feature_name = %s,
                    value = %s,
                    confidence = %s,
                    episode_citations = %s,
                    updated_at = NOW()
                WHERE uid = %s
                """,
                (
                    canonical_tag,
                    canonical_feature,
                    canonical_value,
                    confidence,
                    citations,
                    target_uid,
                ),
            )
            await cur.execute(
                "DELETE FROM semantic_features WHERE uid = %s",
                (source_uid,),
            )
            log.info(
                "Feature MERGE: %s <- %s (%s)",
                target_uid[:8],
                source_uid[:8],
                action.reason[:80],
            )

    async def _persist_command_async(
        self, episode_uid: str, category: str, cmd: FeatureCommand
    ) -> None:
        """Persist feature command to PostgreSQL."""
        feature_uid = self._feature_uid(category, cmd.tag, cmd.feature)
        embedding: list[float] = []
        if cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE}:
            # Run synchronous HTTP embedding in a thread so the event loop stays free
            embedding = await asyncio.to_thread(
                self._embedder.embed_query, cmd.value or cmd.feature
            )

        async with self._pg_pool.connection() as conn, conn.cursor() as cur:
            if cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE}:
                await cur.execute(
                    """
                    INSERT INTO semantic_features (
                        uid, category, tag, feature_name, value,
                        episode_citations, confidence, embedding, updated_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector, NOW())
                    ON CONFLICT (uid) DO UPDATE
                    SET
                        value = EXCLUDED.value,
                        confidence = EXCLUDED.confidence,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW(),
                        episode_citations = (
                            SELECT ARRAY(
                                SELECT DISTINCT citation
                                FROM unnest(
                                    semantic_features.episode_citations
                                    || EXCLUDED.episode_citations
                                ) AS citation
                            )
                        )
                    """,
                    (
                        feature_uid,
                        category,
                        cmd.tag,
                        cmd.feature,
                        cmd.value,
                        [episode_uid],
                        max(0.0, min(1.0, cmd.confidence)),
                        embedding,
                    ),
                )
            elif cmd.command is FeatureCommandType.DELETE:
                await cur.execute(
                    """
                    DELETE FROM semantic_features
                    WHERE category = %s
                      AND tag = %s
                      AND feature_name = %s
                    """,
                    (category, cmd.tag, cmd.feature),
                )
