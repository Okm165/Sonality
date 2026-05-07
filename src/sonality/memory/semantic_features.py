"""Semantic feature extraction and storage (personality, preferences, knowledge, relationships).

Background ingestion worker extracts features from episodes using category-specific
LLM prompts. Features stored in Qdrant with vector embeddings.
"""

from __future__ import annotations

import asyncio
import queue
import re
import threading
import time as _time
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

import structlog
from pydantic import BaseModel, Field, field_validator, model_validator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from shared.llm.parse import normalize_llm_list_response
from shared.types import deterministic_id

from .. import config
from ..caller import llm_call
from ..prompts import FEATURE_CONSOLIDATION_PROMPT, FEATURE_EXTRACTION_PROMPT, FEATURE_TAGS
from ..provider import interaction_in_progress, llm_semaphore_idle
from ..schema import DENSE_VECTOR, Collection, SemanticCategory
from .embedder import Embedder

log = structlog.get_logger()

SEMANTIC_CATEGORIES: list[SemanticCategory] = list(SemanticCategory)


class FeatureCommandType(StrEnum):
    """Operations the LLM can request on semantic features."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"


class FeatureConsolidationDecision(StrEnum):
    """Whether duplicate/overlapping features should be merged."""

    CONSOLIDATE = "CONSOLIDATE"
    SKIP = "SKIP"


_VALID_FEATURE_TAGS: frozenset[str] = frozenset(
    tag.strip().lower().replace(" ", "_")
    for tags_str in FEATURE_TAGS.values()
    for tag in tags_str.split(",")
)


class FeatureCommand(BaseModel):
    command: FeatureCommandType
    tag: str = ""
    feature: str
    value: str = ""
    confidence: float = 0.5
    reason: str = ""

    @field_validator("command", mode="before")
    @classmethod
    def normalize_command(cls, v: object) -> object:
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("tag", mode="before")
    @classmethod
    def normalize_tag(cls, v: object) -> str:
        return v.strip().lower().replace(" ", "_") if isinstance(v, str) else ""

    @field_validator("feature", "reason", mode="before")
    @classmethod
    def truncate_text(cls, v: object) -> str:
        return str(v)[:150] if v is not None else ""

    @field_validator("value", mode="before")
    @classmethod
    def coerce_value(cls, v: object) -> str:
        if isinstance(v, str):
            result = re.sub(r"\s*\(conf=[\d.]+\)\s*$", "", v).strip()
            return result[:200]
        if isinstance(v, dict):
            first = next(iter(v.values()), "")
            return str(first)[:200] if isinstance(first, (str, int, float)) else ""
        return ""


class FeatureExtractionResponse(BaseModel):
    commands: list[FeatureCommand] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_commands(cls, data: object) -> object:
        return normalize_llm_list_response(
            data,
            list_key="commands",
            item_required_key="command",
        )

    @model_validator(mode="after")
    def validate_tags(self) -> FeatureExtractionResponse:
        """Filter out commands with unrecognised tags — silently discard rather than repair."""
        before = len(self.commands)
        self.commands = [cmd for cmd in self.commands if cmd.tag and cmd.tag in _VALID_FEATURE_TAGS]
        dropped = before - len(self.commands)
        if dropped:
            log.debug("feature_extraction_unknown_tags_dropped", dropped_count=dropped)
        return self


class FeatureConsolidationAction(BaseModel):
    source_uid: str
    target_uid: str
    canonical_tag: str = ""
    canonical_feature: str = ""
    canonical_value: str = ""
    reason: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_uid_keys(cls, data: object) -> object:
        """LLM sometimes returns variant key names for source/target UIDs."""
        if isinstance(data, dict):
            for alt, canonical in (
                ("from", "source_uid"),
                ("source_id", "source_uid"),
                ("source", "source_uid"),
                ("to", "target_uid"),
                ("target_id", "target_uid"),
                ("target", "target_uid"),
            ):
                if alt in data and canonical not in data:
                    data[canonical] = data.pop(alt)
        return data


class FeatureConsolidationResponse(BaseModel):
    consolidation_decision: FeatureConsolidationDecision = FeatureConsolidationDecision.SKIP
    reasoning: str = ""
    actions: list[FeatureConsolidationAction] = Field(default_factory=list)

    @field_validator("consolidation_decision", mode="before")
    @classmethod
    def normalize_decision(cls, v: object) -> object:
        """Normalize case: models return 'skip'/'consolidate' but enum is uppercase."""
        if isinstance(v, str):
            return v.upper()
        return v


@dataclass(frozen=True, slots=True)
class SemanticFeatureRow:
    """Read-only view of a semantic feature from Qdrant."""

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
    """

    def __init__(
        self,
        qdrant_url: str,
        embedder: Embedder,
    ) -> None:
        self._embedder = embedder
        self._qdrant = AsyncQdrantClient(url=qdrant_url)
        self._queue: queue.Queue[tuple[str, str, tuple[SemanticCategory, ...]]] = queue.Queue()
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="semantic-ingestion-loop", daemon=True
        )
        self._thread = threading.Thread(target=self._run, name="semantic-ingestion", daemon=True)
        self._stop_event = threading.Event()
        self._last_defer_log: float = 0.0

    def _run_async[T](self, coro: Coroutine[object, object, T]) -> T:
        if not self._loop.is_running() or self._stop_event.is_set():
            coro.close()
            from shared.errors import ServiceUnavailableError

            raise ServiceUnavailableError("semantic ingestion loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=config.settings.async_timeout)
        except TimeoutError:
            future.cancel()
            raise

    def start(self) -> None:
        if not self._thread.is_alive():
            self._loop_thread.start()
            # Wait for the event loop to be running before starting the processing
            # thread, preventing a race where the first queue item is processed
            # before run_forever() has been called.
            deadline = _time.monotonic() + 5.0
            while not self._loop.is_running() and _time.monotonic() < deadline:
                _time.sleep(0.005)
            self._thread.start()
        log.info("semantic_worker_started")

    async def _cancel_all_and_close_client(self) -> None:
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self._qdrant.close()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            # Use config.settings.async_timeout as the join deadline so that a long LLM
            # call inside _process_episode has time to finish before the event
            # loop is torn down. If the thread doesn't finish in time, we still
            # proceed so the agent shutdown doesn't hang indefinitely.
            self._thread.join(timeout=float(config.settings.async_timeout))
        if self._loop.is_running():
            try:
                self._run_async(self._cancel_all_and_close_client())
            except Exception:
                log.debug("semantic_worker_shutdown_error", exc_info=True)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

    def enqueue(
        self,
        episode_uid: str,
        content: str,
        categories: tuple[SemanticCategory, ...] = (),
    ) -> None:
        """Queue an episode for feature extraction. Empty categories = all."""
        log.info("semantic_enqueue", episode_uid=episode_uid[:8], categories=len(categories) or "all", queue_depth=self._queue.qsize())
        self._queue.put((episode_uid, content, categories))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                first = self._queue.get(timeout=60.0)
            except queue.Empty:
                continue

            batch = [first]
            while len(batch) < 5:
                try:
                    batch.append(self._queue.get_nowait())
                except queue.Empty:
                    break

            requeue: list[tuple[str, str, tuple[SemanticCategory, ...]]] = []
            for item in batch:
                episode_uid, content, categories = item
                if interaction_in_progress():
                    now = _time.monotonic()
                    if now - self._last_defer_log > 30.0:
                        log.debug(
                            "semantic_worker_deferring",
                            queued_count=self._queue.qsize() + len(requeue) + 1,
                        )
                        self._last_defer_log = now
                    requeue.append(item)
                else:
                    try:
                        self._process_episode(episode_uid, content, categories)
                    except Exception:
                        log.error("semantic_ingestion_failed_retry", episode_uid=episode_uid[:8], exc_info=True)
                        _time.sleep(5.0)
                        try:
                            self._process_episode(episode_uid, content, categories)
                        except Exception:
                            log.error("semantic_ingestion_failed_dropped", episode_uid=episode_uid[:8], exc_info=True)
                    _time.sleep(30.0)
                self._queue.task_done()

            for item in requeue:
                self._queue.put(item)

            if requeue:
                _time.sleep(5.0)

    def _process_episode(
        self,
        episode_uid: str,
        content: str,
        categories: tuple[SemanticCategory, ...],
    ) -> None:
        target_cats = list(categories) if categories else SEMANTIC_CATEGORIES
        for category in target_cats:
            if self._stop_event.is_set():
                log.debug("semantic_worker_stopping_mid_episode", category=category)
                return
            if interaction_in_progress():
                log.debug(
                    "semantic_worker_pausing_mid_episode",
                    category=category,
                )
                self._queue.put(
                    (episode_uid, content, tuple(target_cats[target_cats.index(category) :]))
                )
                return
            try:
                self._extract_features(episode_uid, content, category)
            except Exception:
                log.error(
                    "feature_extraction_episode_failed",
                    episode_uid=episode_uid[:8],
                    category=category,
                    exc_info=True,
                )

    def _extract_features(self, episode_uid: str, content: str, category: SemanticCategory) -> None:
        if self._stop_event.is_set():
            return
        if not llm_semaphore_idle():
            log.debug(
                "semantic_worker_llm_busy_requeue",
                episode_uid=episode_uid[:8],
                category=category,
            )
            remaining = SEMANTIC_CATEGORIES[SEMANTIC_CATEGORIES.index(category) :]
            self._queue.put((episode_uid, content, tuple(remaining)))
            return
        existing = self._load_existing_features(category)
        tags = FEATURE_TAGS.get(category, "")
        prompt = FEATURE_EXTRACTION_PROMPT.format(
            category=category,
            episode_content=content[:4000],
            tags=tags,
            existing_features=existing[:3000],
        )

        result = llm_call(
            prompt=prompt,
            response_model=FeatureExtractionResponse,
            fallback=FeatureExtractionResponse(),
            model=config.settings.structured_model,
        )
        if not result.success:
            log.debug(
                "feature_extraction_llm_failed",
                error=(result.error or "")[:80],
            )
            return

        response = result.value

        seen_tags: dict[str, int] = {}
        for i, cmd in enumerate(response.commands):
            if cmd.tag:
                seen_tags[cmd.tag] = i
        if len(seen_tags) < len(response.commands):
            response.commands = [response.commands[i] for i in sorted(seen_tags.values())]

        upsert_indices = [
            i
            for i, cmd in enumerate(response.commands)
            if cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE}
        ]
        upsert_texts = [
            response.commands[i].value or response.commands[i].feature for i in upsert_indices
        ]
        batch_embeddings: list[list[float]] = []
        if upsert_texts:
            try:
                batch_embeddings = self._embedder.embed_documents(upsert_texts)
            except Exception:
                log.warning(
                    "batch_embedding_failed",
                    feature_count=len(upsert_texts),
                    category=category,
                )
        embedding_map: dict[int, list[float]] = {
            idx: emb for idx, emb in zip(upsert_indices, batch_embeddings, strict=True)
        }

        if self._stop_event.is_set():
            return
        for i, cmd in enumerate(response.commands):
            if cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE}:
                log.info(
                    "feature_upsert",
                    category=category,
                    tag=cmd.tag,
                    feature=cmd.feature,
                    value=(cmd.value or "")[:80],
                    confidence=cmd.confidence,
                )
            elif cmd.command is FeatureCommandType.DELETE:
                if not cmd.reason.strip():
                    continue
                log.info(
                    "feature_delete",
                    category=category,
                    tag=cmd.tag,
                    feature=cmd.feature,
                    reason=cmd.reason[:80],
                )
            else:
                continue
            embedding = embedding_map.get(i, [])
            if cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE} and not embedding:
                log.warning(
                    "embedding_missing_for_feature",
                    category=category,
                    tag=cmd.tag,
                    feature=cmd.feature,
                )
                continue
            try:
                self._run_async(self._persist_command_async(episode_uid, category, cmd, embedding))
            except Exception:
                if self._stop_event.is_set():
                    log.debug("feature_persist_skipped", episode=episode_uid[:8])
                else:
                    log.warning(
                        "feature_persist_failed",
                        episode=episode_uid[:8],
                        category=category,
                        tag=cmd.tag,
                        feature=cmd.feature,
                        exc_info=True,
                    )

        try:
            self._consolidate_features(category)
        except Exception:
            if self._stop_event.is_set():
                log.debug("consolidation_skipped", category=category)
            else:
                log.warning("consolidation_failed", category=category, exc_info=True)

    def _load_existing_features(self, category: SemanticCategory) -> str:
        try:
            rows = self._run_async(self._load_feature_rows_async(category, limit=30))
        except Exception:
            log.debug("semantic_features_load_failed", exc_info=True)
            return "None yet"
        if not rows:
            return "None yet"
        return "\n".join(
            f"- [{row.tag}] {row.feature_name}: {row.value} (conf={row.confidence:.2f})"
            for row in rows
        )

    def _consolidate_features(self, category: SemanticCategory) -> None:
        features = self._run_async(self._load_feature_rows_async(category, limit=40))
        if len(features) < 2:
            return
        log.debug(
            "consolidating_semantic_features",
            feature_count=len(features),
            category=category,
        )
        features_text = "\n".join(
            f"[{f.uid[:8]}] [{f.tag}] {f.feature_name}: {f.value} (conf={f.confidence:.2f})"
            for f in features
        )
        result = llm_call(
            prompt=FEATURE_CONSOLIDATION_PROMPT.format(category=category, features=features_text),
            response_model=FeatureConsolidationResponse,
            fallback=FeatureConsolidationResponse(),
            model=config.settings.structured_model,
        )
        if not result.success:
            log.debug(
                "feature_consolidation_llm_failed",
                error=(result.error or "")[:80],
            )
            return
        response = result.value
        if response.consolidation_decision is FeatureConsolidationDecision.SKIP:
            return
        feature_uids = {f.uid for f in features}
        for action in response.actions:
            source_uid = action.source_uid
            target_uid = action.target_uid
            if (
                not source_uid
                or not target_uid
                or source_uid == target_uid
                or source_uid not in feature_uids
                or target_uid not in feature_uids
            ):
                continue
            self._run_async(self._merge_features_async(category, source_uid, target_uid, action))

    async def _load_feature_rows_async(
        self, category: SemanticCategory, limit: int
    ) -> list[SemanticFeatureRow]:
        results, _ = await self._qdrant.scroll(
            collection_name=Collection.SEMANTIC_FEATURES,
            scroll_filter=Filter(
                must=[FieldCondition(key="category", match=MatchValue(value=category))]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        rows = []
        for point in results:
            if not point.payload:
                continue
            p = point.payload
            rows.append(
                SemanticFeatureRow(
                    uid=str(p.get("uid", "")),
                    tag=str(p.get("tag", "")),
                    feature_name=str(p.get("feature_name", "")),
                    value=str(p.get("value", "")),
                    confidence=float(p.get("confidence", 0.0)),
                    citations=list(p.get("episode_citations", []))
                    if p.get("episode_citations")
                    else [],
                )
            )
        rows.sort(key=lambda r: -r.confidence)
        return rows

    async def _merge_features_async(
        self,
        category: SemanticCategory,
        source_uid: str,
        target_uid: str,
        action: FeatureConsolidationAction,
    ) -> None:
        results, _ = await self._qdrant.scroll(
            collection_name=Collection.SEMANTIC_FEATURES,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="category", match=MatchValue(value=category)),
                    FieldCondition(key="uid", match=MatchValue(value=target_uid)),
                ]
            ),
            limit=1,
            with_payload=True,
        )
        if not results:
            return
        target = results[0].payload or {}
        citations = list(target.get("episode_citations", []))
        confidence = float(target.get("confidence", 0.5))
        await self._qdrant.set_payload(
            collection_name=Collection.SEMANTIC_FEATURES,
            payload={
                "tag": action.canonical_tag or target.get("tag", ""),
                "feature_name": action.canonical_feature or target.get("feature_name", ""),
                "value": action.canonical_value or target.get("value", ""),
                "confidence": confidence,
                "episode_citations": citations,
                "updated_at": datetime.now(UTC).isoformat(),
            },
            points=[target_uid],
        )
        await self._qdrant.delete(
            collection_name=Collection.SEMANTIC_FEATURES,
            points_selector=Filter(
                must=[FieldCondition(key="uid", match=MatchValue(value=source_uid))]
            ),
        )
        log.info(
            "feature_merge",
            target_uid=target_uid[:8],
            source_uid=source_uid[:8],
            reason=action.reason[:80],
        )

    async def _persist_command_async(
        self,
        episode_uid: str,
        category: SemanticCategory,
        cmd: FeatureCommand,
        embedding: list[float],
    ) -> None:
        seed = f"semantic:{category}:{cmd.tag.strip().lower()}:{cmd.feature.strip().lower()}"
        feature_uid = deterministic_id(seed)
        now = datetime.now(UTC).isoformat()

        if cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE}:
            existing, _ = await self._qdrant.scroll(
                collection_name=Collection.SEMANTIC_FEATURES,
                scroll_filter=Filter(
                    must=[FieldCondition(key="uid", match=MatchValue(value=feature_uid))]
                ),
                limit=1,
                with_payload=True,
            )
            citations = [episode_uid]
            if existing and existing[0].payload:
                old_citations = existing[0].payload.get("episode_citations", [])
                if isinstance(old_citations, list):
                    citations = list(dict.fromkeys([*old_citations, episode_uid]))

            point = PointStruct(
                id=feature_uid,
                vector={DENSE_VECTOR: embedding},
                payload={
                    "uid": feature_uid,
                    "category": category,
                    "tag": cmd.tag,
                    "feature_name": cmd.feature,
                    "value": cmd.value,
                    "episode_citations": citations,
                    "confidence": max(0.0, min(1.0, cmd.confidence)),
                    "created_at": now,
                    "updated_at": now,
                },
            )
            await self._qdrant.upsert(collection_name=Collection.SEMANTIC_FEATURES, points=[point])
        elif cmd.command is FeatureCommandType.DELETE:
            await self._qdrant.delete(
                collection_name=Collection.SEMANTIC_FEATURES,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="category", match=MatchValue(value=category)),
                        FieldCondition(key="tag", match=MatchValue(value=cmd.tag)),
                        FieldCondition(key="feature_name", match=MatchValue(value=cmd.feature)),
                    ]
                ),
            )
