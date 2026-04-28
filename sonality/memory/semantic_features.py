"""Semantic feature extraction and storage (personality, preferences, knowledge, relationships).

Background ingestion worker extracts features from episodes using category-specific
LLM prompts. Features stored in Qdrant with vector embeddings.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import re
import threading
import time as _time
import uuid
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from .. import config
from ..llm.caller import llm_call
from ..prompts import FEATURE_CONSOLIDATION_PROMPT, FEATURE_EXTRACTION_PROMPT, FEATURE_TAGS
from ..provider import interaction_in_progress, llm_semaphore_idle
from ..schema import DENSE_VECTOR, Collection, SemanticCategory
from .embedder import Embedder

log = logging.getLogger(__name__)

SEMANTIC_CATEGORIES: list[SemanticCategory] = list(SemanticCategory)


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
        if isinstance(data, list):
            return {"commands": data}
        if isinstance(data, dict) and "command" in data and "commands" not in data:
            return {"commands": [data]}
        if isinstance(data, dict) and "commands" in data:
            # Drop malformed/empty command dicts the LLM emits as "no features" signals.
            cmds = data.get("commands", [])
            if isinstance(cmds, list):
                data = {
                    **data,
                    "commands": [c for c in cmds if isinstance(c, dict) and c.get("command")],
                }
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
            raise RuntimeError("semantic ingestion loop is not running")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=config.ASYNC_TIMEOUT)
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
        log.info("Semantic ingestion worker started")

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
            # Use config.ASYNC_TIMEOUT as the join deadline so that a long LLM
            # call inside _process_episode has time to finish before the event
            # loop is torn down. If the thread doesn't finish in time, we still
            # proceed so the agent shutdown doesn't hang indefinitely.
            self._thread.join(timeout=float(config.ASYNC_TIMEOUT))
        if self._loop.is_running():
            try:
                self._run_async(self._cancel_all_and_close_client())
            except Exception:
                log.debug("Error during semantic worker shutdown", exc_info=True)
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread.is_alive():
            self._loop_thread.join(timeout=5.0)

    def enqueue(
        self,
        episode_uid: str,
        content: str,
        categories: tuple[SemanticCategory, ...] = (),
    ) -> None:
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
                            "Semantic worker deferring: interaction active (%d queued)",
                            self._queue.qsize() + len(requeue) + 1,
                        )
                        self._last_defer_log = now
                    requeue.append(item)
                else:
                    try:
                        self._process_episode(episode_uid, content, categories)
                    except Exception:
                        log.exception("Semantic ingestion error; continuing")
                self._queue.task_done()

            for item in requeue:
                self._queue.put(item)

            if requeue:
                _time.sleep(1.0)

    def _process_episode(
        self,
        episode_uid: str,
        content: str,
        categories: tuple[SemanticCategory, ...],
    ) -> None:
        target_cats = list(categories) if categories else SEMANTIC_CATEGORIES
        for category in target_cats:
            if self._stop_event.is_set():
                log.debug("Semantic worker stopping mid-episode at category=%s", category)
                return
            if interaction_in_progress():
                log.debug(
                    "Semantic worker pausing mid-episode: interaction active at category=%s",
                    category,
                )
                self._queue.put(
                    (episode_uid, content, tuple(target_cats[target_cats.index(category) :]))
                )
                return
            try:
                self._extract_features(episode_uid, content, category)
            except Exception:
                log.exception(
                    "Feature extraction failed for episode=%s category=%s",
                    episode_uid[:8],
                    category,
                )

    def _extract_features(self, episode_uid: str, content: str, category: SemanticCategory) -> None:
        if self._stop_event.is_set():
            return
        if not llm_semaphore_idle():
            log.debug(
                "LLM busy; re-queuing episode=%s from category=%s",
                episode_uid[:8],
                category,
            )
            remaining = SEMANTIC_CATEGORIES[SEMANTIC_CATEGORIES.index(category) :]
            self._queue.put((episode_uid, content, tuple(remaining)))
            return
        existing = self._load_existing_features(category)
        tags = FEATURE_TAGS.get(category, "")
        prompt = FEATURE_EXTRACTION_PROMPT.format(
            category=category,
            episode_content=content,
            tags=tags,
            existing_features=existing,
        )

        # 2-4 short commands; 256-512 tokens is ample. Keeping it small caps the semaphore
        # hold time to ~60s on a 35B model, reducing foreground latency spikes.
        # max_retries=1: if server is busy, skip this episode rather than blocking further.
        result = llm_call(
            prompt=prompt,
            response_model=FeatureExtractionResponse,
            fallback=FeatureExtractionResponse(),
            max_tokens=config.EXTRACTION_MAX_TOKENS,
            max_retries=1,
            assistant_prefix='{"commands": [',
        )
        if not result.success:
            log.debug("Feature extraction failed: %s", result.error)
            return

        response = result.value
        if len(response.commands) > 4:
            response.commands = response.commands[:4]

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
                    "Batch embedding failed for %s features in %s", len(upsert_texts), category
                )
        embedding_map: dict[int, list[float]] = {
            idx: emb for idx, emb in zip(upsert_indices, batch_embeddings, strict=True)
        }

        if self._stop_event.is_set():
            return
        for i, cmd in enumerate(response.commands):
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
                if not cmd.reason.strip():
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
            embedding = embedding_map.get(i, [])
            if cmd.command in {FeatureCommandType.ADD, FeatureCommandType.UPDATE} and not embedding:
                log.warning(
                    "Embedding missing for %s/%s/%s; skipping", category, cmd.tag, cmd.feature
                )
                continue
            try:
                self._run_async(self._persist_command_async(episode_uid, category, cmd, embedding))
            except Exception:
                level = logging.DEBUG if self._stop_event.is_set() else logging.WARNING
                log.log(
                    level,
                    "Feature persistence failed for episode=%s %s/%s/%s",
                    episode_uid[:8],
                    category,
                    cmd.tag,
                    cmd.feature,
                    exc_info=True,
                )

        try:
            self._consolidate_features(category)
        except Exception:
            level = logging.DEBUG if self._stop_event.is_set() else logging.WARNING
            log.log(level, "Feature consolidation failed for category=%s", category, exc_info=True)

    def _load_existing_features(self, category: SemanticCategory) -> str:
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

    def _consolidate_features(self, category: SemanticCategory) -> None:
        features = self._run_async(self._load_feature_rows_async(category, limit=40))
        if len(features) < 2:
            return
        log.debug("Consolidating %d features in category=%s", len(features), category)
        features_text = "\n".join(
            f"[{f.uid[:8]}] [{f.tag}] {f.feature_name}: {f.value} (conf={f.confidence:.2f})"
            for f in features
        )
        result = llm_call(
            prompt=FEATURE_CONSOLIDATION_PROMPT.format(category=category, features=features_text),
            response_model=FeatureConsolidationResponse,
            fallback=FeatureConsolidationResponse(),
            max_tokens=config.STRUCTURED_JSON_MAX_TOKENS,
            max_retries=1,
            assistant_prefix='{"consolidation_decision": "',
        )
        if not result.success:
            log.debug("Feature consolidation LLM failed: %s", result.error)
            return
        response = result.value
        if response.consolidation_decision is FeatureConsolidationDecision.SKIP:
            return
        feature_uids = {f.uid for f in features}
        for action in response.actions[:2]:
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
        confidence = max(float(target.get("confidence", 0.0)), 0.5)
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
        log.info("Feature MERGE: %s <- %s (%s)", target_uid[:8], source_uid[:8], action.reason[:80])

    async def _persist_command_async(
        self,
        episode_uid: str,
        category: SemanticCategory,
        cmd: FeatureCommand,
        embedding: list[float],
    ) -> None:
        seed = f"semantic:{category}:{cmd.tag.strip().lower()}:{cmd.feature.strip().lower()}"
        feature_uid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
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
