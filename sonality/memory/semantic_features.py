"""Semantic feature extraction and storage (personality, preferences, knowledge, relationships).

Background ingestion worker extracts features from episodes using category-specific
LLM prompts. Features stored in PostgreSQL with pgvector embeddings.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

from pydantic import BaseModel

from ..llm.caller import llm_call
from ..llm.prompts import FEATURE_EXTRACTION_PROMPT
from .embedder import ExternalEmbedder

log = logging.getLogger(__name__)

SEMANTIC_CATEGORIES: list[str] = ["personality", "preferences", "knowledge", "relationships"]


class FeatureCommand(BaseModel):
    command: str  # "add" | "update" | "delete"
    tag: str
    feature: str
    value: str = ""
    confidence: float = 0.5
    reason: str = ""


class FeatureExtractionResponse(BaseModel):
    commands: list[FeatureCommand]


@dataclass(frozen=True, slots=True)
class SemanticFeature:
    uid: str
    category: str
    tag: str
    feature_name: str
    value: str
    episode_citations: list[str]
    confidence: float
    created_at: str
    updated_at: str


class SemanticIngestionWorker:
    """Background daemon thread that extracts semantic features from episodes.

    Receives episode UIDs via thread-safe queue. Processes in adaptive batches.
    Uses LLM for category-specific feature extraction.
    """

    def __init__(
        self,
        pg_pool: object,  # AsyncConnectionPool - typed loosely for sync context
        embedder: ExternalEmbedder,
    ) -> None:
        self._pg_pool = pg_pool
        self._embedder = embedder
        self._queue: queue.Queue[tuple[str, str]] = queue.Queue()  # (episode_uid, content)
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the background ingestion thread."""
        self._thread = threading.Thread(
            target=self._run, name="semantic-ingestion", daemon=True
        )
        self._thread.start()
        log.info("Semantic ingestion worker started")

    def stop(self) -> None:
        """Signal the worker to stop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)

    def enqueue(self, episode_uid: str, content: str) -> None:
        """Queue an episode for semantic feature extraction."""
        self._queue.put((episode_uid, content))

    def _run(self) -> None:
        """Main worker loop: wait for events, process in batches."""
        while not self._stop_event.is_set():
            try:
                # Block until first item arrives (with timeout for shutdown check)
                first = self._queue.get(timeout=60.0)
                batch = [first]

                # Adaptive batching: collect more if available
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
                    episode_uid[:8], category,
                )

    def _extract_features(self, episode_uid: str, content: str, category: str) -> None:
        """Use LLM to extract features for one category, then apply commands."""
        # Get existing features for context (would need async-to-sync bridge in prod)
        existing_features = "None yet"

        prompt = FEATURE_EXTRACTION_PROMPT.format(
            episode_content=content,
            category=category,
            existing_features=existing_features,
        )
        result = llm_call(
            prompt=prompt,
            response_model=FeatureExtractionResponse,
            fallback=FeatureExtractionResponse(commands=[]),
        )
        if not result.success or not result.value:
            return

        response = result.value
        assert isinstance(response, FeatureExtractionResponse)

        for cmd in response.commands:
            if cmd.command == "add":
                log.info(
                    "Feature ADD: %s/%s/%s = %s (conf=%.2f)",
                    category, cmd.tag, cmd.feature, cmd.value, cmd.confidence,
                )
                # In production, INSERT INTO semantic_features via pg_pool
            elif cmd.command == "update":
                log.info(
                    "Feature UPDATE: %s/%s/%s = %s",
                    category, cmd.tag, cmd.feature, cmd.value,
                )
            elif cmd.command == "delete":
                log.info(
                    "Feature DELETE: %s/%s/%s reason=%s",
                    category, cmd.tag, cmd.feature, cmd.reason,
                )
