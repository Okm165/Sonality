"""LLM-based semantic chunking for episode derivatives.

Replaces regex/tokenization-based chunking with LLM semantic understanding.
Each episode is split into 1-15 self-contained derivative chunks for granular
embedding and retrieval (MemMachine derivative model).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, field_validator, model_validator

from ..llm.caller import llm_call
from ..llm.prompts import CHUNKING_PROMPT
from .embedder import ExternalEmbedder
from .graph import DerivativeNode

log = logging.getLogger(__name__)


class ChunkImportance(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ChunkItem(BaseModel):
    text: str
    key_concept: str
    importance: ChunkImportance = ChunkImportance.MEDIUM

    @field_validator("importance", mode="before")
    @classmethod
    def coerce_importance(cls, v: object) -> object:
        """Accept placeholder patterns ('...', 'high/medium/low') → MEDIUM fallback."""
        if not isinstance(v, str):
            return v
        # Take first slash-separated option, then try to match enum
        candidate = v.split("/")[0].strip().lower()
        if candidate in ("", "...", "none"):
            return ChunkImportance.MEDIUM
        return candidate


class ChunkingResponse(BaseModel):
    chunks: list[ChunkItem]

    @model_validator(mode="before")
    @classmethod
    def normalize_chunks(cls, data: object) -> object:
        """Handle LLM responses that omit the outer chunks wrapper."""
        if isinstance(data, list):
            return {"chunks": data}
        if isinstance(data, dict) and "text" in data and "chunks" not in data:
            return {"chunks": [data]}
        return data


@dataclass(frozen=True, slots=True)
class DerivativeWithEmbedding:
    """A derivative chunk with its pre-computed embedding."""

    node: DerivativeNode
    embedding: list[float]


class DerivativeChunker:
    """LLM-based semantic chunking of episode text into derivatives."""

    def __init__(self, embedder: ExternalEmbedder) -> None:
        self._embedder = embedder

    def chunk_and_embed(
        self,
        text: str,
        episode_uid: str,
    ) -> list[DerivativeWithEmbedding]:
        """Split text into semantic chunks and embed each one.

        Returns a list of DerivativeWithEmbedding. On LLM failure, falls back
        to treating the entire text as a single derivative.
        """
        result = llm_call(
            prompt=CHUNKING_PROMPT.format(text=text),
            response_model=ChunkingResponse,
            fallback=ChunkingResponse(chunks=[]),
        )
        if result.success:
            chunks: list[ChunkItem] = result.value.chunks
        else:
            log.warning("LLM chunking failed: %s. Using whole text.", result.error)
            chunks = []

        if not chunks:
            chunks = [ChunkItem(text=text, key_concept="full_content")]

        embeddings = self._embedder.embed_documents([c.text for c in chunks])

        results: list[DerivativeWithEmbedding] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            node = DerivativeNode(
                uid=f"{episode_uid}_d{i}",
                source_episode_uid=episode_uid,
                text=chunk.text,
                key_concept=chunk.key_concept,
                sequence_num=i,
            )
            results.append(DerivativeWithEmbedding(node=node, embedding=embedding))

        log.debug("Chunked episode %s into %d derivatives", episode_uid[:8], len(results))
        return results
