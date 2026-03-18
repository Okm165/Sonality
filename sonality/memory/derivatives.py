"""LLM-based semantic chunking for episode derivatives.

Replaces regex/tokenization-based chunking with LLM semantic understanding.
Each episode is split into 1-15 self-contained derivative chunks for granular
embedding and retrieval (MemMachine derivative model).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, field_validator, model_validator

from ..llm.caller import llm_call
from ..llm.prompts import CHUNKING_PROMPT
from .embedder import Embedder
from .graph import DerivativeNode

log = logging.getLogger(__name__)


class ChunkImportance(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ChunkItem(BaseModel):
    text: str
    key_concept: str = ""
    importance: ChunkImportance = ChunkImportance.MEDIUM

    @field_validator("importance", mode="before")
    @classmethod
    def coerce_importance(cls, v: object) -> object:
        """Accept placeholder patterns ('...', 'high/medium/low') → MEDIUM fallback."""
        if not isinstance(v, str):
            return v
        candidate = v.split("/")[0].strip().lower()
        return ChunkImportance.MEDIUM if candidate in ("", "...", "none") else candidate


class ChunkingResponse(BaseModel):
    chunks: list[ChunkItem]

    @model_validator(mode="before")
    @classmethod
    def normalize_chunks(cls, data: object) -> object:
        """Handle LLM responses that omit the outer chunks wrapper or return empty items."""
        if isinstance(data, list):
            return {"chunks": [x for x in data if isinstance(x, dict) and "text" in x]}
        if isinstance(data, dict) and "text" in data and "chunks" not in data:
            return {"chunks": [data]}
        if isinstance(data, dict) and "chunks" in data and isinstance(data["chunks"], list):
            data = {"chunks": [x for x in data["chunks"] if isinstance(x, dict) and "text" in x]}
        return data


@dataclass(frozen=True, slots=True)
class DerivativeWithEmbedding:
    """A derivative chunk with its pre-computed dense embedding."""

    node: DerivativeNode
    embedding: list[float]


class DerivativeChunker:
    """LLM-based semantic chunking of episode text into derivatives."""

    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def chunk_and_embed(self, text: str, episode_uid: str) -> list[DerivativeWithEmbedding]:
        """Split text into semantic chunks and embed each one."""
        result = llm_call(
            prompt=CHUNKING_PROMPT.format(text=text),
            response_model=ChunkingResponse,
            fallback=ChunkingResponse(chunks=[]),
            max_tokens=512,
            max_retries=1,
            assistant_prefix='{"chunks": [',
        )
        if result.success:
            chunks: list[ChunkItem] = [
                c for c in result.value.chunks if c.text.strip() and c.text.strip() != "..."
            ]
        else:
            log.warning("LLM chunking failed: %s. Using whole text.", result.error)
            chunks = []

        if not chunks:
            chunks = [ChunkItem(text=text, key_concept="full_content")]

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_documents(texts)
        results: list[DerivativeWithEmbedding] = []

        for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
            node = DerivativeNode(
                uid=str(uuid.uuid5(uuid.NAMESPACE_OID, f"{episode_uid}:{i}")),
                source_episode_uid=episode_uid,
                text=chunk.text,
                key_concept=chunk.key_concept,
                sequence_num=i,
            )
            results.append(DerivativeWithEmbedding(node=node, embedding=emb))

        log.debug("Chunked episode %s into %d derivatives", episode_uid[:8], len(results))
        return results
