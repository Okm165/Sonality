"""LLM-based semantic chunking for episode derivatives.

Each episode is split into 1-15 self-contained derivative chunks for granular
embedding and retrieval (MemMachine derivative model).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, model_validator

from .. import config
from ..llm.caller import llm_call
from ..prompts import CHUNKING_PROMPT
from .embedder import Embedder
from .graph import DerivativeNode

log = logging.getLogger(__name__)


class ChunkItem(BaseModel):
    text: str
    key_concept: str = ""


class ChunkingResponse(BaseModel):
    chunks: list[ChunkItem]

    @model_validator(mode="before")
    @classmethod
    def normalize_chunks(cls, data: object) -> object:
        if isinstance(data, list):
            return {"chunks": [x for x in data if _is_valid_chunk(x)]}
        if isinstance(data, dict) and "text" in data and "chunks" not in data:
            return {"chunks": [data]}
        if isinstance(data, dict) and "chunks" in data and isinstance(data["chunks"], list):
            data = {"chunks": [x for x in data["chunks"] if _is_valid_chunk(x)]}
        return data


def _is_valid_chunk(x: object) -> bool:
    if isinstance(x, ChunkItem):
        return True
    return isinstance(x, dict) and "text" in x


@dataclass(frozen=True, slots=True)
class DerivativeWithEmbedding:
    """A derivative chunk with its pre-computed dense embedding."""

    node: DerivativeNode
    embedding: list[float]


def chunk_and_embed(embedder: Embedder, text: str, episode_uid: str) -> list[DerivativeWithEmbedding]:
    """Split text into semantic chunks and embed each one."""
    result = llm_call(
        prompt=CHUNKING_PROMPT.format(text=text),
        response_model=ChunkingResponse,
        fallback=ChunkingResponse(chunks=[]),
        max_tokens=config.LLM_MAX_TOKENS,
        max_retries=1,
        assistant_prefix='{"chunks": [',
    )
    if result.success:
        chunks = [c for c in result.value.chunks if c.text.strip() and c.text.strip() != "..."]
    else:
        log.warning("LLM chunking failed: %s. Using whole text.", result.error)
        chunks = []

    if not chunks:
        chunks = [ChunkItem(text=text, key_concept="full_content")]

    texts = [c.text for c in chunks]
    embeddings = embedder.embed_documents(texts)
    derivatives: list[DerivativeWithEmbedding] = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
        node = DerivativeNode(
            uid=str(uuid.uuid5(uuid.NAMESPACE_OID, f"{episode_uid}:{i}")),
            source_episode_uid=episode_uid,
            text=chunk.text,
            key_concept=chunk.key_concept,
            sequence_num=i,
        )
        derivatives.append(DerivativeWithEmbedding(node=node, embedding=emb))

    log.debug("Chunked episode %s into %d derivatives", episode_uid[:8], len(derivatives))
    return derivatives
