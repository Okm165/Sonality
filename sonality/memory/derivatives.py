"""LLM-based semantic chunking for episode derivatives.

Each episode is split into 1-15 self-contained derivative chunks for granular
embedding and retrieval (MemMachine derivative model).
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from pydantic import BaseModel, model_validator

from ..llm.caller import llm_call
from ..llm.parse import normalize_llm_list_response
from ..prompts import CHUNKING_PROMPT
from .embedder import Embedder
from .graph import DerivativeNode

log = logging.getLogger(__name__)


class ChunkItem(BaseModel):
    """A single semantic chunk produced by LLM chunking."""

    text: str
    key_concept: str = ""


class ChunkingResponse(BaseModel):
    """LLM response containing semantic chunks for an episode."""

    chunks: list[ChunkItem]

    @model_validator(mode="before")
    @classmethod
    def normalize_chunks(cls, data: object) -> object:
        return normalize_llm_list_response(
            data,
            list_key="chunks",
            item_required_key="text",
            item_filter=lambda x: (
                isinstance(x, (ChunkItem, dict))
                and (isinstance(x, ChunkItem) or (isinstance(x, dict) and "text" in x))
            ),
        )


@dataclass(frozen=True, slots=True)
class DerivativeWithEmbedding:
    """A derivative chunk with its pre-computed dense embedding."""

    node: DerivativeNode
    embedding: list[float]


def chunk_and_embed(
    embedder: Embedder, text: str, episode_uid: str
) -> list[DerivativeWithEmbedding]:
    """Split text into semantic chunks and embed each one.

    Derivative UIDs are deterministic (uuid5 of episode_uid:index) so
    re-processing the same episode produces identical UIDs.
    """
    result = llm_call(
        prompt=CHUNKING_PROMPT.format(text=text),
        response_model=ChunkingResponse,
        fallback=ChunkingResponse(chunks=[]),
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
