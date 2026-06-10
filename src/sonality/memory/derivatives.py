"""LLM-based semantic chunking for episode derivatives.

Each episode is split into 1-15 self-contained derivative chunks for granular
embedding and retrieval (MemMachine derivative model).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from pydantic import BaseModel, Field, model_validator

from shared.embedder import Embedder
from shared.llm.parse import normalize_llm_list_response
from shared.types import deterministic_id

from .. import config
from ..caller import async_embed_documents, async_llm_call, format_prompt
from ..prompts import CHUNKING_PROMPT
from .graph import DerivativeNode

log = structlog.get_logger(__name__)

_MAX_CHUNKS = 15


class ChunkItem(BaseModel):
    """A single semantic chunk produced by LLM chunking."""

    text: str = Field(max_length=5000)
    key_concept: str = Field(default="", max_length=200)


class ChunkingResponse(BaseModel):
    """LLM response containing semantic chunks for an episode."""

    chunks: list[ChunkItem] = Field(max_length=_MAX_CHUNKS)

    @model_validator(mode="before")
    @classmethod
    def normalize_chunks(cls, data: object) -> object:
        data = normalize_llm_list_response(
            data,
            list_key="chunks",
            item_required_key="text",
            item_filter=lambda x: (
                isinstance(x, (dict, BaseModel))
                and bool(x.get("text") if isinstance(x, dict) else getattr(x, "text", None))
            ),
        )
        if isinstance(data, dict) and isinstance(data.get("chunks"), list):
            data["chunks"] = data["chunks"][:_MAX_CHUNKS]
        return data


@dataclass(frozen=True, slots=True)
class DerivativeWithEmbedding:
    """A derivative chunk with its pre-computed dense embedding."""

    node: DerivativeNode
    embedding: list[float]


async def chunk_and_embed(
    embedder: Embedder, text: str, episode_uid: str
) -> list[DerivativeWithEmbedding]:
    """Split text into semantic chunks and embed each one.

    LLM call gated by sonality's ``_llm_gate``; embeddings gated by
    ``_embedding_gate``.  Derivative UIDs are deterministic (uuid5 of
    episode_uid:index) so re-processing produces identical UIDs.
    """
    result = await async_llm_call(
        instructions=format_prompt(CHUNKING_PROMPT, text=text),
        response_model=ChunkingResponse,
        fallback=ChunkingResponse(chunks=[]),
        model=config.settings.fast_model,
    )
    if result.success:
        chunks = [c for c in result.value.chunks if c.text.strip() and c.text.strip() != "..."]
    else:
        log.warning(
            "derivative_chunking_failed",
            error=(result.error or "")[:80],
        )
        chunks = []

    if not chunks:
        if not text.strip():
            return []
        chunks = [ChunkItem(text=text, key_concept="full_content")]
    elif len(chunks) > _MAX_CHUNKS:
        log.warning("derivative_chunk_cap", raw=len(chunks), cap=_MAX_CHUNKS)
        chunks = chunks[:_MAX_CHUNKS]

    texts = [c.text for c in chunks]
    embeddings = await async_embed_documents(embedder, texts)
    derivatives: list[DerivativeWithEmbedding] = []

    for i, (chunk, emb) in enumerate(zip(chunks, embeddings, strict=True)):
        node = DerivativeNode(
            uid=deterministic_id(f"{episode_uid}:{i}"),
            source_episode_uid=episode_uid,
            text=chunk.text,
            key_concept=chunk.key_concept,
            sequence_num=i,
        )
        derivatives.append(DerivativeWithEmbedding(node=node, embedding=emb))

    log.debug(
        "episode_derivatives_chunked",
        episode_uid=episode_uid[:8],
        derivative_count=len(derivatives),
    )
    return derivatives
