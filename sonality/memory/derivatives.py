"""LLM-based semantic chunking for episode derivatives.

Replaces regex/tokenization-based chunking with LLM semantic understanding.
Each episode is split into 1-15 self-contained derivative chunks for granular
embedding and retrieval (MemMachine derivative model).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pydantic import BaseModel

from ..llm.caller import llm_call
from ..llm.prompts import CHUNKING_PROMPT
from .embedder import ExternalEmbedder
from .graph import DerivativeNode

log = logging.getLogger(__name__)


class ChunkItem(BaseModel):
    text: str
    key_concept: str
    importance: str = "medium"


class ChunkingResponse(BaseModel):
    chunks: list[ChunkItem]


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
        chunks = self._llm_chunk(text)
        if not chunks:
            # Fallback: treat entire text as single chunk
            chunks = [ChunkItem(text=text, key_concept="full_content", importance="medium")]

        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_documents(texts)

        results: list[DerivativeWithEmbedding] = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings, strict=True)):
            deriv_uid = f"{episode_uid}_d{i}"
            node = DerivativeNode(
                uid=deriv_uid,
                source_episode_uid=episode_uid,
                text=chunk.text,
                key_concept=chunk.key_concept,
                sequence_num=i,
            )
            results.append(DerivativeWithEmbedding(node=node, embedding=embedding))

        log.debug("Chunked episode %s into %d derivatives", episode_uid[:8], len(results))
        return results

    def _llm_chunk(self, text: str) -> list[ChunkItem]:
        """Use LLM to split text into semantic chunks."""
        # Short texts don't need chunking
        if len(text) < 100:
            return [ChunkItem(text=text, key_concept="brief_content", importance="medium")]

        prompt = CHUNKING_PROMPT.format(text=text)
        result = llm_call(
            prompt=prompt,
            response_model=ChunkingResponse,
            fallback=ChunkingResponse(chunks=[]),
        )
        if result.success and result.value:
            response = result.value
            assert isinstance(response, ChunkingResponse)
            return response.chunks
        log.warning("LLM chunking failed: %s. Using whole text.", result.error)
        return []
