"""Derivative chunking logic tests (no LLM calls)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from sonality.memory.derivatives import ChunkingResponse, ChunkItem


class TestChunkingResponseNormalization:
    def test_dict_with_chunks_passthrough(self) -> None:
        data = {"chunks": [{"text": "foo", "key_concept": "bar"}]}
        resp = ChunkingResponse.model_validate(data)
        assert len(resp.chunks) == 1
        assert resp.chunks[0].text == "foo"

    def test_bare_list_of_dicts_wrapped(self) -> None:
        data = [{"text": "alpha", "key_concept": "k1"}, {"text": "beta", "key_concept": "k2"}]
        resp = ChunkingResponse.model_validate(data)
        assert len(resp.chunks) == 2

    def test_single_chunk_dict_without_chunks_key(self) -> None:
        data = {"text": "solo chunk", "key_concept": "solo"}
        resp = ChunkingResponse.model_validate(data)
        assert len(resp.chunks) == 1


class TestChunkAndEmbed:
    def test_falls_back_on_llm_failure(self, monkeypatch) -> None:
        from sonality.caller import LLMCallResult
        from sonality.memory.derivatives import chunk_and_embed

        async def failing_call(**_: object) -> LLMCallResult[ChunkingResponse]:
            return LLMCallResult(
                value=ChunkingResponse(chunks=[]),
                success=False,
                error="Provider timeout",
                attempts=1,
                raw_text="",
            )

        monkeypatch.setattr("sonality.memory.derivatives.async_llm_call", failing_call)
        monkeypatch.setattr(
            "sonality.memory.derivatives.async_embed_documents",
            AsyncMock(return_value=[[0.1] * 4]),
        )

        embedder = AsyncMock()
        results = asyncio.run(chunk_and_embed(embedder, "Some long text about science.", "ep-001"))
        assert len(results) == 1
        assert results[0].node.key_concept == "full_content"

    def test_filters_placeholder_chunks(self, monkeypatch) -> None:
        from sonality.caller import LLMCallResult
        from sonality.memory.derivatives import chunk_and_embed

        payload = ChunkingResponse(
            chunks=[
                ChunkItem(text="Real content", key_concept="real"),
                ChunkItem(text="...", key_concept="placeholder"),
                ChunkItem(text="  ", key_concept="whitespace"),
            ]
        )
        mock_result = LLMCallResult[ChunkingResponse](
            value=payload, success=True, attempts=1, raw_text=""
        )

        async def ok_call(**_: object) -> LLMCallResult[ChunkingResponse]:
            return mock_result

        monkeypatch.setattr("sonality.memory.derivatives.async_llm_call", ok_call)
        monkeypatch.setattr(
            "sonality.memory.derivatives.async_embed_documents",
            AsyncMock(return_value=[[0.1] * 4]),
        )

        embedder = AsyncMock()
        results = asyncio.run(chunk_and_embed(embedder, "Some text.", "ep-002"))
        assert len(results) == 1
        assert results[0].node.text == "Real content"
