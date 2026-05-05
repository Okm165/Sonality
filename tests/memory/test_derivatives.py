"""Unit tests for derivative chunking logic (no LLM calls)."""

from __future__ import annotations

import pytest

from sonality.memory.derivatives import ChunkingResponse, ChunkItem


class _FakeEmbedder:
    """Stub embedder returning fixed-dimension vectors."""

    def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[0.1] * 4 for _ in documents]


class TestChunkingResponseNormalization:
    """ChunkingResponse.normalize_chunks handles all LLM output variants."""

    def test_dict_with_chunks_passthrough(self) -> None:
        data = {"chunks": [{"text": "foo", "key_concept": "bar"}]}
        resp = ChunkingResponse.model_validate(data)
        assert len(resp.chunks) == 1
        assert resp.chunks[0].text == "foo"

    def test_bare_list_of_dicts_wrapped(self) -> None:
        """LLM omits 'chunks' wrapper and returns a bare list of chunk objects."""
        data = [{"text": "alpha", "key_concept": "k1"}, {"text": "beta", "key_concept": "k2"}]
        resp = ChunkingResponse.model_validate(data)
        assert len(resp.chunks) == 2
        assert resp.chunks[0].text == "alpha"
        assert resp.chunks[1].text == "beta"

    def test_single_chunk_dict_without_chunks_key(self) -> None:
        """LLM returns a single chunk object directly (no list, no 'chunks' key)."""
        data = {"text": "solo chunk", "key_concept": "solo"}
        resp = ChunkingResponse.model_validate(data)
        assert len(resp.chunks) == 1
        assert resp.chunks[0].text == "solo chunk"

    def test_empty_chunks_list(self) -> None:
        resp = ChunkingResponse.model_validate({"chunks": []})
        assert resp.chunks == []

    def test_fallback_chunk_item_defaults(self) -> None:
        item = ChunkItem(text="just text")
        assert item.key_concept == ""


class TestChunkingResponseFallbackPath:
    """Verify the no-LLM fallback (empty chunks → single full_content chunk)."""

    def test_chunk_and_embed_falls_back_on_llm_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When llm_call fails, DerivativeChunker falls back to single full-text chunk."""
        from pydantic import BaseModel

        from sonality.llm.caller import LLMCallResult
        from sonality.memory.derivatives import chunk_and_embed

        def failing_call[T: BaseModel](
            *,
            prompt: str,
            response_model: type[T],
            fallback: T,
            **_: object,
        ) -> LLMCallResult[T]:
            del prompt, response_model
            return LLMCallResult(
                value=fallback,
                success=False,
                error="Provider timeout",
                attempts=1,
                raw_text="",
            )

        monkeypatch.setattr("sonality.memory.derivatives.llm_call", failing_call)

        results = chunk_and_embed(_FakeEmbedder(), "Some long text about science.", "ep-001")

        assert len(results) == 1
        assert results[0].node.key_concept == "full_content"
        assert "Some long text" in results[0].node.text

    def test_chunk_and_embed_filters_placeholder_chunks(self) -> None:
        """Chunks with '...' text are filtered out; only non-empty chunks are kept."""
        from unittest.mock import patch

        from sonality.llm.caller import LLMCallResult
        from sonality.memory.derivatives import ChunkingResponse, ChunkItem, chunk_and_embed

        payload = ChunkingResponse(
            chunks=[
                ChunkItem(text="Real content", key_concept="real"),
                ChunkItem(text="...", key_concept="placeholder"),
                ChunkItem(text="  ", key_concept="whitespace"),
            ]
        )
        mock_result = LLMCallResult[ChunkingResponse](
            value=payload,
            success=True,
            attempts=1,
            raw_text="",
        )

        with patch("sonality.memory.derivatives.llm_call", return_value=mock_result):
            results = chunk_and_embed(_FakeEmbedder(), "Some text.", "ep-002")

        assert len(results) == 1
        assert results[0].node.text == "Real content"
