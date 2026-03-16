"""Unit tests for derivative chunking logic (no LLM calls)."""

from __future__ import annotations

import pytest

from sonality.memory.derivatives import ChunkImportance, ChunkingResponse, ChunkItem


class TestChunkItemImportanceCoercion:
    """ChunkItem.coerce_importance accepts placeholders and slash-separated options."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("high", ChunkImportance.HIGH),
            ("medium", ChunkImportance.MEDIUM),
            ("low", ChunkImportance.LOW),
            # Slash-separated — first token wins
            ("high/medium/low", ChunkImportance.HIGH),
            ("medium/low", ChunkImportance.MEDIUM),
            # Template placeholders → MEDIUM fallback
            ("...", ChunkImportance.MEDIUM),
            ("", ChunkImportance.MEDIUM),
            ("none", ChunkImportance.MEDIUM),
        ],
    )
    def test_coerce_importance(self, raw: str, expected: ChunkImportance) -> None:
        item = ChunkItem(text="hello", importance=raw)  # type: ignore[arg-type]
        assert item.importance is expected


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
        assert item.importance is ChunkImportance.MEDIUM


class TestChunkingResponseFallbackPath:
    """Verify the no-LLM fallback (empty chunks → single full_content chunk)."""

    def test_chunk_and_embed_falls_back_on_llm_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When llm_call fails, DerivativeChunker falls back to single full-text chunk."""
        from pydantic import BaseModel

        from sonality.llm.caller import LLMCallResult
        from sonality.memory.derivatives import DerivativeChunker

        def fake_embed(texts: list[str]) -> list[list[float]]:
            return [[0.1] * 4 for _ in texts]

        class FakeEmbedder:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return fake_embed(texts)

        def failing_call[T: BaseModel](
            *,
            prompt: str,
            response_model: type[T],
            fallback: T,
            **_: object,
        ) -> LLMCallResult[T]:
            del prompt
            return LLMCallResult(
                value=fallback,
                success=False,
                error="Provider timeout",
                attempts=1,
                raw_text="",
            )

        monkeypatch.setattr("sonality.memory.derivatives.llm_call", failing_call)

        chunker = DerivativeChunker(FakeEmbedder())  # type: ignore[arg-type]
        results = chunker.chunk_and_embed("Some long text about science.", "ep-001")

        assert len(results) == 1
        assert results[0].node.key_concept == "full_content"
        assert "Some long text" in results[0].node.text

    def test_chunk_and_embed_filters_placeholder_chunks(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Chunks with '...' text are filtered out; only non-empty chunks are kept."""
        from pydantic import BaseModel

        from sonality.llm.caller import LLMCallResult
        from sonality.memory.derivatives import ChunkingResponse, ChunkItem, DerivativeChunker

        def fake_embed(texts: list[str]) -> list[list[float]]:
            return [[0.1] * 4 for _ in texts]

        class FakeEmbedder:
            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                return fake_embed(texts)

        def stub_call[T: BaseModel](
            *,
            prompt: str,
            response_model: type[T],
            fallback: T,
            **_: object,
        ) -> LLMCallResult[T]:
            del prompt, fallback
            payload = ChunkingResponse(
                chunks=[
                    ChunkItem(text="Real content", key_concept="real"),
                    ChunkItem(text="...", key_concept="placeholder"),
                    ChunkItem(text="  ", key_concept="whitespace"),
                ]
            )
            return LLMCallResult(
                value=response_model.model_validate(payload.model_dump()),  # type: ignore[arg-type]
                success=True,
                attempts=1,
                raw_text="",
            )

        monkeypatch.setattr("sonality.memory.derivatives.llm_call", stub_call)

        chunker = DerivativeChunker(FakeEmbedder())  # type: ignore[arg-type]
        results = chunker.chunk_and_embed("Some text.", "ep-002")

        assert len(results) == 1
        assert results[0].node.text == "Real content"
