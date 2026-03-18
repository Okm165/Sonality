"""Unit tests for provider.py timeout and error handling (no live API calls)."""

from __future__ import annotations

import threading
import time
from urllib.error import URLError

import pytest


class TestTimeoutFailsFast:
    """TimeoutError should raise immediately without retry waits."""

    def test_urlopen_timeout_in_urlerror_raises_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """URLError wrapping TimeoutError propagates as RuntimeError with no retry sleep."""

        from sonality import provider

        call_count = 0

        def raising_urlopen(*_args: object, **_kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            raise URLError(TimeoutError("timed out"))

        monkeypatch.setattr(provider, "urlopen", raising_urlopen)

        start = time.perf_counter()
        with pytest.raises(RuntimeError, match="Provider transport error"):
            provider._post_json("/chat/completions", {"model": "test"})
        elapsed = time.perf_counter() - start

        # Only 1 attempt because TimeoutError is fail-fast (no retry).
        assert call_count == 1, f"Expected 1 attempt but got {call_count}"
        # No sleep() between attempts means elapsed should be < 1s.
        assert elapsed < 1.0, f"Elapsed {elapsed:.2f}s suggests retry sleep was called"

    def test_direct_timeout_error_raises_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bare TimeoutError (not wrapped in URLError) propagates as RuntimeError."""

        from sonality import provider

        call_count = 0

        def raising_urlopen(*_args: object, **_kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("socket timed out")

        monkeypatch.setattr(provider, "urlopen", raising_urlopen)

        start = time.perf_counter()
        with pytest.raises(RuntimeError, match="Provider transport error"):
            provider._post_json("/chat/completions", {"model": "test"})
        elapsed = time.perf_counter() - start

        assert call_count == 1
        assert elapsed < 1.0

    def test_connection_error_retries_with_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ConnectionError is retried (unlike TimeoutError) — verifies asymmetric policy."""
        from sonality import provider

        call_count = 0

        def raising_urlopen(*_args: object, **_kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("connection refused")

        monkeypatch.setattr(provider, "urlopen", raising_urlopen)
        # Patch sleep to avoid actual delays while still counting retries
        sleep_calls: list[float] = []
        monkeypatch.setattr(provider.time, "sleep", lambda s: sleep_calls.append(s))

        with pytest.raises(RuntimeError, match="Provider transport error"):
            provider._post_json("/chat/completions", {"model": "test"})

        # 3 attempts total (first + 2 retries)
        assert call_count == 3
        # Two backoff sleeps applied between retries
        assert len(sleep_calls) == 2

    def test_dns_failure_raises_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DNS resolution failure raises immediately without retrying (name resolution)."""

        from sonality import provider

        call_count = 0

        def raising_urlopen(*_args: object, **_kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            raise URLError("Name or service not known: errno -3")

        monkeypatch.setattr(provider, "urlopen", raising_urlopen)

        start = time.perf_counter()
        with pytest.raises(RuntimeError, match="Provider network error"):
            provider._post_json("/chat/completions", {"model": "test"})
        elapsed = time.perf_counter() - start

        assert call_count == 1
        assert elapsed < 1.0


class TestLLMSemaphoreContention:
    """Background workers must respect llm_semaphore_idle() to avoid preempting foreground calls."""

    def test_semaphore_idle_when_free(self) -> None:
        from sonality.provider import llm_semaphore_idle

        assert llm_semaphore_idle() is True

    def test_semaphore_busy_when_held(self) -> None:
        from sonality import provider

        barrier = threading.Barrier(2)
        released = threading.Event()

        def hold_semaphore() -> None:
            with provider._LLM_SEMAPHORE:
                barrier.wait()
                released.wait(timeout=2.0)

        t = threading.Thread(target=hold_semaphore, daemon=True)
        t.start()
        barrier.wait()
        try:
            assert provider.llm_semaphore_idle() is False, (
                "Semaphore should appear busy while held by another thread"
            )
        finally:
            released.set()
            t.join(timeout=2.0)

    def test_semantic_worker_skips_when_semaphore_busy(self) -> None:
        """SemanticIngestionWorker._extract_features skips the LLM call when semaphore is busy."""
        from unittest.mock import MagicMock, patch

        from sonality import provider
        from sonality.memory.semantic_features import SemanticIngestionWorker

        embedder = MagicMock()
        worker = SemanticIngestionWorker(qdrant_url="http://localhost:6333", embedder=embedder)

        llm_call_count = 0

        def fake_llm_call(**_kwargs: object) -> object:
            nonlocal llm_call_count
            llm_call_count += 1
            from sonality.llm.caller import LLMCallResult
            from sonality.memory.semantic_features import FeatureExtractionResponse

            return LLMCallResult(value=FeatureExtractionResponse(), success=False, error="skipped")

        # Acquire the semaphore to simulate main thread being busy
        with (
            provider._LLM_SEMAPHORE,
            patch("sonality.memory.semantic_features.llm_call", side_effect=fake_llm_call),
        ):
            worker._extract_features("ep-uid-test", "some content", "personality")

        # The semaphore was held → _extract_features should have skipped without calling llm_call
        assert llm_call_count == 0, (
            f"Expected 0 LLM calls when semaphore is busy but got {llm_call_count}. "
            "SemanticIngestionWorker must check llm_semaphore_idle() before calling LLM."
        )

    def test_semantic_worker_proceeds_when_semaphore_free(self) -> None:
        """SemanticIngestionWorker._extract_features runs the LLM call when semaphore is idle."""
        from unittest.mock import MagicMock, patch

        from sonality.llm.caller import LLMCallResult
        from sonality.memory.semantic_features import (
            FeatureExtractionResponse,
            SemanticIngestionWorker,
        )

        embedder = MagicMock()
        worker = SemanticIngestionWorker(qdrant_url="http://localhost:6333", embedder=embedder)

        llm_call_count = 0

        def fake_llm_call(**_kwargs: object) -> object:
            nonlocal llm_call_count
            llm_call_count += 1
            return LLMCallResult(value=FeatureExtractionResponse(), success=False, error="test")

        # Semaphore is free — worker should attempt the LLM call
        with (
            patch("sonality.memory.semantic_features.llm_call", side_effect=fake_llm_call),
            patch.object(worker, "_load_existing_features", return_value=""),
        ):
            worker._extract_features("ep-uid-test", "some content", "personality")

        assert llm_call_count == 1, (
            f"Expected 1 LLM call when semaphore is free but got {llm_call_count}"
        )


class TestNormalizeSchemaNotation:
    """_normalize_schema_notation correctly coerces LLM schema-template outputs."""

    def test_pipe_separated_enum_keeps_first_option(self) -> None:
        from sonality.provider import _normalize_schema_notation

        text = '{"field": "OPTION_A" | "OPTION_B" | "OPTION_C"}'
        result = _normalize_schema_notation(text)
        assert '"OPTION_A"' in result
        assert '"OPTION_B"' not in result

    def test_ellipsis_placeholder_cleared(self) -> None:
        from sonality.provider import _normalize_schema_notation

        text = '{"field": "...", "other": 1}'
        result = _normalize_schema_notation(text)
        # The "..." value should be normalized to ""
        assert '""' in result

    def test_angle_bracket_float_replaced(self) -> None:
        from sonality.provider import _normalize_schema_notation

        text = '{"score": <float>}'
        result = _normalize_schema_notation(text)
        assert "0.5" in result

    def test_angle_bracket_string_replaced(self) -> None:
        from sonality.provider import _normalize_schema_notation

        text = '{"label": <string>}'
        result = _normalize_schema_notation(text)
        assert '""' in result
