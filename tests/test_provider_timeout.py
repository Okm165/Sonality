"""Unit tests for provider.py timeout and error handling (no live API calls)."""

from __future__ import annotations

import time
from urllib.error import URLError

import pytest

from sonality.schema import SemanticCategory


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
        with pytest.raises(RuntimeError, match="Provider timeout"):
            provider.default_provider._post_json("/chat/completions", {"model": "test"})
        elapsed = time.perf_counter() - start

        assert call_count == 1, f"Expected 1 attempt but got {call_count}"
        assert elapsed < 1.0, f"Elapsed {elapsed:.2f}s suggests retry sleep was called"

    def test_direct_timeout_error_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Bare TimeoutError (not wrapped in URLError) retries before failing."""
        from sonality import provider

        call_count = 0

        def raising_urlopen(*_args: object, **_kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            raise TimeoutError("socket timed out")

        monkeypatch.setattr(provider, "urlopen", raising_urlopen)
        sleep_calls: list[float] = []
        monkeypatch.setattr(provider.time, "sleep", lambda s: sleep_calls.append(s))

        with pytest.raises(RuntimeError, match="Provider transport error"):
            provider.default_provider._post_json("/chat/completions", {"model": "test"})

        assert call_count == 3
        assert len(sleep_calls) == 2

    def test_connection_error_retries_with_backoff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ConnectionError is retried — verifies retry policy."""
        from sonality import provider

        call_count = 0

        def raising_urlopen(*_args: object, **_kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            raise ConnectionError("connection refused")

        monkeypatch.setattr(provider, "urlopen", raising_urlopen)
        sleep_calls: list[float] = []
        monkeypatch.setattr(provider.time, "sleep", lambda s: sleep_calls.append(s))

        with pytest.raises(RuntimeError, match="Provider transport error"):
            provider.default_provider._post_json("/chat/completions", {"model": "test"})

        assert call_count == 3
        assert len(sleep_calls) == 2

    def test_dns_failure_raises_immediately(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """DNS resolution failure raises immediately without retrying."""
        from sonality import provider

        call_count = 0

        def raising_urlopen(*_args: object, **_kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            # Create URLError with reason containing "name resolution"
            exc = URLError("name resolution failed")
            exc.reason = "Temporary failure in name resolution"
            raise exc

        monkeypatch.setattr(provider, "urlopen", raising_urlopen)

        start = time.perf_counter()
        with pytest.raises(RuntimeError, match="Provider DNS failure"):
            provider.default_provider._post_json("/chat/completions", {"model": "test"})
        elapsed = time.perf_counter() - start

        assert call_count == 1
        assert elapsed < 1.0


class TestLLMSemaphoreContention:
    """Background workers must respect llm_semaphore_idle() to avoid preempting foreground calls."""

    def test_semaphore_idle_when_free(self) -> None:
        from sonality.provider import llm_semaphore_idle

        assert llm_semaphore_idle() is True

    def test_semaphore_busy_when_held(self) -> None:
        import threading

        from sonality import provider

        barrier = threading.Barrier(2)
        released = threading.Event()

        def hold_semaphore() -> None:
            with provider.default_provider._semaphore:
                barrier.wait()
                released.wait(timeout=2.0)

        t = threading.Thread(target=hold_semaphore, daemon=True)
        t.start()
        barrier.wait()
        try:
            assert provider.llm_semaphore_idle() is False
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

        with (
            provider.default_provider._semaphore,
            patch("sonality.memory.semantic_features.llm_call", side_effect=fake_llm_call),
        ):
            worker._extract_features("ep-uid-test", "some content", SemanticCategory.PERSONALITY)

        assert llm_call_count == 0

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

        with (
            patch("sonality.memory.semantic_features.llm_call", side_effect=fake_llm_call),
            patch.object(worker, "_load_existing_features", return_value=""),
        ):
            worker._extract_features("ep-uid-test", "some content", SemanticCategory.PERSONALITY)

        assert llm_call_count == 1
