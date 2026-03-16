"""Unit tests for provider.py timeout and error handling (no live API calls)."""

from __future__ import annotations

from urllib.error import URLError

import pytest


class TestTimeoutFailsFast:
    """TimeoutError should raise immediately without retry waits."""

    def test_urlopen_timeout_in_urlerror_raises_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """URLError wrapping TimeoutError propagates as RuntimeError with no retry sleep."""
        import time

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

    def test_direct_timeout_error_raises_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Bare TimeoutError (not wrapped in URLError) propagates as RuntimeError."""
        import time

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

    def test_connection_error_retries_with_backoff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
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

    def test_dns_failure_raises_immediately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """DNS resolution failure raises immediately without retrying (name resolution)."""
        import time

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
