"""Deterministic ESS parsing and coercion behavior tests."""

from __future__ import annotations

from collections.abc import Mapping
from unittest.mock import MagicMock, patch

from sonality.ess import (
    InternalConsistencyStatus,
    OpinionDirection,
    ReasoningType,
    SourceReliability,
    classify,
)
from sonality.provider import ChatResult


def _mock_completion(payloads: list[Mapping[str, object]]) -> MagicMock:
    """Create a mock that returns tool_call responses from payloads."""
    state = {"calls": 0}

    def side_effect(**_: object) -> ChatResult:
        index = min(state["calls"], len(payloads) - 1)
        state["calls"] += 1
        return ChatResult(
            text="",
            input_tokens=11,
            output_tokens=7,
            raw={
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "classify_evidence",
                                        "arguments": payloads[index],
                                    }
                                }
                            ]
                        }
                    }
                ]
            },
        )

    mock = MagicMock(side_effect=side_effect)
    return mock


def test_classify_normalizes_labels_and_boolean_strings() -> None:
    """Test that classify normalizes labels and boolean strings."""
    payload = {
        "score": "0.72",
        "reasoning_type": "Logical Argument",
        "source_reliability": "Peer Reviewed",
        "internal_consistency": "false",
        "novelty": "0.35",
        "topics": "governance, open_source",
        "summary": "Structured governance evidence.",
        "opinion_direction": "Support",
    }
    with patch("sonality.ess.default_provider") as mock_provider:
        mock_provider.chat_completion = _mock_completion([payload])
        result = classify("message", "snapshot")

    assert result.score == 0.72
    assert result.novelty == 0.35
    assert result.reasoning_type == ReasoningType.LOGICAL_ARGUMENT
    assert result.source_reliability == SourceReliability.PEER_REVIEWED
    assert result.opinion_direction == OpinionDirection.SUPPORTS
    assert result.internal_consistency is InternalConsistencyStatus.INCONSISTENT
    assert result.topics == ("governance", "open_source")
    assert result.attempt_count == 1
    assert result.input_tokens == 11
    assert result.output_tokens == 7
    assert not result.used_defaults
    assert result.defaulted_fields == ()
    assert result.default_severity == "none"


def test_classify_marks_defaults_on_invalid_fields() -> None:
    """Test that classify marks defaults on invalid fields."""
    payload = {
        "score": "not-a-number",
        "reasoning_type": "vibes_only",
        "source_reliability": "trust_me_bro",
        "internal_consistency": "unknown",
        "novelty": "n/a",
        "topics": ["policy"],
        "summary": "Unreliable claim",
        "opinion_direction": "mixedish",
    }
    with patch("sonality.ess.default_provider") as mock_provider:
        mock_provider.chat_completion = _mock_completion([payload])
        result = classify("message", "snapshot")

    assert result.score == 0.0
    assert result.novelty == 0.0
    assert result.reasoning_type == ReasoningType.NO_ARGUMENT
    assert result.source_reliability == SourceReliability.NOT_APPLICABLE
    assert result.opinion_direction == OpinionDirection.NEUTRAL
    assert result.internal_consistency is InternalConsistencyStatus.CONSISTENT
    assert result.topics == ("policy",)
    assert result.used_defaults
    assert "coerced:score" in result.defaulted_fields
    assert "coerced:reasoning_type" in result.defaulted_fields
    assert "coerced:source_reliability" in result.defaulted_fields
    assert "coerced:opinion_direction" in result.defaulted_fields
    assert result.default_severity == "coercion"


def test_classify_retries_on_malformed_required_fields() -> None:
    """Test that classify retries on malformed required fields."""
    payloads: list[Mapping[str, object]] = [
        {
            "score": "0.71",
            "reasoning_type": "vibes_only",
            "source_reliability": "peer_reviewed",
            "internal_consistency": True,
            "novelty": 0.4,
            "topics": ["governance"],
            "summary": "First pass with malformed required enum.",
            "opinion_direction": "supports",
        },
        {
            "score": "0.71",
            "reasoning_type": "logical_argument",
            "source_reliability": "peer_reviewed",
            "internal_consistency": True,
            "novelty": 0.4,
            "topics": ["governance"],
            "summary": "Second pass corrected enum.",
            "opinion_direction": "supports",
        },
    ]
    with patch("sonality.ess.default_provider") as mock_provider:
        mock_provider.chat_completion = _mock_completion(payloads)
        result = classify("message", "snapshot")

    assert result.attempt_count == 2
    assert result.reasoning_type == ReasoningType.LOGICAL_ARGUMENT
    assert not result.used_defaults
    assert result.defaulted_fields == ()
    assert result.default_severity == "none"


def test_classify_debunked_claim_aliases_resolve() -> None:
    """debunked, misinformation, conspiracy all map to debunked_claim."""
    for alias in ("debunked", "misinformation", "conspiracy", "debunked_claim"):
        payload = {
            "score": "0.04",
            "reasoning_type": alias,
            "source_reliability": "unverified_claim",
            "internal_consistency": True,
            "novelty": "0.1",
            "topics": ["climate", "conspiracy"],
            "summary": "Debunked conspiracy theory.",
            "opinion_direction": "opposes",
        }
        with patch("sonality.ess.default_provider") as mock_provider:
            mock_provider.chat_completion = _mock_completion([payload])
            result = classify("message", "snapshot")

        assert result.reasoning_type == ReasoningType.DEBUNKED_CLAIM, (
            f"alias={alias!r} did not resolve to debunked_claim, got {result.reasoning_type!r}"
        )


def test_classify_marks_missing_when_required_field_absent_after_retries() -> None:
    """Test that classify marks missing when required field absent after retries."""
    payload = {
        "score": "0.55",
        "source_reliability": "informed_opinion",
        "internal_consistency": True,
        "novelty": 0.2,
        "topics": ["policy"],
        "summary": "Missing required field",
        "opinion_direction": "neutral",
    }
    with patch("sonality.ess.default_provider") as mock_provider:
        mock_provider.chat_completion = _mock_completion([payload])
        result = classify("message", "snapshot")

    assert result.attempt_count == 2
    assert result.used_defaults
    assert "missing:reasoning_type" in result.defaulted_fields
    assert result.default_severity == "missing"
