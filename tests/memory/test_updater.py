from __future__ import annotations

from collections.abc import Callable

from sonality.ess import (
    ESSResult,
    InternalConsistencyStatus,
    OpinionDirection,
    ReasoningType,
    SourceReliability,
)
from sonality.memory.updater import extract_insight


def _ess() -> ESSResult:
    return ESSResult(
        score=0.7,
        reasoning_type=ReasoningType.LOGICAL_ARGUMENT,
        source_reliability=SourceReliability.INFORMED_OPINION,
        internal_consistency=InternalConsistencyStatus.CONSISTENT,
        novelty=0.5,
        topics=("topic",),
        summary="summary",
        opinion_direction=OpinionDirection.NEUTRAL,
    )


def test_extract_insight_returns_text_when_decision_extract(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "What identity-forming observation emerged": {
                "insight_decision": "EXTRACT",
                "insight_text": "Prefers explicit reasoning over anecdotes.",
            }
        }
    )
    text = extract_insight(_ess(), "user text", "agent text")
    assert text == "Prefers explicit reasoning over anecdotes."


def test_extract_insight_returns_empty_on_invalid_payload(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    """LLM parse failure degrades gracefully — returns empty string, no exception."""
    mock_llm_call({})
    result = extract_insight(_ess(), "user text", "agent text")
    assert result == ""
