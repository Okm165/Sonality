from __future__ import annotations

import pytest
from pydantic import BaseModel

from sonality.llm.caller import LLMCallResult
from sonality.memory.segmentation import BoundaryDecision, EventBoundaryDetector


def test_boundary_detector_uses_llm_boundary_boolean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call[T: BaseModel](
        *,
        prompt: str,
        response_model: type[T],
        fallback: T,
        **_: object,
    ) -> LLMCallResult[T]:
        del prompt, fallback
        return LLMCallResult(
            value=response_model.model_validate(
                {
                    "boundary_decision": "BOUNDARY",
                    "confidence": 0.12,
                    "boundary_type": "topic_shift",
                    "reasoning": "Hard shift in topic",
                    "suggested_segment_label": "new topic",
                }
            ),
            success=True,
            attempts=1,
            raw_text="",
        )

    monkeypatch.setattr("sonality.memory.segmentation.llm_call", fake_call, raising=False)

    detector = EventBoundaryDetector()
    result = detector.check_boundary("Let's switch to another topic now.")
    assert result.boundary_decision is BoundaryDecision.BOUNDARY
    assert result.segment_id == "segment_1"
    assert result.label == "new topic"
