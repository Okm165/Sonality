"""LLM-based event boundary detection for conversation segmentation.

Replaces cosine-distance threshold approaches with LLM contextual analysis
to identify topic shifts, goal changes, and explicit transitions.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from pydantic import BaseModel

from ..llm.caller import llm_call
from ..llm.prompts import BOUNDARY_DETECTION_PROMPT

log = logging.getLogger(__name__)


class BoundaryDetectionResponse(BaseModel):
    is_boundary: bool
    confidence: float = 0.0
    boundary_type: str = "none"
    reasoning: str = ""
    suggested_segment_label: str | None = None


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    is_boundary: bool
    segment_id: str
    label: str = ""
    boundary_type: str = "none"
    reasoning: str = ""


class EventBoundaryDetector:
    """LLM-based conversation boundary detector.

    Maintains a sliding window of recent messages and uses LLM to determine
    whether each new message represents a significant topic boundary.
    """

    def __init__(self) -> None:
        self._recent_messages: deque[str] = deque(maxlen=5)
        self._current_segment_id: str = "segment_0"
        self._segment_counter: int = 0

    @property
    def current_segment_id(self) -> str:
        return self._current_segment_id

    def check_boundary(self, message: str) -> BoundaryResult:
        """Check if the message represents a conversation boundary.

        Returns BoundaryResult with is_boundary=True and a new segment_id
        if a significant boundary is detected.
        """
        recent_context = "\n".join(self._recent_messages) if self._recent_messages else "No previous context"

        prompt = BOUNDARY_DETECTION_PROMPT.format(
            recent_context=recent_context,
            current_message=message,
        )
        result = llm_call(
            prompt=prompt,
            response_model=BoundaryDetectionResponse,
            fallback=BoundaryDetectionResponse(is_boundary=False),
        )

        self._recent_messages.append(message)

        if result.success and result.value:
            response = result.value
            assert isinstance(response, BoundaryDetectionResponse)
            if response.is_boundary and response.confidence > 0.6:
                self._segment_counter += 1
                self._current_segment_id = f"segment_{self._segment_counter}"
                self._recent_messages.clear()
                log.info(
                    "Boundary detected: %s (%s, conf=%.2f)",
                    response.suggested_segment_label,
                    response.boundary_type,
                    response.confidence,
                )
                return BoundaryResult(
                    is_boundary=True,
                    segment_id=self._current_segment_id,
                    label=response.suggested_segment_label or "",
                    boundary_type=response.boundary_type,
                    reasoning=response.reasoning,
                )

        return BoundaryResult(
            is_boundary=False,
            segment_id=self._current_segment_id,
        )
