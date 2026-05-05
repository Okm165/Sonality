"""LLM-based event boundary detection for conversation segmentation.

Uses contextual analysis to identify topic shifts, goal changes, and
explicit transitions between conversation segments.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel

from .. import config
from ..llm.caller import llm_call
from ..prompts import BOUNDARY_DETECTION_PROMPT

log = logging.getLogger(__name__)


class BoundaryDecision(StrEnum):
    """Whether the current message starts a new conversation segment."""

    BOUNDARY = "BOUNDARY"
    CONTINUE = "CONTINUE"


class BoundaryDetectionResponse(BaseModel):
    """LLM-returned boundary classification."""

    boundary_decision: BoundaryDecision = BoundaryDecision.CONTINUE
    confidence: float = 0.0
    boundary_type: str = "none"
    reasoning: str = ""
    suggested_segment_label: str = ""


@dataclass(frozen=True, slots=True)
class BoundaryResult:
    """Result of boundary check: decision + current segment identifier.

    label is only populated when boundary_decision is BOUNDARY.
    """

    boundary_decision: BoundaryDecision
    segment_id: str
    label: str = ""


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

    def set_segment_counter(self, counter: int) -> None:
        """Restore persisted segment numbering after restart."""
        self._segment_counter = max(counter, 0)
        self._current_segment_id = f"segment_{self._segment_counter}"

    def check_boundary(self, message: str) -> BoundaryResult:
        """Check if the message represents a conversation boundary.

        Returns BoundaryResult with boundary_decision=BOUNDARY and a new segment_id
        if a significant boundary is detected.
        """
        # First interaction is trivially a new segment — skip LLM call.
        if not self._recent_messages:
            self._recent_messages.append(message)
            self._segment_counter += 1
            self._current_segment_id = f"segment_{self._segment_counter}"
            log.debug("Boundary: first message, starting segment_%d", self._segment_counter)
            return BoundaryResult(
                boundary_decision=BoundaryDecision.BOUNDARY,
                segment_id=self._current_segment_id,
            )

        recent_context = "\n".join(self._recent_messages)
        prompt = BOUNDARY_DETECTION_PROMPT.format(
            recent_context=recent_context,
            current_message=message,
        )
        result = llm_call(
            prompt=prompt,
            response_model=BoundaryDetectionResponse,
            fallback=BoundaryDetectionResponse(),
            model=config.STRUCTURED_MODEL,
        )
        if not result.success:
            log.warning(
                "Boundary detection parse failed (using CONTINUE fallback): %s", result.error
            )
        self._recent_messages.append(message)
        response = result.value
        if response.boundary_decision is BoundaryDecision.BOUNDARY:
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
                boundary_decision=BoundaryDecision.BOUNDARY,
                segment_id=self._current_segment_id,
                label=response.suggested_segment_label or "",
            )

        return BoundaryResult(
            boundary_decision=BoundaryDecision.CONTINUE,
            segment_id=self._current_segment_id,
        )
