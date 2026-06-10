"""Agent progress events for real-time UX streaming.

Events are emitted during the agentic loop (agent.py) and serialized as
SSE events by the API (api.py). Chat clients consume them to render live
progress: which tool is running, when the agent is thinking, etc.
"""

from __future__ import annotations

from dataclasses import dataclass

from .schema import EventType


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """Single event in the agent's processing stream.

    Emitted at each significant step so UIs can render real-time progress.
    """

    type: EventType
    detail: str = ""
    tool_name: str = ""
    tool_args: str = ""
    tool_result_summary: str = ""
    iteration: int = 0
    sources_count: int = 0


def noop_progress(_event: AgentEvent) -> None:
    """No-op callback for code paths that don't need progress reporting."""
