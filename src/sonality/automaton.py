"""2-phase automaton state machine: data models and pure helpers.

The agentic loop alternates between THINKING (LLM reasoning) and ACTING
(tool dispatch + mandatory consolidation). This module contains the
state representation and stateless helpers — no I/O, no LLM calls.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Final, NamedTuple

import structlog
from pydantic import BaseModel, model_validator

from shared.llm.parse import ParsedToolCall
from shared.types import ChatRole

from . import config
from .caller import ChatResult
from .schema import Phase

log = structlog.get_logger(__name__)

__all__ = [
    "TERMINAL_PHASES",
    "ActingContext",
    "LoopState",
    "LoopStep",
    "MemoryUpdate",
    "build_scaffolding",
    "build_step_context",
    "dedup_tool_calls",
    "summarize_for_step_log",
    "synthesis_prompt",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MemoryUpdate(BaseModel):
    """Structured output from the consolidation LLM call.

    No decision field — the finish signal emerges from STM content
    ("Next: answer the user"), not from an explicit enum.
    """

    long_term_memory: str = ""
    short_term_memory: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_to_strings(cls, data: object) -> object:
        """LLMs sometimes return dicts instead of flat strings for memory fields."""
        if not isinstance(data, dict):
            return data
        for key in ("long_term_memory", "short_term_memory"):
            val = data.get(key)
            if isinstance(val, (dict, list)):
                data[key] = json.dumps(val, indent=2)
        return data


class LoopStep(NamedTuple):
    """Compact record of one tool call for the step-context injector.

    summary    — one-sentence digest used in the compact history (older steps).
    raw_output — tool output prefix (≤2000 chars) fed to the state update for LTM extraction.
    """

    step_index: int
    tool: str
    query: str
    summary: str
    raw_output: str = ""


@dataclasses.dataclass(frozen=True, slots=True)
class ActingContext:
    """Transient data produced in THINKING, consumed in ACTING.

    Frozen: created once per transition, never mutated.
    """

    completion: ChatResult
    calls: tuple[ParsedToolCall, ...]


@dataclasses.dataclass(slots=True)
class LoopState:
    """Per-request mutable state for the agentic loop.

    Isolates loop state from the agent instance so concurrent requests
    never risk cross-contamination.

    Groups:
      Memory    — distilled LTM/STM/beliefs from consolidation
      Context   — ephemeral messages cleared each transition
      History   — append-only tool log and dedup set
      Control   — phase, iteration counter, timing
      Output    — response accumulation for truncated outputs
      Transient — acting-phase data, not checkpointed
    """

    # Memory (distilled)
    long_term_memory: str = ""
    short_term_memory: str = ""
    relevant_beliefs: str = ""
    # Context (ephemeral — cleared after consolidation)
    context_messages: list[dict[str, object]] = dataclasses.field(default_factory=list)
    # History (compact, never cleared)
    tool_history: list[str] = dataclasses.field(default_factory=list)
    step_log: list[LoopStep] = dataclasses.field(default_factory=list)
    recent_calls: set[tuple[str, str]] = dataclasses.field(default_factory=set)
    last_assistant_msg: str = ""
    user_message: str = ""
    # Control
    phase: Phase = Phase.THINKING
    iteration: int = 0
    stall_count: int = 0
    nudged: bool = False
    # Output continuity
    output_buffer: str = ""
    extend_count: int = 0
    # Timing
    loop_start_time: float = 0.0
    run_id: str = ""
    # Transient (THINKING → ACTING, cleared after each cycle)
    acting_ctx: ActingContext | None = None


TERMINAL_PHASES: Final[frozenset[Phase]] = frozenset({Phase.COMPLETED, Phase.FAILED})


# ---------------------------------------------------------------------------
# Pure helpers (no I/O, no self)
# ---------------------------------------------------------------------------


def build_step_context(state: LoopState) -> str:
    """Build structured context block for each THINKING phase.

    Priority: iteration header → STM → LTM → beliefs → tool log (last 8).
    No budget arithmetic — compose_guarded handles compression.
    """
    remaining = config.settings.agent_loop_hard_ceiling - state.iteration - 1
    header = (
        f"[Iteration {state.iteration + 1} / {config.settings.agent_loop_hard_ceiling}"
        f" — {remaining} iterations remaining]"
    )
    sections: list[str] = [header]
    if state.short_term_memory:
        sections.append(f"## Your Plan\n{state.short_term_memory}")
    if state.long_term_memory:
        sections.append(f"## Research Findings\n{state.long_term_memory}")
    if state.relevant_beliefs:
        sections.append(f"## Your Beliefs\n{state.relevant_beliefs}")
    if state.step_log:
        lines = [
            f"  {s.step_index}. {s.tool}({s.query[:50]!r}) → {s.summary}"
            for s in state.step_log[-8:]
        ]
        sections.append("## Actions Taken\n" + "\n".join(lines))
    sections.append(
        "## Discipline Reminder\n"
        "Pick ONE next action. State what sub-question it addresses before calling the tool."
    )
    return "\n\n".join(sections)


def synthesis_prompt(state: LoopState) -> str:
    """Build the synthesis instruction for forced final response (nudge safety net)."""
    parts = [f"Based on your research, answer the user's question: {state.user_message}"]
    if state.long_term_memory:
        parts.append(f"\nYour findings:\n{state.long_term_memory}")
    if state.short_term_memory:
        parts.append(f"\nYour plan:\n{state.short_term_memory}")
    parts.append("\nProvide your best answer now. Do not call any tools.")
    return "\n".join(parts)


def build_scaffolding(system_prompt: str, conv: list[dict[str, str]]) -> list[dict[str, object]]:
    """Build the immutable scaffolding prefix: system prompt + conversation history."""
    scaffolding: list[dict[str, object]] = [{"role": ChatRole.SYSTEM, "content": system_prompt}]
    scaffolding.extend(
        {"role": m.get("role", ChatRole.USER), "content": m.get("content", "")} for m in conv
    )
    return scaffolding


def dedup_tool_calls(
    tool_calls: list[ParsedToolCall],
    recent_calls: set[tuple[str, str]],
) -> list[ParsedToolCall]:
    """Filter exact duplicate tool calls (same name + same args).

    Returns only new calls and registers them in recent_calls (mutates the set).
    """
    fresh: list[ParsedToolCall] = []
    for tc in tool_calls:
        key = (tc.name, json.dumps(tc.args, sort_keys=True))
        if key in recent_calls:
            log.info("dedup_skip", tool=tc.name)
        else:
            recent_calls.add(key)
            fresh.append(tc)
    return fresh


def summarize_for_step_log(result: str) -> str:
    """First sentence or 150-char prefix of tool result for the step log."""
    text = result.strip()[:200]
    for sep in (". ", ".\n"):
        idx = text.find(sep)
        if idx > 0:
            return text[: idx + 1]
    return text[:150]
