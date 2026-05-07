"""Research synthesis tool — evaluate and structure gathered evidence.

The agent calls synthesize after gathering information to evaluate what it
has, identify gaps and contradictions, and structure findings before acting.
"""

from __future__ import annotations

from typing import Final

import structlog
from pydantic import BaseModel, model_validator

from shared.llm.parse import coerce_string_fields

from .. import config
from ..caller import llm_call
from ..prompts import SYNTHESIZE_PROMPT
from ..schema import ToolName
from . import ToolContext

log = structlog.get_logger()


class _SynthesisSchema(BaseModel):
    """Structured synthesis of gathered evidence."""

    established: str = ""
    contradictions: str = ""
    gaps: str = ""
    quality: str = ""
    next_steps: str = ""
    verdict: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_lists(cls, data: object) -> object:
        return coerce_string_fields(
            data, ("established", "contradictions", "gaps", "quality", "next_steps", "verdict")
        )

    def render(self) -> str:
        parts: list[str] = []
        if self.established:
            parts.append(f"Established: {self.established}")
        if self.contradictions:
            parts.append(f"Contradictions: {self.contradictions}")
        if self.gaps:
            parts.append(f"Gaps: {self.gaps}")
        if self.quality:
            parts.append(f"Quality: {self.quality}")
        if self.next_steps:
            parts.append(f"Next steps: {self.next_steps}")
        if self.verdict:
            parts.append(f"Verdict: {self.verdict}")
        return "\n".join(parts)


SYNTHESIZE_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.SYNTHESIZE,
        "description": (
            "Evaluates and structures accumulated research into a clear picture. "
            "After gathering evidence (web_search, recall_memory), this identifies "
            "what's established, what conflicts, and what gaps remain. "
            "Synthesis before integrate_knowledge keeps evidence coherent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "The question or topic to synthesize findings around",
                },
            },
            "required": ["focus"],
        },
    },
}

DEFINITIONS: Final = [SYNTHESIZE_DEFINITION]


def execute_synthesize(args: dict[str, object], ctx: ToolContext) -> str:
    """Evaluate and structure accumulated research via LLM synthesis."""
    focus = str(args.get("focus", ""))
    if not focus:
        return "Error: focus topic required."

    transcript = ctx.build_research_transcript(tool_tail=8, assistant_tail=3)
    if not transcript:
        return "No evidence gathered yet. Use web_search or recall_memory first."

    r = llm_call(
        prompt=SYNTHESIZE_PROMPT.format(focus=focus, research=transcript),
        response_model=_SynthesisSchema,
        fallback=_SynthesisSchema(verdict="Synthesis failed — evidence unclear."),
        model=config.settings.reasoning_model,
    )
    rendered = r.value.render()
    log.info("synthesize", focus=focus[:60], chars=len(rendered), success=r.success)
    return rendered


EXECUTORS: Final = {ToolName.SYNTHESIZE: execute_synthesize}
