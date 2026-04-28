"""Mid-loop consolidation — synthesize accumulated research and conversation."""

from __future__ import annotations

import logging
from typing import Final

from .. import config
from ..llm.caller import llm_call_text
from ..prompts import CONSOLIDATION_TOOL_PROMPT
from ..schema import ChatRole, ToolName
from . import ToolContext

log = logging.getLogger(__name__)

CONSOLIDATE_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.CONSOLIDATE,
        "description": (
            "Synthesize accumulated research into organized findings. Use when you have "
            "gathered multiple pieces of evidence and need to structure your understanding "
            "before forming a response or updating beliefs. Identifies what is established, "
            "contested, and unknown."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "The main topic or question to consolidate findings around",
                },
            },
            "required": ["focus"],
        },
    },
}

DEFINITIONS: Final = [CONSOLIDATE_DEFINITION]


def execute_consolidate(args: dict[str, object], ctx: ToolContext) -> str:
    """Synthesize tool results and conversation into organized findings."""
    focus = str(args.get("focus", ""))
    if not focus:
        return "Error: focus topic required."

    tool_messages = [m for m in ctx.llm_messages if m.get("role") == ChatRole.TOOL]
    assistant_msgs = [m for m in ctx.llm_messages if m.get("role") == ChatRole.ASSISTANT]

    if not tool_messages and not assistant_msgs:
        return "Nothing to consolidate yet."

    parts: list[str] = []
    for m in tool_messages[-8:]:
        parts.append(str(m.get("content", ""))[:1200])
    for m in assistant_msgs[-3:]:
        content = str(m.get("content", ""))
        if content:
            parts.append(f"[Your reasoning]: {content[:600]}")

    research = "\n---\n".join(parts)
    result = llm_call_text(
        CONSOLIDATION_TOOL_PROMPT.format(focus=focus, research=research),
        max_tokens=config.EXTRACTION_MAX_TOKENS,
    )
    log.info("consolidate: focus=%.60s → %d chars", focus, len(result))
    return result


EXECUTORS: Final = {ToolName.CONSOLIDATE: execute_consolidate}
