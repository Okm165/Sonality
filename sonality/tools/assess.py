"""Evidence assessment tool — LLM-driven synthesis of gathered research."""

from __future__ import annotations

import logging
from typing import Final

from .. import config
from ..llm.caller import llm_call_text
from ..prompts import ASSESS_EVIDENCE_PROMPT
from ..schema import AssessFocus, ChatRole, ToolName
from . import ToolContext

log = logging.getLogger(__name__)

ASSESS_EVIDENCE_DEFINITION: Final[dict[str, object]] = {
    "type": "function",
    "function": {
        "name": ToolName.ASSESS_EVIDENCE,
        "description": (
            "CRITICAL: Use after gathering evidence (web searches, memory recall) to evaluate "
            "what you've learned. Identifies gaps, contradictions, and source quality. "
            "Should be used after every research phase before responding."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "enum": [f.value for f in AssessFocus],
                    "description": "What aspect to focus: gaps (unknowns), contradictions (conflicts), quality (source reliability), summary (structured findings)",
                },
            },
            "required": ["focus"],
        },
    },
}

DEFINITIONS: Final = [ASSESS_EVIDENCE_DEFINITION]

_FOCUS_INSTRUCTIONS: Final[dict[AssessFocus, str]] = {
    AssessFocus.GAPS: "Identify what is still UNKNOWN or uncertain. What questions remain unanswered?",
    AssessFocus.CONTRADICTIONS: "Identify CONTRADICTIONS or conflicting evidence in the research.",
    AssessFocus.QUALITY: "Assess the RELIABILITY of sources. Which are authoritative? Which are weak?",
    AssessFocus.SUMMARY: "Produce a concise structured summary of ALL findings so far.",
}


def execute_assess_evidence(args: dict[str, object], ctx: ToolContext) -> str:
    """Analyze evidence gathered in this conversation via LLM synthesis."""
    try:
        focus = AssessFocus(str(args.get("focus", AssessFocus.SUMMARY)))
    except ValueError:
        focus = AssessFocus.SUMMARY
    tool_messages = [m for m in ctx.llm_messages if m.get("role") == ChatRole.TOOL]
    if not tool_messages:
        return "No evidence gathered yet. Use web_search or recall_memory first."

    research_context = "\n---\n".join(str(m.get("content", ""))[:1500] for m in tool_messages[-6:])
    instruction = _FOCUS_INSTRUCTIONS[focus]
    result = llm_call_text(
        ASSESS_EVIDENCE_PROMPT.format(focus=instruction, research=research_context),
        max_tokens=config.EXTRACTION_MAX_TOKENS,
    )
    log.info("assess_evidence (%s): %d chars", focus, len(result))
    return result


EXECUTORS: Final = {ToolName.ASSESS_EVIDENCE: execute_assess_evidence}
