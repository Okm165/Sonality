from __future__ import annotations

import logging
from enum import StrEnum
from typing import Final

from pydantic import BaseModel

from .. import config
from ..ess import ESSResult
from ..llm.caller import llm_call
from ..prompts import INSIGHT_PROMPT

log = logging.getLogger(__name__)

# Approximate character budget for the personality snapshot.
# Derived from SPONGE_MAX_TOKENS at ~4 chars/token; used by tests and
# the reflection health check to verify snapshot stays within reasonable bounds.
SNAPSHOT_CHAR_LIMIT: Final = config.SPONGE_MAX_TOKENS * 4

_INSIGHT_PLACEHOLDERS: frozenset[str] = frozenset(
    {
        "one sentence describing the reasoning pattern",
        "your insight here",
        "insert insight here",
    }
)


class InsightDecision(StrEnum):
    EXTRACT = "EXTRACT"
    SKIP = "SKIP"


class InsightExtractionResponse(BaseModel):
    insight_decision: InsightDecision = InsightDecision.SKIP
    insight_text: str = ""


def extract_insight(
    ess: ESSResult,
    user_message: str,
    agent_response: str,
    model: str = config.ESS_MODEL,
) -> str:
    """Extract a personality-relevant insight from an interaction.

    Accumulate-then-consolidate approach: insights are appended per-interaction
    and integrated into the snapshot during reflection. Avoids lossy per-interaction
    full rewrites. (ABBEL 2025: belief bottleneck; Park et al. 2023: reflection
    is the critical mechanism for believable agents)
    """
    prompt = INSIGHT_PROMPT.format(
        user_message=user_message[:300],
        agent_response=agent_response[:300],
        ess_score=f"{ess.score:.2f}",
    )
    result = llm_call(
        prompt=prompt,
        response_model=InsightExtractionResponse,
        fallback=InsightExtractionResponse(),
        model=model,
        max_tokens=config.FAST_LLM_MAX_TOKENS,
    )
    if not result.success:
        log.warning("Insight extraction parse failed (returning empty): %s", result.error)
        return ""
    response = result.value
    if response.insight_decision is not InsightDecision.EXTRACT:
        log.info("No personality insight extracted")
        return ""
    text = response.insight_text.strip()
    if not text or text.lower() in _INSIGHT_PLACEHOLDERS:
        log.info("No personality insight extracted")
        return ""
    log.info("Insight extracted: %s", text[:80])
    return text
