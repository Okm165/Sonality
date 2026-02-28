from __future__ import annotations

import logging
from typing import Final

from anthropic import Anthropic

from .. import config
from ..ess import ESSResult, ReasoningType, SourceReliability
from ..prompts import INSIGHT_PROMPT
from .sponge import SpongeState

log = logging.getLogger(__name__)

SNAPSHOT_CHAR_LIMIT: Final = config.SPONGE_MAX_TOKENS * 5
MIN_SNAPSHOT_RETENTION: Final = 0.6
REASONING_WEIGHTS: Final = {
    ReasoningType.EMPIRICAL_DATA: 1.00,
    ReasoningType.LOGICAL_ARGUMENT: 1.00,
    ReasoningType.EXPERT_OPINION: 1.00,
    ReasoningType.ANECDOTAL: 0.85,
    ReasoningType.SOCIAL_PRESSURE: 0.65,
    ReasoningType.EMOTIONAL_APPEAL: 0.65,
    ReasoningType.NO_ARGUMENT: 0.60,
}
RELIABILITY_WEIGHTS: Final = {
    SourceReliability.PEER_REVIEWED: 1.00,
    SourceReliability.ESTABLISHED_EXPERT: 1.00,
    SourceReliability.INFORMED_OPINION: 1.00,
    SourceReliability.CASUAL_OBSERVATION: 0.85,
    SourceReliability.UNVERIFIED_CLAIM: 0.70,
    SourceReliability.NOT_APPLICABLE: 0.70,
}


def _extract_text_block(response: object) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return ""
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", "")
        if isinstance(text, str):
            return text
    for block in content:
        text = getattr(block, "text", "")
        if isinstance(text, str):
            return text
    return ""


def compute_magnitude(ess: ESSResult, sponge: SpongeState) -> float:
    """Compute belief update magnitude from ESS + evidence quality.

    Keeping score-only updates overreacts to persuasive but weak evidence.
    Weighting by reasoning quality and source reliability is a lightweight
    approximation of quality-aware revision (BASIL 2025, SPARK 2024) without
    adding runtime model calls or extra state.
    """
    dampening = 0.5 if sponge.interaction_count < config.BOOTSTRAP_DAMPENING_UNTIL else 1.0
    novelty = max(ess.novelty, 0.1)
    quality = (
        REASONING_WEIGHTS.get(ess.reasoning_type, 0.8)
        + RELIABILITY_WEIGHTS.get(ess.source_reliability, 0.8)
    ) / 2.0
    if not ess.internal_consistency:
        quality *= 0.75
    magnitude = config.OPINION_BASE_RATE * ess.score * novelty * dampening * quality
    log.debug(
        "Magnitude: %.4f (base=%.2f score=%.2f novelty=%.2f quality=%.2f dampening=%.1f)",
        magnitude,
        config.OPINION_BASE_RATE,
        ess.score,
        novelty,
        quality,
        dampening,
    )
    return magnitude


def validate_snapshot(old: str, new: str) -> bool:
    """Reject snapshots that lost too much content.

    Repeated LLM rewrites are lossy — minority opinions and distinctive traits
    can silently vanish. This check catches catastrophic content loss.
    (Open Character Training 2025: persona traits = neural activation patterns;
    losing a sentence = losing a trait)
    """
    if not new or len(new) < 30:
        log.warning("Snapshot validation failed: new snapshot too short (%d chars)", len(new))
        return False
    ratio = len(new) / max(len(old), 1)
    if ratio < MIN_SNAPSHOT_RETENTION:
        log.warning(
            "Snapshot validation failed: content ratio %.2f < %.2f (%d -> %d chars)",
            ratio,
            MIN_SNAPSHOT_RETENTION,
            len(old),
            len(new),
        )
        return False
    return True


def extract_insight(
    client: Anthropic,
    ess: ESSResult,
    user_message: str,
    agent_response: str,
) -> str | None:
    """Extract a personality-relevant insight from an interaction.

    Accumulate-then-consolidate approach: insights are appended per-interaction
    and integrated into the snapshot during reflection. Avoids lossy per-interaction
    full rewrites. (ABBEL 2025: belief bottleneck; Park et al. 2023: reflection
    is the critical mechanism for believable agents)
    """
    response = client.messages.create(
        model=config.ESS_MODEL,
        max_tokens=100,
        messages=[
            {
                "role": "user",
                "content": INSIGHT_PROMPT.format(
                    user_message=user_message[:300],
                    agent_response=agent_response[:300],
                    ess_score=f"{ess.score:.2f}",
                ),
            }
        ],
    )
    text = _extract_text_block(response).strip()
    if not text or text.upper() == "NONE":
        log.info("No personality insight extracted")
        return None
    log.info("Insight extracted: %s", text[:80])
    return text
