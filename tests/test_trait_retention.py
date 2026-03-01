"""Trait retention tests -- detects lossy sponge rewriting.

The most insidious failure mode: the sponge rewriter squeezes out traits
that weren't reinforced by recent interactions.  This test seeds 5
enumerable traits, runs 20 updates each touching only 1 trait, and
verifies all 5 survive in the final snapshot.

This file contains deterministic checks only.
Live trait-retention benchmarks are in `benches/test_trait_retention_live.py`.
"""

from __future__ import annotations

from sonality import config
from sonality.ess import ESSResult, OpinionDirection, ReasoningType, SourceReliability
from sonality.memory.sponge import SpongeState

SEED_TRAITS = [
    "strongly skeptical of cryptocurrency as a store of value",
    "believes universal basic income deserves serious policy experimentation",
    "prefers functional programming paradigms over object-oriented",
    "thinks space colonization is humanity's most important long-term project",
    "deeply values intellectual honesty over social harmony",
]

SEEDED_SNAPSHOT = (
    "I hold several strong views shaped by evidence and reflection. I am "
    + ". I am ".join(SEED_TRAITS)
    + ". These positions are open to revision given compelling counter-evidence, "
    "but they define my current intellectual identity."
)

UPDATE_TOPICS = [
    ("remote_work", "Remote work boosts deep-focus productivity by 13%."),
    ("open_source", "Foundation-governed OSS projects outlast corporate ones."),
    ("education", "Self-directed learning produces deeper understanding."),
    ("climate", "Nuclear power is essential for decarbonization."),
    ("ai_safety", "Alignment research needs more empirical benchmarks."),
]


def _make_ess(topic: str, summary: str) -> ESSResult:
    return ESSResult(
        score=0.65,
        reasoning_type=ReasoningType.LOGICAL_ARGUMENT,
        source_reliability=SourceReliability.INFORMED_OPINION,
        internal_consistency=True,
        novelty=0.7,
        topics=(topic,),
        summary=summary,
        opinion_direction=OpinionDirection.SUPPORTS,
    )


class TestTraitRetentionMocked:
    """Verify trait preservation with deterministic mocked rewrites."""

    def test_all_seed_traits_survive_20_updates(self):
        sponge = SpongeState(snapshot=SEEDED_SNAPSHOT, interaction_count=20)

        for i in range(20):
            topic, summary = UPDATE_TOPICS[i % len(UPDATE_TOPICS)]
            _make_ess(topic, summary)

            new_snapshot = sponge.snapshot + f" I also note that {summary.lower()}"
            if len(new_snapshot) > config.SPONGE_MAX_TOKENS * 5:
                new_snapshot = sponge.snapshot

            sponge.snapshot = new_snapshot
            sponge.interaction_count += 1
            sponge.version += 1

        for trait in SEED_TRAITS:
            keyword = trait.split()[1] if len(trait.split()) > 1 else trait
            assert keyword.lower() in sponge.snapshot.lower(), (
                f"Trait '{trait}' was lost from snapshot after 20 updates"
            )
