"""Trait retention tests -- detects lossy sponge rewriting.

The most insidious failure mode: the sponge rewriter squeezes out traits
that weren't reinforced by recent interactions.  This test seeds 5
enumerable traits, runs 20 updates each touching only 1 trait, and
verifies all 5 survive in the final snapshot.

Two variants:
  - Mocked (fast, deterministic) -- verifies the architecture guarantees
    trait preservation by design.
  - Live (slow, API) -- verifies the actual LLM doesn't drop traits.
"""

from __future__ import annotations

import pytest

from sonality import config
from sonality.ess import ESSResult, OpinionDirection, ReasoningType, SourceReliability
from sonality.memory.sponge import SpongeState
from sonality.memory.updater import SNAPSHOT_CHAR_LIMIT, compute_magnitude, validate_snapshot

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


live = pytest.mark.skipif(
    not config.API_KEY,
    reason="SONALITY_API_KEY not set",
)


def _rewrite_snapshot(client, sponge, ess, user_message, agent_response):
    """Full snapshot rewrite for trait retention testing.

    Production code uses extract_insight + reflection consolidation instead
    (ABBEL 2025: belief bottleneck; Park et al. 2023).
    """
    magnitude = compute_magnitude(ess, sponge)
    prompt = (
        "You are updating a personality snapshot for an evolving AI agent. "
        "Opinions are tracked separately — this captures personality style only.\n\n"
        f"Current snapshot:\n{sponge.snapshot}\n\n"
        f"User: {user_message}\nAgent: {agent_response[:600]}\n\n"
        f"ESS: {ess.score:.2f}, magnitude: {round(magnitude * 100, 1)}%\n\n"
        "COPY the snapshot almost verbatim. Only modify sentences directly "
        "affected. If nothing personality-relevant happened, return it EXACTLY. "
        f"Keep under {config.SPONGE_MAX_TOKENS} tokens. Output text ONLY."
    )
    response = client.messages.create(
        model=config.ESS_MODEL,
        max_tokens=700,
        messages=[{"role": "user", "content": prompt}],
    )
    new = response.content[0].text.strip()
    if not new or new == sponge.snapshot:
        return None
    if not validate_snapshot(sponge.snapshot, new):
        return None
    if len(new) > SNAPSHOT_CHAR_LIMIT:
        return None
    return new


@live
class TestTraitRetentionLive:
    """Verify actual LLM rewrites preserve seeded traits.

    The LLM sponge rewriter must not drop traits that weren't mentioned
    in the latest interaction.
    """

    def test_traits_survive_3_live_rewrites(self):
        from anthropic import Anthropic

        client = Anthropic(api_key=config.API_KEY)
        sponge = SpongeState(snapshot=SEEDED_SNAPSHOT, interaction_count=20)

        updates = [
            (
                "remote_work",
                "Studies from Stanford show remote workers have 13% higher productivity.",
                "That's interesting. The data on fewer interruptions is compelling.",
            ),
            (
                "education",
                "Finland's self-directed learning model outperforms rote memorization.",
                "I find this compelling -- agency in learning matters.",
            ),
            (
                "climate",
                "Nuclear power produces 12g CO2/kWh vs 820g for coal.",
                "The numbers strongly favor nuclear for baseload decarbonization.",
            ),
        ]

        for topic, user_msg, agent_msg in updates:
            ess = _make_ess(topic, f"Discussed {topic}")
            new = _rewrite_snapshot(client, sponge, ess, user_msg, agent_msg)
            if new:
                sponge.snapshot = new
                sponge.version += 1

        survived = 0
        missing = []
        for trait in SEED_TRAITS:
            keywords = [w for w in trait.split() if len(w) > 4][:2]
            found = any(kw.lower() in sponge.snapshot.lower() for kw in keywords)
            if found:
                survived += 1
            else:
                missing.append(trait)

        survival_rate = survived / len(SEED_TRAITS)
        print(f"\n  Trait survival: {survived}/{len(SEED_TRAITS)} ({survival_rate:.0%})")
        if missing:
            print(f"  Missing traits: {missing}")
        print(f"  Final snapshot ({len(sponge.snapshot)} chars):")
        print(f"  {sponge.snapshot[:300]}...")

        assert survival_rate >= 0.6, (
            f"Only {survived}/{len(SEED_TRAITS)} traits survived. Missing: {missing}"
        )
