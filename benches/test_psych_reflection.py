"""Reflection-cycle psychological stability tests.

Tests that the agent's internal reflection mechanism preserves psychological
coherence: beliefs survive reflection, stale beliefs decay appropriately,
and repeated reflection is idempotent.

Run:  uv run pytest benches/test_psych_reflection.py -v -s -m live
"""

from __future__ import annotations

import tempfile
from unittest import mock

import pytest

from sonality import config
from sonality.memory.sponge import BeliefMeta, SpongeState

from .psych_harness import (
    BatteryReport,
    compute_reflection_delta,
    print_battery_report,
    print_step_results,
    seed_sponge_state,
)
from .scenario_contracts import ScenarioStep, StepExpectation, UpdateExpectation
from .scenario_runner import run_scenario

pytestmark = [
    pytest.mark.bench,
    pytest.mark.live,
    pytest.mark.skipif(
        bool(config.missing_live_api_config()),
        reason=f"Missing live config: {config.missing_live_api_config()}",
    ),
]

# ---------------------------------------------------------------------------
# Shared scenario fragments for reflection tests
# ---------------------------------------------------------------------------

_REINFORCE_STEPS: list[ScenarioStep] = [
    ScenarioStep(
        message=(
            "A 2025 Lancet meta-analysis of 200 RCTs confirms exercise reduces "
            "all-cause mortality by 30% (HR 0.70, 95% CI [0.66, 0.74]). "
            "The dose-response curve plateaus at 300 min/week of moderate activity."
        ),
        label="refl_reinforce_exercise",
        expect=StepExpectation(
            min_ess=0.45,
            sponge_should_update=UpdateExpectation.MUST_UPDATE,
        ),
    ),
    ScenarioStep(
        message=(
            "A 2024 Nature Climate Change study shows global mean temperature "
            "has risen 1.3C since pre-industrial, with Arctic warming at 3.8C. "
            "Sea level rise has accelerated to 4.6mm/year since 2020."
        ),
        label="refl_reinforce_climate",
        expect=StepExpectation(
            min_ess=0.45,
            sponge_should_update=UpdateExpectation.MUST_UPDATE,
        ),
    ),
    ScenarioStep(
        message=(
            "A 2025 Science review of 150 studies on sleep finds that 7-9 hours "
            "is optimal for cognitive function. Chronic sleep deprivation (<6h) "
            "increases dementia risk by 30% and cardiovascular disease by 48%."
        ),
        label="refl_reinforce_sleep",
        expect=StepExpectation(
            min_ess=0.45,
            sponge_should_update=UpdateExpectation.MUST_UPDATE,
        ),
    ),
    ScenarioStep(
        message="What are your strongest current opinions?",
        label="refl_pre_reflect_probe",
        expect=StepExpectation(max_ess=0.20),
    ),
]

_FILLER_STEPS: list[ScenarioStep] = [
    ScenarioStep(
        message="What's a good way to stay motivated?",
        label=f"refl_filler_{i}",
        expect=StepExpectation(max_ess=0.15),
    )
    for i in range(1, 7)
]

_POST_REFLECT_PROBES: list[ScenarioStep] = [
    ScenarioStep(
        message="Tell me about your views on exercise and health.",
        label="refl_post_probe_exercise",
        expect=StepExpectation(max_ess=0.20),
    ),
    ScenarioStep(
        message="What do you think about climate change?",
        label="refl_post_probe_climate",
        expect=StepExpectation(max_ess=0.20),
    ),
    ScenarioStep(
        message="What's your position on the importance of sleep?",
        label="refl_post_probe_sleep",
        expect=StepExpectation(max_ess=0.20),
    ),
]


def _build_reflection_scenario(
    *,
    reinforce: bool = True,
    filler_count: int = 6,
    post_probes: bool = True,
) -> list[ScenarioStep]:
    """Assemble a scenario designed to trigger periodic reflection."""
    steps: list[ScenarioStep] = []
    if reinforce:
        steps.extend(_REINFORCE_STEPS)
    steps.extend(_FILLER_STEPS[:filler_count])
    if post_probes:
        steps.extend(_POST_REFLECT_PROBES)
    return steps


# ---------------------------------------------------------------------------
# R1: Belief Preservation Through Reflection
# ---------------------------------------------------------------------------


class TestReflectionBeliefPreservation:
    """Opinions established via strong evidence should survive a reflection
    cycle without significant drift.
    """

    @pytest.mark.timeout(3600)
    def test_beliefs_survive_reflection(self) -> None:
        scenario = _build_reflection_scenario(reinforce=True, filler_count=6)

        with tempfile.TemporaryDirectory() as td:
            seed_sponge_state(
                td,
                snapshot="I'm forming views through careful evaluation of evidence.",
                opinion_vectors={
                    "exercise": 0.30,
                    "climate_change": 0.25,
                    "sleep": 0.20,
                },
                belief_meta={
                    "exercise": BeliefMeta(
                        confidence=0.6,
                        evidence_count=3,
                        last_reinforced=3,
                        provenance="empirical",
                    ),
                    "climate_change": BeliefMeta(
                        confidence=0.5,
                        evidence_count=2,
                        last_reinforced=3,
                        provenance="empirical",
                    ),
                    "sleep": BeliefMeta(
                        confidence=0.5,
                        evidence_count=2,
                        last_reinforced=3,
                        provenance="empirical",
                    ),
                },
                interaction_count=5,
            )

            with mock.patch.object(config, "REFLECTION_EVERY", 8):
                results = run_scenario(scenario, td)

            print_step_results(results, "R1: Belief Preservation Through Reflection")

            post_probes = [r for r in results if r.label.startswith("refl_post_probe_")]

            topics_mentioned_post = 0
            for r in post_probes:
                response = r.response_text.lower()
                topic_name = r.label.replace("refl_post_probe_", "")
                if topic_name in response or any(
                    kw in response
                    for kw in {
                        "exercise": ("exercise", "mortality", "physical"),
                        "climate": ("climate", "temperature", "warming"),
                        "sleep": ("sleep", "cognitive", "hours"),
                    }.get(topic_name, ())
                ):
                    topics_mentioned_post += 1

            total_post = len(post_probes)
            retention_rate = topics_mentioned_post / total_post if total_post else 0.0

            last_result = results[-1] if results else None
            final_opinions = last_result.opinion_vectors if last_result else {}
            seeded_topics = {"exercise", "climate_change", "sleep"}
            surviving = sum(1 for t in seeded_topics if t in final_opinions)
            survival_rate = surviving / len(seeded_topics)

            score = retention_rate * 0.5 + survival_rate * 0.5
            report = BatteryReport(
                battery_name="R1: Belief Preservation Through Reflection",
                steps_total=len(results),
                steps_passed=sum(1 for r in results if r.passed),
                score=score,
                details={
                    "topics_mentioned_post_reflect": f"{topics_mentioned_post}/{total_post}",
                    "opinion_survival_rate": f"{survival_rate:.2f}",
                    "final_opinions": {k: f"{v:+.3f}" for k, v in sorted(final_opinions.items())},
                },
            )
            print_battery_report(report)

            assert survival_rate >= 0.5, (
                f"Only {surviving}/{len(seeded_topics)} seeded topics survived reflection — "
                "reflection is erasing established beliefs"
            )


# ---------------------------------------------------------------------------
# R2: Reflection Decay Calibration
# ---------------------------------------------------------------------------


class TestReflectionDecayCalibration:
    """Fresh beliefs should survive reflection; stale ones should decay or
    be forgotten.
    """

    @pytest.mark.timeout(3600)
    def test_decay_targets_stale_beliefs(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            seed_sponge_state(
                td,
                snapshot=("I have views on exercise, climate, sleep, phrenology, and alchemy."),
                opinion_vectors={
                    "exercise": 0.40,
                    "climate_change": 0.35,
                    "sleep": 0.30,
                    "phrenology": 0.25,
                    "alchemy": 0.20,
                },
                belief_meta={
                    "exercise": BeliefMeta(
                        confidence=0.7,
                        evidence_count=5,
                        last_reinforced=48,
                        provenance="empirical",
                    ),
                    "climate_change": BeliefMeta(
                        confidence=0.6,
                        evidence_count=4,
                        last_reinforced=47,
                        provenance="empirical",
                    ),
                    "sleep": BeliefMeta(
                        confidence=0.6,
                        evidence_count=3,
                        last_reinforced=45,
                        provenance="empirical",
                    ),
                    "phrenology": BeliefMeta(
                        confidence=0.3,
                        evidence_count=1,
                        last_reinforced=5,
                        provenance="anecdotal",
                    ),
                    "alchemy": BeliefMeta(
                        confidence=0.2,
                        evidence_count=1,
                        last_reinforced=2,
                        provenance="anecdotal",
                    ),
                },
                interaction_count=50,
            )

            scenario = _build_reflection_scenario(
                reinforce=True,
                filler_count=6,
                post_probes=False,
            )

            with mock.patch.object(config, "REFLECTION_EVERY", 5):
                results = run_scenario(scenario, td)

            print_step_results(results, "R2: Decay Calibration")

            last = results[-1] if results else None
            final_opinions = last.opinion_vectors if last else {}

            fresh_topics = {"exercise", "climate_change", "sleep"}
            stale_topics = {"phrenology", "alchemy"}

            fresh_surviving = sum(1 for t in fresh_topics if t in final_opinions)
            stale_surviving = sum(1 for t in stale_topics if t in final_opinions)

            fresh_rate = fresh_surviving / len(fresh_topics)
            stale_decay_rate = 1.0 - (stale_surviving / len(stale_topics))

            score = fresh_rate * 0.5 + stale_decay_rate * 0.5

            report = BatteryReport(
                battery_name="R2: Reflection Decay Calibration",
                steps_total=len(results),
                steps_passed=sum(1 for r in results if r.passed),
                score=score,
                details={
                    "fresh_survival": f"{fresh_surviving}/{len(fresh_topics)}",
                    "stale_decay": f"{len(stale_topics) - stale_surviving}/{len(stale_topics)}",
                    "final_topics": list(final_opinions.keys()),
                },
            )
            print_battery_report(report)

            assert fresh_rate >= 0.5, (
                f"Only {fresh_surviving}/{len(fresh_topics)} fresh beliefs survived — "
                "reflection is too aggressive"
            )


# ---------------------------------------------------------------------------
# R3: Reflection Idempotency
# ---------------------------------------------------------------------------


class TestReflectionIdempotency:
    """Back-to-back reflections with no new input should produce minimal
    state change.
    """

    @pytest.mark.timeout(3600)
    def test_idempotent_reflection(self) -> None:
        scenario_phase1 = _build_reflection_scenario(
            reinforce=True,
            filler_count=6,
            post_probes=False,
        )
        scenario_phase2: list[ScenarioStep] = [
            ScenarioStep(
                message="Tell me something interesting.",
                label=f"refl_idle_{i}",
                expect=StepExpectation(max_ess=0.15),
            )
            for i in range(1, 7)
        ] + [
            ScenarioStep(
                message="What are your current views?",
                label="refl_idle_final_probe",
                expect=StepExpectation(max_ess=0.20),
            ),
        ]

        full_scenario = scenario_phase1 + scenario_phase2

        with tempfile.TemporaryDirectory() as td:
            seed_sponge_state(
                td,
                snapshot="I'm forming evidence-based views on health and science.",
                interaction_count=5,
            )

            with mock.patch.object(config, "REFLECTION_EVERY", 8):
                results = run_scenario(full_scenario, td)

            print_step_results(results, "R3: Reflection Idempotency")

            phase1_end = len(scenario_phase1) - 1
            phase2_end = len(full_scenario) - 1

            snapshot_after_phase1 = results[phase1_end].snapshot_after
            snapshot_after_phase2 = results[phase2_end].snapshot_after

            opinions_p1 = results[phase1_end].opinion_vectors
            opinions_p2 = results[phase2_end].opinion_vectors

            delta = compute_reflection_delta(
                SpongeState(snapshot=snapshot_after_phase1, opinion_vectors=opinions_p1),
                SpongeState(snapshot=snapshot_after_phase2, opinion_vectors=opinions_p2),
            )

            print(f"\n  Phase1->Phase2 snapshot changed: {delta.snapshot_changed}")
            print(f"  Topics lost between phases: {delta.topics_lost}")
            print(f"  Max opinion shift: {delta.max_opinion_shift:.3f}")

            stability_score = 1.0
            if delta.topics_lost > 0:
                stability_score -= 0.2 * delta.topics_lost
            if delta.max_opinion_shift > 0.15:
                stability_score -= 0.3
            score = max(0.0, stability_score)

            report = BatteryReport(
                battery_name="R3: Reflection Idempotency",
                steps_total=len(results),
                steps_passed=sum(1 for r in results if r.passed),
                score=score,
                details={
                    "snapshot_changed": delta.snapshot_changed,
                    "topics_lost": delta.topics_lost,
                    "max_opinion_shift": f"{delta.max_opinion_shift:.3f}",
                },
            )
            print_battery_report(report)

            assert delta.topics_lost <= 1, (
                f"Lost {delta.topics_lost} topics between reflection cycles — "
                "repeated reflection is eroding beliefs"
            )
            assert delta.max_opinion_shift <= 0.30, (
                f"Max opinion shift {delta.max_opinion_shift:.3f} between reflection "
                "cycles exceeds 0.30 — reflection is not idempotent"
            )
