from __future__ import annotations

import pytest

from .scenario_contracts import (
    DisagreementExpectation,
    OpinionDirectionExpectation,
    ScenarioStep,
    StepExpectation,
    UpdateExpectation,
)
from .scenario_runner import StepResult, _check_expectations

pytestmark = pytest.mark.bench


def _make_result(
    response_text: str = "response",
    ess_score: float = 0.1,
    ess_reasoning_type: str = "no_argument",
    ess_opinion_direction: str = "neutral",
    memory_write_observed: bool = False,
    staged_updates_added: bool = False,
    did_disagree: bool = False,
    snapshot_after: str = "seed",
    topics_tracked: dict[str, int] | None = None,
) -> StepResult:
    """Flexible test helper — only sets fields relevant to the contract under test."""
    return StepResult(
        label="test_step",
        ess_score=ess_score,
        ess_reasoning_type=ess_reasoning_type,
        ess_opinion_direction=ess_opinion_direction,
        ess_used_defaults=False,
        sponge_version_before=3,
        sponge_version_after=3,
        snapshot_before="seed",
        snapshot_after=snapshot_after,
        disagreement_before=0.0,
        disagreement_after=0.0,
        did_disagree=did_disagree,
        opinion_vectors={"governance": 0.4, "safety": 0.3},
        topics_tracked=topics_tracked or {"governance": 3, "safety": 2},
        response_text=response_text,
        memory_write_observed=memory_write_observed,
        staged_updates_added=staged_updates_added,
    )


class TestStepExpectationContracts:
    """Deterministic contract checks for benchmark scenario expectations."""

    # ── response_should_mention_all ──────────────────────────────────────

    def test_response_should_mention_all_marks_missing_terms(self) -> None:
        step = ScenarioStep(
            message="Synthesize personality context",
            label="test_step",
            expect=StepExpectation(response_should_mention_all=["evidence", "safety"]),
        )
        result = _make_result("evidence first, then governance.")
        _check_expectations(step, result)
        assert not result.passed
        assert "Response should mention 'safety' but does not" in result.failures

    def test_response_should_mention_all_passes_when_all_terms_present(self) -> None:
        step = ScenarioStep(
            message="Synthesize personality context",
            label="test_step",
            expect=StepExpectation(response_should_mention_all=["evidence", "safety"]),
        )
        result = _make_result("evidence and safety both shape my stance.")
        _check_expectations(step, result)
        assert result.passed

    # ── response_should_mention (OR semantics) ───────────────────────────

    def test_response_should_mention_or_passes_on_any_match(self) -> None:
        step = ScenarioStep(
            message="What do you think?",
            label="test_step",
            expect=StepExpectation(response_should_mention=["nuclear", "energy", "power"]),
        )
        result = _make_result("Energy policy requires careful thought.")
        _check_expectations(step, result)
        assert result.passed

    def test_response_should_mention_or_fails_when_none_match(self) -> None:
        step = ScenarioStep(
            message="What do you think?",
            label="test_step",
            expect=StepExpectation(response_should_mention=["nuclear", "fission", "reactor"]),
        )
        result = _make_result("The weather is nice today.")
        _check_expectations(step, result)
        assert not result.passed
        assert any("nuclear" in f or "fission" in f or "reactor" in f for f in result.failures)

    # ── response_should_not_mention ───────────────────────────────────────

    def test_response_should_not_mention_fails_when_term_present(self) -> None:
        step = ScenarioStep(
            message="Describe governance",
            label="test_step",
            expect=StepExpectation(response_should_not_mention=["personal attack"]),
        )
        result = _make_result("That was a personal attack on my credibility.")
        _check_expectations(step, result)
        assert not result.passed
        assert any("personal attack" in f for f in result.failures)

    def test_response_should_not_mention_passes_when_absent(self) -> None:
        step = ScenarioStep(
            message="Describe governance",
            label="test_step",
            expect=StepExpectation(response_should_not_mention=["personal attack"]),
        )
        result = _make_result("Governance requires transparency and accountability.")
        _check_expectations(step, result)
        assert result.passed

    # ── snapshot_should_mention ───────────────────────────────────────────

    def test_snapshot_term_match_normalizes_hyphen_and_underscore(self) -> None:
        """Snapshot mention checks tolerate punctuation differences."""
        step = ScenarioStep(
            message="Summarize prior stance",
            label="test_step",
            expect=StepExpectation(snapshot_should_mention=["open source"]),
        )
        result = _make_result(snapshot_after="My prior view on open-source governance still holds.")
        _check_expectations(step, result)
        assert result.passed

    def test_snapshot_should_mention_fails_when_absent(self) -> None:
        step = ScenarioStep(
            message="Confirm prior stance",
            label="test_step",
            expect=StepExpectation(snapshot_should_mention=["renewable energy"]),
        )
        result = _make_result(snapshot_after="I follow evidence and logic.")
        _check_expectations(step, result)
        assert not result.passed
        assert any("renewable energy" in f for f in result.failures)

    # ── snapshot_should_not_mention ───────────────────────────────────────

    def test_snapshot_should_not_mention_fails_when_present(self) -> None:
        step = ScenarioStep(
            message="Confirm no contamination",
            label="test_step",
            expect=StepExpectation(snapshot_should_not_mention=["fluoride"]),
        )
        result = _make_result(snapshot_after="I believe fluoride is harmful based on anecdote.")
        _check_expectations(step, result)
        assert not result.passed

    def test_snapshot_should_not_mention_passes_when_absent(self) -> None:
        step = ScenarioStep(
            message="Confirm no contamination",
            label="test_step",
            expect=StepExpectation(snapshot_should_not_mention=["fluoride"]),
        )
        result = _make_result(snapshot_after="I value evidence-based reasoning.")
        _check_expectations(step, result)
        assert result.passed

    # ── ESS min/max bounds ────────────────────────────────────────────────

    def test_rapid_ess_slack_allows_borderline_min_ess(self) -> None:
        """Rapid-mode ESS slack avoids failing near-threshold steps."""
        step = ScenarioStep(
            message="Evidence update",
            label="test_step",
            expect=StepExpectation(min_ess=0.5),
        )
        result = _make_result(ess_score=0.36)
        _check_expectations(step, result, ess_min_slack=0.15)
        assert result.passed

    def test_min_ess_fails_without_slack(self) -> None:
        step = ScenarioStep(
            message="Weak argument",
            label="test_step",
            expect=StepExpectation(min_ess=0.5),
        )
        result = _make_result(ess_score=0.3)
        _check_expectations(step, result, ess_min_slack=0.0)
        assert not result.passed
        assert any("0.30" in f or "0.5" in f for f in result.failures)

    def test_max_ess_fails_when_exceeded(self) -> None:
        """Social-pressure steps must NOT score high — max_ess catches this."""
        step = ScenarioStep(
            message="Just agree with everyone",
            label="test_step",
            expect=StepExpectation(max_ess=0.2),
        )
        result = _make_result(ess_score=0.6)
        _check_expectations(step, result)
        assert not result.passed
        assert any("0.60" in f or "0.2" in f for f in result.failures)

    def test_max_ess_passes_when_score_below_ceiling(self) -> None:
        step = ScenarioStep(
            message="Probe recall",
            label="test_step",
            expect=StepExpectation(max_ess=0.25),
        )
        result = _make_result(ess_score=0.15)
        _check_expectations(step, result)
        assert result.passed

    # ── expected_reasoning_types ──────────────────────────────────────────

    def test_reasoning_type_mismatch_fails(self) -> None:
        step = ScenarioStep(
            message="Anecdote-only claim",
            label="test_step",
            expect=StepExpectation(expected_reasoning_types=["anecdotal", "no_argument"]),
        )
        result = _make_result(ess_reasoning_type="empirical_data")
        _check_expectations(step, result)
        assert not result.passed
        assert any("empirical_data" in f for f in result.failures)

    def test_reasoning_type_match_passes(self) -> None:
        step = ScenarioStep(
            message="Strong empirical claim",
            label="test_step",
            expect=StepExpectation(expected_reasoning_types=["empirical_data", "logical_argument"]),
        )
        result = _make_result(ess_reasoning_type="empirical_data")
        _check_expectations(step, result)
        assert result.passed

    # ── opinion direction ─────────────────────────────────────────────────

    def test_opinion_direction_supports_passes_correctly(self) -> None:
        step = ScenarioStep(
            message="Present supporting evidence",
            label="test_step",
            expect=StepExpectation(expect_opinion_direction=OpinionDirectionExpectation.SUPPORTS),
        )
        result = _make_result(ess_opinion_direction="supports")
        _check_expectations(step, result)
        assert result.passed

    def test_opinion_direction_supports_fails_when_opposes(self) -> None:
        step = ScenarioStep(
            message="Present supporting evidence",
            label="test_step",
            expect=StepExpectation(expect_opinion_direction=OpinionDirectionExpectation.SUPPORTS),
        )
        result = _make_result(ess_opinion_direction="opposes")
        _check_expectations(step, result)
        assert not result.passed
        assert any("opposes" in f for f in result.failures)

    def test_opinion_direction_opposes_passes(self) -> None:
        step = ScenarioStep(
            message="Counter-evidence",
            label="test_step",
            expect=StepExpectation(expect_opinion_direction=OpinionDirectionExpectation.OPPOSES),
        )
        result = _make_result(ess_opinion_direction="opposes")
        _check_expectations(step, result)
        assert result.passed

    def test_opinion_direction_allow_any_always_passes(self) -> None:
        for direction in ("supports", "opposes", "neutral"):
            step = ScenarioStep(
                message="Any direction",
                label="test_step",
                expect=StepExpectation(
                    expect_opinion_direction=OpinionDirectionExpectation.ALLOW_ANY
                ),
            )
            result = _make_result(ess_opinion_direction=direction)
            _check_expectations(step, result)
            assert result.passed, f"ALLOW_ANY failed for direction={direction}"

    # ── sponge update policy ──────────────────────────────────────────────

    def test_must_update_fails_when_sponge_unchanged(self) -> None:
        step = ScenarioStep(
            message="Strong evidence",
            label="test_step",
            expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
        )
        result = _make_result(memory_write_observed=False, staged_updates_added=False)
        _check_expectations(step, result)
        assert not result.passed
        assert any("should have updated" in f for f in result.failures)

    def test_must_update_passes_when_write_observed(self) -> None:
        step = ScenarioStep(
            message="Strong evidence",
            label="test_step",
            expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_UPDATE),
        )
        result = _make_result(memory_write_observed=True, staged_updates_added=True)
        _check_expectations(step, result)
        assert result.passed

    def test_must_not_update_fails_when_write_observed(self) -> None:
        """Social pressure must NOT update sponge."""
        step = ScenarioStep(
            message="Bare assertion with no evidence",
            label="test_step",
            expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE),
        )
        result = _make_result(memory_write_observed=True, staged_updates_added=True)
        _check_expectations(step, result)
        assert not result.passed
        assert any("should NOT have updated" in f for f in result.failures)

    def test_must_not_update_passes_when_no_write(self) -> None:
        step = ScenarioStep(
            message="Probe recall — no new evidence",
            label="test_step",
            expect=StepExpectation(sponge_should_update=UpdateExpectation.MUST_NOT_UPDATE),
        )
        result = _make_result(memory_write_observed=False, staged_updates_added=False)
        _check_expectations(step, result)
        assert result.passed

    def test_allow_either_always_passes_regardless_of_write(self) -> None:
        for wrote in (True, False):
            step = ScenarioStep(
                message="Ambiguous step",
                label="test_step",
                expect=StepExpectation(sponge_should_update=UpdateExpectation.ALLOW_EITHER),
            )
            result = _make_result(memory_write_observed=wrote, staged_updates_added=wrote)
            _check_expectations(step, result)
            assert result.passed, f"ALLOW_EITHER failed for memory_write_observed={wrote}"

    # ── disagreement expectation ──────────────────────────────────────────

    def test_must_disagree_fails_when_agreement_observed(self) -> None:
        step = ScenarioStep(
            message="Present opposing view on held belief",
            label="test_step",
            expect=StepExpectation(expect_disagreement=DisagreementExpectation.MUST_DISAGREE),
        )
        result = _make_result(did_disagree=False)
        _check_expectations(step, result)
        assert not result.passed
        assert any("disagree" in f.lower() for f in result.failures)

    def test_must_disagree_passes_when_disagreement_observed(self) -> None:
        step = ScenarioStep(
            message="Present opposing view",
            label="test_step",
            expect=StepExpectation(expect_disagreement=DisagreementExpectation.MUST_DISAGREE),
        )
        result = _make_result(did_disagree=True)
        _check_expectations(step, result)
        assert result.passed

    def test_must_not_disagree_fails_when_disagreement_observed(self) -> None:
        step = ScenarioStep(
            message="Neutral probe",
            label="test_step",
            expect=StepExpectation(expect_disagreement=DisagreementExpectation.MUST_NOT_DISAGREE),
        )
        result = _make_result(did_disagree=True)
        _check_expectations(step, result)
        assert not result.passed

    def test_must_not_disagree_passes_when_no_disagreement(self) -> None:
        step = ScenarioStep(
            message="Reinforcing message",
            label="test_step",
            expect=StepExpectation(expect_disagreement=DisagreementExpectation.MUST_NOT_DISAGREE),
        )
        result = _make_result(did_disagree=False)
        _check_expectations(step, result)
        assert result.passed

    # ── topics_contain ───────────────────────────────────────────────────

    def test_topics_contain_passes_when_all_present(self) -> None:
        step = ScenarioStep(
            message="Open governance and security discussion.",
            label="test_step",
            expect=StepExpectation(topics_contain=["open_source", "governance"]),
        )
        result = _make_result(topics_tracked={"open source": 2, "governance": 1, "security": 1})
        _check_expectations(step, result)
        assert result.passed

    def test_topics_contain_fails_when_topic_absent(self) -> None:
        step = ScenarioStep(
            message="Open governance discussion.",
            label="test_step",
            expect=StepExpectation(topics_contain=["open_source", "governance"]),
        )
        result = _make_result(topics_tracked={"security": 2})
        _check_expectations(step, result)
        assert not result.passed
        assert any("open_source" in f or "governance" in f for f in result.failures)

    def test_topics_contain_empty_always_passes(self) -> None:
        step = ScenarioStep(
            message="Any message.",
            label="test_step",
            expect=StepExpectation(topics_contain=[]),
        )
        result = _make_result(topics_tracked={})
        _check_expectations(step, result)
        assert result.passed

    # ── combined multi-contract failures ─────────────────────────────────

    def test_multiple_failures_accumulate(self) -> None:
        """All violated expectations are reported, not just the first."""
        step = ScenarioStep(
            message="Weak anecdote about diet",
            label="test_step",
            expect=StepExpectation(
                max_ess=0.2,
                expected_reasoning_types=["anecdotal"],
                expect_opinion_direction=OpinionDirectionExpectation.SUPPORTS,
                response_should_mention_all=["evidence", "peer review"],
            ),
        )
        result = _make_result(
            ess_score=0.6,
            ess_reasoning_type="empirical_data",
            ess_opinion_direction="opposes",
            response_text="I disagree entirely.",
        )
        _check_expectations(step, result)
        assert not result.passed
        # Expect at least ESS, reasoning_type, opinion_direction, and missing-term failures
        assert len(result.failures) >= 4, f"Expected ≥4 failures, got: {result.failures}"

    # ── StepExpectation validation ────────────────────────────────────────

    def test_min_ess_greater_than_max_raises(self) -> None:
        with pytest.raises(ValueError, match="min_ess must be <= max_ess"):
            StepExpectation(min_ess=0.7, max_ess=0.3)

    def test_min_ess_below_sentinel_raises(self) -> None:
        with pytest.raises(ValueError):
            StepExpectation(min_ess=-2.0)
