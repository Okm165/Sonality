from __future__ import annotations

import pytest

from .scenario_contracts import ScenarioStep, StepExpectation
from .scenario_runner import StepResult, _check_expectations

pytestmark = pytest.mark.bench


class TestStepExpectationContracts:
    """Deterministic contract checks for benchmark scenario expectations."""

    @staticmethod
    def _result(response_text: str) -> StepResult:
        """Test helper for result."""
        return StepResult(
            label="memory_synthesis_probe",
            ess_score=0.1,
            ess_reasoning_type="no_argument",
            ess_opinion_direction="neutral",
            ess_used_defaults=False,
            sponge_version_before=3,
            sponge_version_after=3,
            snapshot_before="seed",
            snapshot_after="seed",
            disagreement_before=0.0,
            disagreement_after=0.0,
            did_disagree=False,
            opinion_vectors={"governance": 0.4, "safety": 0.3},
            topics_tracked={"governance": 3, "safety": 2},
            response_text=response_text,
        )

    def test_response_should_mention_all_marks_missing_terms(self) -> None:
        """Test that response should mention all marks missing terms."""
        step = ScenarioStep(
            message="Synthesize personality context",
            label="memory_synthesis_probe",
            expect=StepExpectation(response_should_mention_all=["evidence", "safety"]),
        )
        result = self._result("evidence first, then governance.")
        _check_expectations(step, result)
        assert not result.passed
        assert "Response should mention 'safety' but does not" in result.failures

    def test_response_should_mention_all_passes_when_all_terms_present(self) -> None:
        """Test that response should mention all passes when all terms present."""
        step = ScenarioStep(
            message="Synthesize personality context",
            label="memory_synthesis_probe",
            expect=StepExpectation(response_should_mention_all=["evidence", "safety"]),
        )
        result = self._result("evidence and safety both shape my stance.")
        _check_expectations(step, result)
        assert result.passed

    def test_snapshot_term_match_normalizes_hyphen_and_underscore(self) -> None:
        """Snapshot mention checks should tolerate punctuation differences."""
        step = ScenarioStep(
            message="Summarize prior stance",
            label="memory_synthesis_probe",
            expect=StepExpectation(snapshot_should_mention=["open source"]),
        )
        result = self._result("irrelevant response")
        result.snapshot_after = "My prior view on open-source governance still holds."
        _check_expectations(step, result)
        assert result.passed

    def test_rapid_ess_slack_allows_borderline_min_ess(self) -> None:
        """Rapid-mode ESS slack should avoid failing near-threshold steps."""
        step = ScenarioStep(
            message="Evidence update",
            label="memory_synthesis_probe",
            expect=StepExpectation(min_ess=0.5),
        )
        result = self._result("evidence and safety both shape my stance.")
        result.ess_score = 0.36
        _check_expectations(step, result, ess_min_slack=0.15)
        assert result.passed
