from __future__ import annotations

import pytest

from .scenario_contracts import ScenarioStep, StepExpectation
from .scenario_runner import StepResult, _check_expectations

pytestmark = pytest.mark.bench


class TestStepExpectationContracts:
    """Deterministic contract checks for benchmark scenario expectations."""

    @staticmethod
    def _result(response_text: str) -> StepResult:
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
        step = ScenarioStep(
            message="Synthesize personality context",
            label="memory_synthesis_probe",
            expect=StepExpectation(response_should_mention_all=["evidence", "safety"]),
        )
        result = self._result("evidence and safety both shape my stance.")
        _check_expectations(step, result)
        assert result.passed
