"""Shared scenario execution helpers for live benchmarks/tests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from sonality import config

from .scenario_contracts import ScenarioStep


@dataclass(slots=True)
class StepResult:
    label: str
    ess_score: float
    ess_reasoning_type: str
    ess_opinion_direction: str
    ess_used_defaults: bool
    sponge_version_before: int
    sponge_version_after: int
    snapshot_before: str
    snapshot_after: str
    disagreement_before: float
    disagreement_after: float
    did_disagree: bool
    opinion_vectors: dict[str, float]
    topics_tracked: dict[str, int]
    response_text: str
    response_calls: int = 0
    ess_calls: int = 0
    response_input_tokens: int = 0
    response_output_tokens: int = 0
    ess_input_tokens: int = 0
    ess_output_tokens: int = 0
    ess_defaulted_fields: tuple[str, ...] = ()
    ess_default_severity: str = "none"
    passed: bool = True
    failures: list[str] = field(default_factory=list)


def run_scenario(
    scenario: Sequence[ScenarioStep], tmp_dir: str, session_split_at: int | None = None
) -> list[StepResult]:
    """Run a scenario with optional session split for continuity probes."""
    if session_split_at is not None and not (0 < session_split_at < len(scenario)):
        raise ValueError("session_split_at must be within scenario bounds")

    import unittest.mock as mock

    with (
        mock.patch.object(config, "SPONGE_FILE", Path(tmp_dir) / "sponge.json"),
        mock.patch.object(config, "SPONGE_HISTORY_DIR", Path(tmp_dir) / "history"),
        mock.patch.object(config, "CHROMADB_DIR", Path(tmp_dir) / "chromadb"),
        mock.patch.object(config, "ESS_AUDIT_LOG_FILE", Path(tmp_dir) / "ess_log.jsonl"),
    ):
        from sonality.agent import SonalityAgent

        agent = SonalityAgent()
        results: list[StepResult] = []

        for idx, step in enumerate(scenario):
            if session_split_at is not None and idx == session_split_at:
                agent = SonalityAgent()

            interaction_before = agent.sponge.interaction_count
            disagreement_before = agent.sponge.behavioral_signature.disagreement_rate
            version_before = agent.sponge.version
            snapshot_before = agent.sponge.snapshot

            response = agent.respond(step.message)

            interaction_after = agent.sponge.interaction_count
            disagreement_after = agent.sponge.behavioral_signature.disagreement_rate
            disagreement_delta = (
                disagreement_after * interaction_after - disagreement_before * interaction_before
            )
            did_disagree = disagreement_delta >= 0.5

            ess = agent.last_ess
            usage = getattr(agent, "last_usage", None)
            result = StepResult(
                label=step.label,
                ess_score=ess.score if ess else -1.0,
                ess_reasoning_type=ess.reasoning_type.value if ess else "unknown",
                ess_opinion_direction=ess.opinion_direction.value if ess else "neutral",
                ess_used_defaults=ess.used_defaults if ess else True,
                sponge_version_before=version_before,
                sponge_version_after=agent.sponge.version,
                snapshot_before=snapshot_before,
                snapshot_after=agent.sponge.snapshot,
                disagreement_before=disagreement_before,
                disagreement_after=disagreement_after,
                did_disagree=did_disagree,
                opinion_vectors=dict(agent.sponge.opinion_vectors),
                topics_tracked=dict(agent.sponge.behavioral_signature.topic_engagement),
                response_text=response,
                response_calls=int(getattr(usage, "response_calls", 1)),
                ess_calls=int(getattr(usage, "ess_calls", 1)),
                response_input_tokens=int(getattr(usage, "response_input_tokens", 0)),
                response_output_tokens=int(getattr(usage, "response_output_tokens", 0)),
                ess_input_tokens=int(getattr(usage, "ess_input_tokens", 0)),
                ess_output_tokens=int(getattr(usage, "ess_output_tokens", 0)),
                ess_defaulted_fields=ess.defaulted_fields if ess else (),
                ess_default_severity=ess.default_severity if ess else "exception",
            )
            _check_expectations(step, result)
            results.append(result)

        return results


def _check_expectations(step: ScenarioStep, result: StepResult) -> None:
    e = step.expect

    if e.min_ess is not None and result.ess_score < e.min_ess:
        result.failures.append(f"ESS {result.ess_score:.2f} < min {e.min_ess}")

    if e.max_ess is not None and result.ess_score > e.max_ess:
        result.failures.append(f"ESS {result.ess_score:.2f} > max {e.max_ess}")

    if e.expected_reasoning_types and result.ess_reasoning_type not in e.expected_reasoning_types:
        result.failures.append(
            f"reasoning_type '{result.ess_reasoning_type}' not in {e.expected_reasoning_types}"
        )

    if e.expect_opinion_direction and result.ess_opinion_direction != e.expect_opinion_direction:
        result.failures.append(
            "opinion_direction "
            f"'{result.ess_opinion_direction}' != expected '{e.expect_opinion_direction}'"
        )

    if (
        e.sponge_should_update is True
        and result.sponge_version_after <= result.sponge_version_before
    ):
        result.failures.append("Sponge should have updated but did not")

    if (
        e.sponge_should_update is False
        and result.sponge_version_after > result.sponge_version_before
    ):
        result.failures.append(
            "Sponge should NOT have updated but went "
            f"v{result.sponge_version_before}->v{result.sponge_version_after}"
        )

    if e.expect_disagreement is not None and result.did_disagree != e.expect_disagreement:
        result.failures.append(
            f"disagreement {result.did_disagree} != expected {e.expect_disagreement}"
        )

    # Topic naming is model-dependent; keep this as informational only.
    if e.topics_contain:
        tracked = set(result.topics_tracked)
        for topic_hint in e.topics_contain:
            if not any(topic_hint.lower() in t.lower() for t in tracked):
                continue

    if e.snapshot_should_mention:
        for term in e.snapshot_should_mention:
            if term.lower() not in result.snapshot_after.lower():
                result.failures.append(f"Snapshot should mention '{term}' but does not")

    if e.snapshot_should_not_mention:
        for term in e.snapshot_should_not_mention:
            if term.lower() in result.snapshot_after.lower():
                result.failures.append(f"Snapshot should NOT mention '{term}' but does")

    if e.response_should_mention:
        response_lower = result.response_text.lower()
        if not any(term.lower() in response_lower for term in e.response_should_mention):
            result.failures.append(
                "Response should mention one of "
                f"{e.response_should_mention} but did not"
            )

    if e.response_should_mention_all:
        response_lower = result.response_text.lower()
        for term in e.response_should_mention_all:
            if term.lower() not in response_lower:
                result.failures.append(f"Response should mention '{term}' but does not")

    if e.response_should_not_mention:
        response_lower = result.response_text.lower()
        for term in e.response_should_not_mention:
            if term.lower() in response_lower:
                result.failures.append(f"Response should NOT mention '{term}' but does")

    if result.failures:
        result.passed = False

