"""Shared scenario execution helpers for live benchmarks/tests.

Updated to work with the new Neo4j/Qdrant-based memory architecture.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from .scenario_contracts import (
    MAX_ESS_UNSET,
    MIN_ESS_UNSET,
    DisagreementExpectation,
    OpinionDirectionExpectation,
    ScenarioStep,
    StepExpectation,
    UpdateExpectation,
)

if TYPE_CHECKING:
    from sonality.agent import SonalityAgent


@dataclass(slots=True)
class StepResult:
    """Per-step benchmark artifact used by harness gates and reporting."""

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
    memory_update_observed: bool = False
    memory_write_observed: bool = False
    opinion_vectors_changed: bool = False
    staged_updates_added: bool = False
    staged_updates_committed: bool = False
    staged_updates_before: int = 0
    staged_updates_after: int = 0
    pending_insights_before: int = 0
    pending_insights_after: int = 0
    knowledge_writes: int = 0
    interaction_count_before: int = 0
    interaction_count_after: int = 0
    episode_count_before: int = 0
    episode_count_after: int = 0
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


NO_SESSION_SPLIT: Final = -1
TEXT_TOKEN_PATTERN: Final = re.compile(r"[a-z0-9]+")


StepProgressEvent = Literal["start", "end"]
StepProgressCallback = Callable[[StepProgressEvent, int, int, ScenarioStep, object], None]


def _noop_step_progress(
    event: StepProgressEvent,
    step_index: int,
    total_steps: int,
    step: ScenarioStep,
    payload: object,
) -> None:
    _ = (event, step_index, total_steps, step, payload)


NO_STEP_PROGRESS: Final[StepProgressCallback] = _noop_step_progress


@dataclass(frozen=True, slots=True)
class _StepBaseline:
    """Pre-step state used to build one step result."""

    episode_count: int
    personality_version: int
    snapshot: str
    beliefs: dict[str, float]


def _capture_step_baseline(agent: SonalityAgent) -> _StepBaseline:
    """Capture pre-step state used for result deltas."""
    snapshot = agent._run_async(agent._graph.get_personality_snapshot())
    beliefs = agent._run_async(agent._graph.get_all_beliefs())
    episode_count = agent._run_async(agent._graph.get_episode_count())
    return _StepBaseline(
        episode_count=episode_count,
        personality_version=snapshot.version,
        snapshot=snapshot.text,
        beliefs={b.topic: b.valence for b in beliefs},
    )


def _build_step_result(
    step: ScenarioStep,
    agent: SonalityAgent,
    response: str,
    before: _StepBaseline,
) -> StepResult:
    """Build one benchmark step artifact from pre/post agent state."""
    ess = agent.last_ess
    usage = getattr(agent, "last_usage", None)

    snapshot_after = agent._run_async(agent._graph.get_personality_snapshot())
    beliefs_after = agent._run_async(agent._graph.get_all_beliefs())
    episode_count_after = agent._run_async(agent._graph.get_episode_count())

    beliefs_dict = {b.topic: b.valence for b in beliefs_after}
    topics_dict = {b.topic: b.evidence_count for b in beliefs_after}

    opinions_changed = before.beliefs.keys() != beliefs_dict.keys() or any(
        abs(beliefs_dict.get(t, 0) - v) > 1e-9 for t, v in before.beliefs.items()
    )

    version_bumped = snapshot_after.version > before.personality_version
    episode_added = episode_count_after > before.episode_count

    memory_update_observed = version_bumped or opinions_changed or episode_added
    memory_write_observed = episode_added or opinions_changed

    knowledge_writes = getattr(agent, "last_knowledge_writes", 0)
    if knowledge_writes > 0:
        memory_write_observed = True

    return StepResult(
        label=step.label,
        ess_score=ess.score if ess else -1.0,
        ess_reasoning_type=ess.reasoning_type.value if ess else "unknown",
        ess_opinion_direction=ess.opinion_direction.value if ess else "neutral",
        ess_used_defaults=ess.used_defaults if ess else True,
        sponge_version_before=before.personality_version,
        sponge_version_after=snapshot_after.version,
        snapshot_before=before.snapshot,
        snapshot_after=snapshot_after.text,
        disagreement_before=0.0,
        disagreement_after=0.0,
        did_disagree=False,
        opinion_vectors=beliefs_dict,
        topics_tracked=topics_dict,
        response_text=response,
        memory_update_observed=memory_update_observed,
        memory_write_observed=memory_write_observed,
        opinion_vectors_changed=opinions_changed,
        staged_updates_added=False,
        staged_updates_committed=False,
        staged_updates_before=0,
        staged_updates_after=0,
        pending_insights_before=0,
        pending_insights_after=0,
        knowledge_writes=knowledge_writes,
        interaction_count_before=before.episode_count,
        interaction_count_after=episode_count_after,
        episode_count_before=before.episode_count,
        episode_count_after=episode_count_after,
        response_calls=int(getattr(usage, "response_calls", 1)),
        ess_calls=int(getattr(usage, "ess_calls", 1)),
        response_input_tokens=int(getattr(usage, "response_input_tokens", 0)),
        response_output_tokens=int(getattr(usage, "response_output_tokens", 0)),
        ess_input_tokens=int(getattr(usage, "ess_input_tokens", 0)),
        ess_output_tokens=int(getattr(usage, "ess_output_tokens", 0)),
        ess_defaulted_fields=ess.defaulted_fields if ess else (),
        ess_default_severity=ess.default_severity if ess else "exception",
    )


def run_scenario(
    scenario: Sequence[ScenarioStep],
    neo4j_url: str | None = None,
    qdrant_url: str | None = None,
    session_split_at: int = NO_SESSION_SPLIT,
    step_progress: StepProgressCallback = NO_STEP_PROGRESS,
    ess_min_slack: float = 0.0,
    ess_max_slack: float = 0.0,
) -> list[StepResult]:
    """Run a scenario with an optional session restart boundary.

    Args:
        scenario: Sequence of steps to execute.
        neo4j_url: Optional Neo4j URL override (uses config default if None).
        qdrant_url: Optional Qdrant URL override (uses config default if None).
        session_split_at: Index to restart agent (NO_SESSION_SPLIT to skip).
        step_progress: Callback for progress updates.
        ess_min_slack: Tolerance for ESS minimum thresholds.
        ess_max_slack: Tolerance for ESS maximum thresholds.

    Returns:
        List of StepResult objects for each scenario step.
    """
    scenario_len = len(scenario)
    if session_split_at != NO_SESSION_SPLIT and not (0 < session_split_at < scenario_len):
        raise ValueError("session_split_at must be within scenario bounds")
    split_index = session_split_at

    from sonality.agent import SonalityAgent

    agent = SonalityAgent()
    try:
        results: list[StepResult] = []

        for idx, step in enumerate(scenario):
            step_index = idx + 1
            step_progress("start", step_index, scenario_len, step, "start")
            if idx == split_index:
                agent.shutdown()
                agent = SonalityAgent()
            before = _capture_step_baseline(agent)

            try:
                response = agent.respond(step.message)
                result = _build_step_result(
                    step=step, agent=agent, response=response, before=before
                )
                _check_expectations(
                    step,
                    result,
                    ess_min_slack=ess_min_slack,
                    ess_max_slack=ess_max_slack,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Scenario step failed ({step_index}/{scenario_len}, label='{step.label}')"
                ) from exc
            step_progress("end", step_index, scenario_len, step, result)
            results.append(result)

        return results
    finally:
        agent.shutdown()


def _append_ess_threshold_failures(
    e: StepExpectation,
    result: StepResult,
    *,
    ess_min_slack: float,
    ess_max_slack: float,
) -> None:
    """Append ESS min/max threshold failures for one scenario step result."""
    min_slack = max(ess_min_slack, 0.0)
    max_slack = max(ess_max_slack, 0.0)
    effective_min_ess = max(MIN_ESS_UNSET, e.min_ess - min_slack)
    effective_max_ess = min(MAX_ESS_UNSET, e.max_ess + max_slack)

    if e.min_ess > MIN_ESS_UNSET and result.ess_score < effective_min_ess:
        result.failures.append(
            f"ESS {result.ess_score:.2f} < min {e.min_ess}"
            + (f" (effective {effective_min_ess:.2f})" if effective_min_ess != e.min_ess else "")
        )
    if e.max_ess < MAX_ESS_UNSET and result.ess_score > effective_max_ess:
        result.failures.append(
            f"ESS {result.ess_score:.2f} > max {e.max_ess}"
            + (f" (effective {effective_max_ess:.2f})" if effective_max_ess != e.max_ess else "")
        )


def _normalize_text_for_match(text: str) -> str:
    """Normalize text into lowercase alphanumeric tokens for robust matching."""
    return " ".join(TEXT_TOKEN_PATTERN.findall(text.lower()))


def _contains_term(normalized_text: str, term: str) -> bool:
    """Check if a normalized term phrase is present in normalized text."""
    normalized_term = _normalize_text_for_match(term)
    return bool(normalized_term) and normalized_term in normalized_text


def _append_reasoning_direction_failures(e: StepExpectation, result: StepResult) -> None:
    """Append reasoning-type and opinion-direction contract failures."""
    if e.expected_reasoning_types and result.ess_reasoning_type not in e.expected_reasoning_types:
        result.failures.append(
            f"reasoning_type '{result.ess_reasoning_type}' not in {e.expected_reasoning_types}"
        )
    if (
        e.expect_opinion_direction is OpinionDirectionExpectation.ALLOW_ANY
        or result.ess_opinion_direction == e.expect_opinion_direction.value
    ):
        return
    result.failures.append(
        "opinion_direction "
        f"'{result.ess_opinion_direction}' != expected "
        f"'{e.expect_opinion_direction.value}'"
    )


def _append_update_policy_failures(e: StepExpectation, result: StepResult) -> None:
    """Append memory-update policy failures for one scenario step result."""
    if e.sponge_should_update is UpdateExpectation.MUST_UPDATE and not result.memory_write_observed:
        result.failures.append("Memory should have updated but did not")
    if e.sponge_should_update is UpdateExpectation.MUST_NOT_UPDATE:
        opinion_update_observed = (
            result.opinion_vectors_changed
            or result.staged_updates_added
            or result.pending_insights_after > result.pending_insights_before
        )
        if opinion_update_observed:
            update_signals: list[str] = []
            if result.sponge_version_after > result.sponge_version_before:
                update_signals.append(
                    f"version v{result.sponge_version_before}->v{result.sponge_version_after}"
                )
            if result.opinion_vectors_changed:
                update_signals.append("beliefs changed")
            if result.staged_updates_added:
                update_signals.append(
                    f"staged_updates {result.staged_updates_before}->{result.staged_updates_after}"
                )
            if result.pending_insights_after > result.pending_insights_before:
                update_signals.append(
                    f"pending_insights {result.pending_insights_before}->{result.pending_insights_after}"
                )
            result.failures.append(
                "Memory should NOT have updated but did"
                + (f" ({', '.join(update_signals)})" if update_signals else "")
            )


def _append_disagreement_failures(e: StepExpectation, result: StepResult) -> None:
    """Append disagreement expectation failures for one scenario step result."""
    if e.expect_disagreement is DisagreementExpectation.MUST_DISAGREE and not result.did_disagree:
        result.failures.append(
            f"disagreement {result.did_disagree} != expected "
            f"{DisagreementExpectation.MUST_DISAGREE.value}"
        )
    if e.expect_disagreement is DisagreementExpectation.MUST_NOT_DISAGREE and result.did_disagree:
        result.failures.append(
            f"disagreement {result.did_disagree} != expected "
            f"{DisagreementExpectation.MUST_NOT_DISAGREE.value}"
        )


def _append_topics_contain_failures(e: StepExpectation, result: StepResult) -> None:
    """Append failures when expected topics are absent from the tracked topic set."""
    if not e.topics_contain:
        return
    tracked_normalized = {_normalize_text_for_match(t) for t in result.topics_tracked}
    for expected in e.topics_contain:
        norm = _normalize_text_for_match(expected)
        if not any(norm in tracked or tracked.startswith(norm) for tracked in tracked_normalized):
            result.failures.append(
                f"Topic '{expected}' not found in tracked topics {sorted(result.topics_tracked)}"
            )


def _append_snapshot_text_failures(e: StepExpectation, result: StepResult) -> None:
    """Append snapshot mention/non-mention contract failures."""
    normalized_snapshot = _normalize_text_for_match(result.snapshot_after)
    for term in e.snapshot_should_mention:
        if not _contains_term(normalized_snapshot, term):
            result.failures.append(f"Snapshot should mention '{term}' but does not")
    for term in e.snapshot_should_not_mention:
        if _contains_term(normalized_snapshot, term):
            result.failures.append(f"Snapshot should NOT mention '{term}' but does")


def _append_response_text_failures(e: StepExpectation, result: StepResult) -> None:
    """Append response mention/non-mention contract failures."""
    normalized_response = _normalize_text_for_match(result.response_text)
    if e.response_should_mention and not any(
        _contains_term(normalized_response, term) for term in e.response_should_mention
    ):
        result.failures.append(
            f"Response should mention one of {e.response_should_mention} but did not"
        )
    for term in e.response_should_mention_all:
        if not _contains_term(normalized_response, term):
            result.failures.append(f"Response should mention '{term}' but does not")
    for term in e.response_should_not_mention:
        if _contains_term(normalized_response, term):
            result.failures.append(f"Response should NOT mention '{term}' but does")


def _check_expectations(
    step: ScenarioStep,
    result: StepResult,
    *,
    ess_min_slack: float = 0.0,
    ess_max_slack: float = 0.0,
) -> None:
    """Evaluate scenario expectations and append any contract failures."""
    e = step.expect
    _append_ess_threshold_failures(
        e,
        result,
        ess_min_slack=ess_min_slack,
        ess_max_slack=ess_max_slack,
    )
    _append_reasoning_direction_failures(e, result)
    _append_update_policy_failures(e, result)
    _append_disagreement_failures(e, result)
    _append_topics_contain_failures(e, result)
    _append_snapshot_text_failures(e, result)
    _append_response_text_failures(e, result)

    if result.failures:
        result.passed = False
