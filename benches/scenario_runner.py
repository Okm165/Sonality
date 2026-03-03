"""Shared scenario execution helpers for live benchmarks/tests."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal

from sonality import config

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
StepProgressCallback = Callable[
    [StepProgressEvent, int, int, ScenarioStep, StepResult | None], None
]


@dataclass(frozen=True, slots=True)
class _StepBaseline:
    """Pre-step sponge state used to build one step result."""

    interaction_count: int
    disagreement_rate: float
    sponge_version: int
    snapshot: str
    opinion_vectors: dict[str, float]
    staged_updates: int
    pending_insights: int
    episode_count: int


def _validated_session_split(session_split_at: int, scenario_len: int) -> int:
    """Validate optional session split index and return canonical value."""
    if session_split_at == NO_SESSION_SPLIT:
        return NO_SESSION_SPLIT
    if 0 < session_split_at < scenario_len:
        return session_split_at
    raise ValueError("session_split_at must be within scenario bounds")


def _capture_step_baseline(agent: SonalityAgent) -> _StepBaseline:
    """Capture pre-step sponge state used for result deltas."""
    sponge = agent.sponge
    return _StepBaseline(
        interaction_count=sponge.interaction_count,
        disagreement_rate=sponge.behavioral_signature.disagreement_rate,
        sponge_version=sponge.version,
        snapshot=sponge.snapshot,
        opinion_vectors=dict(sponge.opinion_vectors),
        staged_updates=len(sponge.staged_opinion_updates),
        pending_insights=len(sponge.pending_insights),
        episode_count=agent.episodes.collection.count(),
    )


def _did_disagree(
    interaction_before: int,
    disagreement_before: float,
    interaction_after: int,
    disagreement_after: float,
) -> bool:
    """Infer disagreement event from disagreement-rate delta semantics."""
    disagreement_delta = (
        disagreement_after * interaction_after - disagreement_before * interaction_before
    )
    return disagreement_delta >= 0.5


def _usage_int(usage: object, field_name: str, default: int) -> int:
    """Read integer usage counters from runtime usage snapshots."""
    return int(getattr(usage, field_name, default))


def _opinion_vectors_changed(before: dict[str, float], after: dict[str, float]) -> bool:
    """Return whether opinion-vector keys or values changed this step."""
    if before.keys() != after.keys():
        return True
    return any(abs(after[topic] - before_value) > 1e-9 for topic, before_value in before.items())


def _build_step_result(
    step: ScenarioStep,
    agent: SonalityAgent,
    response: str,
    before: _StepBaseline,
) -> StepResult:
    """Build one benchmark step artifact from pre/post agent state."""
    ess = agent.last_ess
    usage = getattr(agent, "last_usage", None)
    sponge = agent.sponge
    disagreement_after = agent.sponge.behavioral_signature.disagreement_rate
    interaction_after = agent.sponge.interaction_count
    episode_count_after = agent.episodes.collection.count()
    opinion_vectors_after = dict(sponge.opinion_vectors)
    staged_updates_after = len(sponge.staged_opinion_updates)
    pending_insights_after = len(sponge.pending_insights)
    opinions_changed = _opinion_vectors_changed(before.opinion_vectors, opinion_vectors_after)
    staged_updates_added = staged_updates_after > before.staged_updates
    staged_updates_committed = staged_updates_after < before.staged_updates
    version_bumped = sponge.version > before.sponge_version
    pending_insights_added = pending_insights_after > before.pending_insights
    memory_update_observed = (
        version_bumped
        or opinions_changed
        or staged_updates_added
        or staged_updates_committed
        or pending_insights_added
    )
    memory_write_observed = (
        version_bumped
        or staged_updates_added
        or pending_insights_added
        or (opinions_changed and not staged_updates_committed)
    )
    return StepResult(
        label=step.label,
        ess_score=ess.score if ess else -1.0,
        ess_reasoning_type=ess.reasoning_type.value if ess else "unknown",
        ess_opinion_direction=ess.opinion_direction.value if ess else "neutral",
        ess_used_defaults=ess.used_defaults if ess else True,
        sponge_version_before=before.sponge_version,
        sponge_version_after=sponge.version,
        snapshot_before=before.snapshot,
        snapshot_after=sponge.snapshot,
        disagreement_before=before.disagreement_rate,
        disagreement_after=disagreement_after,
        did_disagree=_did_disagree(
            interaction_before=before.interaction_count,
            disagreement_before=before.disagreement_rate,
            interaction_after=interaction_after,
            disagreement_after=disagreement_after,
        ),
        opinion_vectors=opinion_vectors_after,
        topics_tracked=dict(sponge.behavioral_signature.topic_engagement),
        response_text=response,
        memory_update_observed=memory_update_observed,
        memory_write_observed=memory_write_observed,
        opinion_vectors_changed=opinions_changed,
        staged_updates_added=staged_updates_added,
        staged_updates_committed=staged_updates_committed,
        staged_updates_before=before.staged_updates,
        staged_updates_after=staged_updates_after,
        pending_insights_before=before.pending_insights,
        pending_insights_after=pending_insights_after,
        interaction_count_before=before.interaction_count,
        interaction_count_after=interaction_after,
        episode_count_before=before.episode_count,
        episode_count_after=episode_count_after,
        response_calls=_usage_int(usage, "response_calls", 1),
        ess_calls=_usage_int(usage, "ess_calls", 1),
        response_input_tokens=_usage_int(usage, "response_input_tokens", 0),
        response_output_tokens=_usage_int(usage, "response_output_tokens", 0),
        ess_input_tokens=_usage_int(usage, "ess_input_tokens", 0),
        ess_output_tokens=_usage_int(usage, "ess_output_tokens", 0),
        ess_defaulted_fields=ess.defaulted_fields if ess else (),
        ess_default_severity=ess.default_severity if ess else "exception",
    )


def run_scenario(
    scenario: Sequence[ScenarioStep],
    tmp_dir: str,
    session_split_at: int = NO_SESSION_SPLIT,
    step_progress: StepProgressCallback | None = None,
    ess_min_slack: float = 0.0,
    ess_max_slack: float = 0.0,
) -> list[StepResult]:
    """Run a scenario with an optional session restart boundary.

    The split is encoded with the `NO_SESSION_SPLIT` sentinel to avoid nullable
    control flow in benchmark orchestration helpers.
    """
    scenario_len = len(scenario)
    split_index = _validated_session_split(
        session_split_at=session_split_at, scenario_len=scenario_len
    )

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
            step_index = idx + 1
            if step_progress is not None:
                step_progress("start", step_index, scenario_len, step, None)
            if idx == split_index:
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
            if step_progress is not None:
                step_progress("end", step_index, scenario_len, step, result)
            results.append(result)

        return results


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
            + (
                f" (effective {effective_min_ess:.2f})"
                if effective_min_ess != e.min_ess
                else ""
            )
        )
    if e.max_ess < MAX_ESS_UNSET and result.ess_score > effective_max_ess:
        result.failures.append(
            f"ESS {result.ess_score:.2f} > max {e.max_ess}"
            + (
                f" (effective {effective_max_ess:.2f})"
                if effective_max_ess != e.max_ess
                else ""
            )
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
    """Append sponge-update policy failures for one scenario step result."""
    if e.sponge_should_update is UpdateExpectation.MUST_UPDATE and not result.memory_write_observed:
        result.failures.append("Sponge should have updated but did not")
    if e.sponge_should_update is UpdateExpectation.MUST_NOT_UPDATE and result.memory_write_observed:
        update_signals: list[str] = []
        if result.sponge_version_after > result.sponge_version_before:
            update_signals.append(
                f"version v{result.sponge_version_before}->v{result.sponge_version_after}"
            )
        if result.opinion_vectors_changed and not result.staged_updates_committed:
            update_signals.append("opinion_vectors changed")
        if result.staged_updates_added:
            update_signals.append(
                f"staged_updates {result.staged_updates_before}->{result.staged_updates_after}"
            )
        if result.pending_insights_after > result.pending_insights_before:
            update_signals.append(
                f"pending_insights {result.pending_insights_before}->{result.pending_insights_after}"
            )
        result.failures.append(
            "Sponge should NOT have updated but did"
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

    _append_snapshot_text_failures(e, result)
    _append_response_text_failures(e, result)

    if result.failures:
        result.passed = False
