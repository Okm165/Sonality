from __future__ import annotations

from pathlib import Path

import pytest

from sonality import config

from .teaching_harness import (
    CONTRACT_PACK_SPECS,
    BenchProgressLevel,
    EvalProfile,
    MetricOutcome,
    PackDefinition,
    _probe_artifact_names,
    run_teaching_benchmark,
)

bench_live = pytest.mark.skipif(not config.API_KEY, reason="No provider API key configured")

def _expected_run_artifacts() -> tuple[str, ...]:
    """Return expected artifact names derived from harness declarations."""
    contract_traces = tuple(f"{key}_trace.jsonl" for key in CONTRACT_PACK_SPECS)
    return (
        "run_manifest.json",
        "turn_trace.jsonl",
        "ess_trace.jsonl",
        "belief_delta_trace.jsonl",
        "run_isolation_trace.jsonl",
        "memory_validity_trace.jsonl",
        *_probe_artifact_names(),
        *contract_traces,
        "health_metrics_trace.jsonl",
        "observer_verdict_trace.jsonl",
        "stop_rule_trace.jsonl",
        "risk_event_trace.jsonl",
        "cost_ledger.json",
        "judge_calibration_report.json",
        "health_summary_report.json",
        "run_isolation_report.json",
        "memory_validity_report.json",
        "belief_memory_alignment_report.json",
        "dataset_admission_report.json",
        "run_summary.json",
    )


def _missing_artifact_message(artifact_name: str) -> str:
    """Test helper for missing artifact message."""
    normalized = artifact_name.removesuffix(".jsonl").removesuffix(".json").replace("_", "-")
    return f"Missing {normalized} artifact."


@pytest.mark.bench
@pytest.mark.live
@bench_live
def test_teaching_suite_benchmark(
    bench_profile: EvalProfile,
    bench_output_root: Path,
    bench_progress: BenchProgressLevel,
    bench_packs: tuple[PackDefinition, ...],
) -> None:
    """Test that teaching suite benchmark."""
    run_dir, outcomes, replicates, blockers = run_teaching_benchmark(
        profile=bench_profile,
        output_root=bench_output_root,
        progress=bench_progress,
        packs=bench_packs,
    )

    assert run_dir.exists(), "Benchmark run directory was not created."
    for artifact_name in _expected_run_artifacts():
        assert (run_dir / artifact_name).exists(), _missing_artifact_message(artifact_name)

    assert replicates >= bench_profile.min_runs

    if bench_profile.name in {"rapid", "lean"}:
        # Rapid/lean are iteration signal modes, not release gates.
        return

    hard_gates = [metric for metric in outcomes if metric.hard_gate]
    _assert_hard_gates_pass(hard_gates, blockers)


def _assert_hard_gates_pass(hard_gates: list[MetricOutcome], blockers: list[str]) -> None:
    """Test helper for assert hard gates pass."""
    failed = [metric.key for metric in hard_gates if metric.status == "fail"]
    assert not failed, (
        f"Hard gate failures: {failed}; blockers={blockers}. "
        "Check run_summary.json for detailed metric traces."
    )
