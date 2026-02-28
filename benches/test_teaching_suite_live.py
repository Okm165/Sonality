from __future__ import annotations

from pathlib import Path

import pytest

from sonality import config

from .teaching_harness import EvalProfile, MetricOutcome, run_teaching_benchmark

bench_live = pytest.mark.skipif(not config.API_KEY, reason="SONALITY_API_KEY not set")


@pytest.mark.bench
@pytest.mark.live
@bench_live
def test_teaching_suite_benchmark(
    bench_profile: EvalProfile,
    bench_output_root: Path,
) -> None:
    run_dir, outcomes, replicates, blockers = run_teaching_benchmark(
        profile=bench_profile,
        output_root=bench_output_root,
    )

    assert run_dir.exists(), "Benchmark run directory was not created."
    assert (run_dir / "run_manifest.json").exists(), "Missing run manifest artifact."
    assert (run_dir / "turn_trace.jsonl").exists(), "Missing per-turn trace artifact."
    assert (run_dir / "ess_trace.jsonl").exists(), "Missing ESS trace artifact."
    assert (run_dir / "belief_delta_trace.jsonl").exists(), "Missing belief-delta trace artifact."
    assert (run_dir / "continuity_probe_trace.jsonl").exists(), "Missing continuity trace artifact."
    assert (run_dir / "memory_structure_trace.jsonl").exists(), "Missing memory-structure trace."
    assert (run_dir / "memory_leakage_trace.jsonl").exists(), "Missing memory-leakage trace."
    assert (run_dir / "observer_verdict_trace.jsonl").exists(), "Missing observer verdict trace."
    assert (run_dir / "stop_rule_trace.jsonl").exists(), "Missing stop-rule trace artifact."
    assert (run_dir / "cost_ledger.json").exists(), "Missing cost ledger artifact."
    assert (run_dir / "dataset_admission_report.json").exists(), "Missing dataset admission artifact."
    assert (run_dir / "run_summary.json").exists(), "Missing run summary artifact."

    assert replicates >= bench_profile.min_runs

    hard_gates = [metric for metric in outcomes if metric.hard_gate]
    _assert_hard_gates_pass(hard_gates, blockers)


def _assert_hard_gates_pass(hard_gates: list[MetricOutcome], blockers: list[str]) -> None:
    failed = [metric.key for metric in hard_gates if metric.status != "pass"]
    assert not failed, (
        f"Hard gate failures: {failed}; blockers={blockers}. "
        "Check run_summary.json for detailed metric traces."
    )
