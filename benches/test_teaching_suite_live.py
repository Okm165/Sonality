from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from sonality import config

from .teaching_harness import EvalProfile, MetricOutcome, run_teaching_benchmark

bench_live = pytest.mark.skipif(not config.API_KEY, reason="SONALITY_API_KEY not set")

EXPECTED_RUN_ARTIFACTS: Final[tuple[str, ...]] = (
    "run_manifest.json",
    "turn_trace.jsonl",
    "ess_trace.jsonl",
    "belief_delta_trace.jsonl",
    "continuity_probe_trace.jsonl",
    "selective_revision_trace.jsonl",
    "misinformation_trace.jsonl",
    "source_vigilance_trace.jsonl",
    "longmem_trace.jsonl",
    "perturbation_trace.jsonl",
    "argument_defense_trace.jsonl",
    "prebunking_trace.jsonl",
    "narrative_identity_trace.jsonl",
    "contradiction_resolution_trace.jsonl",
    "value_coherence_trace.jsonl",
    "epistemic_calibration_trace.jsonl",
    "trajectory_drift_trace.jsonl",
    "revision_fidelity_trace.jsonl",
    "source_reputation_transfer_trace.jsonl",
    "identity_threat_resilience_trace.jsonl",
    "counterfactual_recovery_trace.jsonl",
    "consensus_pressure_resilience_trace.jsonl",
    "delayed_regrounding_trace.jsonl",
    "cross_session_reconciliation_trace.jsonl",
    "source_memory_integrity_trace.jsonl",
    "cross_topic_ledger_consistency_trace.jsonl",
    "belief_decay_retention_trace.jsonl",
    "spacing_durability_trace.jsonl",
    "recency_quality_tradeoff_trace.jsonl",
    "causal_replacement_fidelity_trace.jsonl",
    "inoculation_booster_durability_trace.jsonl",
    "motivated_skepticism_resilience_trace.jsonl",
    "source_tag_decay_resilience_trace.jsonl",
    "base_rate_anecdote_resilience_trace.jsonl",
    "interference_partition_retention_trace.jsonl",
    "source_rehabilitation_hysteresis_trace.jsonl",
    "framing_invariance_resilience_trace.jsonl",
    "countermyth_causal_chain_consistency_trace.jsonl",
    "majority_trust_repair_conflict_trace.jsonl",
    "contradictory_confidence_regrounding_trace.jsonl",
    "provenance_conflict_arbitration_trace.jsonl",
    "value_priority_conflict_stability_trace.jsonl",
    "long_delay_identity_consistency_trace.jsonl",
    "cross_domain_provenance_transfer_boundary_trace.jsonl",
    "false_balance_weight_of_evidence_resilience_trace.jsonl",
    "outgroup_source_derogation_resilience_trace.jsonl",
    "commitment_consistency_pressure_resilience_trace.jsonl",
    "authority_bias_evidence_priority_resilience_trace.jsonl",
    "anchoring_adjustment_resilience_trace.jsonl",
    "status_quo_default_resilience_trace.jsonl",
    "sunk_cost_escalation_resilience_trace.jsonl",
    "outcome_bias_process_fidelity_resilience_trace.jsonl",
    "hindsight_certainty_resilience_trace.jsonl",
    "omission_bias_action_inaction_resilience_trace.jsonl",
    "endowment_effect_ownership_resilience_trace.jsonl",
    "ambiguity_aversion_evidence_priority_resilience_trace.jsonl",
    "belief_perseverance_debiasing_resilience_trace.jsonl",
    "correspondence_bias_situational_resilience_trace.jsonl",
    "conjunction_fallacy_probability_resilience_trace.jsonl",
    "memory_structure_trace.jsonl",
    "memory_leakage_trace.jsonl",
    "health_metrics_trace.jsonl",
    "observer_verdict_trace.jsonl",
    "stop_rule_trace.jsonl",
    "cost_ledger.json",
    "judge_calibration_report.json",
    "health_summary_report.json",
    "dataset_admission_report.json",
    "run_summary.json",
)


def _missing_artifact_message(artifact_name: str) -> str:
    normalized = artifact_name.removesuffix(".jsonl").removesuffix(".json").replace("_", "-")
    return f"Missing {normalized} artifact."


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
    for artifact_name in EXPECTED_RUN_ARTIFACTS:
        assert (run_dir / artifact_name).exists(), _missing_artifact_message(artifact_name)

    assert replicates >= bench_profile.min_runs

    hard_gates = [metric for metric in outcomes if metric.hard_gate]
    _assert_hard_gates_pass(hard_gates, blockers)


def _assert_hard_gates_pass(hard_gates: list[MetricOutcome], blockers: list[str]) -> None:
    failed = [metric.key for metric in hard_gates if metric.status != "pass"]
    assert not failed, (
        f"Hard gate failures: {failed}; blockers={blockers}. "
        "Check run_summary.json for detailed metric traces."
    )
