"""Unit tests for bench harness metric and statistical functions.

All tests are deterministic (no API calls).
"""

from __future__ import annotations

import pytest

from .scenario_runner import StepResult
from .teaching_harness import (
    ESSDefaultFlags,
    _binomial_cdf,
    _ess_default_flags,
    _ess_retry_stats,
    _exact_binomial_interval,
    _min_n_for_zero_failures,
    _proportion_interval_95,
    _width_escalation_status,
    metric_status,
    wilson_interval,
)

pytestmark = pytest.mark.bench


def _step(
    *,
    ess_used_defaults: bool = False,
    ess_default_severity: str = "none",
    ess_defaulted_fields: tuple[str, ...] = (),
    ess_calls: int = 1,
) -> StepResult:
    return StepResult(
        label="t",
        ess_score=0.4,
        ess_reasoning_type="logical_argument",
        ess_opinion_direction="supports",
        ess_used_defaults=ess_used_defaults,
        ess_default_severity=ess_default_severity,
        ess_defaulted_fields=ess_defaulted_fields,
        ess_calls=ess_calls,
        sponge_version_before=1,
        sponge_version_after=1,
        snapshot_before="s",
        snapshot_after="s",
        disagreement_before=0.0,
        disagreement_after=0.0,
        did_disagree=False,
        opinion_vectors={},
        topics_tracked={},
        response_text="ok",
    )


# ── wilson_interval ─────────────────────────────────────────────────────


class TestWilsonInterval:
    def test_all_pass_returns_high_lower_bound(self) -> None:
        lo, hi = wilson_interval(10, 10)
        assert lo > 0.7
        assert hi == 1.0

    def test_all_fail_returns_low_upper_bound(self) -> None:
        lo, hi = wilson_interval(0, 10)
        assert lo == 0.0
        assert hi < 0.3

    def test_half_pass_returns_centered_interval(self) -> None:
        lo, hi = wilson_interval(50, 100)
        mid = (lo + hi) / 2.0
        assert abs(mid - 0.5) < 0.02
        assert lo > 0.4
        assert hi < 0.6

    def test_zero_total_returns_full_uncertainty(self) -> None:
        lo, hi = wilson_interval(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_output_bounds_are_valid_probabilities(self) -> None:
        for s, n in [(0, 1), (1, 1), (7, 10), (3, 3), (0, 100), (100, 100)]:
            lo, hi = wilson_interval(s, n)
            assert 0.0 <= lo <= hi <= 1.0, f"Invalid interval [{lo},{hi}] for s={s},n={n}"


# ── _binomial_cdf ───────────────────────────────────────────────────────


class TestBinomialCDF:
    def test_k_gte_n_returns_one(self) -> None:
        assert _binomial_cdf(10, 10, 0.5) == 1.0

    def test_k_negative_returns_zero(self) -> None:
        assert _binomial_cdf(-1, 10, 0.5) == 0.0

    def test_zero_n_returns_one(self) -> None:
        assert _binomial_cdf(0, 0, 0.5) == 1.0

    def test_fair_coin_median_around_half(self) -> None:
        # P(X <= 5 | n=10, p=0.5) should be around 0.62
        result = _binomial_cdf(5, 10, 0.5)
        assert 0.6 < result < 0.7

    def test_certain_success_gives_one(self) -> None:
        assert _binomial_cdf(5, 10, 0.0) == 1.0  # p=0 → all k satisfy X<=k


# ── _exact_binomial_interval ────────────────────────────────────────────


class TestExactBinomialInterval:
    def test_zero_total_returns_full_uncertainty(self) -> None:
        lo, hi = _exact_binomial_interval(0, 0)
        assert lo == 0.0
        assert hi == 1.0

    def test_all_pass_lower_bound_is_positive(self) -> None:
        lo, hi = _exact_binomial_interval(10, 10)
        assert lo > 0.6
        assert hi == 1.0

    def test_all_fail_upper_bound_is_below_half(self) -> None:
        lo, hi = _exact_binomial_interval(0, 10)
        assert lo == 0.0
        assert hi < 0.4

    def test_interval_contains_true_rate(self) -> None:
        # For 3/5 successes (true rate 0.6), the interval should contain 0.6
        lo, hi = _exact_binomial_interval(3, 5)
        assert lo <= 0.6 <= hi

    def test_bounds_are_valid_probabilities(self) -> None:
        for s, n in [(0, 5), (5, 5), (2, 4), (1, 20)]:
            lo, hi = _exact_binomial_interval(s, n)
            assert 0.0 <= lo <= hi <= 1.0, f"Invalid: s={s}, n={n}, [{lo},{hi}]"


# ── _proportion_interval_95 ─────────────────────────────────────────────


class TestProportionInterval95:
    def test_zero_total_returns_full_uncertainty_and_none_family(self) -> None:
        lo, hi, family = _proportion_interval_95(0, 0)
        assert lo == 0.0
        assert hi == 1.0
        assert family == "none"

    def test_small_n_uses_exact_binomial(self) -> None:
        # n < INTERVAL_SWITCH_SMALL_N_LT (40) → exact_binomial
        lo, hi, family = _proportion_interval_95(5, 10)
        assert family == "exact_binomial"
        assert 0.0 <= lo <= hi <= 1.0

    def test_boundary_case_uses_exact(self) -> None:
        # All pass (boundary) always uses exact regardless of n
        _lo, _hi, family = _proportion_interval_95(50, 50)
        assert family == "exact_binomial"

    def test_large_n_non_boundary_uses_wilson(self) -> None:
        # n >= 40 and not at boundary → wilson
        lo, hi, family = _proportion_interval_95(30, 50)
        assert family == "wilson"
        assert 0.0 <= lo <= hi <= 1.0

    def test_intervals_are_valid_probabilities(self) -> None:
        for s, n in [(0, 1), (1, 100), (99, 100), (50, 100), (5, 40)]:
            lo, hi, _ = _proportion_interval_95(s, n)
            assert 0.0 <= lo <= hi <= 1.0, f"Invalid for s={s}, n={n}"


# ── metric_status ───────────────────────────────────────────────────────


class TestMetricStatus:
    def test_ci_entirely_above_threshold_is_pass(self) -> None:
        assert metric_status(0.8, 0.95, 0.75) == "pass"

    def test_ci_entirely_below_threshold_is_fail(self) -> None:
        assert metric_status(0.3, 0.6, 0.75) == "fail"

    def test_ci_straddles_threshold_is_inconclusive(self) -> None:
        assert metric_status(0.6, 0.9, 0.75) == "inconclusive"

    def test_exact_boundary_ci_low_equals_threshold_is_pass(self) -> None:
        assert metric_status(0.75, 0.90, 0.75) == "pass"

    def test_exact_boundary_ci_high_equals_threshold_is_inconclusive(self) -> None:
        # ci_high == threshold means ci_high is NOT strictly below threshold
        assert metric_status(0.5, 0.75, 0.75) == "inconclusive"


# ── _width_escalation_status ────────────────────────────────────────────


class TestWidthEscalationStatus:
    def test_zero_margin_always_returns_decide(self) -> None:
        _hw, status = _width_escalation_status(ci_low=0.4, ci_high=0.9, margin_value=0.0)
        assert status == "decide"

    def test_narrow_interval_returns_decide(self) -> None:
        # half-width = 0.025, margin = 0.1 → half_width <= 0.5 * margin (0.05) → decide
        hw, status = _width_escalation_status(ci_low=0.475, ci_high=0.525, margin_value=0.1)
        assert status == "decide"
        assert abs(hw - 0.025) < 1e-9

    def test_medium_interval_returns_escalate(self) -> None:
        # half-width = 0.075, margin = 0.1 → 0.05 < 0.075 <= 0.1 → escalate
        _hw, status = _width_escalation_status(ci_low=0.425, ci_high=0.575, margin_value=0.1)
        assert status == "escalate"

    def test_wide_interval_returns_no_go(self) -> None:
        # half-width = 0.15, margin = 0.1 → 0.15 > 0.1 → no_go
        _hw, status = _width_escalation_status(ci_low=0.35, ci_high=0.65, margin_value=0.1)
        assert status == "no_go"

    def test_half_width_calculation_is_correct(self) -> None:
        hw, _ = _width_escalation_status(ci_low=0.2, ci_high=0.8, margin_value=0.5)
        assert abs(hw - 0.3) < 1e-9


# ── _min_n_for_zero_failures ────────────────────────────────────────────


class TestMinNForZeroFailures:
    def test_standard_critical_metric_requires_sufficient_n(self) -> None:
        # alpha=0.05, p_target=0.01 → n = ceil(ln(0.05) / 0.01) = ceil(299.6) = 300
        n = _min_n_for_zero_failures(alpha=0.05, p_target=0.01)
        assert n == 300

    def test_high_risk_metric_requires_fewer_samples(self) -> None:
        # alpha=0.05, p_target=0.02 → n = ceil(ln(0.05) / 0.02) = ceil(149.8) = 150
        n = _min_n_for_zero_failures(alpha=0.05, p_target=0.02)
        assert n == 150

    def test_invalid_alpha_returns_zero(self) -> None:
        assert _min_n_for_zero_failures(alpha=0.0, p_target=0.01) == 0
        assert _min_n_for_zero_failures(alpha=1.0, p_target=0.01) == 0

    def test_zero_p_target_returns_zero(self) -> None:
        assert _min_n_for_zero_failures(alpha=0.05, p_target=0.0) == 0

    def test_minimum_is_at_least_one(self) -> None:
        # Even for very generous p_target, result should be >= 1
        n = _min_n_for_zero_failures(alpha=0.99, p_target=0.99)
        assert n >= 1


# ── _ess_default_flags ──────────────────────────────────────────────────


class TestESSDefaultFlags:
    def test_no_defaults_gives_all_free(self) -> None:
        steps = [_step(ess_used_defaults=False) for _ in range(5)]
        flags = _ess_default_flags(steps)
        assert flags == ESSDefaultFlags(defaults_free=True, missing_free=True, exception_free=True)

    def test_exception_severity_marks_all_flags_false(self) -> None:
        steps = [_step(ess_used_defaults=True, ess_default_severity="exception")]
        flags = _ess_default_flags(steps)
        assert not flags.defaults_free
        assert not flags.missing_free
        assert not flags.exception_free

    def test_missing_severity_does_not_set_exception_flag(self) -> None:
        steps = [_step(ess_used_defaults=True, ess_default_severity="missing")]
        flags = _ess_default_flags(steps)
        assert not flags.defaults_free
        assert not flags.missing_free
        assert flags.exception_free  # exception_free stays True

    def test_coercion_only_sets_defaults_flag(self) -> None:
        steps = [
            _step(
                ess_used_defaults=True,
                ess_default_severity="none",
                ess_defaulted_fields=("coerced:score",),
            )
        ]
        flags = _ess_default_flags(steps)
        assert not flags.defaults_free
        assert flags.missing_free  # coercion-only → missing_free stays True
        assert flags.exception_free

    def test_empty_steps_gives_all_free(self) -> None:
        flags = _ess_default_flags([])
        assert flags == ESSDefaultFlags(defaults_free=True, missing_free=True, exception_free=True)

    def test_mixed_steps_reflects_worst_case(self) -> None:
        steps = [
            _step(ess_used_defaults=False),
            _step(ess_used_defaults=True, ess_default_severity="exception"),
            _step(ess_used_defaults=False),
        ]
        flags = _ess_default_flags(steps)
        assert not flags.defaults_free
        assert not flags.exception_free


# ── _ess_retry_stats ────────────────────────────────────────────────────


class TestESSRetryStats:
    def test_no_retries_is_stable(self) -> None:
        steps = [_step(ess_calls=1) for _ in range(10)]
        stats = _ess_retry_stats(steps)
        assert stats.retry_stable
        assert stats.retry_steps == 0
        assert stats.retry_step_rate == 0.0

    def test_all_retries_is_unstable(self) -> None:
        steps = [_step(ess_calls=2) for _ in range(10)]
        stats = _ess_retry_stats(steps)
        assert not stats.retry_stable
        assert stats.retry_steps == 10
        assert stats.retry_step_rate == 1.0

    def test_rate_at_threshold_is_stable(self) -> None:
        # MAX_ESS_RETRY_STEP_RATE = 0.10 → 1 out of 10 steps retried is exactly at boundary
        from .teaching_harness import MAX_ESS_RETRY_STEP_RATE

        steps = [_step(ess_calls=2)] + [_step(ess_calls=1) for _ in range(9)]
        stats = _ess_retry_stats(steps)
        assert abs(stats.retry_step_rate - 0.1) < 1e-9
        assert stats.retry_stable == (MAX_ESS_RETRY_STEP_RATE >= 0.1)

    def test_empty_steps_gives_zero_rate(self) -> None:
        stats = _ess_retry_stats([])
        assert stats.retry_step_rate == 0.0
        assert stats.retry_stable
