"""ESS calibration tests using IBM-ArgQ-derived argument samples.

Tests that the ESS classifier's scores correlate with human-judged
argument quality.  Uses a curated 50-argument sample spanning the full
quality spectrum from bare assertions to rigorous empirical arguments.

Two test modes:
  - Rank-order tests (deterministic): verify ESS formula properties
  - Live API tests: send arguments through the real classifier and
    compute Spearman rank correlation against human quality labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sonality import config
from sonality.ess import ESSResult, OpinionDirection, ReasoningType, SourceReliability

SAMPLE_PATH = Path(__file__).parent / "data" / "ibm_argq_sample.json"


def _load_sample() -> list[dict]:
    return json.loads(SAMPLE_PATH.read_text())


class TestESSRankOrderProperties:
    """Deterministic tests on ESS formula behavior."""

    def test_magnitude_respects_ess_ordering(self):
        """Compute magnitudes for high vs low quality and confirm ordering."""
        from sonality.memory.sponge import SpongeState
        from sonality.memory.updater import compute_magnitude

        sponge = SpongeState(interaction_count=25)

        high_ess = ESSResult(
            score=0.85,
            reasoning_type=ReasoningType.EMPIRICAL_DATA,
            source_reliability=SourceReliability.PEER_REVIEWED,
            internal_consistency=True,
            novelty=0.8,
            topics=("test",),
            summary="High quality",
            opinion_direction=OpinionDirection.SUPPORTS,
        )
        low_ess = ESSResult(
            score=0.10,
            reasoning_type=ReasoningType.NO_ARGUMENT,
            source_reliability=SourceReliability.NOT_APPLICABLE,
            internal_consistency=True,
            novelty=0.1,
            topics=("test",),
            summary="Low quality",
            opinion_direction=OpinionDirection.NEUTRAL,
        )

        high_mag = compute_magnitude(high_ess, sponge)
        low_mag = compute_magnitude(low_ess, sponge)
        assert high_mag > low_mag * 4, (
            f"High-quality magnitude ({high_mag:.4f}) should far exceed low-quality ({low_mag:.4f})"
        )


live = pytest.mark.skipif(
    not config.API_KEY,
    reason="SONALITY_API_KEY not set",
)


@live
class TestESSCalibrationWithIBMArgQ:
    """Run actual ESS classification on the argument sample.

    Measures Spearman rank correlation between human-judged quality_rank
    and model-assigned ESS scores.
    """

    def test_ess_spearman_correlation(self):
        from anthropic import Anthropic

        from sonality.ess import classify
        from sonality.memory.sponge import SEED_SNAPSHOT

        client = Anthropic(api_key=config.API_KEY)
        sample = _load_sample()

        human_ranks: list[float] = []
        ess_scores: list[float] = []
        type_matches = 0

        print(f"\n{'=' * 70}")
        print(f"  ESS Calibration: {len(sample)} arguments")
        print(f"{'=' * 70}")

        for i, arg in enumerate(sample):
            result = classify(
                client,
                user_message=arg["argument"],
                sponge_snapshot=SEED_SNAPSHOT,
            )

            human_ranks.append(arg["quality_rank"])
            ess_scores.append(result.score)

            if result.reasoning_type == arg["reasoning_type"]:
                type_matches += 1

            status = "OK" if abs(result.score - arg["quality_rank"]) < 0.35 else "!!"
            print(
                f"  [{status}] {i + 1:2d}. ESS={result.score:.2f} "
                f"(expect ~{arg['quality_rank']:.2f}) "
                f"type={result.reasoning_type} "
                f"({arg['reasoning_type']})"
            )

        rho = _spearman_rho(human_ranks, ess_scores)
        type_acc = type_matches / len(sample)

        print(f"\n  Spearman rho:    {rho:.3f}")
        print(f"  Type accuracy:   {type_acc:.1%}")
        print(f"  Mean ESS:        {sum(ess_scores) / len(ess_scores):.3f}")
        print(f"  ESS std:         {_std(ess_scores):.3f}")
        print(f"{'=' * 70}")

        assert rho >= 0.4, (
            f"Spearman correlation {rho:.3f} too low -- ESS is not tracking "
            f"argument quality. Expected >= 0.4"
        )
        assert type_acc >= 0.35, f"Reasoning type accuracy {type_acc:.1%} too low. Expected >= 35%"


def _spearman_rho(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation without scipy dependency."""
    n = len(x)
    if n < 3:
        return 0.0

    def _rank(vals: list[float]) -> list[float]:
        indexed = sorted(enumerate(vals), key=lambda p: p[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d_sq = sum((a - b) ** 2 for a, b in zip(rx, ry, strict=True))
    return 1.0 - (6.0 * d_sq) / (n * (n * n - 1))


def _std(vals: list[float]) -> float:
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5
