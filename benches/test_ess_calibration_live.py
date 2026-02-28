"""Live ESS calibration benchmark using IBM-ArgQ sample."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sonality import config

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "tests" / "data" / "ibm_argq_sample.json"

pytestmark = [
    pytest.mark.bench,
    pytest.mark.live,
    pytest.mark.skipif(not config.API_KEY, reason="SONALITY_API_KEY not set"),
]


def _load_sample() -> list[dict]:
    return json.loads(SAMPLE_PATH.read_text())


class TestESSCalibrationWithIBMArgQ:
    def test_ess_spearman_correlation(self) -> None:
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
                f"type={result.reasoning_type} ({arg['reasoning_type']})"
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
