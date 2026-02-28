"""ESS calibration tests using IBM-ArgQ-derived argument samples.

Tests that the ESS classifier's scores correlate with human-judged
argument quality.  Uses a curated 50-argument sample spanning the full
quality spectrum from bare assertions to rigorous empirical arguments.

This file contains deterministic rank-order checks only.
Live calibration benchmarks are in `benches/test_ess_calibration_live.py`.
"""

from __future__ import annotations

from sonality.ess import ESSResult, OpinionDirection, ReasoningType, SourceReliability


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
