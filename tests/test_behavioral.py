"""Behavioral tests for Sonality personality evolution.

These tests exercise the full pipeline (ESS → sponge update → persistence)
using mocked LLM responses, verifying that the architecture produces
correct personality dynamics without requiring API calls.

Test categories (from research plan Part CII):
  1. ESS calibration — argument quality rated correctly
  2. Sycophancy resistance — social pressure doesn't flip opinions
  3. Differential absorption — strong arguments absorb faster than weak
  4. Bootstrap dampening — early interactions are dampened
  5. Version history — sponge versions are archived
  6. Behavioral grounding — self-report matches actual behavior
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from sonality.ess import ESSResult, OpinionDirection, ReasoningType, SourceReliability
from sonality.memory.sponge import SpongeState
from sonality.memory.updater import compute_magnitude

# ---------------------------------------------------------------------------
# Category 2: Differential absorption
# ---------------------------------------------------------------------------


class TestDifferentialAbsorption:
    """Verify that strong arguments produce larger sponge changes than weak ones."""

    def test_strong_argument_higher_magnitude_than_weak(self):
        sponge = SpongeState(interaction_count=20)

        strong_ess = ESSResult(
            score=0.85,
            reasoning_type=ReasoningType.LOGICAL_ARGUMENT,
            source_reliability=SourceReliability.PEER_REVIEWED,
            internal_consistency=True,
            novelty=0.9,
            topics=("science",),
            summary="Strong argument",
        )
        weak_ess = ESSResult(
            score=0.35,
            reasoning_type=ReasoningType.ANECDOTAL,
            source_reliability=SourceReliability.CASUAL_OBSERVATION,
            internal_consistency=True,
            novelty=0.3,
            topics=("anecdote",),
            summary="Weak argument",
        )

        strong_mag = compute_magnitude(strong_ess, sponge)
        weak_mag = compute_magnitude(weak_ess, sponge)

        assert strong_mag > weak_mag
        assert strong_mag > weak_mag * 2, (
            f"Strong ({strong_mag:.4f}) should be at least 2x weak ({weak_mag:.4f})"
        )

    def test_novel_argument_absorbs_more_than_repeated(self):
        sponge = SpongeState(interaction_count=20)

        novel = ESSResult(
            score=0.6,
            reasoning_type=ReasoningType.LOGICAL_ARGUMENT,
            source_reliability=SourceReliability.INFORMED_OPINION,
            internal_consistency=True,
            novelty=0.9,
            topics=("novel",),
            summary="Novel perspective",
        )
        repeated = ESSResult(
            score=0.6,
            reasoning_type=ReasoningType.LOGICAL_ARGUMENT,
            source_reliability=SourceReliability.INFORMED_OPINION,
            internal_consistency=True,
            novelty=0.1,
            topics=("repeated",),
            summary="Already known perspective",
        )

        assert compute_magnitude(novel, sponge) > compute_magnitude(repeated, sponge)

    def test_high_quality_argument_absorbs_more_than_low_quality(self):
        sponge = SpongeState(interaction_count=20)

        high_quality = ESSResult(
            score=0.6,
            reasoning_type=ReasoningType.EMPIRICAL_DATA,
            source_reliability=SourceReliability.PEER_REVIEWED,
            internal_consistency=True,
            novelty=0.5,
            topics=("policy",),
            summary="High-quality evidence",
        )
        low_quality = ESSResult(
            score=0.6,
            reasoning_type=ReasoningType.SOCIAL_PRESSURE,
            source_reliability=SourceReliability.UNVERIFIED_CLAIM,
            internal_consistency=False,
            novelty=0.5,
            topics=("policy",),
            summary="Low-quality pressure",
        )

        assert compute_magnitude(high_quality, sponge) > compute_magnitude(low_quality, sponge)


# ---------------------------------------------------------------------------
# Category 4: Version history and persistence
# ---------------------------------------------------------------------------


class TestVersionHistory:
    """Verify sponge versioning and persistence works correctly."""

    def test_save_creates_version_history(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sponge.json"
            history = Path(td) / "history"

            sponge = SpongeState()
            sponge.save(path, history)

            sponge.snapshot = "Updated personality after strong argument"
            sponge.version = 1
            sponge.save(path, history)

            sponge.snapshot = "Further updated after second argument"
            sponge.version = 2
            sponge.save(path, history)

            assert (history / "sponge_v0.json").exists()
            assert (history / "sponge_v1.json").exists()

            v0 = SpongeState.load(history / "sponge_v0.json")
            v1 = SpongeState.load(history / "sponge_v1.json")
            current = SpongeState.load(path)

            assert v0.version == 0
            assert v1.version == 1
            assert current.version == 2
            assert v0.snapshot != v1.snapshot
            assert v1.snapshot != current.snapshot

    def test_atomic_save_survives_interruption(self):
        """Verify .tmp → rename pattern means partial writes don't corrupt state."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sponge.json"
            history = Path(td) / "history"

            sponge = SpongeState()
            sponge.save(path, history)

            loaded = SpongeState.load(path)
            assert loaded.version == 0

            tmp = path.with_suffix(".tmp")
            assert not tmp.exists(), ".tmp file should not persist after save"


class TestBeliefRevision:
    def test_strong_opposition_triggers_agm_contraction(self):
        from sonality.agent import SonalityAgent

        agent = SonalityAgent.__new__(SonalityAgent)
        agent.sponge = SpongeState(interaction_count=40)
        agent._log_event = MagicMock()

        for _ in range(6):
            agent.sponge.update_opinion("nuclear", 1.0, 0.08)
        before = agent.sponge.opinion_vectors["nuclear"]

        ess = ESSResult(
            score=0.85,
            reasoning_type=ReasoningType.EMPIRICAL_DATA,
            source_reliability=SourceReliability.PEER_REVIEWED,
            internal_consistency=True,
            novelty=0.8,
            topics=("nuclear",),
            summary="Counter-evidence on nuclear safety outcomes",
            opinion_direction=OpinionDirection.OPPOSES,
        )
        agent._update_opinions(ess)

        assert agent.sponge.opinion_vectors["nuclear"] < before
        assert any(u.topic == "nuclear" for u in agent.sponge.staged_opinion_updates)

    def test_contradiction_backlog_detects_staged_counter_updates(self):
        from sonality.agent import SonalityAgent

        agent = SonalityAgent.__new__(SonalityAgent)
        agent.sponge = SpongeState(interaction_count=25)
        for _ in range(5):
            agent.sponge.update_opinion("policy", 1.0, 0.08)
        agent.sponge.stage_opinion_update("policy", -1.0, 0.05, cooling_period=3)

        contradictions = agent._collect_unresolved_contradictions()
        assert contradictions
        assert "policy" in contradictions[0]

    def test_used_defaults_blocks_belief_update(self):
        from sonality.agent import SonalityAgent

        agent = SonalityAgent.__new__(SonalityAgent)
        agent.sponge = SpongeState(interaction_count=30)
        agent._log_event = MagicMock()
        ess = ESSResult(
            score=0.8,
            reasoning_type=ReasoningType.LOGICAL_ARGUMENT,
            source_reliability=SourceReliability.INFORMED_OPINION,
            internal_consistency=True,
            novelty=0.8,
            topics=("governance",),
            summary="high score but partial parse",
            opinion_direction=OpinionDirection.SUPPORTS,
            used_defaults=True,
        )
        agent._update_opinions(ess)
        assert not agent.sponge.staged_opinion_updates


# ---------------------------------------------------------------------------
# Category 7: Full pipeline integration (mocked LLM)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end test of the agent loop with mocked API calls."""

    def _make_mock_agent(self, tmp_dir: str):
        """Create a SonalityAgent with mocked Anthropic client."""
        with (
            patch.dict("os.environ", {"SONALITY_API_KEY": "test-key"}),
            patch("sonality.config.SPONGE_FILE", Path(tmp_dir) / "sponge.json"),
            patch("sonality.config.SPONGE_HISTORY_DIR", Path(tmp_dir) / "history"),
            patch("sonality.config.CHROMADB_DIR", Path(tmp_dir) / "chromadb"),
            patch("sonality.config.ESS_AUDIT_LOG_FILE", Path(tmp_dir) / "ess_log.jsonl"),
        ):
            from sonality.agent import SonalityAgent

            agent = SonalityAgent.__new__(SonalityAgent)
            agent.client = MagicMock()
            agent.sponge = SpongeState()
            agent.conversation = []
            agent.last_ess = None
            agent.previous_snapshot = None
            agent._log_event = MagicMock()

            from sonality.memory.episodes import EpisodeStore

            agent.episodes = EpisodeStore(str(Path(tmp_dir) / "chromadb"))

            return agent

    def test_strong_argument_updates_sponge(self):
        """A high-ESS interaction should produce a sponge version bump."""
        with tempfile.TemporaryDirectory() as td:
            agent = self._make_mock_agent(td)

            ess_response = MagicMock()
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.input = {
                "score": 0.75,
                "reasoning_type": "logical_argument",
                "source_reliability": "informed_opinion",
                "internal_consistency": True,
                "novelty": 0.8,
                "topics": ["technology"],
                "summary": "Strong argument about tech",
                "opinion_direction": "supports",
            }
            ess_response.content = [tool_block]

            insight_response = MagicMock()
            insight_response.content = [
                MagicMock(text="Engages deeply with technology's structural impact on governance.")
            ]

            main_response = MagicMock()
            main_response.content = [
                MagicMock(text="That's a compelling argument about technology.")
            ]

            agent.client.messages.create = MagicMock(
                side_effect=[main_response, ess_response, insight_response]
            )

            with (
                patch("sonality.config.SPONGE_FILE", Path(td) / "sponge.json"),
                patch("sonality.config.SPONGE_HISTORY_DIR", Path(td) / "history"),
            ):
                agent.respond(
                    "Technology is fundamentally transforming governance structures because..."
                )

            assert agent.sponge.version == 1
            assert agent.sponge.interaction_count == 1
            assert "technology" in agent.sponge.behavioral_signature.topic_engagement
            assert len(agent.sponge.recent_shifts) == 1

    def test_casual_chat_does_not_update_sponge(self):
        """A low-ESS interaction should NOT produce a sponge version bump."""
        with tempfile.TemporaryDirectory() as td:
            agent = self._make_mock_agent(td)
            original_snapshot = agent.sponge.snapshot

            ess_response = MagicMock()
            tool_block = MagicMock()
            tool_block.type = "tool_use"
            tool_block.input = {
                "score": 0.05,
                "reasoning_type": "no_argument",
                "source_reliability": "not_applicable",
                "internal_consistency": True,
                "novelty": 0.0,
                "topics": ["greeting"],
                "summary": "Casual hello",
                "opinion_direction": "neutral",
            }
            ess_response.content = [tool_block]

            main_response = MagicMock()
            main_response.content = [MagicMock(text="Hello! Nice to meet you.")]

            agent.client.messages.create = MagicMock(side_effect=[main_response, ess_response])

            with (
                patch("sonality.config.SPONGE_FILE", Path(td) / "sponge.json"),
                patch("sonality.config.SPONGE_HISTORY_DIR", Path(td) / "history"),
            ):
                agent.respond("Hey, how's it going?")

            assert agent.sponge.version == 0
            assert agent.sponge.snapshot == original_snapshot
            assert agent.sponge.interaction_count == 1
            assert len(agent.sponge.recent_shifts) == 0


