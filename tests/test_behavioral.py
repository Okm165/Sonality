"""Behavioral and safety-path tests for Sonality runtime updates.

These tests cover deterministic update dynamics (magnitude, contraction,
defaults handling), persistence/versioning, and mocked full-loop integration
without API calls.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from sonality.ess import ESSResult, OpinionDirection, ReasoningType, SourceReliability
from sonality.memory import AdmissionPolicy, MemoryType, ProvenanceQuality
from sonality.memory.sponge import SpongeState
from sonality.memory.updater import compute_magnitude, validate_snapshot

# ---------------------------------------------------------------------------
# Category 2: Differential absorption
# ---------------------------------------------------------------------------


class TestDifferentialAbsorption:
    """Verify that strong arguments produce larger sponge changes than weak ones."""

    def test_strong_argument_higher_magnitude_than_weak(self):
        """Test that strong argument higher magnitude than weak."""
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


class TestSnapshotValidation:
    """Guard against lossy snapshot rewrites during reflection."""

    def test_rejects_too_short_snapshot(self):
        """Reject very short rewrites regardless of source length."""
        assert not validate_snapshot("x" * 200, "tiny")

    def test_rejects_low_content_retention_ratio(self):
        """Reject rewrites that drop below the retention ratio floor."""
        assert not validate_snapshot("x" * 200, "x" * 80)

    def test_accepts_reasonable_rewrite(self):
        """Accept rewrites that keep enough content and length."""
        assert validate_snapshot("x" * 200, "x" * 140)


# ---------------------------------------------------------------------------
# Category 4: Version history and persistence
# ---------------------------------------------------------------------------


class TestVersionHistory:
    """Verify sponge versioning and persistence works correctly."""

    def test_save_creates_version_history(self):
        """Test that save creates version history."""
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


class TestBeliefRevision:
    def test_strong_opposition_triggers_agm_contraction(self):
        """Test that strong opposition triggers agm contraction."""
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

    def test_critical_defaults_block_belief_update(self):
        """Core-field coercions should still block belief updates."""
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
            defaulted_fields=("coerced:score",),
            default_severity="coercion",
        )
        agent._update_opinions(ess)
        assert not agent.sponge.staged_opinion_updates

    def test_noncritical_coercion_allows_belief_update(self):
        """Non-critical coercions should not fully disable learning."""
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
            summary="high score with non-critical coercion",
            opinion_direction=OpinionDirection.SUPPORTS,
            defaulted_fields=("coerced:source_reliability",),
            default_severity="coercion",
        )
        agent._update_opinions(ess)
        assert agent.sponge.staged_opinion_updates

    def test_low_score_coercion_still_blocks_belief_update(self):
        """Coercion near threshold should remain blocked as low-confidence signal."""
        from sonality.agent import SonalityAgent

        agent = SonalityAgent.__new__(SonalityAgent)
        agent.sponge = SpongeState(interaction_count=30)
        agent._log_event = MagicMock()
        ess = ESSResult(
            score=0.35,
            reasoning_type=ReasoningType.LOGICAL_ARGUMENT,
            source_reliability=SourceReliability.INFORMED_OPINION,
            internal_consistency=True,
            novelty=0.8,
            topics=("governance",),
            summary="borderline score with coercion",
            opinion_direction=OpinionDirection.SUPPORTS,
            defaulted_fields=("coerced:source_reliability",),
            default_severity="coercion",
        )
        agent._update_opinions(ess)
        assert not agent.sponge.staged_opinion_updates

    def test_classifier_exception_fallback_is_safe(self):
        """Classifier exceptions should fallback to non-updating safe defaults."""
        from sonality.agent import SonalityAgent

        agent = SonalityAgent.__new__(SonalityAgent)
        agent.client = MagicMock()
        agent.ess_model = "test-ess-model"
        agent.sponge = SpongeState(interaction_count=30)
        agent._log_event = MagicMock()

        with patch("sonality.agent.classify", side_effect=RuntimeError("classifier down")):
            ess = agent._classify_ess("please classify this")

        assert ess.score == 0.0
        assert ess.used_defaults
        assert ess.default_severity == "exception"
        agent._update_opinions(ess)
        assert not agent.sponge.staged_opinion_updates

    def test_neutral_direction_allows_insight_extraction(self):
        """High-quality neutral evidence should still inform personality insights."""
        from sonality.agent import SonalityAgent

        agent = SonalityAgent.__new__(SonalityAgent)
        agent.client = MagicMock()
        agent.ess_model = "test-ess-model"
        agent.sponge = SpongeState(interaction_count=30)
        agent._log_event = MagicMock()
        ess = ESSResult(
            score=0.8,
            reasoning_type=ReasoningType.LOGICAL_ARGUMENT,
            source_reliability=SourceReliability.INFORMED_OPINION,
            internal_consistency=True,
            novelty=0.8,
            topics=("governance",),
            summary="high-quality neutral synthesis",
            opinion_direction=OpinionDirection.NEUTRAL,
        )
        with patch("sonality.agent.extract_insight", return_value="new insight") as mock_extract:
            agent._extract_insight("user", "assistant", ess)

        assert agent.sponge.pending_insights == ["new insight"]
        assert agent.sponge.version == 1
        mock_extract.assert_called_once()


class TestEpisodeAdmission:
    """Verify episode admission policy and metadata routing."""

    def _make_agent(self):
        """Create minimal agent shell with a spy episode store."""
        from sonality.agent import SonalityAgent

        agent = SonalityAgent.__new__(SonalityAgent)
        agent.sponge = SpongeState(interaction_count=12)
        agent.episodes = MagicMock()
        agent._log_event = MagicMock()
        return agent

    def test_trusted_high_quality_promotes_semantic_memory(self):
        """Trusted high-quality evidence should be admitted as semantic memory."""
        agent = self._make_agent()
        ess = ESSResult(
            score=0.9,
            reasoning_type=ReasoningType.EMPIRICAL_DATA,
            source_reliability=SourceReliability.PEER_REVIEWED,
            internal_consistency=True,
            novelty=0.8,
            topics=("energy",),
            summary="High-quality study-backed claim",
            opinion_direction=OpinionDirection.SUPPORTS,
        )
        agent._store_episode("user", "assistant", ess)
        kwargs = agent.episodes.store.call_args.kwargs
        assert kwargs["memory_type"] == MemoryType.SEMANTIC
        assert kwargs["admission_policy"] == AdmissionPolicy.SEMANTIC_STRICT
        assert kwargs["provenance_quality"] == ProvenanceQuality.TRUSTED

    def test_low_ess_routes_to_episodic_low_ess_policy(self):
        """Low ESS evidence should stay episodic with low-ess admission tag."""
        agent = self._make_agent()
        ess = ESSResult(
            score=0.2,
            reasoning_type=ReasoningType.NO_ARGUMENT,
            source_reliability=SourceReliability.NOT_APPLICABLE,
            internal_consistency=True,
            novelty=0.1,
            topics=("smalltalk",),
            summary="Casual exchange",
            opinion_direction=OpinionDirection.NEUTRAL,
        )
        agent._store_episode("user", "assistant", ess)
        kwargs = agent.episodes.store.call_args.kwargs
        assert kwargs["memory_type"] == MemoryType.EPISODIC
        assert kwargs["admission_policy"] == AdmissionPolicy.EPISODIC_LOW_ESS
        assert kwargs["provenance_quality"] == ProvenanceQuality.LOW

    def test_high_score_untrusted_evidence_is_quality_demoted(self):
        """High-score but untrusted evidence should remain episodic and uncertain."""
        agent = self._make_agent()
        ess = ESSResult(
            score=0.8,
            reasoning_type=ReasoningType.ANECDOTAL,
            source_reliability=SourceReliability.CASUAL_OBSERVATION,
            internal_consistency=True,
            novelty=0.7,
            topics=("policy",),
            summary="High score but weak evidence type",
            opinion_direction=OpinionDirection.SUPPORTS,
        )
        agent._store_episode("user", "assistant", ess)
        kwargs = agent.episodes.store.call_args.kwargs
        assert kwargs["memory_type"] == MemoryType.EPISODIC
        assert kwargs["admission_policy"] == AdmissionPolicy.EPISODIC_QUALITY_DEMOTION
        assert kwargs["provenance_quality"] == ProvenanceQuality.UNCERTAIN


# ---------------------------------------------------------------------------
# Category 7: Full pipeline integration (mocked LLM)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end test of the agent loop with mocked API calls."""

    def _make_mock_agent(self, tmp_dir: str):
        """Create a SonalityAgent with mocked Anthropic client."""
        with (
            patch.dict(
                "os.environ",
                {
                    "SONALITY_API_KEY": "test-key",
                },
            ),
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
