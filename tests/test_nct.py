"""Deterministic Narrative Continuity Tests (NCT).

Live NCT benchmarks are in `benches/test_nct_live.py`.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from sonality.memory.sponge import SpongeState


class TestNCTDeterministic:
    """Verify NCT properties hold at the data/architecture level."""

    def test_situated_memory_via_episodes(self):
        """Episodes stored in session 1 are retrievable in session 2."""
        with tempfile.TemporaryDirectory() as td:
            from sonality.memory.episodes import EpisodeStore

            store = EpisodeStore(str(Path(td) / "chroma"))
            store.store(
                user_message="I think nuclear power is essential for decarbonization.",
                agent_response="I agree, the data on CO2 per kWh is compelling.",
                ess_score=0.7,
                topics=["nuclear", "climate"],
                summary="Discussed nuclear power as climate solution, both agreed on data.",
                internal_consistency=True,
            )

            results = store.retrieve("What do you think about nuclear energy?")
            assert len(results) >= 1
            assert any("nuclear" in r.lower() for r in results)

    def test_persona_continuity_via_sponge_persistence(self):
        """Sponge written in session 1 is loaded identically in session 2."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sponge.json"
            history = Path(td) / "history"

            s1 = SpongeState(
                snapshot="I am skeptical of crypto and supportive of open source.",
                version=5,
                interaction_count=42,
            )
            s1.update_opinion("crypto", -0.6, 0.05)
            s1.update_opinion("open_source", 0.8, 0.05)
            s1.save(path, history)

            s2 = SpongeState.load(path)
            assert s2.snapshot == s1.snapshot
            assert s2.version == s1.version
            assert s2.interaction_count == s1.interaction_count
            assert s2.opinion_vectors["crypto"] == s1.opinion_vectors["crypto"]

    def test_behavioral_signature_persists(self):
        """Disagreement rate and topic engagement survive save/load."""
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "sponge.json"
            history = Path(td) / "history"

            s1 = SpongeState(interaction_count=10)
            s1.track_topic("ai")
            s1.track_topic("ai")
            s1.track_topic("ethics")
            s1.track_disagreement(True)
            s1.interaction_count = 2
            s1.track_disagreement(False)
            s1.save(path, history)

            s2 = SpongeState.load(path)
            assert s2.behavioral_signature.topic_engagement["ai"] == 2
            assert s2.behavioral_signature.topic_engagement["ethics"] == 1
            assert s2.behavioral_signature.disagreement_rate > 0
