from pathlib import Path

from sonality.memory.sponge import SEED_SNAPSHOT, SpongeState


def test_belief_meta_tracked_on_update() -> None:
    """Test that belief meta tracked on update."""
    s = SpongeState()
    s.interaction_count = 5
    s.update_opinion("ai", 1.0, 0.1)
    assert "ai" in s.belief_meta
    assert s.belief_meta["ai"].evidence_count == 1
    assert s.belief_meta["ai"].last_reinforced == 5
    s.interaction_count = 10
    s.update_opinion("ai", 1.0, 0.05)
    assert s.belief_meta["ai"].evidence_count == 2
    assert s.belief_meta["ai"].last_reinforced == 10
    assert s.belief_meta["ai"].confidence > 0


def test_load_nonexistent_returns_seed() -> None:
    """Test that load nonexistent returns seed."""
    s = SpongeState.load(Path("/tmp/nonexistent_sponge_test.json"))
    assert s.version == 0
    assert s.snapshot == SEED_SNAPSHOT


def test_staged_updates_respect_cooling_period() -> None:
    """Test that staged updates respect cooling period."""
    s = SpongeState(interaction_count=10)
    due = s.stage_opinion_update("ai", 1.0, 0.05, cooling_period=3, provenance="test")
    assert due == 13
    assert s.apply_due_staged_updates() == []
    s.interaction_count = 13
    applied = s.apply_due_staged_updates()
    assert applied
    assert "ai" in s.opinion_vectors


def test_staged_updates_net_out_when_conflicting() -> None:
    """Test that staged updates net out when conflicting."""
    s = SpongeState(interaction_count=5)
    s.stage_opinion_update("topic", 1.0, 0.02, cooling_period=1)
    s.stage_opinion_update("topic", -1.0, 0.02, cooling_period=1)
    s.interaction_count = 6
    applied = s.apply_due_staged_updates()
    assert applied == []
    assert "topic" not in s.opinion_vectors


def test_negative_sentiment_opinion_stages_negative_direction() -> None:
    """Negative-sentiment opinion proposition must produce a negative belief vector.

    Regression: previously `sentiment >= 0` mapped neutral (0.0) to +1.0 direction.
    """
    s = SpongeState(interaction_count=3)
    # Simulate what knowledge_extract.py does for a negative-sentiment opinion
    s.stage_opinion_update("vaccines", -1.0, 0.05, cooling_period=1, provenance="test")
    s.interaction_count = 4
    applied = s.apply_due_staged_updates()
    assert applied, "Negative opinion should produce a staged update"
    assert s.opinion_vectors.get("vaccines", 0.0) < 0, (
        "Negative-sentiment opinion must yield negative opinion vector"
    )


def test_apply_due_staged_updates_does_not_bump_version() -> None:
    """Committing staged updates must not increment sponge.version.

    Only insight extraction and reflection bump the version; staged commits
    are not writes attributed to the current turn.
    """
    s = SpongeState(interaction_count=3)
    initial_version = s.version
    s.stage_opinion_update("climate", 1.0, 0.05, cooling_period=1)
    s.interaction_count = 4
    applied = s.apply_due_staged_updates()
    assert applied, "staged update should have committed"
    assert s.version == initial_version, (
        f"apply_due_staged_updates must not bump sponge.version "
        f"(expected {initial_version}, got {s.version})"
    )
