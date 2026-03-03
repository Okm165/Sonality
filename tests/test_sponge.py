from pathlib import Path

import pytest

from sonality.memory.sponge import SEED_SNAPSHOT, SpongeState


def test_belief_meta_tracked_on_update():
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


def test_load_nonexistent_returns_seed():
    """Test that load nonexistent returns seed."""
    s = SpongeState.load(Path("/tmp/nonexistent_sponge_test.json"))
    assert s.version == 0
    assert s.snapshot == SEED_SNAPSHOT


def test_staged_updates_respect_cooling_period():
    """Test that staged updates respect cooling period."""
    s = SpongeState(interaction_count=10)
    due = s.stage_opinion_update("ai", 1.0, 0.05, cooling_period=3, provenance="test")
    assert due == 13
    assert s.apply_due_staged_updates() == []
    s.interaction_count = 13
    applied = s.apply_due_staged_updates()
    assert applied
    assert "ai" in s.opinion_vectors


def test_staged_updates_net_out_when_conflicting():
    """Test that staged updates net out when conflicting."""
    s = SpongeState(interaction_count=5)
    s.stage_opinion_update("topic", 1.0, 0.02, cooling_period=1)
    s.stage_opinion_update("topic", -1.0, 0.02, cooling_period=1)
    s.interaction_count = 6
    applied = s.apply_due_staged_updates()
    assert applied == []
    assert "topic" not in s.opinion_vectors


def test_decay_skips_recently_reinforced_beliefs():
    """Beliefs reinforced within 5 turns should not decay."""
    s = SpongeState(interaction_count=10)
    s.update_opinion("ai", 1.0, 0.05)
    before = s.belief_meta["ai"].confidence
    s.interaction_count = 14  # gap = 4
    dropped = s.decay_beliefs()
    assert dropped == []
    assert s.belief_meta["ai"].confidence == before


def test_decay_applies_reinforcement_floor():
    """Well-reinforced beliefs should not decay below their floor."""
    s = SpongeState(interaction_count=10)
    s.update_opinion("policy", 1.0, 0.05)
    meta = s.belief_meta["policy"]
    meta.evidence_count = 10
    meta.confidence = 0.20
    meta.last_reinforced = 0
    s.interaction_count = 100
    s.decay_beliefs()
    assert s.belief_meta["policy"].confidence == pytest.approx(0.36)


def test_decay_drops_weak_unreinforced_beliefs():
    """Weak single-evidence beliefs should be removed after sufficient decay."""
    s = SpongeState(interaction_count=5)
    s.update_opinion("crypto", 1.0, 0.05)
    s.belief_meta["crypto"].confidence = 0.04
    s.belief_meta["crypto"].evidence_count = 1
    s.belief_meta["crypto"].last_reinforced = 0
    s.interaction_count = 20
    dropped = s.decay_beliefs()
    assert "crypto" in dropped
    assert "crypto" not in s.belief_meta
    assert "crypto" not in s.opinion_vectors
