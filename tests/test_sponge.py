from pathlib import Path

from sonality.memory.sponge import SEED_SNAPSHOT, SpongeState


def test_belief_meta_tracked_on_update():
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
    s = SpongeState.load(Path("/tmp/nonexistent_sponge_test.json"))
    assert s.version == 0
    assert s.snapshot == SEED_SNAPSHOT
