from __future__ import annotations

from collections.abc import Callable

from sonality.agent import SonalityAgent
from sonality.memory.sponge import BeliefMeta, SpongeState


def _minimal_agent() -> SonalityAgent:
    agent = SonalityAgent.__new__(SonalityAgent)
    agent.sponge = SpongeState(interaction_count=50)
    agent._last_entrenched = []
    agent._last_entrenched_interaction = -1
    return agent


def test_decay_beliefs_llm_forgets_topic(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Assess whether this belief should be retained or decayed based on staleness": {
                "action": "FORGET",
                "new_confidence": 0.0,
                "reasoning": "No longer relevant",
            }
        }
    )
    agent = _minimal_agent()
    agent.sponge.opinion_vectors["topic"] = 0.6
    agent.sponge.belief_meta["topic"] = BeliefMeta(
        confidence=0.3,
        evidence_count=1,
        last_reinforced=10,
    )
    dropped = agent._decay_beliefs_with_llm()
    assert dropped == ["topic"]
    assert "topic" not in agent.sponge.belief_meta


def test_entrenchment_detection_llm_flags_topic(
    mock_llm_call: Callable[[dict[str, dict[str, object]]], None],
) -> None:
    mock_llm_call(
        {
            "Assess if this belief shows signs of entrenchment": {
                "entrenchment_status": "ENTRENCHED",
                "confidence": 0.9,
                "reasoning": "Echo-chamber pattern",
                "recommendation": "Seek contradictory evidence",
            }
        }
    )
    agent = _minimal_agent()
    agent.sponge.opinion_vectors["policy"] = 0.8
    agent.sponge.belief_meta["policy"] = BeliefMeta(
        confidence=0.9,
        evidence_count=6,
        last_reinforced=49,
        recent_updates=[0.05, 0.04, 0.03, 0.04],
    )
    entrenched = agent._detect_entrenched_beliefs_llm()
    assert entrenched == ["policy"]
