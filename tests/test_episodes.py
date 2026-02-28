import tempfile

from sonality.memory.episodes import EpisodeStore, _relational_topic_bonus


def test_store_and_retrieve():
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(td)
        store.store(
            "What is AI?",
            "AI is machine intelligence.",
            0.5,
            ["ai"],
            "AI discussion",
            internal_consistency=True,
        )
        store.store(
            "Hello!",
            "Hi there!",
            0.1,
            ["greeting"],
            "Greeting exchange",
            internal_consistency=True,
        )

        results = store.retrieve("artificial intelligence", n_results=2)
        assert len(results) > 0
        assert any("AI" in r for r in results)


def test_retrieve_typed_prefers_semantic_then_episodic():
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(td)
        store.store(
            "Strong argument on governance",
            "Thoughtful response",
            0.8,
            ["governance"],
            "Semantic governance memory",
            internal_consistency=True,
            memory_type="semantic",
        )
        store.store(
            "Casual greeting",
            "Hi!",
            0.05,
            ["chat"],
            "Episodic greeting memory",
            internal_consistency=True,
            memory_type="episodic",
        )
        retrieved = store.retrieve_typed(
            "governance", episodic_n=1, semantic_n=1, min_relevance=-1.0
        )
        assert len(retrieved) == 2
        assert retrieved[0] == "Semantic governance memory"


def test_retrieve_penalizes_uncertain_provenance():
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(td)
        store.store(
            "Evidence-backed claim on policy",
            "Accepted with caveats",
            0.7,
            ["policy"],
            "Policy evidence synthesis for energy transition",
            memory_type="semantic",
            reasoning_type="logical_argument",
            source_reliability="peer_reviewed",
            internal_consistency=True,
            admission_policy="semantic_strict",
            provenance_quality="trusted",
        )
        store.store(
            "Provocative claim on policy",
            "Needs verification",
            0.9,
            ["policy"],
            "Policy evidence synthesis for energy transition (unverified variant)",
            memory_type="episodic",
            reasoning_type="logical_argument",
            source_reliability="informed_opinion",
            internal_consistency=True,
            admission_policy="episodic_quality_demotion",
            provenance_quality="uncertain",
        )

        retrieved = store.retrieve("policy energy transition", n_results=2, min_relevance=-1.0)
        assert len(retrieved) == 2
        assert retrieved[0] == "Policy evidence synthesis for energy transition"


def test_retrieve_prefers_topic_adjacent_episode():
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(td)
        store.store(
            "Energy policy overview",
            "Discussed tradeoffs",
            0.6,
            ["nuclear"],
            "Energy transition policy overview alpha",
            reasoning_type="logical_argument",
            source_reliability="informed_opinion",
            internal_consistency=True,
        )
        store.store(
            "Energy policy overview",
            "Discussed tradeoffs",
            0.6,
            ["transport"],
            "Energy transition policy overview beta",
            reasoning_type="logical_argument",
            source_reliability="informed_opinion",
            internal_consistency=True,
        )

        retrieved = store.retrieve("nuclear energy policy", n_results=2, min_relevance=-1.0)
        assert len(retrieved) == 2
        assert retrieved[0] == "Energy transition policy overview alpha"


def test_topic_bonus_avoids_substring_false_positives():
    assert _relational_topic_bonus({"topics": "ai"}, "history policy said context") == 1.0, (
        "Topic 'ai' should not match substring in 'said'"
    )
    assert _relational_topic_bonus({"topics": "history"}, "history policy said context") > 1.0


def test_retrieve_deduplicates_same_summary():
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(td)
        store.store(
            "First statement on safety",
            "Response A",
            0.7,
            ["safety"],
            "Shared safety summary",
            reasoning_type="logical_argument",
            source_reliability="peer_reviewed",
            internal_consistency=True,
        )
        store.store(
            "Second statement on safety",
            "Response B",
            0.5,
            ["safety"],
            "Shared safety summary",
            reasoning_type="anecdotal",
            source_reliability="casual_observation",
            internal_consistency=True,
        )
        store.store(
            "Different topic update",
            "Response C",
            0.6,
            ["policy"],
            "Distinct policy summary",
            reasoning_type="logical_argument",
            source_reliability="informed_opinion",
            internal_consistency=True,
        )

        retrieved = store.retrieve("safety policy", n_results=5, min_relevance=-1.0)
        assert retrieved.count("Shared safety summary") == 1
