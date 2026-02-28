import tempfile

from sonality.memory.episodes import EpisodeStore


def test_store_and_retrieve():
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(td)
        store.store("What is AI?", "AI is machine intelligence.", 0.5, ["ai"], "AI discussion")
        store.store("Hello!", "Hi there!", 0.1, ["greeting"], "Greeting exchange")

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
            memory_type="semantic",
        )
        store.store(
            "Casual greeting",
            "Hi!",
            0.05,
            ["chat"],
            "Episodic greeting memory",
            memory_type="episodic",
        )
        retrieved = store.retrieve_typed("governance", episodic_n=1, semantic_n=1, min_relevance=-1.0)
        assert len(retrieved) == 2
        assert retrieved[0] == "Semantic governance memory"
