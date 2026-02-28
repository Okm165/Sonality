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
