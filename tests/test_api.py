"""API endpoint tests for sonality/api.py.

Tests all HTTP endpoints using FastAPI's TestClient with a mocked SonalityAgent.
Covers happy paths, error cases, and response schema correctness.

Run: uv run pytest tests/test_api.py -v --tb=short
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from sonality.api import _agent_store, app
from sonality.ess import (
    ESSResult,
    InternalConsistencyStatus,
    KnowledgeDensity,
    OpinionDirection,
    ReasoningType,
    SourceReliability,
    UrgencyLevel,
)
from sonality.memory.graph import BeliefCorrelation, EdgeType
from sonality.memory.sponge import BeliefState, ProbabilityEstimate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ess(**kwargs: Any) -> ESSResult:
    defaults: dict[str, Any] = {
        "score": 0.55,
        "reasoning_type": ReasoningType.EMPIRICAL_DATA,
        "source_reliability": SourceReliability.UNVERIFIED_CLAIM,
        "internal_consistency": InternalConsistencyStatus.CONSISTENT,
        "novelty": 0.6,
        "topics": ("climate", "energy"),
        "summary": "User asserts that renewable energy reduces emissions.",
        "opinion_direction": OpinionDirection.SUPPORTS,
        "knowledge_density": KnowledgeDensity.MODERATE,
        "belief_update_recommended": True,
        "urgency": UrgencyLevel.STANDARD,
    }
    defaults.update(kwargs)
    return ESSResult(**defaults)


def _make_belief(topic: str = "climate", position: float = 0.4) -> BeliefState:
    return BeliefState(
        topic=topic,
        position=position,
        confidence=0.7,
        uncertainty=0.3,
        evidence_count=3,
        supporting_count=2,
        contradicting_count=1,
    )


def _make_probability(topic: str = "climate") -> ProbabilityEstimate:
    return ProbabilityEstimate(
        topic=topic,
        probability=0.72,
        evidence_weight=0.6,
        opinion=0.4,
        confidence=0.7,
        evidence_count=3,
        raw_probability=0.65,
    )


def _make_correlation() -> BeliefCorrelation:
    return BeliefCorrelation(
        source_topic="climate",
        target_topic="energy",
        correlation_type=EdgeType.CORRELATES_WITH,
        strength=0.8,
        reasoning="Climate mitigation often involves energy transition.",
    )


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock()
    ess = _make_ess()
    prob = _make_probability()

    agent.respond.return_value = "Renewable energy does reduce emissions significantly."
    agent.last_ess = ess
    agent.ingest.return_value = ess
    agent.sponge.opinion_vectors = {"climate": 0.4, "energy": 0.3}
    agent.sponge.get_belief.side_effect = lambda t: _make_belief(topic=t)
    agent.sponge.estimate_probability.return_value = prob
    agent.sponge.version = 5
    agent.sponge.interaction_count = 42
    agent.sponge.belief_count = 2
    agent.sponge.behavioral_signature.topic_engagement = {"climate": 5, "energy": 3}
    agent.sponge.staged_opinion_updates = []
    agent._run_async.return_value = [_make_correlation()]
    return agent


@pytest.fixture
def client(mock_agent: MagicMock) -> TestClient:
    _agent_store["agent"] = mock_agent
    yield TestClient(app, raise_server_exceptions=True)
    _agent_store.pop("agent", None)


@pytest.fixture
def client_no_agent() -> TestClient:
    _agent_store.pop("agent", None)
    yield TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_correct_structure(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["version"] == 5
        assert body["interaction_count"] == 42
        assert body["belief_count"] == 2
        assert body["topic_count"] == 2
        assert body["staged_updates"] == 0

    def test_503_when_agent_not_initialized(self, client_no_agent: TestClient) -> None:
        r = client_no_agent.get("/health")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Models endpoints
# ---------------------------------------------------------------------------


class TestModels:
    def test_list_models(self, client: TestClient) -> None:
        r = client.get("/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["id"] == "sonality"

    def test_get_known_model(self, client: TestClient) -> None:
        r = client.get("/v1/models/sonality")
        assert r.status_code == 200
        assert r.json()["id"] == "sonality"

    def test_get_unknown_model_returns_404(self, client: TestClient) -> None:
        r = client.get("/v1/models/gpt-4o")
        assert r.status_code == 404


# ---------------------------------------------------------------------------
# Chat completions (OpenAI-compatible)
# ---------------------------------------------------------------------------


class TestChatCompletions:
    def test_valid_request(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Tell me about renewables."}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert "emissions" in body["choices"][0]["message"]["content"]
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["usage"]["total_tokens"] > 0

    def test_no_user_message_returns_400(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "system", "content": "You are helpful."}]},
        )
        assert r.status_code == 400
        assert "No user message" in r.json()["detail"]

    def test_stream_true_returns_501(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert r.status_code == 501
        assert "Streaming" in r.json()["detail"]

    def test_uses_last_user_message(self, client: TestClient, mock_agent: MagicMock) -> None:
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "Some reply"},
                    {"role": "user", "content": "Second message"},
                ]
            },
        )
        mock_agent.respond.assert_called_once_with("Second message")

    def test_503_when_agent_not_initialized(self, client_no_agent: TestClient) -> None:
        r = client_no_agent.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class TestEmbeddings:
    def test_single_text_embedding(self, client: TestClient, mock_agent: MagicMock) -> None:
        mock_agent._embedder.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        r = client.post("/v1/embeddings", json={"input": "Hello world"})
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "list"
        assert len(body["data"]) == 1
        assert body["data"][0]["embedding"] == [0.1, 0.2, 0.3]
        assert body["data"][0]["index"] == 0

    def test_multiple_texts_embedding(self, client: TestClient, mock_agent: MagicMock) -> None:
        mock_agent._embedder.embed_documents.return_value = [[0.1], [0.2], [0.3]]
        r = client.post("/v1/embeddings", json={"input": ["text one", "text two", "text three"]})
        assert r.status_code == 200
        body = r.json()
        assert len(body["data"]) == 3
        assert body["data"][1]["index"] == 1

    def test_usage_counts_tokens(self, client: TestClient, mock_agent: MagicMock) -> None:
        mock_agent._embedder.embed_documents.return_value = [[0.0] * 10]
        r = client.post("/v1/embeddings", json={"input": "five words in here"})
        assert r.status_code == 200
        assert r.json()["usage"]["completion_tokens"] == 0
        assert r.json()["usage"]["total_tokens"] > 0


# ---------------------------------------------------------------------------
# Simple chat
# ---------------------------------------------------------------------------


class TestSimpleChat:
    def test_returns_ess_metadata(self, client: TestClient) -> None:
        r = client.post("/chat", json={"message": "What do you think about solar energy?"})
        assert r.status_code == 200
        body = r.json()
        assert "emissions" in body["response"]
        assert body["ess_score"] == pytest.approx(0.55)
        assert body["reasoning_type"] == "empirical_data"
        assert "climate" in body["topics"]

    def test_503_when_agent_not_initialized(self, client_no_agent: TestClient) -> None:
        r = client_no_agent.post("/chat", json={"message": "Hello"})
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class TestIngest:
    def test_valid_ingest(self, client: TestClient) -> None:
        r = client.post("/ingest", json={"text": "Solar output grew 25% globally in 2024."})
        assert r.status_code == 200
        body = r.json()
        assert body["success"] is True
        assert body["score"] == pytest.approx(0.55)
        assert body["reasoning_type"] == "empirical_data"
        assert body["belief_update_recommended"] is True
        assert "climate" in body["topics"]

    def test_ingest_with_topic_override(self, client: TestClient, mock_agent: MagicMock) -> None:
        client.post(
            "/ingest",
            json={"text": "Some text.", "topic_override": "renewable_energy"},
        )
        mock_agent.ingest.assert_called_once_with("Some text.", topic_override="renewable_energy")

    def test_ingest_no_topic_override(self, client: TestClient, mock_agent: MagicMock) -> None:
        client.post("/ingest", json={"text": "Some text."})
        mock_agent.ingest.assert_called_once_with("Some text.", topic_override="")


# ---------------------------------------------------------------------------
# Beliefs
# ---------------------------------------------------------------------------


class TestBeliefs:
    def test_get_all_beliefs(self, client: TestClient) -> None:
        r = client.get("/beliefs")
        assert r.status_code == 200
        beliefs = r.json()
        assert isinstance(beliefs, list)
        assert len(beliefs) == 2
        for b in beliefs:
            assert "topic" in b
            assert "position" in b
            assert "confidence" in b
            assert "evidence_count" in b
            assert "uncertainty" in b

    def test_beliefs_sorted_by_absolute_position(
        self, client: TestClient, mock_agent: MagicMock
    ) -> None:
        mock_agent.sponge.opinion_vectors = {"low": 0.1, "high": 0.8, "mid": -0.5}
        mock_agent.sponge.get_belief.side_effect = lambda t: _make_belief(
            topic=t, position={"low": 0.1, "high": 0.8, "mid": -0.5}[t]
        )
        r = client.get("/beliefs")
        positions = [abs(b["position"]) for b in r.json()]
        assert positions == sorted(positions, reverse=True)

    def test_get_specific_belief(self, client: TestClient) -> None:
        r = client.get("/beliefs/climate")
        assert r.status_code == 200
        body = r.json()
        assert body["topic"] == "climate"
        assert "position" in body

    def test_get_unknown_belief_returns_defaults(
        self, client: TestClient, mock_agent: MagicMock
    ) -> None:
        mock_agent.sponge.get_belief.side_effect = None
        mock_agent.sponge.get_belief.return_value = _make_belief(topic="unknown", position=0.0)
        r = client.get("/beliefs/unknown_topic")
        assert r.status_code == 200
        assert r.json()["position"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Probability
# ---------------------------------------------------------------------------


class TestProbability:
    def test_post_probability(self, client: TestClient) -> None:
        r = client.post("/beliefs/climate/probability", json={"base_rate": 0.5})
        assert r.status_code == 200
        body = r.json()
        assert body["topic"] == "climate"
        assert body["probability"] == pytest.approx(0.72)
        assert body["evidence_count"] == 3

    def test_get_probability_compat(self, client: TestClient) -> None:
        r = client.get("/probability/climate?base_rate=0.3")
        assert r.status_code == 200
        assert r.json()["topic"] == "climate"

    def test_default_base_rate(self, client: TestClient, mock_agent: MagicMock) -> None:
        client.get("/probability/climate")
        mock_agent.sponge.estimate_probability.assert_called_with("climate", base_rate=0.5)


# ---------------------------------------------------------------------------
# Correlations
# ---------------------------------------------------------------------------


class TestCorrelations:
    def test_get_correlations(self, client: TestClient) -> None:
        r = client.get("/beliefs/climate/correlations")
        assert r.status_code == 200
        corrs = r.json()
        assert isinstance(corrs, list)
        assert len(corrs) == 1
        c = corrs[0]
        assert c["source_topic"] == "climate"
        assert c["target_topic"] == "energy"
        assert c["strength"] == pytest.approx(0.8)

    def test_get_correlations_compat(self, client: TestClient) -> None:
        r = client.get("/correlations/climate")
        assert r.status_code == 200
        assert len(r.json()) == 1

    def test_empty_correlations(self, client: TestClient, mock_agent: MagicMock) -> None:
        mock_agent._run_async.return_value = []
        r = client.get("/beliefs/niche_topic/correlations")
        assert r.status_code == 200
        assert r.json() == []
