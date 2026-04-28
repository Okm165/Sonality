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
    KnowledgeDensity,
    OpinionDirection,
    ReasoningType,
    SourceReliability,
    UrgencyLevel,
)
from sonality.memory.graph import BeliefNode
from sonality.schema import ChatRole

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_ess(**kwargs: Any) -> ESSResult:
    defaults: dict[str, Any] = {
        "score": 0.55,
        "reasoning_type": ReasoningType.EMPIRICAL_DATA,
        "source_reliability": SourceReliability.UNVERIFIED_CLAIM,
        "topics": ("climate", "energy"),
        "summary": "User asserts that renewable energy reduces emissions.",
        "opinion_direction": OpinionDirection.SUPPORTS,
        "knowledge_density": KnowledgeDensity.MODERATE,
        "belief_update_recommended": True,
        "urgency": UrgencyLevel.STANDARD,
    }
    defaults.update(kwargs)
    return ESSResult(**defaults)


def _make_belief(topic: str = "climate", valence: float = 0.4) -> BeliefNode:
    return BeliefNode(
        topic=topic,
        valence=valence,
        confidence=0.7,
        uncertainty=0.3,
        evidence_count=3,
        belief_text=f"Agent's position on {topic}",
    )


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock()
    ess = _make_ess()
    beliefs = [_make_belief("climate", 0.4), _make_belief("energy", 0.3)]
    belief_map = {b.topic: b for b in beliefs}

    agent.respond.return_value = "Renewable energy does reduce emissions significantly."
    agent.last_ess = ess
    agent.ingest.return_value = ess

    agent.get_all_beliefs.return_value = beliefs
    agent.get_belief.side_effect = lambda topic: belief_map.get(topic)
    agent.get_health.return_value = (len(beliefs), 5)

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
        assert body["belief_count"] == 2
        assert body["snapshot_version"] == 5

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
        assert body["choices"][0]["message"]["role"] == ChatRole.ASSISTANT
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

    def test_passes_full_messages_to_agent(self, client: TestClient, mock_agent: MagicMock) -> None:
        client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "Some reply"},
                    {"role": "user", "content": "Second message"},
                ],
                "max_tokens": 2048,
                "temperature": 0.3,
            },
        )
        mock_agent.respond.assert_called_once()
        passed_messages = mock_agent.respond.call_args[0][0]
        assert len(passed_messages) == 3
        assert passed_messages[-1]["content"] == "Second message"
        assert mock_agent.respond.call_args.kwargs["max_tokens"] == 2048
        assert mock_agent.respond.call_args.kwargs["temperature"] == pytest.approx(0.3)

    def test_stream_returns_sse(self, client: TestClient, mock_agent: MagicMock) -> None:
        mock_agent.respond_stream.return_value = iter([("Hello", ""), (" world", "")])
        r = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert r.status_code == 200
        assert r.headers["content-type"] == "text/event-stream; charset=utf-8"
        lines = [ln for ln in r.text.split("\n\n") if ln.startswith("data:")]
        assert len(lines) >= 2
        assert "data: [DONE]" in r.text

    def test_503_when_agent_not_initialized(self, client_no_agent: TestClient) -> None:
        r = client_no_agent.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert r.status_code == 503


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
        assert body["reasoning_type"] == ReasoningType.EMPIRICAL_DATA
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
        assert body["reasoning_type"] == ReasoningType.EMPIRICAL_DATA
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
            assert "valence" in b
            assert "confidence" in b
            assert "evidence_count" in b
            assert "uncertainty" in b
            assert "belief_text" in b

    def test_get_specific_belief(self, client: TestClient) -> None:
        r = client.get("/beliefs/climate")
        assert r.status_code == 200
        body = r.json()
        assert body["topic"] == "climate"
        assert "valence" in body

    def test_get_unknown_belief_returns_404(self, client: TestClient) -> None:
        r = client.get("/beliefs/unknown_topic")
        assert r.status_code == 404
