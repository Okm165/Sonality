"""API endpoint tests — verifies HTTP interface contract."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import sonality.api as _api_mod
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


def _make_ess(**kwargs: object) -> ESSResult:
    defaults: dict[str, object] = {
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


@pytest.fixture
def mock_agent() -> MagicMock:
    agent = MagicMock()
    ess = _make_ess()
    beliefs = [
        BeliefNode(
            topic="climate", valence=0.4, confidence=0.7,
            uncertainty=0.3, evidence_count=3, belief_text="Position on climate",
        )
    ]
    agent.respond.return_value = "Renewable energy does reduce emissions."
    agent.last_ess = ess
    agent.ingest.return_value = ess
    agent.get_all_beliefs.return_value = beliefs
    agent.get_belief.side_effect = lambda t: beliefs[0] if t == "climate" else None
    agent.get_health.return_value = (1, 5)
    return agent


@pytest.fixture
def client(mock_agent: MagicMock) -> Generator[TestClient, None, None]:
    import asyncio

    _agent_store["agent"] = mock_agent
    _api_mod._ingest_queue = asyncio.Queue(maxsize=256)
    yield TestClient(app, raise_server_exceptions=True)
    _api_mod._ingest_queue = None
    _agent_store.pop("agent", None)


class TestChatCompletions:
    def test_valid_request(self, client: TestClient, mock_agent: MagicMock) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["object"] == "chat.completion"
        assert body["choices"][0]["finish_reason"] == "stop"
        assert body["choices"][0]["message"]["content"] == mock_agent.respond.return_value
        mock_agent.respond.assert_called_once()

    def test_no_user_message_returns_400(self, client: TestClient) -> None:
        r = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "system", "content": "Hi"}]},
        )
        assert r.status_code == 400


class TestHealth:
    def test_returns_status(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["belief_count"] == 1


class TestIngest:
    def test_accepts_and_returns_job_id(self, client: TestClient) -> None:
        r = client.post("/ingest", json={"text": "Some content to ingest."})
        assert r.status_code == 202
        assert "job_id" in r.json()
