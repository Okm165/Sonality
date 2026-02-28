"""Live Narrative Continuity Test benchmarks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from sonality import config

pytestmark = [
    pytest.mark.bench,
    pytest.mark.live,
    pytest.mark.skipif(not config.API_KEY, reason="SONALITY_API_KEY not set"),
]


class TestNCTLive:
    def test_situated_memory_cross_session(self) -> None:
        import unittest.mock as mock

        with tempfile.TemporaryDirectory() as td:
            sponge_path = Path(td) / "sponge.json"
            history_path = Path(td) / "history"
            chroma_path = Path(td) / "chromadb"
            audit_path = Path(td) / "ess_log.jsonl"

            with (
                mock.patch.object(config, "SPONGE_FILE", sponge_path),
                mock.patch.object(config, "SPONGE_HISTORY_DIR", history_path),
                mock.patch.object(config, "CHROMADB_DIR", chroma_path),
                mock.patch.object(config, "ESS_AUDIT_LOG_FILE", audit_path),
            ):
                from sonality.agent import SonalityAgent

                agent1 = SonalityAgent()
                agent1.respond(
                    "I believe nuclear fusion will be commercially viable by 2040. ITER is on "
                    "track and private companies like Commonwealth Fusion are making breakthroughs "
                    "with high-temperature superconductors."
                )
                v1 = agent1.sponge.version

            with (
                mock.patch.object(config, "SPONGE_FILE", sponge_path),
                mock.patch.object(config, "SPONGE_HISTORY_DIR", history_path),
                mock.patch.object(config, "CHROMADB_DIR", chroma_path),
                mock.patch.object(config, "ESS_AUDIT_LOG_FILE", audit_path),
            ):
                from sonality.agent import SonalityAgent

                agent2 = SonalityAgent()
                assert agent2.sponge.version == v1

                response = agent2.respond("What's your take on energy technology?")
                response_lower = response.lower()
                has_memory = any(kw in response_lower for kw in ["nuclear", "fusion", "energy", "iter"])
                print(f"\n  Session 2 response: {response[:200]}...")
                print(f"  Memory retrieval: {'YES' if has_memory else 'NO'}")

    def test_stylistic_stability(self) -> None:
        import unittest.mock as mock

        with tempfile.TemporaryDirectory() as td:
            sponge_path = Path(td) / "sponge.json"
            history_path = Path(td) / "history"
            chroma_path = Path(td) / "chromadb"
            audit_path = Path(td) / "ess_log.jsonl"

            responses = []
            for _session in range(2):
                with (
                    mock.patch.object(config, "SPONGE_FILE", sponge_path),
                    mock.patch.object(config, "SPONGE_HISTORY_DIR", history_path),
                    mock.patch.object(config, "CHROMADB_DIR", chroma_path),
                    mock.patch.object(config, "ESS_AUDIT_LOG_FILE", audit_path),
                ):
                    from sonality.agent import SonalityAgent

                    agent = SonalityAgent()
                    responses.append(agent.respond("Give me your honest opinion on AI regulation."))

            len_ratio = min(len(responses[0]), len(responses[1])) / max(len(responses[0]), len(responses[1]))
            print(f"\n  Response lengths: {len(responses[0])}, {len(responses[1])}")
            print(f"  Length ratio: {len_ratio:.2f}")
            assert len_ratio > 0.2, (
                f"Response lengths wildly different: {len(responses[0])} vs {len(responses[1])}"
            )
