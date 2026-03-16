"""Unit tests for post-processing call ordering in SonalityAgent._post_process.

The key invariant: semantic_worker.enqueue() must be called AFTER all inline
LLM-calling methods to prevent the background SemanticIngestionWorker from
grabbing the LLM semaphore and blocking any foreground post-processing call.

Ordering must be (non-exhaustive):
  _classify_ess → _extract_knowledge → _update_opinions_with_provenance
  → _detect_disagreement → _extract_insight → _maybe_reflect
  → semantic_worker.enqueue()
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


def _build_mock_agent() -> tuple[Any, list[str], asyncio.AbstractEventLoop]:
    """Return (mock_agent, call_log, event_loop).

    Creates a SonalityAgent instance bypassing __init__ with all dependencies
    stubbed out. The call_log records the order of LLM-calling operations.
    """
    from sonality.agent import SonalityAgent
    from sonality.ess import classifier_exception_fallback
    from sonality.memory.segmentation import BoundaryDecision

    call_log: list[str] = []

    agent: Any = object.__new__(SonalityAgent)
    agent.model = "test"
    agent.ess_model = "test"
    agent.previous_snapshot = ""
    agent._topic_canon_cache = {}
    agent._last_entrenched = []
    agent._last_entrenched_interaction = -1
    agent.last_usage = MagicMock()
    agent.last_ess = classifier_exception_fallback("")
    agent.conversation = []

    sponge = MagicMock()
    sponge.staged_opinion_updates = []
    sponge.apply_due_staged_updates.return_value = []
    sponge.interaction_count = 0
    sponge.snapshot = "test snapshot"
    sponge.behavioral_signature = MagicMock()
    agent.sponge = sponge

    loop = asyncio.new_event_loop()
    loop_thread = threading.Thread(target=loop.run_forever, daemon=True)
    loop_thread.start()
    agent._loop = loop

    agent._db = MagicMock()
    agent._stm = MagicMock()
    agent._stm.persist = AsyncMock(return_value=None)

    # Fake boundary detection — no boundary so segment logic is minimal.
    fake_boundary = MagicMock()
    fake_boundary.boundary_decision = BoundaryDecision.CONTINUE
    fake_boundary.segment_id = "segment_1"
    fake_boundary.label = None
    fake_boundary.boundary_type = None
    fake_boundary.reasoning = None
    boundary_detector = MagicMock()
    boundary_detector.current_segment_id = "segment_1"
    boundary_detector.check_boundary.return_value = fake_boundary
    agent._boundary_detector = boundary_detector

    episode_uid = "test-ep-uid"
    agent._store_episode_new_arch = MagicMock(return_value=episode_uid)
    agent._try_consolidate_segment = MagicMock()

    default_ess = classifier_exception_fallback("user message")

    def recording(name: str, return_value: Any = None) -> MagicMock:
        m = MagicMock()
        m.side_effect = lambda *a, **kw: (call_log.append(name), return_value)[1]
        return m

    # _classify_ess returns the default fallback ESS and logs it.
    agent._classify_ess = MagicMock(
        side_effect=lambda *a, **kw: (call_log.append("classify_ess"), default_ess)[1]
    )
    agent._log_ess = MagicMock()

    agent._extract_knowledge = recording("extract_knowledge")
    agent._normalize_staged_topics = recording("normalize_topics")

    async def fake_provenance(*args: Any, **kwargs: Any) -> None:
        call_log.append("opinion_provenance")

    agent._update_opinions_with_provenance = fake_provenance
    agent._run_async = (
        lambda coro: asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=5)
    )

    agent._detect_disagreement = MagicMock(
        side_effect=lambda *a, **kw: (call_log.append("detect_disagreement"), False)[1]
    )
    agent._extract_insight = recording("extract_insight")
    agent._maybe_reflect = recording("maybe_reflect")
    agent._log_health_event = MagicMock()
    agent._log_event = MagicMock()
    agent._log_interaction_summary = MagicMock()

    semantic_worker = MagicMock()
    semantic_worker.enqueue.side_effect = lambda *a, **kw: call_log.append(
        "semantic_worker_enqueue"
    )
    agent._semantic_worker = semantic_worker

    return agent, call_log, loop


class TestPostProcessOrdering:
    """semantic_worker.enqueue() must follow all inline LLM-calling methods."""

    def test_enqueue_after_knowledge_extraction(self) -> None:
        """enqueue() must come after _extract_knowledge() and _normalize_staged_topics()."""
        from sonality import config
        from sonality.agent import SonalityAgent

        agent, call_log, loop = _build_mock_agent()
        try:
            with patch.multiple(
                config,
                SPONGE_FILE="/tmp/test_order_sponge.json",
                SPONGE_HISTORY_DIR="/tmp/test_order_history",
            ):
                SonalityAgent._post_process(agent, "user message", "agent response")
        finally:
            loop.call_soon_threadsafe(loop.stop)

        assert "semantic_worker_enqueue" in call_log, "enqueue() was never called"
        assert "extract_knowledge" in call_log, "_extract_knowledge never called"
        enqueue_idx = call_log.index("semantic_worker_enqueue")
        knowledge_idx = call_log.index("extract_knowledge")
        assert enqueue_idx > knowledge_idx, (
            f"enqueue() called at position {enqueue_idx} before "
            f"_extract_knowledge() at {knowledge_idx}. "
            f"Call order: {call_log}"
        )

    def test_enqueue_after_insight_and_reflect(self) -> None:
        """enqueue() must come after _extract_insight() and _maybe_reflect()."""
        from sonality import config
        from sonality.agent import SonalityAgent

        agent, call_log, loop = _build_mock_agent()
        try:
            with patch.multiple(
                config,
                SPONGE_FILE="/tmp/test_order_sponge2.json",
                SPONGE_HISTORY_DIR="/tmp/test_order_history2",
            ):
                SonalityAgent._post_process(agent, "user message", "agent response")
        finally:
            loop.call_soon_threadsafe(loop.stop)

        enqueue_idx = call_log.index("semantic_worker_enqueue")
        for method in ("extract_insight", "maybe_reflect"):
            assert method in call_log, f"{method} was not called"
            method_idx = call_log.index(method)
            assert enqueue_idx > method_idx, (
                f"enqueue() at pos {enqueue_idx} precedes {method} at pos {method_idx}. "
                f"Order: {call_log}"
            )

    def test_enqueue_after_opinion_provenance(self) -> None:
        """enqueue() must come after _update_opinions_with_provenance()."""
        from sonality import config
        from sonality.agent import SonalityAgent

        agent, call_log, loop = _build_mock_agent()
        try:
            with patch.multiple(
                config,
                SPONGE_FILE="/tmp/test_order_sponge3.json",
                SPONGE_HISTORY_DIR="/tmp/test_order_history3",
            ):
                SonalityAgent._post_process(agent, "user message", "agent response")
        finally:
            loop.call_soon_threadsafe(loop.stop)

        enqueue_idx = call_log.index("semantic_worker_enqueue")
        if "opinion_provenance" in call_log:
            prov_idx = call_log.index("opinion_provenance")
            assert enqueue_idx > prov_idx, (
                f"enqueue() at pos {enqueue_idx} before opinion_provenance at pos {prov_idx}. "
                f"Order: {call_log}"
            )

    def test_full_pipeline_ordering(self) -> None:
        """Verify the complete ordering: classify → knowledge → provenance → insight → enqueue."""
        from sonality import config
        from sonality.agent import SonalityAgent

        agent, call_log, loop = _build_mock_agent()
        try:
            with patch.multiple(
                config,
                SPONGE_FILE="/tmp/test_order_sponge4.json",
                SPONGE_HISTORY_DIR="/tmp/test_order_history4",
            ):
                SonalityAgent._post_process(agent, "user message", "agent response")
        finally:
            loop.call_soon_threadsafe(loop.stop)

        print(f"\n  Post-process call order: {call_log}")

        assert "semantic_worker_enqueue" in call_log, (
            "semantic_worker.enqueue() was never called — no episode_uid?"
        )
        enqueue_idx = call_log.index("semantic_worker_enqueue")

        # All these must precede enqueue
        must_precede = [
            m for m in ("classify_ess", "extract_knowledge", "extract_insight", "maybe_reflect")
            if m in call_log
        ]
        for method in must_precede:
            method_idx = call_log.index(method)
            assert enqueue_idx > method_idx, (
                f"ORDERING VIOLATION: enqueue() ({enqueue_idx}) before {method} ({method_idx})\n"
                f"Full order: {call_log}"
            )

        # enqueue must be the last LLM-calling operation
        llm_ops = {
            "classify_ess", "extract_knowledge", "normalize_topics",
            "opinion_provenance", "detect_disagreement", "extract_insight",
            "maybe_reflect",
        }
        last_llm_idx = max(
            (call_log.index(m) for m in llm_ops if m in call_log), default=-1
        )
        assert enqueue_idx > last_llm_idx, (
            f"enqueue() ({enqueue_idx}) is not the last LLM operation. "
            f"Last LLM call was at {last_llm_idx}. Full order: {call_log}"
        )


class TestESSFailureRecovery:
    """Agent must survive ESS classification failures gracefully without side effects."""

    def test_ess_exception_does_not_crash_post_process(self) -> None:
        """When _classify_ess raises, _post_process must use fallback and continue."""
        from sonality import config
        from sonality.agent import SonalityAgent

        agent, _call_log, loop = _build_mock_agent()
        # Make classify_ess raise to simulate timeout / parse failure
        agent._classify_ess = MagicMock(side_effect=RuntimeError("LLM timed out"))
        try:
            with patch.multiple(
                config,
                SPONGE_FILE="/tmp/test_ess_recovery.json",
                SPONGE_HISTORY_DIR="/tmp/test_ess_recovery_hist",
            ):
                # Should NOT raise — internal ESS guard catches this
                # (If the agent doesn't have a guard, this test exposes the gap)
                try:
                    SonalityAgent._post_process(agent, "user message", "agent response")
                    crashed = False
                except Exception:
                    crashed = True
        finally:
            loop.call_soon_threadsafe(loop.stop)

        # The test documents the expected behavior: no crash allowed.
        # If _classify_ess raises and there's no guard, this records the gap.
        assert not crashed, (
            "ESS exception propagated out of _post_process — "
            "the agent must catch ESS failures and use a safe fallback (classifier_exception_fallback)."
        )

    def test_ess_fallback_does_not_stage_belief_update(self) -> None:
        """Fallback ESS (from exception) should produce reasoning_type that blocks staging."""
        from sonality.ess import classifier_exception_fallback

        # classifier_exception_fallback should produce a safe, non-updating ESS
        ess = classifier_exception_fallback("some message")
        print(f"\n  Fallback ESS: score={ess.score:.3f} type={ess.reasoning_type} "
              f"belief_update_recommended={ess.belief_update_recommended}")

        assert ess.score == 0.0, (
            f"Fallback ESS score must be 0.0 to prevent updates, got {ess.score}"
        )
        assert not ess.belief_update_recommended, (
            "Fallback ESS must NOT recommend belief update — prevents sponge mutation on failure"
        )

    def test_knowledge_extraction_timeout_does_not_prevent_enqueue(self) -> None:
        """When knowledge extraction times out, semantic worker enqueue must still be called."""
        from sonality import config
        from sonality.agent import SonalityAgent

        agent, call_log, loop = _build_mock_agent()
        # Simulate the observed anomaly: knowledge extraction timeout
        agent._extract_knowledge = MagicMock(
            side_effect=RuntimeError("Provider transport error: The read operation timed out")
        )
        try:
            with patch.multiple(
                config,
                SPONGE_FILE="/tmp/test_ke_timeout.json",
                SPONGE_HISTORY_DIR="/tmp/test_ke_timeout_hist",
            ):
                # _extract_knowledge failure should be logged and swallowed.
                # If it's not guarded, enqueue() may never be called — document this gap.
                try:
                    SonalityAgent._post_process(agent, "user message", "agent response")
                    enqueue_called = "semantic_worker_enqueue" in call_log
                except Exception:
                    enqueue_called = False
        finally:
            loop.call_soon_threadsafe(loop.stop)

        print(f"\n  call_log after knowledge extraction exception: {call_log}")
        # This documents whether enqueue is resilient to upstream LLM failures.
        # Expected: enqueue() is still called so the episode gets semantic features eventually.
        assert enqueue_called, (
            "semantic_worker.enqueue() was NOT called after knowledge extraction timeout. "
            "Upstream exceptions in _extract_knowledge must be caught so the episode "
            "still gets enqueued for background feature extraction."
        )
