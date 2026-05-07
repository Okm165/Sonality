"""Sonality agent: stateless, graph-backed personality with LLM-driven tool use.

Each request starts from zero in-memory state. Identity (personality snapshot +
beliefs) is loaded from Neo4j per request. Conversation context is managed by
the caller (chat client / API). The agent uses a unified agentic loop where
all cognitive stages — recall, research, synthesis, and knowledge integration —
are tools the LLM invokes autonomously.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import Future
from contextvars import Token
from typing import Final, Literal, NamedTuple

import structlog
from pydantic import BaseModel, model_validator

from shared.llm.parse import (
    ParsedToolCall,
    coerce_string_fields,
    extract_tool_calls,
    get_raw_tool_calls,
    strip_markdown,
)

from . import config
from .caller import llm_call
from .ess import ESS_FALLBACK, ESSResult, classify
from .memory import (
    BoundaryDecision,
    DatabaseConnections,
    DualEpisodeStore,
    Embedder,
    EventBoundaryDetector,
    MemoryGraph,
    QueryCategory,
    RoutingDecision,
    SemanticIngestionWorker,
    SemanticMemoryDecision,
    StoredEpisode,
    TemporalExpansionDecision,
    assess_belief_evidence_batch,
    chain_retrieve,
    rerank_episodes,
    route_query,
    split_retrieve,
)
from .memory.consolidation import maybe_consolidate_segment
from .memory.graph import (
    BeliefNode,
    EpisodeNode,
    PersonalitySnapshot,
    format_beliefs_for_prompt_from_nodes,
    format_episode_line,
)
from .progress import (
    CONTEXT_BUILD,
    DONE,
    SUMMARIZING,
    THINKING,
    TOOL_CALL,
    TOOL_RESULT,
    AgentEvent,
    noop_progress,
)
from .prompts import (
    INGEST_SYSTEM_PROMPT,
    LOOP_HANDOFF_PROMPT,
    STATE_COMPRESSION_PROMPT,
    build_system_prompt,
)
from .provider import (
    ChatResult,
    StreamChunk,
    default_provider,
    interaction_active,
    interaction_in_progress,
)
from .request_identity import (
    IdentityBundle,
    get_request_identity,
    reset_request_identity,
    set_request_identity,
)
from .schema import DENSE_VECTOR, ChatRole, Collection, SemanticCategory, ToolName
from .token_budget import (
    SUMMARIZE_THRESHOLD,
    message_tokens_budget_for_system,
    summarize_and_trim,
)
from .tools import ToolContext, dispatch_tool, get_definitions
from .tools.reflect import run_forgetting
from .web_client import ResearchClient

log = structlog.get_logger()
_FOLD_PREFIX: Final = "[Folded]"  # marker for folded tool output summaries




class _StateCompressionSchema(BaseModel):
    """Structured state summary for context compression."""

    findings: str = ""
    established: str = ""
    integrated: str = ""
    open_questions: str = ""
    guidance: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_fields(cls, data: object) -> object:
        if isinstance(data, dict) and "stored" in data and "integrated" not in data:
            data["integrated"] = data.pop("stored")
        return coerce_string_fields(
            data, ("findings", "established", "integrated", "open_questions", "guidance")
        )

    def render(self) -> str:
        parts: list[str] = []
        if self.findings:
            parts.append(f"Findings: {self.findings}")
        if self.established:
            parts.append(f"Established: {self.established}")
        if self.integrated:
            parts.append(f"Integrated: {self.integrated}")
        if self.open_questions:
            parts.append(f"Open: {self.open_questions}")
        if self.guidance:
            parts.append(f"Guidance: {self.guidance}")
        return "\n".join(parts)


class _LoopHandoffSchema(BaseModel):
    """Pydantic schema for the structured handoff between agent iterations."""

    action: str = "continue"
    next_focus: str = ""
    established: list[str] = []
    gaps: list[str] = []
    rationale: str = ""
    guidance: str = ""
    critique: str = ""
    knowledge: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_guidance(cls, data: object) -> object:
        """Coerce guidance to string — LLM sometimes returns a dict with 'query'/'text' keys."""
        if isinstance(data, dict):
            raw = data.get("guidance")
            if isinstance(raw, dict):
                data["guidance"] = raw.get("query") or raw.get("text") or json.dumps(raw)
        return coerce_string_fields(
            data, ("guidance", "next_focus", "rationale", "critique", "knowledge")
        )

    @model_validator(mode="after")
    def normalize_and_trim(self) -> _LoopHandoffSchema:
        self.action = self.action.strip().lower()
        self.next_focus = self.next_focus[:120]
        self.established = [s[:80] for s in self.established[:4] if s.strip()]
        self.gaps = [s[:80] for s in self.gaps[:3] if s.strip()]
        self.rationale = self.rationale[:120]
        self.guidance = self.guidance[:400]
        self.critique = self.critique[:300]
        self.knowledge = self.knowledge[:600]
        return self


class _LoopStep(NamedTuple):
    """Compact record of one tool call for the step-context injector.

    summary — one-sentence digest used in the compact history (older steps).
    brief   — first ~800 chars of raw output, shown verbatim for recent steps.
              With Context Folding active, this is the primary source of raw tool
              output available to subsequent iterations. Older tool messages in
              llm_messages are folded to their summaries.
    """

    n: int
    tool: str
    query: str
    summary: str
    brief: str = ""


class SonalityAgent:
    """Stateless personality agent backed by Neo4j (graph) and Qdrant (vectors).

    No in-memory state is carried between requests. Identity and beliefs are
    loaded from persistent stores per request. Conversation context is provided
    by the caller.
    """

    def __init__(self, model: str = config.settings.agent_model) -> None:
        """Create agent, initialize databases, embedder, and background workers."""
        from shared.errors import ConfigError

        missing = config.missing_live_api_config()
        if missing:
            raise ConfigError(f"Missing required API config: {', '.join(missing)}")
        self.model = model
        self.last_ess = ESS_FALLBACK

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="agent-async-loop", daemon=True
        )
        self._loop_thread.start()

        try:
            self._run_async(self._init_runtime())
            log.info("memory_ready")
        except Exception as exc:
            log.error("memory_init_failed", exc_info=True)
            from shared.errors import ServiceUnavailableError

            raise ServiceUnavailableError("Neo4j + Qdrant required and failed to initialize") from exc

        self._boundary_detector = EventBoundaryDetector()
        counter = self._run_async(self._graph.get_latest_segment_counter())
        self._boundary_detector.set_segment_counter(counter)
        self._semantic_worker = SemanticIngestionWorker(config.settings.qdrant_url, self._embedder)
        self._semantic_worker.start()

        self._web_client = ResearchClient(config.settings.fathom_url) if config.settings.web_search_enabled else None
        self._loop_tool_history: list[str] = []
        self._step_log: list[_LoopStep] = []
        self._current_handoff: _LoopHandoffSchema | None = None
        self._knowledge_block: str = ""
        self._last_assistant_msg = ""

        log.info("agent_ready", model=self.model, web="enabled" if self._web_client else "disabled")

    async def _init_runtime(self) -> None:
        """Initialize async resources: databases, embedder, dual store, segment counter."""
        db = await DatabaseConnections.create()
        self._db = db
        self._embedder = Embedder()
        self._graph = MemoryGraph(db.neo4j_driver)
        self._dual_store = DualEpisodeStore(self._graph, db.qdrant, self._embedder)
        last_uid = await self._graph.get_last_episode_uid()
        if last_uid:
            self._dual_store.restore_last_episode(last_uid)

    def _run_async[T](self, coro: Coroutine[object, object, T]) -> T:
        """Bridge sync → async: submit coroutine to the agent's event loop and block."""
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=config.settings.async_timeout)
        except TimeoutError:
            future.cancel()
            raise

    def shutdown(self) -> None:
        """Stop background workers and close database connections."""
        self._semantic_worker.stop()
        if self._web_client:
            self._run_async(self._web_client.close())
        self._run_async(self._db.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        log.info("agent_shutdown")

    # --- Public API ---

    def respond(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_progress: Callable[[AgentEvent], None] | None = None,
    ) -> str:
        """Generate a response given the full conversation history from the caller."""
        with interaction_active():
            return self._respond_inner(
                messages,
                max_tokens=max_tokens if max_tokens is not None else config.settings.llm_max_tokens,
                temperature=temperature if temperature is not None else config.settings.agent_temperature,
                on_progress=on_progress or noop_progress,
            )

    def respond_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_progress: Callable[[AgentEvent], None] | None = None,
    ) -> Iterator[StreamChunk | AgentEvent]:
        """Streaming response variant. Yields StreamChunk for text and AgentEvent for progress."""
        with interaction_active():
            yield from self._respond_stream_inner(
                messages,
                max_tokens=max_tokens if max_tokens is not None else config.settings.llm_max_tokens,
                temperature=temperature if temperature is not None else config.settings.agent_temperature,
                on_progress=on_progress or noop_progress,
            )

    def ingest(self, text: str, *, topic_override: str = "") -> ESSResult:
        """Non-conversational data ingestion with agentic processing.

        High-value content (belief_update_recommended) triggers the full
        agentic loop — the model researches, verifies, reflects, and stores
        knowledge autonomously. Low-value content is triaged and skipped.

        topic_override is forwarded to the agentic loop as a category hint
        inside the user message; it does NOT override ESS topics. The ESS
        model is responsible for extracting topics from the source text.
        """
        _t0 = time.perf_counter()
        id_token = None
        try:
            bundle = self._run_async(self._fetch_identity_bundle())
            id_token = set_request_identity(bundle)
            ess = self._classify_ess(text)
            self.last_ess = ess
            log.info(
                "ingest",
                ess=f"{ess.score:.2f}",
                type=ess.reasoning_type,
                topics=list(ess.topics),
                text=repr(text[:80]),
            )

            if not ess.belief_update_recommended:
                log.info("ingest_skip", type=ess.reasoning_type, score=f"{ess.score:.3f}")
                return ess

            self._ingest_agentic(text, bundle, ess, topic_override=topic_override)
            log.info(
                "ingest_done",
                elapsed=f"{time.perf_counter() - _t0:.1f}s",
                ess=f"{ess.score:.2f}",
                type=ess.reasoning_type,
                topics=list(ess.topics),
            )
            return ess
        finally:
            if id_token is not None:
                reset_request_identity(id_token)

    def _ingest_agentic(
        self, text: str, bundle: IdentityBundle, ess: ESSResult, *, topic_override: str = ""
    ) -> None:
        """Run the full agentic loop on ingested content.

        The model uses tools (recall, web search, synthesize, integrate_knowledge)
        to deeply process the incoming information before it's stored.
        """
        system_prompt = INGEST_SYSTEM_PROMPT.format(
            snapshot_text=bundle.snapshot_text,
            beliefs_text=bundle.beliefs_prompt_text,
        )
        topic_hint = f"\nCategory hint: {topic_override}" if topic_override else ""
        user_content = (
            f"{text}\n\n"
            f"[ESS: score={ess.score:.2f}, type={ess.reasoning_type}, "
            f"topics={', '.join(ess.topics)}{topic_hint}]"
        )
        conv = [{"role": ChatRole.USER, "content": user_content}]

        agent_response = ""
        with interaction_active():
            for item in self._run_agentic_loop(system_prompt, conv, temperature=0.3):
                if isinstance(item, StreamChunk):
                    agent_response = item.content

        # If the agent didn't integrate during the loop, force integration
        integrated = ToolName.INTEGRATE_KNOWLEDGE in self._loop_tool_history
        if not integrated and ess.belief_update_recommended:
            log.info("ingest_force_integrate", topics=list(ess.topics))
            from .tools.memory import execute_integrate_knowledge

            topic = topic_override or (ess.topics[0] if ess.topics else "general")
            ctx = self._make_tool_context([])
            execute_integrate_knowledge({"text": text[:2000], "topic": topic}, ctx)

        try:
            episode_uid = self._store_episode(text, agent_response, ess, "", "")
        except Exception:
            log.error("ingest_episode_storage_failed", exc_info=True)
            raise
        self._assess_provenance(list(ess.topics), episode_uid, text, ess)
        content = f"Content: {text}\nESS: {ess.score:.2f} ({ess.reasoning_type})"
        if agent_response:
            content += f"\nAnalysis: {agent_response[:500]}"
        self._semantic_worker.enqueue(episode_uid, content, (SemanticCategory.KNOWLEDGE,))

    def get_all_beliefs(self) -> list[BeliefNode]:
        """Return all belief nodes from graph."""
        return self._run_async(self._graph.get_all_beliefs())

    def get_belief(self, topic: str) -> BeliefNode | None:
        """Return a single belief node by topic, or None."""
        return self._run_async(self._graph.get_belief(topic))

    def get_snapshot(self) -> PersonalitySnapshot:
        """Return current personality snapshot."""
        return self._run_async(self._graph.get_personality_snapshot())

    def get_health(self) -> tuple[int, int]:
        """Return (belief_count, snapshot_version)."""
        beliefs = self._run_async(self._graph.get_all_beliefs())
        snapshot = self._run_async(self._graph.get_personality_snapshot())
        return len(beliefs), snapshot.version

    # --- Response pipeline ---

    def _prepare_context(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        on_progress: Callable[[AgentEvent], None],
    ) -> tuple[Token[IdentityBundle | None], str, str, list[dict[str, str]]]:
        """Load identity, build system prompt, trim messages."""
        bundle = self._run_async(self._fetch_identity_bundle())
        id_token = set_request_identity(bundle)
        on_progress(AgentEvent(type=CONTEXT_BUILD, detail="Loading identity..."))

        user_message = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == ChatRole.USER),
            "",
        )
        log.info("context", snapshot_chars=len(bundle.snapshot_text), beliefs=len(bundle.all_beliefs))
        system_prompt = build_system_prompt(
            snapshot_text=bundle.snapshot_text,
            beliefs_text=bundle.beliefs_prompt_text,
        )

        msg_count_before = len(messages)
        conv = summarize_and_trim(
            list(messages),
            max_message_tokens=message_tokens_budget_for_system(
                total_budget=config.settings.chat_input_token_budget,
                system_prompt=system_prompt,
                reserve_completion=max_tokens,
            ),
        )
        if len(conv) < msg_count_before and msg_count_before >= SUMMARIZE_THRESHOLD:
            on_progress(
                AgentEvent(
                    type=SUMMARIZING,
                    detail=f"Compressed {msg_count_before} messages to {len(conv)}",
                )
            )
        return id_token, user_message, system_prompt, conv

    def _respond_inner(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        on_progress: Callable[[AgentEvent], None],
    ) -> str:
        """Execute full respond pipeline: prepare context → agentic loop → bookkeep."""
        _t0 = time.perf_counter()
        log.info("request", messages=len(messages), max_tokens=max_tokens, temp=f"{temperature:.2f}")
        id_token = None
        try:
            id_token, user_message, system_prompt, conv = self._prepare_context(
                messages, max_tokens, on_progress
            )
            log.info("user_message", text=user_message[:120])

            assistant_msg = self._agentic_loop(
                system_prompt,
                conv,
                max_tokens=max_tokens,
                temperature=temperature,
                on_progress=on_progress,
            )
            log.info("agent_response", text=assistant_msg[:200])
            elapsed = time.perf_counter() - _t0
            log.info("response_ready", elapsed=f"{elapsed:.1f}s")
            threading.Thread(
                target=self._bookkeep,
                args=(user_message, assistant_msg),
                daemon=True,
                name="bookkeep",
            ).start()
            return assistant_msg
        finally:
            if id_token is not None:
                reset_request_identity(id_token)

    def _respond_stream_inner(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        on_progress: Callable[[AgentEvent], None],
    ) -> Iterator[StreamChunk | AgentEvent]:
        """Agentic streaming: tool calls yield AgentEvents, final text yields StreamChunks."""
        _t0 = time.perf_counter()
        log.info("stream_request", messages=len(messages), max_tokens=max_tokens, temp=f"{temperature:.2f}")
        id_token = None
        try:
            id_token, user_message, system_prompt, conv = self._prepare_context(
                messages, max_tokens, on_progress
            )
            log.info("user_message_stream", text=user_message[:120])

            yield from self._run_agentic_loop(
                system_prompt,
                conv,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            elapsed = time.perf_counter() - _t0
            yield AgentEvent(
                type=DONE,
                detail=json.dumps(
                    {
                        "tool_count": len(self._loop_tool_history),
                        "elapsed": round(elapsed, 1),
                    }
                ),
            )
            log.info("stream_done", elapsed=f"{elapsed:.1f}s")
            threading.Thread(
                target=self._bookkeep,
                args=(user_message, self._last_assistant_msg),
                daemon=True,
                name="bookkeep",
            ).start()
        finally:
            if id_token is not None:
                reset_request_identity(id_token)

    # --- Tool context and loop status ---

    def _make_tool_context(self, llm_messages: list[dict[str, object]]) -> ToolContext:
        """Create ToolContext bridging agent state to the tools package."""
        return ToolContext(
            run_async=self._run_async,
            web_client=self._web_client,
            graph=self._graph,
            dual_store=self._dual_store,
            qdrant=self._db.qdrant,
            embedder=self._embedder,
            identity=get_request_identity(),
            llm_messages=llm_messages,
            retrieve=lambda q: self._run_async(self._retrieve(q)),
        )

    def _summarize_for_step_log(self, tool: str, args: dict[str, object], result: str) -> str:
        """First sentence or 150-char prefix of tool result for the step log."""
        text = result.strip()[:200]
        for sep in (". ", ".\n"):
            idx = text.find(sep)
            if idx > 0:
                return text[: idx + 1]
        return text[:150]

    def _fold_prior_tool_results(
        self, llm_messages: list[dict[str, object]], current_batch_start: int
    ) -> None:
        """Replace older tool results with step-log summaries (Context Folding).

        Only the current batch (messages at index >= current_batch_start) retains
        raw tool output — the LLM will reason over it now. All earlier tool results
        are replaced with their 1-sentence summary from the step log.

        Matches tool messages to step-log entries via _step_n tag (set during
        dispatch), so folding remains correct even after compression removes
        earlier messages and shifts indices.
        """
        step_map = {s.n: s for s in self._step_log}
        folded = 0
        for i in range(current_batch_start):
            msg = llm_messages[i]
            if msg.get("role") != ChatRole.TOOL:
                continue
            raw = str(msg.get("content", ""))
            if len(raw) <= config.settings.episode_content_limit or raw.startswith(_FOLD_PREFIX):
                continue
            step_n = msg.get("_step_n")
            if isinstance(step_n, int) and step_n in step_map:
                s = step_map[step_n]
                msg["content"] = f"{_FOLD_PREFIX} {s.tool}({s.query[:60]}) → {s.summary}"
                folded += 1
        if folded:
            log.info("context_folded", compacted=folded)

    def _generate_handoff(
        self, llm_messages: list[dict[str, object]], iteration: int
    ) -> _LoopHandoffSchema:
        """Produce a structured assessment of loop state + knowledge synthesis.

        Single LLM call that replaces both _update_knowledge and the old handoff.
        The handoff is injected into the next iteration so the agent knows what's
        established, what gaps remain, and what to focus on.
        """
        step_history = "\n".join(
            f"  {s.n}. {s.tool}({s.query[:50]!r}) → {s.summary}" for s in self._step_log[-5:]
        )
        recent_results = "\n".join(
            f"- {s.tool}({s.query[:40]}): {s.brief[:300]}" for s in self._step_log[-3:]
        )
        tool_context = (
            f"Recent results:\n{recent_results}\n"
            f"Prior knowledge: {self._knowledge_block[:400] or '(none yet)'}"
        )
        result = llm_call(
            prompt=LOOP_HANDOFF_PROMPT.format(
                iteration=iteration + 1,
                step_history=step_history or "(none yet)",
                tool_context=tool_context,
            ),
            response_model=_LoopHandoffSchema,
            fallback=_LoopHandoffSchema(),
        )
        h = result.value
        if h.knowledge.strip():
            self._knowledge_block = h.knowledge

        rationale = h.rationale or (
            f"Investigating: {h.gaps[0][:60] if h.gaps else h.next_focus[:60] or 'further detail needed'}"
        )
        action: Literal["finish", "continue"] = "finish" if h.action == "finish" else "continue"

        handoff = _LoopHandoffSchema(
            action=action,
            next_focus=h.next_focus,
            established=h.established,
            gaps=h.gaps,
            rationale=rationale,
            guidance=h.guidance,
            critique=h.critique,
        )
        log.info(
            "handoff",
            step=iteration + 1,
            action=handoff.action,
            gaps=handoff.gaps,
            rationale=handoff.rationale[:80],
            guidance=handoff.guidance[:60] if handoff.guidance else "",
        )
        return handoff

    def _compress_messages(
        self,
        llm_messages: list[dict[str, object]],
        iteration: int,
    ) -> list[dict[str, object]]:
        """Compress accumulated messages to prevent context explosion.

        Second layer of context management (tool output folding is first layer).
        With folding active, tool results in the middle section are already compact.
        This compression targets assistant reasoning and system messages that have
        accumulated beyond the threshold. Preserves head (system + user) and tail.
        """
        if len(llm_messages) <= config.settings.agent_compress_threshold:
            return llm_messages

        preserve_head = llm_messages[:2]
        keep_tail = config.settings.agent_compress_keep_tail
        compress_section = llm_messages[2:-keep_tail]
        preserve_tail = llm_messages[-keep_tail:]

        history = "\n\n".join(
            f"[{msg.get('role', 'unknown')}]: {str(msg.get('content', ''))[:600]}"
            for msg in compress_section
            if msg.get("content")
        )
        if not history:
            return llm_messages

        comp_result = llm_call(
            prompt=STATE_COMPRESSION_PROMPT.format(history=history),
            response_model=_StateCompressionSchema,
            fallback=_StateCompressionSchema(findings=history[:500]),
            model=config.settings.structured_model,
        )
        summary = comp_result.value.render()
        if len(summary) < 50:
            log.warning("compress_too_short", chars=len(summary))
            return llm_messages
        log.info(
            "context_compressed",
            iteration=iteration + 1,
            messages=len(compress_section),
            summary_chars=len(summary),
        )
        compressed: dict[str, object] = {
            "role": ChatRole.SYSTEM,
            "content": f"[State summary — {len(compress_section)} messages compressed]:\n{summary}",
        }
        return [*preserve_head, compressed, *preserve_tail]

    def _build_step_context(self, iteration: int) -> str:
        """Build a structured context block injected each iteration.

        Three separated channels (cf. ContextWeaver 2025, HiAgent 2025):

        1. **Thread State** — handoff guidance, current reasoning direction. This is
           the "prompt from the previous agent" that tells the next iteration what to do.
        2. **Tool Execution Log** — compact history of every tool call with 1-line
           summaries (older) and a raw excerpt for the most recent step. The agent uses
           this to avoid redundant calls and reason about what's been tried.
        3. **Knowledge Anchor** — accumulated verified facts. A persistent, growing
           summary that outlives message compression and folding.
        """
        header = f"[Iteration {iteration + 1}]"
        if not self._step_log and self._current_handoff is None:
            return header

        sections: list[str] = [header]

        # --- Section 1: Thread State (handoff as "prompt to next agent") ---
        if self._current_handoff is not None:
            h = self._current_handoff
            thread: list[str] = ["## Thread State"]
            if h.guidance:
                thread.append(f"Instructions: {h.guidance}")
            if h.next_focus:
                thread.append(f"Focus: {h.next_focus}")
            if h.established:
                thread.append("Confirmed: " + "; ".join(h.established))
            if h.gaps:
                thread.append("Open: " + "; ".join(h.gaps))
            if h.rationale:
                thread.append(f"Rationale: {h.rationale}")
            sections.append("\n".join(thread))

        # --- Section 2: Tool Execution Log ---
        if self._step_log:
            recent_threshold = max(1, len(self._step_log) - 1)
            lines: list[str] = []
            for s in self._step_log:
                lines.append(f"  {s.n}. {s.tool}({s.query[:50]!r}) → {s.summary}")
                if s.n > recent_threshold and s.brief:
                    lines.append(f"     [Raw]: {s.brief[:400]}")
            log.debug(
                "iteration_steps",
                iteration=iteration + 1,
                steps=len(self._step_log),
                tools=" | ".join(f"{s.tool}({s.query[:25]!r})" for s in self._step_log),
            )
            sections.append("## Tool Execution Log\n" + "\n".join(lines))

        # --- Section 3: Knowledge Anchor ---
        if self._knowledge_block:
            sections.append(f"## Verified Knowledge\n{self._knowledge_block}")

        return "\n\n".join(sections)

    _STEP_CTX_TAG: Final = "__step_context__"

    def _prepare_iteration(
        self,
        iteration: int,
        llm_messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Per-iteration setup: replace (not accumulate) step context and return tool definitions.

        A tagged USER message carries the latest step context. On subsequent
        iterations the previous one is overwritten, preventing stale handoff
        data and tool-log duplicates from piling up in the conversation.
        """
        ctx_msg: dict[str, object] = {
            "role": ChatRole.USER,
            "content": self._build_step_context(iteration),
            "_tag": self._STEP_CTX_TAG,
        }
        for i in range(len(llm_messages) - 1, -1, -1):
            if llm_messages[i].get("_tag") == self._STEP_CTX_TAG:
                llm_messages[i] = ctx_msg
                break
        else:
            llm_messages.append(ctx_msg)
        return get_definitions()

    # --- Agentic tool-calling loop ---

    def _dedup_tool_calls(
        self,
        tool_calls: list[ParsedToolCall],
        recent_calls: set[tuple[str, str]],
    ) -> list[ParsedToolCall]:
        """Filter exact duplicate tool calls (same name + same args) as a safety measure.

        No warnings or behavioral nudges are injected — the LLM manages its own
        tool strategy via prompts. This only prevents wasting compute on truly
        identical repeat calls.
        """
        calls: list[ParsedToolCall] = []
        for tc in tool_calls:
            sig = (tc.name, json.dumps(tc.args, sort_keys=True))
            if sig in recent_calls:
                log.info("dedup_skip", tool=tc.name)
            else:
                recent_calls.add(sig)
                calls.append(tc)
        return calls

    def _dispatch_tools(
        self,
        calls: list[ParsedToolCall],
        llm_messages: list[dict[str, object]],
    ) -> list[tuple[ParsedToolCall, str]]:
        """Execute tool calls, record results, and append a compressed step-log entry."""
        ctx = self._make_tool_context(llm_messages)
        results: list[tuple[ParsedToolCall, str]] = []
        for tc in calls:
            step_n = len(self._step_log) + 1
            key_arg = str(
                tc.args.get("query")
                or tc.args.get("goal")
                or tc.args.get("url")
                or tc.args.get("focus")
                or tc.args.get("topic")
                or tc.args.get("text")
                or ""
            )[:120]
            log.info("tool_start", step=step_n, tool=tc.name, arg=key_arg)
            t_tool = time.perf_counter()
            result_text = dispatch_tool(tc.name, tc.args, ctx)
            tool_elapsed = time.perf_counter() - t_tool
            self._loop_tool_history.append(tc.name)
            llm_messages.append(
                {
                    "role": ChatRole.TOOL,
                    "content": result_text,
                    "tool_call_id": tc.id,
                    "_step_n": step_n,
                }
            )
            summary = self._summarize_for_step_log(tc.name, tc.args, result_text)
            log.info(
                "tool_done",
                step=step_n,
                tool=tc.name,
                summary=summary[:100],
                elapsed=f"{tool_elapsed:.1f}s",
                chars=len(result_text),
            )
            self._step_log.append(
                _LoopStep(
                    n=step_n, tool=tc.name, query=key_arg, summary=summary, brief=result_text[:800]
                )
            )
            results.append((tc, result_text))
        return results

    def _run_agentic_loop(
        self,
        system_prompt: str,
        conv: list[dict[str, str]],
        *,
        max_tokens: int = config.settings.llm_max_tokens,
        temperature: float = config.settings.agent_temperature,
    ) -> Iterator[AgentEvent | StreamChunk]:
        """Core agentic loop — yields events during reasoning, StreamChunk for the final answer.

        The LLM decides the full workflow via tool calls. Stall detection and
        dedup prevent infinite loops. A generous hard ceiling is a pure circuit breaker.
        """
        self._loop_tool_history.clear()
        self._step_log.clear()
        self._current_handoff = None
        self._knowledge_block = ""
        llm_messages: list[dict[str, object]] = [
            {"role": ChatRole.SYSTEM, "content": system_prompt},
        ]
        llm_messages.extend(dict(m) for m in conv)
        completion = ChatResult(text="", input_tokens=0, output_tokens=0, raw={})

        t_loop_start = time.perf_counter()
        for iteration in range(config.settings.agent_loop_hard_ceiling):
            t_iter = time.perf_counter()
            recent_calls: set[tuple[str, str]] = set()
            yield AgentEvent(type=THINKING, iteration=iteration, detail="Analyzing...")
            log.debug("loop_step", iteration=iteration + 1)

            llm_messages = self._compress_messages(llm_messages, iteration)
            tools = self._prepare_iteration(iteration, llm_messages)
            ctx_chars = sum(len(str(m.get("content", ""))) for m in llm_messages)
            log.info(
                "iteration_context",
                iteration=iteration + 1,
                messages=len(llm_messages),
                chars_k=ctx_chars // 1000,
            )
            finishing = (
                self._current_handoff is not None and self._current_handoff.action == "finish"
            )
            finish_budget = max(max_tokens, config.settings.llm_max_tokens) if finishing else max_tokens
            try:
                completion = default_provider.chat_completion(
                    model=self.model,
                    max_tokens=finish_budget,
                    temperature=temperature,
                    messages=llm_messages,
                    tools=tools if not finishing else [],
                    tool_choice="none" if finishing else "auto",
                )
            except Exception:
                log.error("llm_failed", step=iteration, exc_info=True)
                completion = ChatResult(
                    text="I encountered a processing error. Please try again.",
                    input_tokens=0,
                    output_tokens=0,
                    raw={},
                )
                break

            tool_calls = extract_tool_calls(completion.raw)
            llm_elapsed = time.perf_counter() - t_iter
            log.info(
                "iteration_llm",
                iteration=iteration + 1,
                elapsed=f"{llm_elapsed:.1f}s",
                in_tokens=completion.input_tokens,
                out_tokens=completion.output_tokens,
                tools=len(tool_calls),
                text_chars=len(completion.text),
            )
            if not tool_calls:
                if self._step_log:
                    log.info(
                        "loop_finished",
                        step=iteration + 1,
                        elapsed=f"{time.perf_counter() - t_loop_start:.1f}s",
                        trace=" → ".join(f"{s.tool}({s.query[:25]!r})" for s in self._step_log),
                    )
                clean = strip_markdown(completion.text)
                self._last_assistant_msg = clean
                yield StreamChunk(clean)
                return

            calls_to_run = self._dedup_tool_calls(tool_calls, recent_calls)
            if not calls_to_run:
                # All calls deduplicated; append assistant message WITHOUT tool_calls
                # to avoid leaving dangling tool_calls without results.
                llm_messages.append(
                    {
                        "role": ChatRole.ASSISTANT,
                        "content": completion.text or "",
                    }
                )
                continue

            assistant_msg: dict[str, object] = {
                "role": ChatRole.ASSISTANT,
                "content": completion.text or "",
            }
            raw_tc = get_raw_tool_calls(completion.raw)
            if raw_tc:
                assistant_msg["tool_calls"] = raw_tc
            llm_messages.append(assistant_msg)

            for tc in calls_to_run:
                yield AgentEvent(
                    type=TOOL_CALL,
                    tool_name=tc.name,
                    tool_args=json.dumps(tc.args),
                    iteration=iteration,
                )
            batch_start = len(llm_messages)
            t_dispatch = time.perf_counter()
            for tc, result_text in self._dispatch_tools(calls_to_run, llm_messages):
                yield AgentEvent(
                    type=TOOL_RESULT,
                    tool_name=tc.name,
                    tool_result_summary=result_text[:200],
                    iteration=iteration,
                    sources_count=(
                        result_text.count("[SOURCE ") if tc.name == ToolName.WEB_SEARCH else 0
                    ),
                )
            log.info("tool_dispatch", calls=len(calls_to_run), elapsed=f"{time.perf_counter() - t_dispatch:.1f}s")
            self._fold_prior_tool_results(llm_messages, batch_start)

            t_meta = time.perf_counter()
            self._current_handoff = self._generate_handoff(llm_messages, iteration)

            if self._current_handoff.critique:
                llm_messages.append(
                    {
                        "role": ChatRole.SYSTEM,
                        "content": (
                            f"[Problem identified — address this next]: "
                            f"{self._current_handoff.critique}"
                        ),
                    }
                )
                log.info("critique_injected", text=self._current_handoff.critique[:100])

            log.info(
                "handoff_done",
                handoff_elapsed=f"{time.perf_counter() - t_meta:.1f}s",
                iteration=iteration + 1,
                iter_total=f"{time.perf_counter() - t_iter:.1f}s",
            )
            if self._current_handoff.action == "finish":
                log.info("handoff_finish_signal")
                h = self._current_handoff
                wrap_parts = ["[Loop coordinator]: Research complete."]
                if self._knowledge_block:
                    wrap_parts.append(f"Verified knowledge:\n{self._knowledge_block}")
                if h.established:
                    wrap_parts.append("Confirmed: " + "; ".join(h.established))
                if h.guidance:
                    wrap_parts.append(f"Guidance: {h.guidance}")
                wrap_parts.append(
                    "Now give your final response. Lead with your conclusion, "
                    "then support it with the strongest evidence. When multiple "
                    "sources agree, say so. When claims rest on limited evidence, "
                    "note the limitation. If something remains genuinely uncertain, "
                    "say what you'd need to resolve it."
                )
                llm_messages.append(
                    {
                        "role": ChatRole.SYSTEM,
                        "content": "\n\n".join(wrap_parts),
                    }
                )

        log.warning("loop_ceiling", steps=config.settings.agent_loop_hard_ceiling)
        clean = strip_markdown(completion.text)
        self._last_assistant_msg = clean
        yield StreamChunk(clean)

    def _agentic_loop(
        self,
        system_prompt: str,
        conv: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        on_progress: Callable[[AgentEvent], None],
    ) -> str:
        """Non-streaming wrapper: forwards events to on_progress, returns final text."""
        for item in self._run_agentic_loop(
            system_prompt, conv, max_tokens=max_tokens, temperature=temperature
        ):
            if isinstance(item, AgentEvent):
                on_progress(item)
        return self._last_assistant_msg

    # --- Identity loading ---

    async def _fetch_identity_bundle(self) -> IdentityBundle:
        """Load snapshot and beliefs once (sorted by |valence|, same window as prompt formatting)."""
        snapshot = await self._graph.get_personality_snapshot()
        all_beliefs = await self._graph.get_all_beliefs()
        window = all_beliefs[: config.settings.belief_prompt_window]
        beliefs_text = format_beliefs_for_prompt_from_nodes(window)
        return IdentityBundle(
            snapshot_text=snapshot.text,
            beliefs_prompt_text=beliefs_text,
            all_beliefs=tuple(all_beliefs),
        )

    # --- Retrieval ---

    async def _retrieve(self, user_message: str) -> list[str]:
        """Full retrieval pipeline: route → category-specific search → expand → rerank."""
        if not self._dual_store.has_episodes:
            return []

        decision = await asyncio.to_thread(route_query, user_message)
        log.debug("route", category=decision.category, n=decision.n_results)
        if decision.category == QueryCategory.NONE:
            return []

        episodes = await self._fetch_episodes_for_category(decision, user_message)
        episodes = list({ep.uid: ep for ep in episodes}.values())

        if decision.temporal_expansion is TemporalExpansionDecision.EXPAND and episodes:
            expanded_uids: set[str] = set()
            for ep in episodes:
                for n in await self._graph.traverse_temporal_context(ep.uid):
                    expanded_uids.add(n.uid)
            new_uids = [u for u in expanded_uids if u not in {e.uid for e in episodes}]
            if new_uids:
                episodes.extend(await self._graph.get_episodes(new_uids))

        if len(episodes) > 1 and len(episodes) > decision.n_results:
            episodes = await asyncio.to_thread(rerank_episodes, user_message, episodes)

        selected = episodes[: decision.n_results]
        semantic_context: list[str] = []
        if decision.semantic_memory is SemanticMemoryDecision.SEARCH:
            semantic_context = await self._search_semantic_features(
                user_message, top_k=decision.n_results
            )

        episode_context = [
            format_episode_line(
                created_at=ep.created_at,
                summary=ep.summary,
                content=ep.content,
                content_limit=config.settings.episode_content_limit,
            )
            for ep in selected
        ]
        log.info("retrieval", category=decision.category, episodes=len(selected), semantic=len(semantic_context))
        return [*episode_context, *semantic_context]

    async def _fetch_episodes_for_category(
        self, decision: RoutingDecision, user_message: str
    ) -> list[EpisodeNode]:
        """Dispatch episode fetching based on query category."""
        cat = decision.category
        over_fetch = decision.n_results * config.settings.retrieval_over_fetch_factor

        if cat in (QueryCategory.TEMPORAL, QueryCategory.AGGREGATION):
            return await chain_retrieve(
                self._dual_store, self._graph, user_message, base_n=decision.n_results
            )

        if cat == QueryCategory.MULTI_ENTITY and decision.should_decompose:
            return await split_retrieve(
                self._dual_store, self._graph, user_message, n_per_sub=decision.n_results
            )

        vector_hits = await self._dual_store.vector_search(user_message, top_k=over_fetch)
        vector_uids = list({h.episode_uid for h in vector_hits})
        topic_hits = await self._graph.find_topic_related_episodes(user_message, limit=over_fetch)
        belief_hits = (
            await self._graph.find_belief_related_episodes(user_message, limit=over_fetch)
            if cat == QueryCategory.BELIEF_QUERY
            else []
        )
        return belief_hits + topic_hits + await self._graph.get_episodes(vector_uids)

    async def _search_semantic_features(self, query: str, *, top_k: int) -> list[str]:
        query_embedding = await asyncio.to_thread(self._embedder.embed_query, query)
        response = await self._db.qdrant.query_points(
            collection_name=Collection.SEMANTIC_FEATURES,
            query=query_embedding,
            using=DENSE_VECTOR,
            limit=top_k,
            with_payload=True,
        )
        return [
            f"[semantic/{p.payload.get('category', '')}] "
            f"{p.payload.get('tag', '')}.{p.payload.get('feature_name', '')}: "
            f"{p.payload.get('value', '')} "
            f"(conf={float(p.payload.get('confidence', 0)):.2f}, score={p.score:.3f})"
            for p in response.points
            if p.payload
        ]

    # --- Bookkeeping ---

    def _bookkeep(self, user_message: str, agent_response: str) -> None:
        """Mechanical bookkeeping after the agent responds.

        ESS classification, episode storage, provenance assessment,
        semantic feature ingestion, and periodic forgetting.
        Yields to new interactions — aborts if another request arrives.
        """
        try:
            ess = self._classify_ess(user_message)
        except Exception:
            log.error("ess_classification_failed", exc_info=True)
            ess = ESS_FALLBACK
        self.last_ess = ess
        log.info(
            "ess_result",
            score=f"{ess.score:.2f}",
            type=ess.reasoning_type,
            update=ess.belief_update_recommended,
            topics=list(ess.topics),
        )

        if interaction_in_progress():
            log.info("bookkeeping_yield_after_ess")
            return

        previous_segment_id = self._boundary_detector.current_segment_id
        segment_id = segment_label = ""
        closed_segment_id = ""
        try:
            boundary = self._boundary_detector.check_boundary(user_message)
            segment_id = boundary.segment_id
            if boundary.boundary_decision is BoundaryDecision.BOUNDARY:
                closed_segment_id = previous_segment_id
                segment_label = boundary.label
        except Exception:
            log.error("boundary_detection_failed", exc_info=True)

        try:
            episode_uid = self._store_episode(
                user_message, agent_response, ess, segment_id, segment_label
            )
        except Exception:
            log.error("episode_storage_failed", exc_info=True)
            return

        if interaction_in_progress():
            log.info("bookkeeping_yield_after_store")
            return

        if closed_segment_id:
            try:
                self._run_async(maybe_consolidate_segment(self._graph, closed_segment_id))
            except Exception:
                log.warning("consolidation_failed", segment=closed_segment_id, exc_info=True)

        if episode_uid:
            if ess.belief_update_recommended:
                try:
                    self._assess_provenance(list(ess.topics), episode_uid, user_message, ess)
                except Exception:
                    log.error("provenance_assessment_failed", episode_uid=episode_uid[:8], exc_info=True)

            content = (
                f"User: {user_message}\nAssistant: {agent_response}\n"
                f"ESS: {ess.score:.2f} ({ess.reasoning_type})"
                if agent_response
                else f"Content: {user_message}\nESS: {ess.score:.2f} ({ess.reasoning_type})"
            )
            categories = (SemanticCategory.KNOWLEDGE,) if not agent_response else ()
            self._semantic_worker.enqueue(episode_uid, content, categories)

        if not interaction_in_progress():
            ctx = self._make_tool_context([])
            run_forgetting(ctx)

    def _classify_ess(self, user_message: str) -> ESSResult:
        """Run ESS classification."""
        result = classify(user_message)
        if result.topics:
            normalized = tuple(
                re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_") for t in result.topics
            )
            result = dataclasses.replace(result, topics=normalized)
        return result

    def _store_episode(
        self,
        user_message: str,
        agent_response: str,
        ess: ESSResult,
        segment_id: str,
        segment_label: str,
    ) -> str:
        stored: StoredEpisode = self._run_async(
            self._dual_store.store(
                user_message=user_message,
                agent_response=agent_response,
                summary=ess.summary[:300],
                topics=list(ess.topics),
                ess_score=ess.score,
                reasoning_type=ess.reasoning_type,
                segment_id=segment_id,
                segment_label=segment_label,
            )
        )
        log.info(
            "episode_stored",
            uid=stored.episode_uid[:8],
            segment=segment_id[:8] if segment_id else "none",
            topics=list(ess.topics),
        )
        return stored.episode_uid

    def _assess_provenance(
        self,
        topics: list[str],
        episode_uid: str,
        content: str,
        ess: ESSResult,
    ) -> None:
        """Create provenance edges (SUPPORTS/CONTRADICTS) for relevant beliefs."""
        if not topics:
            return
        topics = [re.sub(r"[^a-z0-9]+", "_", t.lower()).strip("_") for t in topics if t]
        if not topics:
            return
        bundle = get_request_identity()
        if bundle is not None:
            beliefs_dict = {b.topic: b for b in bundle.all_beliefs}
        else:
            all_beliefs = self._run_async(self._graph.get_all_beliefs())
            beliefs_dict = {b.topic: b for b in all_beliefs}
        self._run_async(
            assess_belief_evidence_batch(
                topics=topics,
                episode_uid=episode_uid,
                episode_content=content,
                ess_score=ess.score,
                reasoning_type=ess.reasoning_type,
                source_reliability=ess.source_reliability,
                beliefs=beliefs_dict,
                graph=self._graph,
            )
        )
