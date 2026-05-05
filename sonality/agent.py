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
import logging
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import Future
from contextvars import Token
from typing import Final, Literal, NamedTuple

from pydantic import BaseModel, model_validator

from . import config
from .ess import ESSResult, classifier_exception_fallback, classify
from .llm.caller import llm_call, quorum_critique
from .llm.parse import (
    ParsedToolCall,
    coerce_string_fields,
    extract_tool_calls,
    get_raw_tool_calls,
    strip_markdown,
)
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
    BELIEF_PROMPT_WINDOW,
    BeliefNode,
    EpisodeNode,
    PersonalitySnapshot,
    format_beliefs_for_prompt_from_nodes,
    format_episode_line,
)
from .progress import (
    CONTEXT_BUILD,
    DONE,
    REVIEWING,
    SUMMARIZING,
    THINKING,
    TOOL_CALL,
    TOOL_RESULT,
    AgentEvent,
    noop_progress,
)
from .prompts import (
    INGEST_SYSTEM_PROMPT,
    KNOWLEDGE_UPDATE_PROMPT,
    LOOP_HANDOFF_PROMPT,
    STATE_COMPRESSION_PROMPT,
    STEP_SUMMARY_PROMPT,
    TOOL_RESULT_DIGEST_PROMPT,
    build_system_prompt,
)
from .provider import (
    ChatResult,
    StreamChunk,
    default_provider,
    interaction_active,
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
from .web import get_client

log = logging.getLogger(__name__)

_HARD_CEILING: Final = 12  # pure safety circuit breaker — resource exhaustion guard
_COMPRESS_THRESHOLD: Final = (
    12  # compress when message count exceeds this; folding is the primary mechanism
)
_COMPRESS_KEEP_TAIL: Final = 4  # always keep the most recent N messages intact
_FOLD_MIN_CHARS: Final = 300  # only fold tool results longer than this
_FOLD_PREFIX: Final = "[Folded]"  # marker for folded tool output summaries
_QUORUM_CRITIQUE: Final = "quorum_critique"  # internal-only step, not a ToolName member


class _StepSummarySchema(BaseModel):
    """One-sentence summary of a tool result."""

    summary: str = ""


class _YieldDigestSchema(BaseModel):
    """Structured assessment of what recent tool calls produced."""

    new_facts: str = ""
    had_empty_results: bool = False
    recommendation: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_types(cls, data: object) -> object:
        if isinstance(data, dict):
            raw = data.get("had_empty_results")
            if isinstance(raw, str):
                data["had_empty_results"] = raw.strip().lower() in ("true", "yes", "1")
        return data

    @model_validator(mode="after")
    def trim(self) -> _YieldDigestSchema:
        self.new_facts = self.new_facts[:300]
        self.recommendation = self.recommendation[:200]
        return self


class _KnowledgeBlockSchema(BaseModel):
    """Updated knowledge block with accumulated findings."""

    block: str = ""

    @model_validator(mode="before")
    @classmethod
    def coerce_block(cls, data: object) -> object:
        return coerce_string_fields(data, ("block",), sep="\n")

    @model_validator(mode="after")
    def trim(self) -> _KnowledgeBlockSchema:
        self.block = self.block[:1200]
        return self


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

    @model_validator(mode="before")
    @classmethod
    def coerce_guidance(cls, data: object) -> object:
        """Coerce guidance to string — LLM sometimes returns a dict with 'query'/'text' keys."""
        if isinstance(data, dict):
            raw = data.get("guidance")
            if isinstance(raw, dict):
                data["guidance"] = raw.get("query") or raw.get("text") or json.dumps(raw)
        return coerce_string_fields(data, ("guidance", "next_focus", "rationale"))

    @model_validator(mode="after")
    def normalize_and_trim(self) -> _LoopHandoffSchema:
        self.action = self.action.strip().lower()
        self.next_focus = self.next_focus[:120]
        self.established = [s[:80] for s in self.established[:4] if s.strip()]
        self.gaps = [s[:80] for s in self.gaps[:3] if s.strip()]
        self.rationale = self.rationale[:120]
        self.guidance = self.guidance[:400]
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

    def __init__(
        self,
        model: str = config.AGENT_MODEL,
        ess_model: str = config.STRUCTURED_MODEL,
    ) -> None:
        """Create agent, initialize databases, embedder, and background workers."""
        missing = config.missing_live_api_config()
        if missing:
            raise ValueError(f"Missing required API config: {', '.join(missing)}")
        self.model = model
        self.ess_model = ess_model
        self.last_ess = classifier_exception_fallback("")

        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="agent-async-loop", daemon=True
        )
        self._loop_thread.start()

        try:
            self._run_async(self._init_runtime())
            log.info("Memory architecture initialized (Neo4j + Qdrant)")
        except Exception as exc:
            log.exception("Memory architecture initialization failed")
            raise RuntimeError("Neo4j + Qdrant required and failed to initialize") from exc

        self._boundary_detector = EventBoundaryDetector()
        counter = self._run_async(self._graph.get_latest_segment_counter())
        self._boundary_detector.set_segment_counter(counter)
        self._semantic_worker = SemanticIngestionWorker(config.QDRANT_URL, self._embedder)
        self._semantic_worker.start()

        self._web_client = get_client()
        self._loop_tool_history: list[str] = []
        self._step_log: list[_LoopStep] = []
        self._current_handoff: _LoopHandoffSchema | None = None
        self._knowledge_block: str = ""
        self._last_assistant_msg = ""

        log.info(
            "SonalityAgent ready (model=%s, ess=%s, web=%s)",
            self.model,
            self.ess_model,
            "enabled" if self._web_client else "disabled",
        )

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
            return future.result(timeout=config.ASYNC_TIMEOUT)
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
        log.info("Agent shut down")

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
                max_tokens=max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
                temperature=temperature if temperature is not None else config.AGENT_TEMPERATURE,
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
                max_tokens=max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
                temperature=temperature if temperature is not None else config.AGENT_TEMPERATURE,
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
                "Ingest: ess=%.2f type=%s topics=%s text=%.80r",
                ess.score,
                ess.reasoning_type,
                list(ess.topics),
                text,
            )

            if not ess.belief_update_recommended:
                log.info(
                    "Ingest: update not recommended (type=%s score=%.3f)",
                    ess.reasoning_type,
                    ess.score,
                )
                return ess

            self._ingest_agentic(text, bundle, ess, topic_override=topic_override)
            log.info(
                "Ingest done: %.1fs | ess=%.2f type=%s topics=%s",
                time.perf_counter() - _t0,
                ess.score,
                ess.reasoning_type,
                list(ess.topics),
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

        episode_uid = self._store_episode(text, agent_response, ess, "", "")
        if episode_uid:
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
        log.info(
            "Context: snapshot=%d chars, %d beliefs",
            len(bundle.snapshot_text),
            len(bundle.all_beliefs),
        )
        system_prompt = build_system_prompt(
            snapshot_text=bundle.snapshot_text,
            beliefs_text=bundle.beliefs_prompt_text,
        )

        msg_count_before = len(messages)
        conv = summarize_and_trim(
            list(messages),
            max_message_tokens=message_tokens_budget_for_system(
                total_budget=config.CHAT_INPUT_TOKEN_BUDGET,
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
        log.info(
            "Request: %d messages, max_tokens=%d, temp=%.2f", len(messages), max_tokens, temperature
        )
        id_token = None
        try:
            id_token, user_message, system_prompt, conv = self._prepare_context(
                messages, max_tokens, on_progress
            )
            log.info("User: %.120s", user_message)

            assistant_msg = self._agentic_loop(
                system_prompt,
                conv,
                max_tokens=max_tokens,
                temperature=temperature,
                on_progress=on_progress,
            )
            log.info("Agent: %.200s", assistant_msg)
            self._bookkeep(user_message, assistant_msg)
            elapsed = time.perf_counter() - _t0
            ess = self.last_ess
            on_progress(
                AgentEvent(
                    type=DONE,
                    detail=json.dumps(
                        {
                            "tool_count": len(self._loop_tool_history),
                            "elapsed": round(elapsed, 1),
                            "ess_score": round(ess.score, 2),
                            "reasoning_type": ess.reasoning_type,
                            "topics": list(ess.topics),
                        }
                    ),
                )
            )
            log.info("Total: %.1fs", elapsed)
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
        log.info(
            "Stream request: %d messages, max_tokens=%d, temp=%.2f",
            len(messages),
            max_tokens,
            temperature,
        )
        id_token = None
        try:
            id_token, user_message, system_prompt, conv = self._prepare_context(
                messages, max_tokens, on_progress
            )
            log.info("User (stream): %.120s", user_message)

            yield from self._run_agentic_loop(
                system_prompt,
                conv,
                max_tokens=max_tokens,
                temperature=temperature,
            )

            self._bookkeep(user_message, self._last_assistant_msg)
            elapsed = time.perf_counter() - _t0
            ess = self.last_ess
            yield AgentEvent(
                type=DONE,
                detail=json.dumps(
                    {
                        "tool_count": len(self._loop_tool_history),
                        "elapsed": round(elapsed, 1),
                        "ess_score": round(ess.score, 2),
                        "reasoning_type": ess.reasoning_type,
                        "topics": list(ess.topics),
                    }
                ),
            )
            log.info("Stream total: %.1fs", elapsed)
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
        """Compress a tool result to a 1-sentence summary via FAST_MODEL.

        Uses FAST_MODEL (small, co-loads with AGENT) to avoid forcing a model swap
        to STRUCTURED after every tool call — the most expensive switching pattern.
        """
        query = str(
            args.get("query") or args.get("focus") or args.get("topic") or args.get("text") or ""
        )
        prompt = STEP_SUMMARY_PROMPT.format(
            tool=tool,
            query=query[:120],
            result=result[:600],
        )
        r = llm_call(
            prompt=prompt,
            response_model=_StepSummarySchema,
            fallback=_StepSummarySchema(summary=result[:120]),
            model=config.FAST_MODEL,
        )
        return r.value.summary.strip() or result[:120]

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
        step_map = {s.n: s for s in self._step_log if s.tool != _QUORUM_CRITIQUE}
        folded = 0
        for i in range(current_batch_start):
            msg = llm_messages[i]
            if msg.get("role") != ChatRole.TOOL:
                continue
            raw = str(msg.get("content", ""))
            if len(raw) <= _FOLD_MIN_CHARS or raw.startswith(_FOLD_PREFIX):
                continue
            step_n = msg.get("_step_n")
            if isinstance(step_n, int) and step_n in step_map:
                s = step_map[step_n]
                msg["content"] = f"{_FOLD_PREFIX} {s.tool}({s.query[:60]}) → {s.summary}"
                folded += 1
        if folded:
            log.info("Context folding: compacted %d prior tool results", folded)

    def _generate_handoff(
        self, llm_messages: list[dict[str, object]], iteration: int
    ) -> _LoopHandoffSchema:
        """Produce a structured assessment of loop state using STRUCTURED_MODEL.

        Runs after each tool batch. The handoff is injected into the next iteration
        so the agent knows what's established, what gaps remain, and what to focus on.
        """
        step_history = "\n".join(
            f"  {s.n}. {s.tool}({s.query[:50]!r}) → {s.summary}" for s in self._step_log
        )
        knowledge = self._knowledge_block[:400] if self._knowledge_block else "(nothing stored yet)"
        recent_for_digest = "\n".join(
            f"- {s.tool}({s.query[:40]}): {s.brief[:250]}" for s in self._step_log[-4:]
        )
        digest_result = llm_call(
            prompt=TOOL_RESULT_DIGEST_PROMPT.format(tool_results=recent_for_digest),
            response_model=_YieldDigestSchema,
            fallback=_YieldDigestSchema(),
            model=config.STRUCTURED_MODEL,
        )
        yd = digest_result.value
        log.info(
            "Yield digest: facts=%s empty=%s rec=%s",
            yd.new_facts[:80],
            yd.had_empty_results,
            yd.recommendation[:60],
        )

        # Build rich context for the handoff model — all facts, no heuristic overrides.
        tools_used = {s.tool for s in self._step_log}
        all_tools = tuple(ToolName)
        coverage = ", ".join(t for t in all_tools if t in tools_used) or "none"
        unused = ", ".join(t for t in all_tools if t not in tools_used) or "none"
        tool_context = (
            f"Yield assessment: new_facts={yd.new_facts or 'none'} | "
            f"empty_results={yd.had_empty_results} | recommendation={yd.recommendation}\n"
            f"Tools used: {coverage}\n"
            f"Tools NOT yet used: {unused}\n"
            f"Accumulated knowledge: {knowledge}"
        )
        result = llm_call(
            prompt=LOOP_HANDOFF_PROMPT.format(
                iteration=iteration + 1,
                step_history=step_history or "(none yet)",
                tool_context=tool_context,
            ),
            response_model=_LoopHandoffSchema,
            fallback=_LoopHandoffSchema(),
            model=config.STRUCTURED_MODEL,
        )
        h = result.value

        rationale = h.rationale or (
            f"Investigating: {h.gaps[0][:60] if h.gaps else h.next_focus[:60] or 'further detail needed'}"
        )
        action: Literal["finish", "continue"] = "finish" if h.action == "finish" else "continue"

        # Pipeline enforcement: override premature "finish" when required steps are missing.
        # Only trigger when there is substantive evidence worth synthesizing (2+ info tools
        # or a single tool with meaningful content). This avoids forcing synthesize when
        # recall returned nothing and there's nothing to structure.
        pipeline_tools = (ToolName.SYNTHESIZE, ToolName.INTEGRATE_KNOWLEDGE)
        missing_pipeline = [t for t in pipeline_tools if t not in tools_used]
        info_tools = {ToolName.RECALL_MEMORY, ToolName.WEB_SEARCH, ToolName.WEB_EXTRACT}
        info_steps = [s for s in self._step_log if s.tool in info_tools]
        has_substantive = len(info_steps) >= 2 or (
            len(info_steps) == 1 and len(info_steps[0].brief) > 300
        )
        has_gathered = bool(info_steps)
        if action == "finish" and missing_pipeline and has_gathered and has_substantive:
            next_tool = missing_pipeline[0]
            action = "continue"
            rationale = f"Pipeline incomplete — {next_tool} not yet used"
            guidance_map = {
                ToolName.SYNTHESIZE: "synthesize structures the evidence gathered so far.",
                ToolName.INTEGRATE_KNOWLEDGE: "integrate_knowledge persists what was learned.",
            }
            h.guidance = guidance_map.get(next_tool, f"{next_tool} completes the pipeline.")
            h.gaps = [f"{next_tool} not yet run"]
            log.info("Handoff override: finish→continue (missing %s)", ", ".join(missing_pipeline))

        handoff = _LoopHandoffSchema(
            action=action,
            next_focus=h.next_focus,
            established=h.established,
            gaps=h.gaps,
            rationale=rationale,
            guidance=h.guidance,
        )
        log.info(
            "Handoff step %d: action=%s gaps=%s rationale=%s%s",
            iteration + 1,
            handoff.action,
            handoff.gaps,
            handoff.rationale[:80],
            f" | guidance={handoff.guidance[:60]!r}" if handoff.guidance else "",
        )
        return handoff

    def _update_knowledge(self, iteration: int) -> None:
        """Accumulate verified findings into a persistent knowledge block.

        Runs only when recent steps include information-gathering tools (web_search,
        recall_memory, synthesize). Skips when only quorum_critique or integrate_knowledge
        ran — those don't produce new facts to accumulate.
        """
        if not self._step_log:
            return
        info_tools = {ToolName.WEB_SEARCH, ToolName.RECALL_MEMORY, ToolName.SYNTHESIZE}
        recent = self._step_log[-3:]
        if not any(s.tool in info_tools for s in recent):
            return
        finding_parts: list[str] = []
        for s in recent:
            line = f"- [{s.tool}] {s.summary}"
            if s.brief and s.tool in info_tools:
                line += f"\n  Excerpt: {s.brief[:200]}"
            finding_parts.append(line)
        kb_result = llm_call(
            prompt=KNOWLEDGE_UPDATE_PROMPT.format(
                current_block=self._knowledge_block[:800] or "(empty)",
                new_findings="\n".join(finding_parts),
            ),
            response_model=_KnowledgeBlockSchema,
            fallback=_KnowledgeBlockSchema(block=self._knowledge_block),
            model=config.STRUCTURED_MODEL,
        )
        updated = kb_result.value.block
        if updated.strip():
            self._knowledge_block = strip_markdown(updated)
            log.info(
                "Knowledge block updated: %d chars — %s",
                len(self._knowledge_block),
                self._knowledge_block[:80].replace("\n", " "),
            )

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
        if len(llm_messages) <= _COMPRESS_THRESHOLD:
            return llm_messages

        preserve_head = llm_messages[:2]
        compress_section = llm_messages[2:-_COMPRESS_KEEP_TAIL]
        preserve_tail = llm_messages[-_COMPRESS_KEEP_TAIL:]

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
            model=config.STRUCTURED_MODEL,
        )
        summary = comp_result.value.render()
        if len(summary) < 50:
            log.warning(
                "Context compression produced too-short summary (%d chars), keeping original",
                len(summary),
            )
            return llm_messages
        log.info(
            "Context compressed at iteration %d: %d messages → 1 state summary (%d chars)",
            iteration + 1,
            len(compress_section),
            len(summary),
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
                "Iteration %d — %d steps: %s",
                iteration + 1,
                len(self._step_log),
                " | ".join(f"{s.tool}({s.query[:25]!r})" for s in self._step_log),
            )
            tool_counts = Counter(s.tool for s in self._step_log)
            coverage = " | ".join(f"{t}:{tool_counts[t]}" for t in ToolName if tool_counts[t])
            unused = [t for t in ToolName if not tool_counts[t]]
            sections.append(
                "## Tool Execution Log\n"
                + "\n".join(lines)
                + f"\nCoverage: {coverage or 'none'}"
                + (f" | Unused: {', '.join(unused)}" if unused else "")
            )

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
                log.info("Skipping duplicate call: %s", tc.name)
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
        t_tools = time.perf_counter()
        results: list[tuple[ParsedToolCall, str]] = []
        for tc in calls:
            step_n = len(self._step_log) + 1
            key_arg = str(
                tc.args.get("query")
                or tc.args.get("focus")
                or tc.args.get("topic")
                or tc.args.get("text")
                or ""
            )[:120]
            log.info("Step %d | %s ← %r", step_n, tc.name, key_arg)
            result_text = dispatch_tool(tc.name, tc.args, ctx)
            tool_elapsed = time.perf_counter() - t_tools
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
                "Step %d | %s → %s (%.1fs, %d chars)",
                step_n,
                tc.name,
                summary[:100],
                tool_elapsed,
                len(result_text),
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
        max_tokens: int = config.LLM_MAX_TOKENS,
        temperature: float = config.AGENT_TEMPERATURE,
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
        for iteration in range(_HARD_CEILING):
            t_iter = time.perf_counter()
            recent_calls: set[tuple[str, str]] = set()
            yield AgentEvent(type=THINKING, iteration=iteration, detail="Analyzing...")
            log.debug("Agentic loop step %d", iteration + 1)

            llm_messages = self._compress_messages(llm_messages, iteration)
            tools = self._prepare_iteration(iteration, llm_messages)
            ctx_chars = sum(len(str(m.get("content", ""))) for m in llm_messages)
            log.info(
                "Iteration %d context: %d messages, ~%dk chars",
                iteration + 1,
                len(llm_messages),
                ctx_chars // 1000,
            )
            finishing = (
                self._current_handoff is not None and self._current_handoff.action == "finish"
            )
            try:
                completion = default_provider.chat_completion(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=llm_messages,
                    tools=tools if not finishing else [],
                    tool_choice="none" if finishing else "auto",
                )
            except RuntimeError:
                log.exception("LLM failed at step %d", iteration)
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
                "Iteration %d LLM: %.1fs | in=%d out=%d tools=%d text=%d chars",
                iteration + 1,
                llm_elapsed,
                completion.input_tokens,
                completion.output_tokens,
                len(tool_calls),
                len(completion.text),
            )
            if not tool_calls:
                if self._step_log:
                    log.info(
                        "Loop finished at step %d (%.1fs total) | %s",
                        iteration + 1,
                        time.perf_counter() - t_loop_start,
                        " → ".join(f"{s.tool}({s.query[:25]!r})" for s in self._step_log),
                    )
                clean = strip_markdown(completion.text)
                self._last_assistant_msg = clean
                yield StreamChunk(clean)
                return

            assistant_msg: dict[str, object] = {
                "role": ChatRole.ASSISTANT,
                "content": completion.text or "",
            }
            raw_tc = get_raw_tool_calls(completion.raw)
            if raw_tc:
                assistant_msg["tool_calls"] = raw_tc
            llm_messages.append(assistant_msg)
            calls_to_run = self._dedup_tool_calls(tool_calls, recent_calls)
            if not calls_to_run:
                continue

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
            log.info(
                "Tool dispatch: %d calls in %.1fs",
                len(calls_to_run),
                time.perf_counter() - t_dispatch,
            )
            self._fold_prior_tool_results(llm_messages, batch_start)

            # Run quorum critique when tool results contain substantive content.
            # Skip for: trivial results (empty recalls, errors), synthesize output
            # (already a distilled evaluation), or integrate_knowledge (action, not info).
            batch_tools = {tc.name for tc in calls_to_run}
            skip_critique_tools = {ToolName.SYNTHESIZE, ToolName.INTEGRATE_KNOWLEDGE}
            tool_content = "\n".join(
                str(m.get("content", ""))[:600]
                for m in llm_messages[batch_start:]
                if m.get("role") == ChatRole.TOOL
            )
            substantive = (
                len(tool_content) > 200
                and "no relevant" not in tool_content.lower()
                and not batch_tools.issubset(skip_critique_tools)
            )
            if substantive:
                yield AgentEvent(
                    type=REVIEWING, iteration=iteration, detail="Cross-checking evidence..."
                )
                original_task = next(
                    (
                        str(m.get("content", ""))[:200]
                        for m in conv
                        if m.get("role") == ChatRole.USER
                    ),
                    "",
                )
                critique = quorum_critique(
                    f"Original task: {original_task}\n\n"
                    f"Recent tool outputs:\n{tool_content}\n\n"
                    "What's missing, contradictory, or assumed without evidence? "
                    "Stay focused on the original task."
                )
                if critique:
                    llm_messages.append(
                        {
                            "role": ChatRole.SYSTEM,
                            "content": f"[Reasoning advisor]: {critique}",
                        }
                    )
                    step_n = len(self._step_log) + 1
                    self._step_log.append(
                        _LoopStep(
                            n=step_n,
                            tool=_QUORUM_CRITIQUE,
                            query="critique",
                            summary=critique[:150],
                            brief=critique[:400],
                        )
                    )
                    log.info("Step %d | quorum_critique: %s", step_n, critique[:100])
            else:
                if batch_tools.issubset(skip_critique_tools):
                    log.info(
                        "Skipping quorum_critique: action tools only (%s)", ", ".join(batch_tools)
                    )
                else:
                    log.info(
                        "Skipping quorum_critique: trivial output (%d chars)", len(tool_content)
                    )

            t_meta = time.perf_counter()
            self._update_knowledge(iteration)
            self._current_handoff = self._generate_handoff(llm_messages, iteration)
            log.info(
                "Meta-calls (knowledge+handoff): %.1fs | iteration %d total: %.1fs",
                time.perf_counter() - t_meta,
                iteration + 1,
                time.perf_counter() - t_iter,
            )
            if self._current_handoff.action == "finish":
                log.info("Handoff action=finish — injecting wrap-up signal")
                h = self._current_handoff
                wrap_parts = ["[Loop coordinator]: Research complete."]
                if self._knowledge_block:
                    wrap_parts.append(f"Verified knowledge:\n{self._knowledge_block}")
                if h.established:
                    wrap_parts.append("Confirmed: " + "; ".join(h.established))
                if h.guidance:
                    wrap_parts.append(f"Guidance: {h.guidance}")
                wrap_parts.append("A clear, grounded response from your findings.")
                llm_messages.append(
                    {
                        "role": ChatRole.SYSTEM,
                        "content": "\n\n".join(wrap_parts),
                    }
                )

        log.warning("Agentic loop: hard ceiling reached (%d steps)", _HARD_CEILING)
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
        window = all_beliefs[:BELIEF_PROMPT_WINDOW]
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
        log.debug("Route: category=%s n=%d", decision.category, decision.n_results)
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
                user_message, top_k=max(2, min(decision.n_results, 6))
            )

        episode_context = [
            format_episode_line(
                created_at=ep.created_at,
                summary=ep.summary,
                content=ep.content,
                content_limit=config.EPISODE_CONTENT_LIMIT,
            )
            for ep in selected
        ]
        log.info(
            "Retrieval: category=%s episodes=%d semantic=%d",
            decision.category,
            len(selected),
            len(semantic_context),
        )
        return [*episode_context, *semantic_context]

    async def _fetch_episodes_for_category(
        self, decision: RoutingDecision, user_message: str
    ) -> list[EpisodeNode]:
        """Dispatch episode fetching based on query category."""
        cat = decision.category
        over_fetch = decision.n_results * config.RETRIEVAL_OVER_FETCH_FACTOR

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
        topic_hits = await self._graph.find_topic_related_episodes(
            user_message, limit=max(2, over_fetch // 2)
        )
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
        """
        try:
            ess = self._classify_ess(user_message)
        except Exception:
            log.exception("ESS classification failed — using fallback")
            ess = classifier_exception_fallback(user_message)
        self.last_ess = ess
        log.info(
            "ESS: score=%.2f type=%s update=%s topics=%s",
            ess.score,
            ess.reasoning_type,
            ess.belief_update_recommended,
            list(ess.topics),
        )

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
            log.exception("Boundary detection failed")

        episode_uid = self._store_episode(
            user_message, agent_response, ess, segment_id, segment_label
        )

        if closed_segment_id:
            try:
                self._run_async(maybe_consolidate_segment(self._graph, closed_segment_id))
            except Exception:
                log.warning("Consolidation failed for segment %s", closed_segment_id, exc_info=True)

        if episode_uid:
            if ess.belief_update_recommended:
                self._assess_provenance(list(ess.topics), episode_uid, user_message, ess)

            content = (
                f"User: {user_message}\nAssistant: {agent_response}\n"
                f"ESS: {ess.score:.2f} ({ess.reasoning_type})"
                if agent_response
                else f"Content: {user_message}\nESS: {ess.score:.2f} ({ess.reasoning_type})"
            )
            categories = (SemanticCategory.KNOWLEDGE,) if not agent_response else ()
            self._semantic_worker.enqueue(episode_uid, content, categories)

        ctx = self._make_tool_context([])
        run_forgetting(ctx)

    def _classify_ess(self, user_message: str) -> ESSResult:
        """Run ESS classification."""
        result = classify(
            user_message=user_message,
            model=self.ess_model,
        )
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
        try:
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
        except Exception:
            log.exception("Episode storage failed — continuing without episode")
            return ""
        log.info(
            "Episode stored: uid=%s segment=%s topics=%s",
            stored.episode_uid[:8],
            segment_id[:8] if segment_id else "none",
            list(ess.topics),
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
        try:
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
        except Exception:
            log.exception("Provenance assessment failed")
