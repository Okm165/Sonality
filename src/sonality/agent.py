"""Sonality agent: stateless, graph-backed personality with LLM-driven tool use.

Each request starts from zero in-memory state. Identity (personality snapshot +
beliefs) is loaded from Neo4j per request. Conversation context is managed by
the caller (chat client / API). The agent uses a unified agentic loop where
all cognitive stages — recall, research, synthesis, and knowledge integration —
are tools the LLM invokes autonomously.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
import time
from collections.abc import Callable, Coroutine, Generator, Iterator
from concurrent.futures import Future
from uuid import uuid4

import structlog

from shared.llm.caller import compose_guarded
from shared.llm.parse import (
    ParsedToolCall,
    extract_tool_calls,
    get_raw_tool_calls,
    strip_markdown,
)
from shared.types import ChatRole

from . import config
from .automaton import (
    TERMINAL_PHASES,
    ActingContext,
    LoopState,
    LoopStep,
    MemoryUpdate,
    build_scaffolding,
    build_step_context,
    dedup_tool_calls,
    summarize_for_step_log,
    synthesis_prompt,
)
from .bookkeeping import (
    BookkeepingItem,
    classify_ess,
    enqueue_bookkeeping,
    post_ingest,
    process_bookkeeping,
)
from .caller import StreamChunk, llm_call
from .caller import provider as default_provider
from .ess import ESS_FALLBACK, ESSResult
from .memory import (
    DatabaseConnections,
    DualEpisodeStore,
    MemoryGraph,
    SemanticFeatureExtractor,
    retrieve,
)
from .memory.graph import (
    BeliefNode,
    PersonalitySnapshot,
    format_beliefs_for_prompt_from_nodes,
)
from .progress import AgentEvent, noop_progress
from .prompts import (
    CONSOLIDATION_INSTRUCTIONS,
    build_ingest_system,
    build_system_prompt,
)
from .request_identity import (
    IdentityBundle,
    get_request_identity,
    reset_request_identity,
    set_request_identity,
)
from .schema import EventType, Phase, ToolName
from .token_budget import (
    SUMMARIZE_THRESHOLD,
    message_tokens_budget_for_system,
    summarize_and_trim,
)
from .tools import ToolContext, dispatch_tool, get_definitions, tool_label
from .tools.reflect import rank_beliefs_by_similarity
from .web_client import ResearchClient

log = structlog.get_logger(__name__)


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

            raise ServiceUnavailableError(
                "Neo4j + Qdrant required and failed to initialize"
            ) from exc

        self._web_client = (
            ResearchClient(config.settings.fathom_url) if config.settings.fathom_url else None
        )

        self._bookkeep_queue: asyncio.Queue[BookkeepingItem] = asyncio.Queue(maxsize=64)
        asyncio.run_coroutine_threadsafe(self._bookkeep_worker(), self._loop)

        log.info("agent_ready", model=self.model, web="enabled" if self._web_client else "disabled")

    async def _init_runtime(self) -> None:
        """Initialize all async resources in a single coroutine.

        One ``_run_async`` call in ``__init__`` covers: embedder probe, DB
        connections, dual store restore, and semantic feature worker creation.
        """
        self._embedder = config.settings.make_embedder()
        db = await DatabaseConnections.create(embedding_dims=self._embedder.dims)
        self._db = db
        self._graph = MemoryGraph(db.neo4j_driver)
        self._dual_store = DualEpisodeStore(self._graph, db.qdrant, self._embedder)
        last_uid = await self._graph.get_last_episode_uid()
        if last_uid:
            self._dual_store.restore_last_episode(last_uid)

        self._semantic_worker = SemanticFeatureExtractor(db.qdrant, self._embedder)

    def _run_async[T](self, coro: Coroutine[object, object, T]) -> T:
        """Bridge sync → async: submit *coro* to the agent's event loop, block until done."""
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return fut.result(timeout=config.settings.async_timeout)
        except TimeoutError:
            fut.cancel()
            raise

    def shutdown(self) -> None:
        """Stop background workers and close database connections.

        Drains the bookkeeping queue (up to 10s) before closing DB connections
        to avoid losing episodes that were enqueued but not yet persisted.
        """

        async def _drain_and_close() -> None:
            try:
                await asyncio.wait_for(self._bookkeep_queue.join(), timeout=10.0)
            except TimeoutError:
                log.warning("bookkeep_drain_timeout", pending=self._bookkeep_queue.qsize())
            if self._web_client:
                await self._web_client.close()
            await self._db.close()

        self._run_async(_drain_and_close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)
        log.info("agent_shutdown")

    def check_dependencies(self) -> dict[str, str]:
        """Verify persistent stores are still reachable."""

        async def _check() -> dict[str, str]:
            status: dict[str, str] = {}
            try:
                from shared.neo4j import ping as neo4j_ping

                await neo4j_ping(self._db.neo4j_driver, config.settings.neo4j_database)
                status["neo4j"] = "ok"
            except Exception as exc:
                status["neo4j"] = f"error: {exc}"

            try:
                await self._db.qdrant.get_collections()
                status["qdrant"] = "ok"
            except Exception as exc:
                status["qdrant"] = f"error: {exc}"
            return status

        return self._run_async(_check())

    # --- Public API ---

    @staticmethod
    def _resolve_defaults(
        max_tokens: int | None,
        temperature: float | None,
        on_progress: Callable[[AgentEvent], None] | None,
    ) -> tuple[int, float, Callable[[AgentEvent], None]]:
        return (
            max_tokens if max_tokens is not None else config.settings.llm_max_tokens,
            temperature if temperature is not None else config.settings.agent_temperature,
            on_progress or noop_progress,
        )

    def respond(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_progress: Callable[[AgentEvent], None] | None = None,
    ) -> str:
        """Generate a response given the full conversation history from the caller."""
        mt, temp, prog = self._resolve_defaults(max_tokens, temperature, on_progress)
        return self._respond_inner(messages, max_tokens=mt, temperature=temp, on_progress=prog)

    def respond_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        on_progress: Callable[[AgentEvent], None] | None = None,
    ) -> Iterator[StreamChunk | AgentEvent]:
        """Streaming response variant. Yields StreamChunk for text and AgentEvent for progress."""
        mt, temp, prog = self._resolve_defaults(max_tokens, temperature, on_progress)
        yield from self._respond_stream_inner(
            messages, max_tokens=mt, temperature=temp, on_progress=prog
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
        t0 = time.perf_counter()
        id_token = None
        try:
            bundle = self._run_async(self._fetch_identity_bundle())
            id_token = set_request_identity(bundle)
            ess = self._classify_ess(text)
            self.last_ess = ess
            log.info(
                "ingest",
                ess=f"{ess.score:.2f}",
                signals=ess.signals.summary_str(),
                topics=list(ess.topics),
                text=repr(text[:80]),
            )

            if not ess.belief_update_recommended:
                log.info("ingest_skip", score=round(ess.score, 3))
                return ess

            self._ingest_agentic(text, bundle, ess, topic_override=topic_override)
            log.info(
                "ingest_done",
                elapsed_s=round(time.perf_counter() - t0, 1),
                ess=f"{ess.score:.2f}",
                signals=ess.signals.summary_str(),
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

        The model uses tools (recall_memory, web_research, integrate_knowledge)
        to deeply process the incoming information before it's stored.
        """
        system_prompt = build_ingest_system(
            snapshot_text=bundle.snapshot_text,
            beliefs_text=bundle.beliefs_prompt_text,
        )
        topic_hint = f"\nCategory hint: {topic_override}" if topic_override else ""
        user_content = (
            f"{text}\n\n"
            f"[ESS: score={ess.score:.2f}, credibility=[{ess.signals.summary_str()}], "
            f"topics={', '.join(ess.topics)}{topic_hint}]"
        )
        conv = [{"role": ChatRole.USER, "content": user_content}]

        state = LoopState(
            user_message=user_content,
            run_id=uuid4().hex,
            loop_start_time=time.perf_counter(),
        )
        self._pre_seed_memory(user_content, state)
        for _item in self._run_agentic_loop(
            system_prompt, conv, state, temperature=config.settings.ingest_temperature
        ):
            pass

        if state.phase == Phase.FAILED:
            log.error(
                "ingest_loop_failed",
                topics=list(ess.topics),
                steps=len(state.step_log),
                tools_used=state.tool_history,
            )
            return

        agent_response = state.last_assistant_msg
        if ToolName.INTEGRATE_KNOWLEDGE not in state.tool_history:
            log.warning(
                "ingest_not_integrated",
                topics=list(ess.topics),
                steps=len(state.step_log),
                tools_used=state.tool_history,
            )

        self._flush_ltm_to_episode(state)

        try:
            self._run_async(
                post_ingest(
                    text,
                    agent_response,
                    ess,
                    graph=self._graph,
                    dual_store=self._dual_store,
                    semantic_worker=self._semantic_worker,
                    qdrant=self._db.qdrant,
                    embedder=self._embedder,
                    ltm_content=state.long_term_memory,
                )
            )
        except Exception:
            log.error("post_ingest_failed", exc_info=True)

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

        async def _h() -> tuple[int, int]:
            beliefs, snapshot = await asyncio.gather(
                self._graph.get_all_beliefs(),
                self._graph.get_personality_snapshot(),
            )
            return len(beliefs), snapshot.version

        return self._run_async(_h())

    # --- Response pipeline ---

    def _prepare_context(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        on_progress: Callable[[AgentEvent], None],
    ) -> tuple[IdentityBundle, str, str, list[dict[str, str]]]:
        """Load identity, build system prompt, trim messages.

        Returns the IdentityBundle (not the ContextVar token). Callers are
        responsible for calling ``set_request_identity`` inside their own
        try/finally to guarantee cleanup.
        """
        bundle = self._run_async(self._fetch_identity_bundle())
        on_progress(AgentEvent(type=EventType.CONTEXT_BUILD, detail="Loading identity..."))

        user_message = next(
            (m.get("content", "") for m in reversed(messages) if m.get("role") == ChatRole.USER),
            "",
        )
        log.debug(
            "context", snapshot_chars=len(bundle.snapshot_text), beliefs=len(bundle.all_beliefs)
        )
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
                    type=EventType.SUMMARIZING,
                    detail=f"Compressed {msg_count_before} messages to {len(conv)}",
                )
            )
        return bundle, user_message, system_prompt, conv

    def _respond_inner(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        on_progress: Callable[[AgentEvent], None],
    ) -> str:
        """Execute full respond pipeline: prepare context → agentic loop → enqueue bookkeeping."""
        t0 = time.perf_counter()
        id_token = None
        try:
            bundle, user_message, system_prompt, conv = self._prepare_context(
                messages,
                max_tokens,
                on_progress,
            )
            id_token = set_request_identity(bundle)
            state = LoopState(user_message=user_message, run_id=uuid4().hex, loop_start_time=t0)
            self._pre_seed_memory(user_message, state)
            for item in self._run_agentic_loop(
                system_prompt, conv, state, max_tokens=max_tokens, temperature=temperature
            ):
                if isinstance(item, AgentEvent):
                    on_progress(item)
            self._flush_ltm_to_episode(state)
            self.last_ess = ess = self._classify_ess(user_message)
            enqueue_bookkeeping(
                self._bookkeep_queue,
                self._loop,
                user_message,
                state.last_assistant_msg,
                ess,
                state.long_term_memory,
            )
            return state.last_assistant_msg
        finally:
            if id_token is not None:
                with contextlib.suppress(ValueError):
                    reset_request_identity(id_token)

    def _respond_stream_inner(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        on_progress: Callable[[AgentEvent], None],
    ) -> Iterator[StreamChunk | AgentEvent]:
        """Agentic streaming: tool calls yield AgentEvents, final text yields StreamChunks.

        Flush + bookkeeping run in ``finally`` so they execute even if the
        consumer disconnects mid-stream (GeneratorExit during a yield).
        """
        t0 = time.perf_counter()
        id_token = None
        state: LoopState | None = None
        user_message = ""
        try:
            bundle, user_message, system_prompt, conv = self._prepare_context(
                messages,
                max_tokens,
                on_progress,
            )
            id_token = set_request_identity(bundle)
            state = LoopState(
                user_message=user_message,
                run_id=uuid4().hex,
                loop_start_time=t0,
            )
            self._pre_seed_memory(user_message, state)
            yield from self._run_agentic_loop(
                system_prompt,
                conv,
                state,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            elapsed = time.perf_counter() - t0
            yield AgentEvent(
                type=EventType.DONE,
                detail=json.dumps(
                    {
                        "run_id": state.run_id,
                        "tool_count": len(state.tool_history),
                        "elapsed": round(elapsed, 1),
                    }
                ),
            )
        finally:
            if state is not None:
                self._flush_ltm_to_episode(state)
                self.last_ess = ess = self._classify_ess(user_message)
                enqueue_bookkeeping(
                    self._bookkeep_queue,
                    self._loop,
                    user_message,
                    state.last_assistant_msg,
                    ess,
                    state.long_term_memory,
                )
            if id_token is not None:
                with contextlib.suppress(ValueError):
                    reset_request_identity(id_token)

    # --- Tool context and loop status ---

    def _make_tool_context(
        self,
        state: LoopState,
        progress: Callable[[str], None] | None = None,
    ) -> ToolContext:
        """Create ToolContext bridging agent state to the tools package."""
        return ToolContext(
            run_async=self._run_async,
            web_client=self._web_client,
            graph=self._graph,
            dual_store=self._dual_store,
            qdrant=self._db.qdrant,
            embedder=self._embedder,
            identity=get_request_identity(),
            retrieve=lambda q: self._run_async(
                retrieve(
                    q,
                    graph=self._graph,
                    dual_store=self._dual_store,
                    qdrant=self._db.qdrant,
                    embedder=self._embedder,
                )
            ),
            research_transcript=lambda: (
                f"LTM:\n{state.long_term_memory}\n\nSTM:\n{state.short_term_memory}"
            ),
            short_term_memory=state.short_term_memory,
            progress=progress or (lambda _: None),
        )

    # --- Memory seeding and flushing ---

    def _pre_seed_memory(self, user_message: str, state: LoopState) -> None:
        """Seed LTM from episodic store and beliefs from graph before the loop starts.

        Failures are isolated — a transient DB error degrades to empty LTM/beliefs
        rather than killing the entire response.
        """
        try:
            episodes = self._run_async(
                retrieve(
                    user_message,
                    graph=self._graph,
                    dual_store=self._dual_store,
                    qdrant=self._db.qdrant,
                    embedder=self._embedder,
                )
            )
            if episodes:
                state.long_term_memory = "## Prior Knowledge\n" + "\n".join(
                    f"- {ep[:200]}" for ep in episodes[:5]
                )
        except Exception:
            log.warning("pre_seed_retrieval_failed", exc_info=True)
        try:
            beliefs = self._rank_beliefs(user_message)
            if beliefs:
                state.relevant_beliefs = beliefs
        except Exception:
            log.warning("pre_seed_beliefs_failed", exc_info=True)

    def _flush_ltm_to_episode(self, state: LoopState) -> None:
        """Enqueue LTM research findings for storage and bookkeeping.

        The LTM content gets its own ESS classification. Bookkeeping
        handles both storage (as an episode) and downstream processing
        (provenance, semantic features, knowledge extraction, forgetting).
        """
        if len(state.long_term_memory) < 50:
            return
        try:
            ltm_text = state.long_term_memory[: config.settings.episode_content_limit * 10]
            ltm_ess = self._classify_ess(ltm_text[:2000])
            enqueue_bookkeeping(
                self._bookkeep_queue,
                self._loop,
                state.user_message[:500],
                ltm_text,
                ltm_ess,
                ltm_text,
            )
        except Exception:
            log.warning("ltm_flush_failed", exc_info=True)

    def _rank_beliefs(self, context: str) -> str:
        """Return formatted beliefs relevant to the given context.

        Isolated from embedder failures — returns empty string rather than
        aborting the request, since beliefs are supplementary context.
        """
        identity = get_request_identity()
        if not identity or not identity.all_beliefs:
            return ""
        try:
            relevant = rank_beliefs_by_similarity(
                context, list(identity.all_beliefs), self._embedder, max_results=8
            )
            return format_beliefs_for_prompt_from_nodes(relevant) if relevant else ""
        except Exception:
            log.warning("belief_ranking_failed", exc_info=True)
            return format_beliefs_for_prompt_from_nodes(list(identity.all_beliefs[:8]))

    # --- Agentic tool-calling loop ---

    def _dispatch_tools(
        self,
        calls: list[ParsedToolCall],
        state: LoopState,
    ) -> Iterator[AgentEvent | tuple[ParsedToolCall, str]]:
        """Execute tool calls, yielding tool progress events and (call, result) tuples.

        All tools can emit progress via ctx.progress(). For sync tools, events
        are collected in a list and yielded after the tool returns. For web_research,
        events stream live from Fathom's SSE via a thread-safe queue.
        """
        pending_progress: list[str] = []

        def _progress_sink(detail: str) -> None:
            pending_progress.append(detail)

        ctx = self._make_tool_context(state, progress=_progress_sink)
        for tc in calls:
            step_n = len(state.step_log) + 1
            key_arg = tool_label(tc.name, tc.args)
            log.info("tool_start", step=step_n, tool=tc.name, arg=key_arg)
            t_tool = time.perf_counter()
            pending_progress.clear()

            if tc.name == ToolName.WEB_RESEARCH and self._web_client:
                log.info("dispatch_streaming_research", tool=tc.name)
                result_text = yield from self._dispatch_streaming_research(
                    tc,
                    short_term_memory=state.short_term_memory,
                )
            else:
                result_text = dispatch_tool(tc.name, tc.args, ctx)
                for detail in pending_progress:
                    yield AgentEvent(type=EventType.TOOL_PROGRESS, detail=detail, tool_name=tc.name)

            tool_elapsed = time.perf_counter() - t_tool
            state.tool_history.append(tc.name)
            summary = summarize_for_step_log(result_text)
            log.info(
                "tool_done",
                step=step_n,
                tool=tc.name,
                summary=summary[:100],
                elapsed_s=round(tool_elapsed, 1),
                chars=len(result_text),
            )
            state.step_log.append(
                LoopStep(
                    step_index=step_n,
                    tool=tc.name,
                    query=key_arg,
                    summary=summary,
                    raw_output=result_text[:2000],
                )
            )
            yield (tc, result_text)

    def _dispatch_streaming_research(
        self,
        tc: ParsedToolCall,
        *,
        short_term_memory: str = "",
    ) -> Generator[AgentEvent, None, str]:
        """Run web_research with live SSE progress streaming via a thread-safe queue.

        Yields AgentEvent(TOOL_PROGRESS) as Fathom streams, then returns
        the final result text (via generator return value consumed by yield from).
        """
        import queue as thread_queue

        from .tools.web import format_facts, merge_facts, parse_research_args

        ra = parse_research_args(tc.args)
        if not ra.goal:
            return "Error: no research goal provided."

        goal, depth, seeds = ra.goal, ra.depth, ra.seeds
        max_pages, n = ra.max_pages, ra.pages_per_round
        if short_term_memory:
            goal = f"{goal} [Context: {short_term_memory[:200]}]"
        web_client = self._web_client
        assert web_client is not None  # Caller verifies _web_client is truthy

        progress_q: thread_queue.Queue[object] = thread_queue.Queue()
        log.info("streaming_research_start", goal=goal[:60], depth=depth)
        yield AgentEvent(
            type=EventType.TOOL_PROGRESS,
            detail=f"Starting research: {goal[:50]}",
            tool_name=tc.name,
        )

        async def _stream_research():
            session = await web_client.start_research(
                goal,
                depth=depth,
                seeds=seeds,
                max_pages=max_pages,
                n=n,
            )
            log.info("research_session_created", session_id=session.session_id[:12])
            event_count = 0
            async for progress in web_client.stream_research(session.session_id):
                event_count += 1
                progress_q.put(progress)
                if progress.event in ("complete", "error"):
                    break
            log.info("research_stream_done", events=event_count)
            result = await web_client.get_research_result(session.session_id)
            progress_q.put(None)
            return result

        future: Future[object] = asyncio.run_coroutine_threadsafe(
            _stream_research(),
            self._loop,
        )

        from .web_client import ResearchFact, ResearchProgress, ResearchResult

        accumulated_facts: list[ResearchFact] = []
        yielded_count = 0
        stream_deadline = time.perf_counter() + config.settings.async_timeout

        def _handle(item: object) -> AgentEvent | None:
            if isinstance(item, ResearchProgress):
                if item.partial_facts:
                    accumulated_facts.extend(item.partial_facts)
                nonlocal yielded_count
                yielded_count += 1
                return AgentEvent(
                    type=EventType.TOOL_PROGRESS,
                    detail=item.detail or item.event,
                    tool_name=tc.name,
                )
            return None

        while not future.done():
            if time.perf_counter() > stream_deadline:
                log.error("streaming_research_timeout")
                future.cancel()
                break
            try:
                item = progress_q.get(timeout=0.3)
            except thread_queue.Empty:
                continue
            if item is None:
                break
            evt = _handle(item)
            if evt:
                yield evt

        while not progress_q.empty():
            evt = _handle(progress_q.get_nowait())
            if evt:
                yield evt

        log.info(
            "streaming_research_events_yielded",
            count=yielded_count,
            accumulated_facts=len(accumulated_facts),
        )

        try:
            result = future.result()
        except Exception as exc:
            log.error("research_failed", error=str(exc), exc_info=True)
            if accumulated_facts:
                formatted = format_facts(tuple(accumulated_facts))
                return (
                    f"[Research failed but recovered {len(accumulated_facts)} facts]\n\n{formatted}"
                )
            return f"Research failed: {exc}"

        pages = result.pages_scraped if isinstance(result, ResearchResult) else 0
        final_facts = merge_facts(
            accumulated_facts, result.facts if isinstance(result, ResearchResult) else ()
        )
        if not final_facts:
            return f"Research completed but found no facts ({pages} pages scraped)."
        formatted = format_facts(final_facts)
        return f"[Research: {len(final_facts)} facts from {pages} pages]\n\n{formatted}"

    def _run_agentic_loop(
        self,
        system_prompt: str,
        conv: list[dict[str, str]],
        state: LoopState,
        *,
        max_tokens: int = config.settings.llm_max_tokens,
        temperature: float = config.settings.agent_temperature,
    ) -> Iterator[AgentEvent | StreamChunk]:
        """2-phase automaton: THINKING ↔ ACTING → COMPLETED | FAILED.

        Scaffolding is built once. Each step dispatches to the current phase handler.
        The hard ceiling forces a nudge (synthesis) rather than terminating silently.
        """
        scaffolding = build_scaffolding(system_prompt, conv)
        state.phase = Phase.THINKING
        while state.phase not in TERMINAL_PHASES:
            if state.iteration >= config.settings.agent_loop_hard_ceiling:
                log.warning("loop_ceiling", iterations=state.iteration)
                state.nudged = True
                state.context_messages = [
                    {"role": ChatRole.SYSTEM, "content": synthesis_prompt(state)}
                ]
            match state.phase:
                case Phase.THINKING:
                    yield from self._phase_thinking(
                        scaffolding,
                        state,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                case Phase.ACTING:
                    yield from self._phase_acting(state)

    def _phase_thinking(
        self,
        scaffolding: list[dict[str, object]],
        state: LoopState,
        *,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[AgentEvent | StreamChunk]:
        """THINKING phase: LLM call → classify response → transition.

        Three response cases:
        - Empty: stall → nudge if max_stalls reached
        - Text only: final answer (or extend if truncated)
        - Tool calls: dedup → transition to ACTING
        """
        yield AgentEvent(type=EventType.THINKING, iteration=state.iteration, detail="Analyzing...")

        inputs = list(state.context_messages)
        if not state.nudged:
            step_ctx = build_step_context(state)
            if step_ctx:
                inputs.append({"role": ChatRole.SYSTEM, "content": step_ctx})

        llm_messages = compose_guarded(
            default_provider,
            scaffolding=scaffolding,
            inputs=inputs,
            model=self.model,
            context_char_limit=config.settings.context_char_limit,
        )

        tools = [] if state.nudged else get_definitions()
        tool_choice = "none" if state.nudged else "auto"
        try:
            completion = default_provider.chat_completion(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=llm_messages,
                tools=tools,
                tool_choice=tool_choice,
            )
        except Exception:
            log.error("llm_failed", iteration=state.iteration, exc_info=True)
            state.last_assistant_msg = "I encountered a processing error. Please try again."
            yield StreamChunk(state.last_assistant_msg)
            state.phase = Phase.FAILED
            return

        tool_calls = extract_tool_calls(completion.raw)

        # Case 1: Empty response — stall
        if not tool_calls and not completion.text.strip():
            state.stall_count += 1
            log.warning("empty_stall", iteration=state.iteration, stall=state.stall_count)
            if state.stall_count >= config.settings.max_stalls:
                if state.nudged:
                    log.error("nudge_failed_empty", stalls=state.stall_count)
                    state.last_assistant_msg = (
                        state.long_term_memory or "I was unable to formulate a response."
                    )
                    state.phase = Phase.FAILED
                    yield StreamChunk(state.last_assistant_msg)
                    return
                log.warning("stall_limit_nudge", stalls=state.stall_count)
                state.nudged = True
                state.context_messages = [
                    {"role": ChatRole.SYSTEM, "content": synthesis_prompt(state)}
                ]
            return

        # Case 2: Text only — final answer or extend
        if not tool_calls:
            if completion.finish_reason == "length" and completion.text:
                state.output_buffer += completion.text
                state.extend_count += 1
                state.stall_count = 0
                if state.extend_count >= config.settings.max_extends:
                    state.last_assistant_msg = state.output_buffer
                    state.phase = Phase.COMPLETED
                else:
                    state.context_messages = [
                        {"role": ChatRole.ASSISTANT, "content": completion.text},
                        {"role": ChatRole.SYSTEM, "content": "Continue from where you left off."},
                    ]
                log.info("extend", count=state.extend_count, buffer_chars=len(state.output_buffer))
                yield StreamChunk(completion.text)
                return

            clean = strip_markdown(completion.text)
            final_text = (state.output_buffer + clean) if state.output_buffer else clean
            state.last_assistant_msg = final_text
            if state.step_log:
                log.debug(
                    "loop_finished",
                    iteration=state.iteration,
                    elapsed_s=round(time.perf_counter() - state.loop_start_time, 1),
                    trace=" → ".join(f"{s.tool}({s.query[:25]!r})" for s in state.step_log),
                )
            state.phase = Phase.COMPLETED
            yield StreamChunk(clean)
            return

        # Case 3: Tool calls — transition to ACTING
        if completion.text and completion.text.strip():
            state.last_assistant_msg = completion.text.strip()

        calls_to_run = dedup_tool_calls(tool_calls, state.recent_calls)
        if not calls_to_run:
            state.stall_count += 1
            log.warning("dedup_stall", iteration=state.iteration, stall=state.stall_count)
            raw_tc = get_raw_tool_calls(completion.raw)
            if raw_tc:
                state.context_messages.append(
                    {
                        "role": ChatRole.ASSISTANT,
                        "content": completion.text or "",
                        "tool_calls": raw_tc,
                    }
                )
                for tc in tool_calls:
                    state.context_messages.append(
                        {
                            "role": ChatRole.TOOL,
                            "tool_call_id": tc.id,
                            "content": "(duplicate — already executed)",
                        }
                    )
            if state.stall_count >= config.settings.max_stalls:
                if state.nudged:
                    log.error("nudge_failed_dedup", stalls=state.stall_count)
                    state.last_assistant_msg = (
                        state.long_term_memory or "I was unable to formulate a response."
                    )
                    state.phase = Phase.FAILED
                    yield StreamChunk(state.last_assistant_msg)
                    return
                log.warning("dedup_stall_nudge", stalls=state.stall_count)
                state.nudged = True
                state.context_messages = [
                    {"role": ChatRole.SYSTEM, "content": synthesis_prompt(state)}
                ]
            return

        state.stall_count = 0
        state.acting_ctx = ActingContext(completion=completion, calls=tuple(calls_to_run))
        state.phase = Phase.ACTING
        state.iteration += 1

    def _phase_acting(
        self,
        state: LoopState,
    ) -> Iterator[AgentEvent | StreamChunk]:
        """ACTING phase: dispatch tools → mandatory consolidation → transition to THINKING.

        Raw tool results never reach the THINKING phase. After dispatch, a consolidation
        LLM call distills results into LTM/STM. Only minimal protocol acks are kept
        in context_messages for the next THINKING call.

        If any step raises, we transition to FAILED to prevent re-entry from
        the outer while-loop (which would re-dispatch the same tools).
        """
        assert state.acting_ctx is not None
        calls = list(state.acting_ctx.calls)
        completion = state.acting_ctx.completion

        for tc in calls:
            yield AgentEvent(
                type=EventType.TOOL_CALL,
                tool_name=tc.name,
                tool_args=json.dumps(tc.args),
                iteration=state.iteration,
            )

        try:
            raw_results: list[tuple[ParsedToolCall, str]] = []
            for item in self._dispatch_tools(calls, state):
                if isinstance(item, AgentEvent):
                    yield item
                    continue
                tc, result_text = item
                raw_results.append((tc, result_text))
                yield AgentEvent(
                    type=EventType.TOOL_RESULT,
                    tool_name=tc.name,
                    tool_result_summary=result_text[:200],
                    iteration=state.iteration,
                    sources_count=0,
                )

            new_observations = "\n".join(
                f"- {s.tool}({s.query[:40]}): {s.raw_output[:800]}"
                for s in state.step_log[-len(raw_results) :]
            )
            if new_observations.strip():
                elapsed_s = int(time.perf_counter() - state.loop_start_time)
                data = (
                    f"## User Question\n{state.user_message[:300]}\n\n"
                    f"## Previous Findings\n{state.long_term_memory or 'None yet.'}\n\n"
                    f"## New Results\n{new_observations}\n\n"
                    f"## Agent Reasoning\n{state.last_assistant_msg or 'N/A'}\n\n"
                    f"## Progress\nIteration {state.iteration} of "
                    f"{config.settings.agent_loop_hard_ceiling}. {elapsed_s}s elapsed."
                )
                consolidation_msgs = compose_guarded(
                    default_provider,
                    scaffolding=[{"role": ChatRole.SYSTEM, "content": CONSOLIDATION_INSTRUCTIONS}],
                    inputs=[{"role": ChatRole.USER, "content": data}],
                    model=config.settings.structured_model,
                    context_char_limit=config.settings.context_char_limit,
                )
                instructions_str = "\n\n".join(
                    str(m.get("content", "")) for m in consolidation_msgs
                )
                result = llm_call(
                    instructions=instructions_str,
                    response_model=MemoryUpdate,
                    fallback=MemoryUpdate(
                        long_term_memory=state.long_term_memory,
                        short_term_memory="",
                    ),
                    model=config.settings.structured_model,
                )
                update = result.value
                if update.long_term_memory.strip():
                    state.long_term_memory = update.long_term_memory
                state.short_term_memory = update.short_term_memory.strip()
                state.relevant_beliefs = self._rank_beliefs(state.long_term_memory)
                log.info(
                    "consolidation",
                    iteration=state.iteration,
                    ltm_chars=len(state.long_term_memory),
                    stm_chars=len(state.short_term_memory),
                )
        except Exception:
            log.error("acting_phase_failed", iteration=state.iteration, exc_info=True)
            state.last_assistant_msg = state.long_term_memory or "I encountered a processing error."
            state.phase = Phase.FAILED
            yield StreamChunk(state.last_assistant_msg)
            return

        executed_ids = {tc.id for tc in calls}
        raw_tc = [tc for tc in get_raw_tool_calls(completion.raw) if tc.get("id") in executed_ids]
        state.context_messages = []
        assistant_msg: dict[str, object] = {
            "role": ChatRole.ASSISTANT,
            "content": completion.text or "",
        }
        if raw_tc:
            assistant_msg["tool_calls"] = raw_tc
        state.context_messages.append(assistant_msg)
        for tc, _ in raw_results:
            state.context_messages.append(
                {
                    "role": ChatRole.TOOL,
                    "tool_call_id": tc.id,
                    "content": "Result processed. See Research Findings.",
                }
            )

        state.acting_ctx = None
        state.phase = Phase.THINKING

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

    # --- Bookkeeping (async queue) ---

    async def _bookkeep_worker(self) -> None:
        """Consume bookkeeping queue sequentially."""
        while True:
            item = await self._bookkeep_queue.get()
            try:
                await process_bookkeeping(
                    item,
                    graph=self._graph,
                    dual_store=self._dual_store,
                    semantic_worker=self._semantic_worker,
                    qdrant=self._db.qdrant,
                    embedder=self._embedder,
                )
            except Exception:
                log.error("bookkeeping_failed", exc_info=True)
            finally:
                self._bookkeep_queue.task_done()

    def _classify_ess(self, user_message: str) -> ESSResult:
        """Run ESS classification with existing topic context."""
        try:
            topics = self._run_async(self._graph.get_topic_names())
            return classify_ess(user_message, ", ".join(topics))
        except Exception:
            log.error("ess_classification_failed", exc_info=True)
            return ESS_FALLBACK
