"""Sonality agent: stateless, graph-backed personality with LLM-driven tool use.

Each request starts from zero in-memory state. Identity (personality snapshot +
beliefs) is loaded from Neo4j per request. Conversation context is managed by
the caller (chat client / API). The agent uses a unified agentic loop where ALL
cognitive stages — memory recall, web research, evidence assessment, reflection,
knowledge storage, and consolidation — are tools the LLM invokes autonomously.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import re
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from concurrent.futures import Future
from contextvars import Token
from typing import Final

from . import config
from .ess import ESSResult, classifier_exception_fallback, classify
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
    SUMMARIZING,
    THINKING,
    TOOL_CALL,
    TOOL_RESULT,
    AgentEvent,
    noop_progress,
)
from .prompts import build_system_prompt
from .provider import (
    ChatResult,
    ParsedToolCall,
    StreamChunk,
    _get_raw_tool_calls,
    default_provider,
    extract_tool_calls,
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
    _SUMMARIZE_THRESHOLD,
    message_tokens_budget_for_system,
    summarize_and_trim,
)
from .tools import ToolContext, dispatch_tool, get_definitions
from .tools.reflect import run_forgetting
from .web import get_client

log = logging.getLogger(__name__)

_HARD_CEILING: Final = 50  # pure circuit breaker, should never be hit

_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_HEADER = re.compile(r"^#{1,4}\s+", re.MULTILINE)


def _strip_markdown(text: str) -> str:
    """Remove markdown bold and headers from LLM output. Plain text only."""
    text = _MD_BOLD.sub(r"\1", text)
    text = _MD_HEADER.sub("", text)
    return text


class SonalityAgent:
    """Stateless personality agent backed by Neo4j (graph) and Qdrant (vectors).

    No in-memory state is carried between requests. Identity and beliefs are
    loaded from persistent stores per request. Conversation context is provided
    by the caller.
    """

    def __init__(
        self,
        model: str = config.MODEL,
        ess_model: str = config.ESS_MODEL,
    ) -> None:
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
        self._last_assistant_msg = ""

        log.info(
            "SonalityAgent ready (model=%s, ess=%s, web=%s)",
            self.model,
            self.ess_model,
            "enabled" if self._web_client else "disabled",
        )

    async def _init_runtime(self) -> None:
        db = await DatabaseConnections.create()
        self._db = db
        self._embedder = Embedder()
        self._graph = MemoryGraph(db.neo4j_driver)
        self._dual_store = DualEpisodeStore(self._graph, db.qdrant, self._embedder)
        last_uid = await self._graph.get_last_episode_uid()
        if last_uid:
            self._dual_store.restore_last_episode(last_uid)

    def _run_async[T](self, coro: Coroutine[object, object, T]) -> T:
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=config.ASYNC_TIMEOUT)
        except TimeoutError:
            future.cancel()
            raise

    def shutdown(self) -> None:
        """Stop background workers and close database connections."""
        self._semantic_worker.stop()
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
        """Non-conversational data ingestion (news, articles, social media)."""
        _t0 = time.perf_counter()
        id_token = None
        try:
            bundle = self._run_async(self._fetch_identity_bundle())
            id_token = set_request_identity(bundle)
            ess = self._classify_ess(text)
            if topic_override and not ess.topics:
                ess = dataclasses.replace(ess, topics=(topic_override.strip().lower(),))
            self.last_ess = ess

            if not ess.belief_update_recommended:
                log.info(
                    "Ingest: update not recommended (type=%s score=%.3f)",
                    ess.reasoning_type,
                    ess.score,
                )
                return ess

            episode_uid = self._store_episode(text, "", ess, "", "")
            if episode_uid:
                self._assess_provenance(list(ess.topics), episode_uid, text, ess)
                content = f"Content: {text}\nESS: {ess.score:.2f} ({ess.reasoning_type})"
                self._semantic_worker.enqueue(episode_uid, content, (SemanticCategory.KNOWLEDGE,))

            ctx = self._make_tool_context([])
            run_forgetting(ctx)
            log.info(
                "Ingest completed in %.1fs | ess=%.2f topics=%s",
                time.perf_counter() - _t0,
                ess.score,
                list(ess.topics),
            )
            return ess
        finally:
            if id_token is not None:
                reset_request_identity(id_token)

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
        if len(conv) < msg_count_before and msg_count_before >= _SUMMARIZE_THRESHOLD:
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
            on_progress(AgentEvent(
                type=DONE,
                detail=json.dumps({
                    "tool_count": len(self._loop_tool_history),
                    "elapsed": round(elapsed, 1),
                    "ess_score": round(ess.score, 2),
                    "reasoning_type": ess.reasoning_type,
                    "topics": list(ess.topics),
                }),
            ))
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
                detail=json.dumps({
                    "tool_count": len(self._loop_tool_history),
                    "elapsed": round(elapsed, 1),
                    "ess_score": round(ess.score, 2),
                    "reasoning_type": ess.reasoning_type,
                    "topics": list(ess.topics),
                }),
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

    def _build_loop_status(self, iteration: int) -> str:
        """Minimal step counter — the LLM decides its own strategy."""
        return f"[Step {iteration + 1}]"

    def _prepare_iteration(
        self,
        iteration: int,
        llm_messages: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """Per-iteration setup: cognitive state injection and tool definitions."""
        llm_messages.append(
            {"role": ChatRole.USER, "content": self._build_loop_status(iteration)}
        )
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
        """Execute tool calls and record results. Returns (call, result_text) pairs."""
        ctx = self._make_tool_context(llm_messages)
        t_tools = time.perf_counter()
        results: list[tuple[ParsedToolCall, str]] = []
        for tc in calls:
            result_text = dispatch_tool(tc.name, tc.args, ctx)
            tool_elapsed = time.perf_counter() - t_tools
            self._loop_tool_history.append(tc.name)
            llm_messages.append(
                {"role": ChatRole.TOOL, "content": result_text, "tool_call_id": tc.id}
            )
            log.info(
                "Tool %s: %.60s → %d chars (%.1fs)",
                tc.name,
                json.dumps(tc.args)[:60],
                len(result_text),
                tool_elapsed,
            )
            results.append((tc, result_text))
        return results

    def _run_agentic_loop(
        self,
        system_prompt: str,
        conv: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> Iterator[AgentEvent | StreamChunk]:
        """Core agentic loop — yields events during reasoning, StreamChunk for the final answer.

        The LLM decides the full workflow via tool calls. Stall detection and
        dedup prevent infinite loops. A generous hard ceiling is a pure circuit breaker.
        """
        self._loop_tool_history.clear()
        llm_messages: list[dict[str, object]] = [
            {"role": ChatRole.SYSTEM, "content": system_prompt},
        ]
        llm_messages.extend(dict(m) for m in conv)
        recent_calls: set[tuple[str, str]] = set()
        completion = ChatResult(text="", input_tokens=0, output_tokens=0, raw={})

        for iteration in range(_HARD_CEILING):
            yield AgentEvent(type=THINKING, iteration=iteration, detail="Analyzing...")
            log.debug("Agentic loop step %d", iteration + 1)

            tools = self._prepare_iteration(iteration, llm_messages)
            try:
                completion = default_provider.chat_completion(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=llm_messages,
                    tools=tools,
                    tool_choice="auto",
                    enable_thinking=False,
                )
            except RuntimeError as exc:
                log.error("LLM failed at step %d: %s", iteration, exc)
                break

            tool_calls = extract_tool_calls(completion.raw)
            if not tool_calls:
                log.debug(
                    "Agentic loop: final response after %d steps (%d chars)",
                    iteration + 1,
                    len(completion.text),
                )
                clean = _strip_markdown(completion.text)
                self._last_assistant_msg = clean
                yield StreamChunk(clean, "")
                return

            assistant_msg: dict[str, object] = {
                "role": ChatRole.ASSISTANT,
                "content": completion.text or "",
            }
            raw_tc = _get_raw_tool_calls(completion.raw)
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

        log.warning("Agentic loop: hard ceiling reached (%d steps)", _HARD_CEILING)
        clean = _strip_markdown(completion.text)
        self._last_assistant_msg = clean
        yield StreamChunk(clean, "")

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
            for ep in episodes[:3]:
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
        """Run ESS classification using the per-request identity bundle."""
        bundle = get_request_identity()
        assert bundle, "Identity bundle must be loaded before ESS classification"
        tracked = ", ".join(b.topic for b in bundle.all_beliefs[:20]) or "(none yet)"
        return classify(
            user_message=user_message,
            snapshot_text=bundle.snapshot_text,
            tracked_topics=tracked,
            model=self.ess_model,
        )

    def _store_episode(
        self,
        user_msg: str,
        agent_resp: str,
        ess: ESSResult,
        segment_id: str,
        segment_label: str,
    ) -> str:
        try:
            stored: StoredEpisode = self._run_async(
                self._dual_store.store(
                    user_message=user_msg,
                    agent_response=agent_resp,
                    summary=ess.summary[:300],
                    topics=list(ess.topics),
                    ess_score=ess.score,
                    reasoning_type=ess.reasoning_type,
                    segment_id=segment_id,
                    segment_label=segment_label,
                )
            )
            log.info(
                "Episode stored: uid=%s segment=%s topics=%s",
                stored.episode_uid[:8],
                segment_id[:8] if segment_id else "none",
                list(ess.topics),
            )
            return stored.episode_uid
        except Exception:
            log.exception("Episode storage failed — continuing without episode")
            return ""

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
