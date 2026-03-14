from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
import time
from collections.abc import Coroutine
from concurrent.futures import Future
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from pydantic import BaseModel

from . import config
from .ess import (
    PROVIDER_CLIENT,
    ESSResult,
    KnowledgeDensity,
    ReasoningType,
    classifier_exception_fallback,
    classify,
)
from .llm.caller import llm_call
from .llm.prompts import (
    BELIEF_DECAY_PROMPT,
    DISAGREEMENT_DETECTION_PROMPT,
    ENTRENCHMENT_DETECTION_PROMPT,
    REFLECTION_GATE_PROMPT,
    TOPIC_CANONICALIZATION_PROMPT,
)
from .memory import (
    BackgroundSummarizer,
    BoundaryDecision,
    ChainOfQueryAgent,
    ConsolidationEngine,
    ContractionAction,
    DatabaseConnections,
    DerivativeChunker,
    DualEpisodeStore,
    EventBoundaryDetector,
    ExternalEmbedder,
    ForgettingEngine,
    MemoryGraph,
    QueryCategory,
    QueryRouter,
    SemanticIngestionWorker,
    SemanticMemoryDecision,
    ShortTermMemory,
    SplitQueryAgent,
    SpongeState,
    StoredEpisode,
    TemporalExpansionDecision,
    UpdateMagnitude,
    assess_belief_evidence,
    assess_health,
    consolidate_knowledge,
    dump_memory_snapshot,
    extract_and_store_knowledge,
    extract_insight,
    prune_stale_knowledge,
    rerank_episodes,
    retrieve_relevant_knowledge,
)
from .memory.context_format import format_episode_line
from .prompts import REFLECTION_PROMPT, build_system_prompt
from .provider import chat_completion

log = logging.getLogger(__name__)

CRITICAL_ESS_DEFAULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "coerced:score",
        "coerced:reasoning_type",
        "coerced:opinion_direction",
    }
)
_UTILITY_USED_DELTA: Final = 0.10
_UTILITY_NOISE_DELTA: Final = -0.05


# Universal per-update magnitude ceiling. The LLM's assess_belief_evidence already
# calibrates evidence_strength per reasoning type and source quality; this only
# clips extreme outliers. Manipulative types are blocked upstream by the
# _NO_UPDATE_REASONING frozenset + manipulative check — they never reach here.
MAX_SINGLE_UPDATE_MAGNITUDE: Final[float] = 0.25


class ReflectionTrigger(StrEnum):
    """Reflection execution mode determined by interaction dynamics."""

    SKIP = "skip"
    PERIODIC = "periodic"
    EVENT_DRIVEN = "event_driven"


class DisagreementVerdict(StrEnum):
    DISAGREEMENT = "DISAGREEMENT"
    NO_DISAGREEMENT = "NO_DISAGREEMENT"


class EntrenchmentStatus(StrEnum):
    ENTRENCHED = "ENTRENCHED"
    NOT_ENTRENCHED = "NOT_ENTRENCHED"


class BeliefDecayAction(StrEnum):
    RETAIN = "RETAIN"
    DECAY = "DECAY"
    FORGET = "FORGET"


class ReflectionGateDecision(StrEnum):
    SKIP = "SKIP"
    PERIODIC = "PERIODIC"
    EVENT_DRIVEN = "EVENT_DRIVEN"


@dataclass(frozen=True, slots=True)
class ReflectionGate:
    """Reflection gate decision carrying trigger metadata for one turn."""

    trigger: ReflectionTrigger
    trigger_label: str
    window_interactions: int


class TopicCanonResponse(BaseModel):
    """Structured response for LLM-based topic canonicalization."""

    mappings: dict[str, str] = {}


class DisagreementDetectionResponse(BaseModel):
    """Structured response for topic-level disagreement checks."""

    disagreement_verdict: DisagreementVerdict = DisagreementVerdict.NO_DISAGREEMENT
    disagreement_strength: float = 0.0
    reasoning: str = ""


class BeliefDecayResponse(BaseModel):
    """Structured response for belief staleness handling."""

    action: BeliefDecayAction = BeliefDecayAction.RETAIN
    new_confidence: float = 0.0
    reasoning: str = ""


class EntrenchmentDetectionResponse(BaseModel):
    """Structured response for belief entrenchment detection."""

    entrenchment_status: EntrenchmentStatus = EntrenchmentStatus.NOT_ENTRENCHED
    confidence: float = 0.0
    reasoning: str = ""
    recommendation: str = ""


class ReflectionGateResponse(BaseModel):
    """Structured response for per-turn reflection trigger decisions."""

    trigger: ReflectionGateDecision = ReflectionGateDecision.SKIP
    reasoning: str = ""


@dataclass(frozen=True, slots=True)
class ModelUsage:
    response_calls: int = 0
    ess_calls: int = 0
    response_input_tokens: int = 0
    response_output_tokens: int = 0
    ess_input_tokens: int = 0
    ess_output_tokens: int = 0


@dataclass(frozen=True, slots=True)
class RuntimeComponents:
    db: DatabaseConnections
    embedder: ExternalEmbedder
    graph: MemoryGraph
    dual_store: DualEpisodeStore
    stm: ShortTermMemory
    summarizer: BackgroundSummarizer
    boundary_detector: EventBoundaryDetector
    query_router: QueryRouter
    chain_agent: ChainOfQueryAgent
    split_agent: SplitQueryAgent
    consolidation: ConsolidationEngine
    forgetting: ForgettingEngine
    semantic_worker: SemanticIngestionWorker


class SonalityAgent:
    model: str = config.MODEL
    ess_model: str = config.ESS_MODEL

    def __init__(
        self,
        model: str = config.MODEL,
        ess_model: str = config.ESS_MODEL,
    ) -> None:
        """Boot the runtime agent and load persistent memory state.

        Assumes one OpenAI-compatible provider endpoint for chat and embeddings.
        """
        missing = config.missing_live_api_config()
        if missing:
            raise ValueError(f"Missing required API config: {', '.join(missing)}")
        self.model = model
        self.ess_model = ess_model
        log.info(
            "Initializing SonalityAgent (model=%s, ess_model=%s, base_url=%s)",
            self.model,
            self.ess_model,
            config.BASE_URL,
        )
        if self.model == self.ess_model:
            log.debug("Main and ESS models are identical (single-model setup)")
        self.sponge = SpongeState.load(config.SPONGE_FILE)
        self.conversation: list[dict[str, str]] = []
        self.last_ess = classifier_exception_fallback("")
        self.last_usage = ModelUsage()
        self.previous_snapshot = ""
        self._last_entrenched: list[str] = []
        self._last_entrenched_interaction: int = -1
        # LLM-based topic normalization cache: raw_lower → canonical_lower
        self._topic_canon_cache: dict[str, str] = {}

        # Background event loop for async database operations
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever, name="agent-async-loop", daemon=True
        )
        self._loop_thread.start()

        try:
            runtime = self._run_async(self._init_new_architecture())
            self._db = runtime.db
            self._embedder = runtime.embedder
            self._graph = runtime.graph
            self._dual_store = runtime.dual_store
            self._stm = runtime.stm
            self._summarizer = runtime.summarizer
            self._boundary_detector = runtime.boundary_detector
            self._query_router = runtime.query_router
            self._chain_agent = runtime.chain_agent
            self._split_agent = runtime.split_agent
            self._consolidation = runtime.consolidation
            self._forgetting = runtime.forgetting
            self._semantic_worker = runtime.semantic_worker
            log.info("New memory architecture initialized (Neo4j + pgvector)")
        except Exception as exc:
            log.exception("New memory architecture initialization failed")
            raise RuntimeError(
                "Path A storage (Neo4j + pgvector) is required and failed to initialize"
            ) from exc

        log.info(
            "Agent ready: sponge v%d, %d prior interactions, %d beliefs",
            self.sponge.version,
            self.sponge.interaction_count,
            len(self.sponge.opinion_vectors),
        )

    def _run_async[T](self, coro: Coroutine[object, object, T]) -> T:
        """Run an async coroutine from sync context via the background event loop."""
        future: Future[T] = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=config.ASYNC_TIMEOUT)

    async def _init_new_architecture(self) -> RuntimeComponents:
        """Initialize Neo4j + pgvector + embedding components."""
        db = await DatabaseConnections.create()
        embedder = ExternalEmbedder()
        graph = MemoryGraph(db.neo4j_driver)
        chunker = DerivativeChunker(embedder)
        dual_store = DualEpisodeStore(graph, db.pg_pool, chunker, embedder)
        stm = await ShortTermMemory.load(db.pg_pool)
        summarizer = BackgroundSummarizer(stm)
        summarizer.start()
        boundary_detector = EventBoundaryDetector()
        latest_segment_counter = await graph.get_latest_segment_counter()
        boundary_detector.set_segment_counter(latest_segment_counter)
        query_router = QueryRouter()
        chain_agent = ChainOfQueryAgent(dual_store, graph)
        split_agent = SplitQueryAgent(dual_store, graph)
        consolidation = ConsolidationEngine(graph)
        forgetting = ForgettingEngine(graph, dual_store)
        semantic_worker = SemanticIngestionWorker(db.pg_pool, embedder)
        semantic_worker.start()
        # Restore last episode UID for temporal linking
        last_uid = await graph.get_last_episode_uid()
        if last_uid:
            dual_store.set_last_episode_uid(last_uid)
        return RuntimeComponents(
            db=db,
            embedder=embedder,
            graph=graph,
            dual_store=dual_store,
            stm=stm,
            summarizer=summarizer,
            boundary_detector=boundary_detector,
            query_router=query_router,
            chain_agent=chain_agent,
            split_agent=split_agent,
            consolidation=consolidation,
            forgetting=forgetting,
            semantic_worker=semantic_worker,
        )

    def shutdown(self) -> None:
        """Gracefully shut down background threads and database connections."""
        self._summarizer.stop()
        self._semantic_worker.stop()
        try:
            self._run_async(self._db.close())
        except Exception:
            log.exception("Error closing database connections")
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._loop_thread.join(timeout=5)

    def respond(self, user_message: str) -> str:
        """Run one interaction turn and persist resulting personality state.

        This is the canonical orchestration entrypoint used by the CLI.
        """
        _t0 = time.perf_counter()
        log.info("=== Interaction #%d ===", self.sponge.interaction_count + 1)
        log.info("User: %.120s", user_message)

        # Step 2: Add to STM buffer
        self._stm.add_message("user", user_message)

        # Step 3-4: Retrieve relevant memories + stored knowledge
        try:
            relevant = self._run_async(self._retrieve_new_arch(user_message))
        except Exception:
            log.exception("Memory retrieval failed")
            relevant = []
        try:
            knowledge_lines = self._run_async(
                retrieve_relevant_knowledge(
                    query=user_message,
                    pg_pool=self._db.pg_pool,
                    embedder=self._embedder,
                )
            )
            log.debug(
                "Knowledge retrieval: %d items | query='%.60s'%s",
                len(knowledge_lines),
                user_message,
                f" | top={knowledge_lines[0][:80]!r}" if knowledge_lines else "",
            )
        except Exception:
            log.debug("Knowledge retrieval failed", exc_info=True)
            knowledge_lines = []
        structured_traits = self._build_structured_traits()

        # Step 5: Build system prompt (with STM running summary if available)
        system_prompt = build_system_prompt(
            sponge_snapshot=self.sponge.snapshot,
            relevant_episodes=relevant,
            structured_traits=structured_traits,
            knowledge_context=knowledge_lines,
        )
        if self._stm.running_summary:
            stm_section = f"\n\n## Recent Context Summary\n{self._stm.running_summary}"
            idx = next(
                (i for marker in (
                    "\n## Personality Traits",
                    "\n## Relevant Past Conversations",
                    "\n## Instructions",
                ) if (i := system_prompt.find(marker)) > 0),
                -1,
            )
            system_prompt = (
                system_prompt[:idx] + stm_section + system_prompt[idx:]
                if idx > 0
                else system_prompt + stm_section
            )

        self._log_context_event(
            user_message=user_message,
            relevant_episodes=relevant,
            structured_traits=structured_traits,
            system_prompt=system_prompt,
        )
        log.debug(
            "System prompt: %d chars (~%d tokens)", len(system_prompt), len(system_prompt) // 4
        )

        self.conversation.append({"role": "user", "content": user_message})
        self._truncate_conversation()

        completion = chat_completion(
            model=self.model,
            max_tokens=config.FAST_LLM_MAX_TOKENS,
            messages=(
                {"role": "system", "content": system_prompt},
                *self.conversation,
            ),
        )
        response_input_tokens = completion.input_tokens
        response_output_tokens = completion.output_tokens
        assistant_msg = completion.text
        if not assistant_msg:
            log.warning("Model response contained no text block; using empty reply")
        self.conversation.append({"role": "assistant", "content": assistant_msg})

        # Add assistant response to STM
        self._stm.add_message("assistant", assistant_msg)

        log.info("Agent: %.200s", assistant_msg)
        log.debug("Agent (full): %s", assistant_msg)
        _llm_elapsed = time.perf_counter() - _t0
        log.info("Interaction #%d LLM: %.1fs", self.sponge.interaction_count + 1, _llm_elapsed)

        self._post_process(user_message, assistant_msg)
        _total_elapsed = time.perf_counter() - _t0
        log.info("Interaction #%d total: %.1fs", self.sponge.interaction_count, _total_elapsed)
        last_ess = self.last_ess
        self.last_usage = ModelUsage(
            response_calls=1,
            ess_calls=last_ess.attempt_count,
            response_input_tokens=response_input_tokens,
            response_output_tokens=response_output_tokens,
            ess_input_tokens=last_ess.input_tokens,
            ess_output_tokens=last_ess.output_tokens,
        )
        self._log_event(
            {
                "event": "model_usage",
                "interaction": self.sponge.interaction_count,
                **asdict(self.last_usage),
            }
        )
        return assistant_msg

    async def _retrieve_new_arch(self, user_message: str) -> list[str]:
        """Full retrieval pipeline: route → search → expand → rerank."""
        # Step 3: Route query
        stm_context = self._stm.get_recent_context()
        decision = self._query_router.route(user_message, context=stm_context)

        log.info(
            "Query routing: category=%s n_results=%d temporal=%s semantic=%s",
            decision.category,
            decision.n_results,
            decision.temporal_expansion,
            decision.semantic_memory,
        )
        if decision.category == QueryCategory.NONE:
            return []

        # Step 4: Retrieve based on category
        if decision.category == QueryCategory.MULTI_ENTITY:
            split_result = await self._split_agent.retrieve(
                user_message, n_per_sub=decision.n_results
            )
            episodes = split_result.episodes
        elif decision.category in (QueryCategory.TEMPORAL, QueryCategory.AGGREGATION):
            chain_result = await self._chain_agent.retrieve(user_message, base_n=decision.n_results)
            episodes = chain_result.episodes
        elif decision.category == QueryCategory.BELIEF_QUERY:
            over_fetch = decision.n_results * config.RETRIEVAL_OVER_FETCH_FACTOR
            belief_hits = await self._graph.find_belief_related_episodes(
                user_message,
                limit=over_fetch,
            )
            topic_hits = await self._graph.find_topic_related_episodes(
                user_message,
                limit=max(2, over_fetch // 2),
            )
            vector_hits = await self._dual_store.vector_search(user_message, top_k=over_fetch)
            vector_uids = list({row[1] for row in vector_hits})
            episodes = belief_hits + topic_hits + await self._graph.get_episodes(vector_uids)
        else:
            # Simple query: direct vector search
            over_fetch = decision.n_results * config.RETRIEVAL_OVER_FETCH_FACTOR
            results = await self._dual_store.vector_search(user_message, top_k=over_fetch)
            episode_uids = list({r[1] for r in results})
            topic_hits = await self._graph.find_topic_related_episodes(
                user_message,
                limit=max(2, over_fetch // 2),
            )
            episodes = topic_hits + await self._graph.get_episodes(episode_uids)
        episodes = list({episode.uid: episode for episode in episodes}.values())

        # Step 5: Temporal expansion
        if decision.temporal_expansion is TemporalExpansionDecision.EXPAND and episodes:
            expanded_uids: set[str] = set()
            for ep in episodes[:3]:  # Expand top 3 only
                neighbors = await self._graph.traverse_temporal_context(ep.uid)
                for n in neighbors:
                    expanded_uids.add(n.uid)
            new_uids = [u for u in expanded_uids if u not in {e.uid for e in episodes}]
            if new_uids:
                extra = await self._graph.get_episodes(new_uids)
                episodes.extend(extra)

        # Step 7: LLM Listwise Rerank
        if len(episodes) > 1:
            episodes = rerank_episodes(user_message, episodes)

        selected = episodes[: decision.n_results]
        semantic_context: list[str] = []
        if decision.semantic_memory is SemanticMemoryDecision.SEARCH:
            semantic_context = await self._search_semantic_features(
                user_message,
                top_k=max(2, min(decision.n_results, 6)),
            )

        # Step 8: Differentiated utility feedback for selected vs. noisy candidates
        for idx, ep in enumerate(episodes):
            try:
                delta = _UTILITY_USED_DELTA if idx < len(selected) else _UTILITY_NOISE_DELTA
                await self._graph.update_utility(ep.uid, delta=delta)
            except Exception:
                log.debug("Utility update failed for %s", ep.uid[:8])

        # Step 10: Format as context strings (matching legacy format)
        episode_context = [
            format_episode_line(
                created_at=ep.created_at,
                summary=ep.summary,
                content=ep.content,
                content_limit=300,
            )
            for ep in selected
        ]
        log.info(
            "Retrieval: category=%s n_episodes=%d n_semantic=%d | episodes=%s",
            decision.category,
            len(selected),
            len(semantic_context),
            [(ep.uid[:8], (ep.summary or ep.content)[:50]) for ep in selected],
        )
        return [*episode_context, *semantic_context]

    async def _search_semantic_features(self, query: str, *, top_k: int) -> list[str]:
        """Search semantic features via pgvector similarity."""
        query_embedding = self._embedder.embed_query(query)
        async with self._db.pg_pool.connection() as conn, conn.cursor() as cur:
            await cur.execute("SET hnsw.iterative_scan = 'relaxed_order'")
            await cur.execute("SET hnsw.ef_search = 100")
            await cur.execute(
                """
                SELECT category, tag, feature_name, value, confidence,
                       embedding <=> %s::vector AS distance
                FROM semantic_features
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector, confidence DESC, updated_at DESC
                LIMIT %s
                """,
                (query_embedding, query_embedding, top_k),
            )
            rows = await cur.fetchall()
        return [
            f"[semantic/{row[0]!s}] {row[1]!s}.{row[2]!s}: {row[3]!s}"
            f" (conf={float(row[4]):.2f}, dist={float(row[5]):.3f})"
            for row in rows
        ]

    def _truncate_conversation(self) -> None:
        """Keep chat history inside a configured character budget.

        Oldest messages are discarded first while preserving at least one recent
        exchange for response continuity.
        """
        total = sum(len(m["content"]) for m in self.conversation)
        removed_count = 0
        while total > config.MAX_CONVERSATION_CHARS and len(self.conversation) > 2:
            removed = self.conversation.pop(0)
            total -= len(removed["content"])
            removed_count += 1
        if removed_count:
            log.info("Truncated %d old messages (conversation now %d chars)", removed_count, total)

    def _post_process(self, user_message: str, agent_response: str) -> None:
        """Apply ESS classification, memory updates, and optional reflection.

        Assumes the main response has already been appended to conversation state.
        """
        log.info("--- Post-processing ---")

        ess = self._classify_ess(user_message)
        self.last_ess = ess
        self._log_ess(ess, user_message)
        log.info(
            "ESS: score=%.3f type=%s dir=%s novelty=%.2f topics=%s severity=%s attempts=%d",
            ess.score,
            ess.reasoning_type,
            ess.opinion_direction,
            ess.novelty,
            list(ess.topics),
            ess.default_severity,
            ess.attempt_count,
        )

        # Event boundary detection + dual-store storage (required architecture)
        previous_segment_id = self._boundary_detector.current_segment_id
        segment_id = ""
        segment_label = ""
        segment_reasoning = ""
        closed_segment_id = ""
        try:
            boundary = self._boundary_detector.check_boundary(user_message)
            segment_id = boundary.segment_id
            if boundary.boundary_decision is BoundaryDecision.BOUNDARY:
                closed_segment_id = previous_segment_id
                segment_label = boundary.label
                segment_reasoning = boundary.reasoning
                log.info("Segment boundary: %s (%s)", boundary.label, boundary.boundary_type)
        except Exception:
            log.exception("Boundary detection failed")

        episode_uid = self._store_episode_new_arch(
            user_message,
            agent_response,
            ess,
            segment_id,
            segment_label,
            segment_reasoning,
        )
        if closed_segment_id:
            self._try_consolidate_segment(closed_segment_id, trigger="boundary")
        if not episode_uid:
            log.warning("Dual-store write failed; skipping post-processing belief update")

        # Queue for semantic feature extraction (include ESS context so the worker
        # can distinguish low-evidence chit-chat from belief-shaping exchanges)
        if episode_uid:
            ess_line = (
                f"ESS: {ess.score:.2f} ({ess.reasoning_type}) | "
                f"direction={ess.opinion_direction} | "
                f"topics={list(ess.topics)}"
            )
            content = f"User: {user_message}\nAssistant: {agent_response}\n{ess_line}"
            self._semantic_worker.enqueue(episode_uid, content)

        # Knowledge proposition extraction (inline, gated by ESS knowledge_density)
        if episode_uid:
            self._extract_knowledge(user_message, agent_response, ess, episode_uid)
            # Normalize topics in freshly-staged knowledge updates (same LLM canon logic as ESS topics)
            self._normalize_staged_topics()

        # Persist STM to PostgreSQL
        try:
            self._run_async(self._stm.persist(self._db.pg_pool))
        except Exception:
            log.debug("STM persistence failed", exc_info=True)

        self.sponge.interaction_count += 1

        # Manipulative/invalid interactions should not mutate personality state.
        # debunked_claim: conclusively refuted claims must never update beliefs.
        # social_pressure / emotional_appeal: coercive but no evidential content.
        # anecdotal: unsourced "experts say" / "studies show" without actual evidence.
        manipulative = ess.reasoning_type in {
            "social_pressure", "emotional_appeal", "debunked_claim", "anecdotal"
        }
        if manipulative:
            log.info(
                "Manipulative interaction (%s, score=%.3f): freezing sponge mutation",
                ess.reasoning_type,
                ess.score,
            )
        else:
            log.debug(
                "Non-manipulative interaction (%s, score=%.3f): sponge mutation allowed",
                ess.reasoning_type,
                ess.score,
            )

        if not manipulative:
            committed = self.sponge.apply_due_staged_updates()
            if committed:
                log.info("Committed staged beliefs: %s", committed)
            else:
                log.debug(
                    "No staged beliefs due (pending=%d)",
                    len(self.sponge.staged_opinion_updates),
                )
            self._log_event(
                {
                    "event": "opinion_commit",
                    "interaction": self.sponge.interaction_count,
                    "committed": committed,
                    "remaining_staged": len(self.sponge.staged_opinion_updates),
                }
            )

        for topic in ess.topics:
            self.sponge.track_topic(topic)
        if episode_uid and not manipulative:
            try:
                self._run_async(
                    self._update_opinions_with_provenance(
                        user_message, agent_response, ess, episode_uid
                    )
                )
            except Exception:
                log.exception("Provenance opinion update failed")
        if self._detect_disagreement(user_message, ess):
            self.sponge.note_disagreement()
        else:
            self.sponge.note_agreement()

        self.previous_snapshot = self.sponge.snapshot
        if not manipulative:
            self._extract_insight(user_message, agent_response, ess)
            self._maybe_reflect()
        self._log_health_event()

        self.sponge.save(config.SPONGE_FILE, config.SPONGE_HISTORY_DIR)
        self._log_interaction_summary(ess)

    def _classify_ess(self, user_message: str) -> ESSResult:
        """Classify user evidence and fallback safely on classifier failures."""
        try:
            ess = classify(
                PROVIDER_CLIENT,
                user_message,
                self.sponge.snapshot,
                model=self.ess_model,
            )
        except Exception:
            log.exception("ESS classification failed, using safe defaults")
            return classifier_exception_fallback(user_message)
        canonical = self._normalize_topics_llm(ess.topics)
        return ess if canonical == ess.topics else dataclasses.replace(ess, topics=canonical)

    def _normalize_topics_llm(self, raw_topics: tuple[str, ...]) -> tuple[str, ...]:
        """Map ESS topic labels to canonical forms already in belief memory.

        Prevents topic fragmentation where "nuclear" and "nuclear energy" accumulate
        as separate beliefs. Uses a session-scoped cache to avoid repeated LLM calls
        for topics seen in previous interactions.
        """
        if not raw_topics:
            return raw_topics

        existing = (
            set(self.sponge.opinion_vectors)
            | set(self.sponge.behavioral_signature.topic_engagement)
        )

        result: list[str] = []
        uncached: list[str] = []
        for raw in raw_topics:
            lower = raw.strip().lower()
            if lower in self._topic_canon_cache:
                result.append(self._topic_canon_cache[lower])
            elif not existing or lower in existing:
                self._topic_canon_cache[lower] = lower
                result.append(lower)
            else:
                uncached.append(lower)

        if uncached:
            prompt = TOPIC_CANONICALIZATION_PROMPT.format(
                existing=json.dumps(sorted(existing)),
                new_topics=json.dumps(uncached),
            )
            llm_result = llm_call(
                prompt=prompt,
                response_model=TopicCanonResponse,
                fallback=TopicCanonResponse(mappings={t: t for t in uncached}),
            )
            mappings = llm_result.value.mappings
            for raw in uncached:
                canonical = mappings.get(raw, raw).strip().lower() or raw
                self._topic_canon_cache[raw] = canonical
                if canonical != raw:
                    log.debug("Topic canonical: '%s' → '%s'", raw, canonical)
                result.append(canonical)

        return tuple(result)

    def _normalize_staged_topics(self) -> None:
        """Canonicalize topics in pending staged opinion updates.

        Called after knowledge extraction stages new opinion updates so that
        propositions with slightly different topic wording (e.g. "nuclear power"
        vs "nuclear energy") map to the same canonical belief entry.
        """
        raw_topics = tuple({u.topic for u in self.sponge.staged_opinion_updates})
        if not raw_topics:
            return
        canonical = self._normalize_topics_llm(raw_topics)
        mapping = dict(zip(raw_topics, canonical, strict=True))
        for update in self.sponge.staged_opinion_updates:
            mapped = mapping.get(update.topic, update.topic)
            if mapped != update.topic:
                update.topic = mapped

    def _extract_knowledge(
        self, user_message: str, agent_response: str, ess: ESSResult, episode_uid: str
    ) -> None:
        """Extract and store knowledge propositions when ESS signals learnable content."""
        if ess.knowledge_density == KnowledgeDensity.NONE:
            log.debug("Knowledge extraction skipped: density=NONE")
            return
        log.debug("Knowledge extraction starting: density=%s", ess.knowledge_density)
        text = f"User: {user_message}\nAssistant: {agent_response}"
        try:
            stored = self._run_async(
                extract_and_store_knowledge(
                    text=text,
                    episode_uid=episode_uid,
                    pg_pool=self._db.pg_pool,
                    embedder=self._embedder,
                    sponge=self.sponge,
                )
            )
            if stored:
                log.info("Knowledge extraction stored %d propositions", stored)
        except Exception:
            log.exception("Knowledge extraction failed")

    def _store_episode_new_arch(
        self,
        user_message: str,
        agent_response: str,
        ess: ESSResult,
        segment_id: str,
        segment_label: str,
        segment_reasoning: str,
    ) -> str:
        """Store episode in Neo4j + pgvector dual store. Returns episode UID on success."""
        try:
            stored: StoredEpisode = self._run_async(
                self._dual_store.store(
                    user_message=user_message,
                    agent_response=agent_response,
                    summary=ess.summary[:300],
                    topics=list(ess.topics),
                    ess_score=ess.score,
                    segment_id=segment_id,
                    segment_label=segment_label,
                    segment_reasoning=segment_reasoning,
                )
            )
            return stored.episode_uid
        except Exception:
            log.exception("Dual-store episode storage failed")
            return ""

    def _detect_disagreement(self, user_message: str, ess: ESSResult) -> bool:
        """Structural disagreement between current user evidence and held beliefs.

        Also checks staged (uncommitted) opinion updates so early-interaction
        disagreement is correctly tracked before beliefs mature.
        """
        sign = ess.opinion_direction.sign
        if sign == 0.0:
            return False
        # Build effective position map: committed + net staged
        staged_net: dict[str, float] = {}
        for s in self.sponge.staged_opinion_updates:
            staged_net[s.topic] = staged_net.get(s.topic, 0.0) + s.signed_magnitude
        for topic in ess.topics:
            committed = self.sponge.opinion_vectors.get(topic, 0.0)
            staged = staged_net.get(topic, 0.0)
            pos = committed + staged
            prompt = DISAGREEMENT_DETECTION_PROMPT.format(
                user_message=user_message[:500],
                topic=topic,
                position_value=f"{pos:+.2f}",
                opinion_direction=f"{sign:+.1f}",
            )
            try:
                result = llm_call(
                    prompt=prompt,
                    response_model=DisagreementDetectionResponse,
                    fallback=DisagreementDetectionResponse(),
                )
            except Exception:
                continue
            if not result.success:
                continue
            response = result.value
            if response.disagreement_verdict is DisagreementVerdict.DISAGREEMENT:
                return True
        return False

    def _collect_unresolved_contradictions(self) -> list[str]:
        """Summarize staged deltas that currently oppose strong held beliefs."""
        candidates: list[tuple[float, str]] = []
        for staged in self.sponge.staged_opinion_updates:
            pos = self.sponge.opinion_vectors.get(staged.topic, 0.0)
            if pos * staged.signed_magnitude >= 0:
                continue
            summary = (
                f"{staged.topic}({pos:+.2f} vs {staged.signed_magnitude:+.3f},"
                f" due #{staged.due_interaction})"
            )
            candidates.append((abs(staged.signed_magnitude), summary))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [summary for _, summary in candidates]

    def _apply_llm_contraction(self, topic: str, evidence_strength: float) -> None:
        """Soften a belief when provenance assessment recommends contraction."""
        old_pos = self.sponge.opinion_vectors.get(topic, 0.0)
        if abs(old_pos) < 1e-9:
            return
        strength = max(0.0, min(1.0, evidence_strength))
        step = min(abs(old_pos), max(0.02, abs(old_pos) * strength))
        new_pos = old_pos - (1.0 if old_pos > 0 else -1.0) * step
        self.sponge.opinion_vectors[topic] = new_pos
        if topic in self.sponge.belief_meta:
            meta = self.sponge.belief_meta[topic]
            meta.confidence = max(0.0, meta.confidence - step * 0.5)
            meta.uncertainty = min(1.0, max(meta.uncertainty, 1.0 - meta.confidence))
        self.sponge.record_shift(
            description=f"LLM-guided contraction on {topic}",
            magnitude=step,
        )
        self._log_event(
            {
                "event": "opinion_contract",
                "interaction": self.sponge.interaction_count,
                "topic": topic,
                "old_pos": round(old_pos, 4),
                "new_pos": round(new_pos, 4),
                "delta": round(step, 4),
                "evidence_strength": round(strength, 4),
            }
        )

    # Reasoning types the LLM classifies as non-substantive (no personality updates).
    _NO_UPDATE_REASONING: Final = frozenset({
        ReasoningType.NO_ARGUMENT,
        ReasoningType.SOCIAL_PRESSURE,
        ReasoningType.EMOTIONAL_APPEAL,
        ReasoningType.DEBUNKED_CLAIM,
    })

    def _ess_allows_update(
        self, ess: ESSResult, *, update_kind: str, require_topics: bool = False,
    ) -> bool:
        """Return whether ESS permits a personality update path.

        Gates on the LLM's semantic classification of reasoning type rather than
        a hardcoded score threshold. The LLM already decided whether the argument
        is substantive — we trust that classification.
        """
        if require_topics and not ess.topics:
            return False
        if ess.reasoning_type in self._NO_UPDATE_REASONING:
            log.info("Skipping %s: reasoning_type=%s is non-substantive", update_kind, ess.reasoning_type)
            return False
        reliable = (
            ess.default_severity not in {"missing", "exception"}
            and not any(field in CRITICAL_ESS_DEFAULT_FIELDS for field in ess.defaulted_fields)
        )
        if reliable:
            return True
        log.info(
            "Skipping %s due to ESS fallback defaults (severity=%s fields=%s)",
            update_kind, ess.default_severity, ess.defaulted_fields,
        )
        return False

    def _stage_topic_opinion_update(
        self,
        *,
        topic: str,
        direction: float,
        magnitude: float,
        provenance: str,
        episode_uid: str = "",
        new_uncertainty: float = -1.0,
    ) -> None:
        """Stage one topic update and emit a consistent audit event."""
        if magnitude <= 0.0:
            return
        due = self.sponge.stage_opinion_update(
            topic=topic,
            direction=direction,
            magnitude=magnitude,
            cooling_period=config.OPINION_COOLING_PERIOD,
            provenance=provenance,
            new_uncertainty=new_uncertainty,
        )
        event: dict[str, object] = {
            "event": "opinion_staged",
            "interaction": self.sponge.interaction_count,
            "topic": topic,
            "signed_magnitude": direction * magnitude,
            "due_interaction": due,
            "staged_total": len(self.sponge.staged_opinion_updates),
        }
        if episode_uid:
            event["provenance_episode"] = episode_uid
        self._log_event(event)

    async def _update_opinions_with_provenance(
        self,
        user_message: str,
        agent_response: str,
        ess: ESSResult,
        episode_uid: str,
    ) -> None:
        """Use LLM-based evidence assessment with episode provenance links."""
        if not self._ess_allows_update(ess, update_kind="provenance opinion update", require_topics=True):
            return
        content = (
            f"User: {user_message}\nAssistant: {agent_response}\n"
            f"ESS summary: {ess.summary}\nESS score: {ess.score:.2f}"
        )
        fallback_direction = ess.opinion_direction.sign

        for topic in ess.topics:
            try:
                update = await assess_belief_evidence(
                    topic=topic,
                    episode_uid=episode_uid,
                    episode_content=content,
                    ess_score=ess.score,
                    reasoning_type=str(ess.reasoning_type),
                    source_reliability=str(ess.source_reliability),
                    sponge=self.sponge,
                    graph=self._graph,
                )
            except Exception:
                log.exception("Belief provenance assessment failed for %s", topic)
                continue

            direction = update.direction if abs(update.direction) > 1e-6 else fallback_direction
            if abs(direction) < 1e-6:
                continue
            if update.contraction_action is ContractionAction.CONTRACT:
                self._apply_llm_contraction(topic, update.evidence_strength)

            old_pos = self.sponge.opinion_vectors.get(topic, 0.0)
            meta = self.sponge.belief_meta.get(topic)
            confidence = (meta.confidence if meta else 0.0) + (abs(old_pos) if old_pos * direction < 0 else 0.0)
            raw_mag = max(0.0, min(1.0, update.evidence_strength)) / (confidence + 1.0)
            effective_mag = min(raw_mag, MAX_SINGLE_UPDATE_MAGNITUDE)
            log.debug(
                "Belief magnitude %s: raw=%.3f effective=%.3f (type=%s)",
                topic, raw_mag, effective_mag, ess.reasoning_type,
            )
            self._stage_topic_opinion_update(
                topic=topic,
                direction=direction,
                magnitude=effective_mag,
                provenance=f"{update.reasoning[:120]} (ep={episode_uid[:8]})",
                episode_uid=episode_uid,
                new_uncertainty=update.new_uncertainty,
            )
            if update.update_magnitude is UpdateMagnitude.MAJOR:
                self.sponge.record_shift(
                    description=f"Belief update: {topic} ({update.reasoning[:60]})",
                    magnitude=abs(direction * effective_mag),
                )

    def _extract_insight(self, user_message: str, agent_response: str, ess: ESSResult) -> None:
        """Extract personality insight per interaction, consolidated during reflection.

        Avoids lossy per-interaction full snapshot rewrites (ABBEL 2025: belief
        bottleneck). Snapshot only changes during reflection (Park et al. 2023).
        """
        if not self._ess_allows_update(ess, update_kind="insight extraction"):
            return
        try:
            insight = extract_insight(
                ess,
                user_message,
                agent_response,
                model=self.ess_model,
            )
            if not insight:
                return
            self.sponge.pending_insights.append(insight)
            self.sponge.version += 1
            magnitude = max(0.01, ess.score * max(ess.novelty, 0.1))
            self.sponge.record_shift(
                description=f"ESS {ess.score:.2f}: {insight[:80]}",
                magnitude=magnitude,
            )
            log.info(
                "Insight (v%d, %d pending): %s",
                self.sponge.version,
                len(self.sponge.pending_insights),
                insight[:80],
            )
        except Exception:
            log.exception("Insight extraction failed")

    def _build_structured_traits(self) -> str:
        """Build a compact trait summary injected into the system prompt.

        Keeps high-signal structured context visible without forcing full JSON
        state into each generation call.
        """
        top_topics = sorted(
            self.sponge.behavioral_signature.topic_engagement.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        topics_line = ", ".join(f"{t}({c})" for t, c in top_topics) if top_topics else "none yet"

        opinions = sorted(
            self.sponge.opinion_vectors.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:5]
        opinions_parts: list[str] = []
        for topic, pos in opinions:
            meta = self.sponge.belief_meta.get(topic)
            conf = f" c={meta.confidence:.1f}" if meta else ""
            opinions_parts.append(f"{topic}={pos:+.2f}{conf}")
        opinions_line = ", ".join(opinions_parts) if opinions_parts else "none yet"

        recent = [s for s in self.sponge.recent_shifts[-3:] if s.magnitude > 0]
        evolution_line = ", ".join(s.description[:50] for s in recent) if recent else "stable"
        staged_topics = [u.topic for u in self.sponge.staged_opinion_updates[-3:]]
        staged_line = ", ".join(staged_topics) if staged_topics else "none"

        return (
            f"Style: {self.sponge.tone}\n"
            f"Top topics: {topics_line}\n"
            f"Strongest opinions: {opinions_line}\n"
            f"Disagreement rate: {self.sponge.behavioral_signature.disagreement_rate:.0%}\n"
            f"Recent evolution: {evolution_line}\n"
            f"Staged beliefs: {staged_line}"
        )

    # Minimum interactions required before event-driven reflection is allowed.
    # Prevents aggressive early reflection on fresh agents with few interactions.
    _MIN_WINDOW_FOR_EVENT_DRIVEN: int = 5

    def _reflection_gate(self) -> ReflectionGate:
        """Determine whether reflection should run for this interaction."""
        window_interactions = self.sponge.interaction_count - self.sponge.last_reflection_at
        recent_mag = sum(
            shift.magnitude
            for shift in self.sponge.recent_shifts
            if shift.interaction > self.sponge.last_reflection_at
        )

        # Hard minimum window: never reflect before 5 interactions have accumulated.
        if window_interactions < self._MIN_WINDOW_FOR_EVENT_DRIVEN:
            return ReflectionGate(
                trigger=ReflectionTrigger.SKIP,
                trigger_label=f"skip (window={window_interactions} < min={self._MIN_WINDOW_FOR_EVENT_DRIVEN})",
                window_interactions=window_interactions,
            )

        prompt = REFLECTION_GATE_PROMPT.format(
            interaction_count=self.sponge.interaction_count,
            window_interactions=window_interactions,
            target_cadence=config.REFLECTION_EVERY,
            pending_insights=len(self.sponge.pending_insights),
            staged_updates=len(self.sponge.staged_opinion_updates),
            recent_shift_magnitude=f"{recent_mag:.3f}",
            disagreement_rate=f"{self.sponge.behavioral_signature.disagreement_rate:.2f}",
            belief_count=len(self.sponge.belief_meta),
        )
        # At-cadence periodic reflection is not gated by the LLM — it always fires.
        # The LLM gate only decides whether to fire *early* (event-driven) before cadence.
        if window_interactions >= config.REFLECTION_EVERY:
            log.info(
                "Periodic reflection: window=%d >= cadence=%d",
                window_interactions, config.REFLECTION_EVERY,
            )
            return ReflectionGate(
                trigger=ReflectionTrigger.PERIODIC,
                trigger_label="periodic",
                window_interactions=window_interactions,
            )

        # Below cadence: consult LLM for event-driven early reflection.
        result = llm_call(
            prompt=prompt,
            response_model=ReflectionGateResponse,
            fallback=ReflectionGateResponse(trigger=ReflectionGateDecision.SKIP),
        )
        if not result.success:
            return ReflectionGate(
                trigger=ReflectionTrigger.SKIP,
                trigger_label="skip (invalid gate payload)",
                window_interactions=window_interactions,
            )
        trigger_name = result.value.trigger
        if trigger_name is ReflectionGateDecision.PERIODIC:
            return ReflectionGate(
                trigger=ReflectionTrigger.PERIODIC,
                trigger_label="periodic",
                window_interactions=window_interactions,
            )
        if trigger_name is ReflectionGateDecision.EVENT_DRIVEN:
            reason = (
                result.value.reasoning[:80] if result.value.reasoning else f"mag={recent_mag:.3f}"
            )
            return ReflectionGate(
                trigger=ReflectionTrigger.EVENT_DRIVEN,
                trigger_label=f"event-driven ({reason})",
                window_interactions=window_interactions,
            )
        return ReflectionGate(
            trigger=ReflectionTrigger.SKIP,
            trigger_label="skip",
            window_interactions=window_interactions,
        )

    def _reflection_prompt(self, trigger_label: str, recent_episodes: list[str]) -> str:
        """Assemble reflection prompt from current belief/shift/insight state."""
        beliefs_text = (
            "\n".join(
                f"- {topic}: {self.sponge.opinion_vectors.get(topic, 0):+.2f} "
                f"(conf={meta.confidence:.2f}, ev={meta.evidence_count}, "
                f"last=#{meta.last_reinforced})"
                for topic, meta in sorted(
                    self.sponge.belief_meta.items(),
                    key=lambda item: -abs(self.sponge.opinion_vectors.get(item[0], 0)),
                )
            )
            or "No beliefs formed yet."
        )
        shifts_text = (
            "\n".join(
                f"- #{shift.interaction} (mag {shift.magnitude:.3f}): {shift.description}"
                for shift in self.sponge.recent_shifts
            )
            or "No recent shifts."
        )
        n = self.sponge.interaction_count
        b = len(self.sponge.opinion_vectors)
        if n < 20:
            maturity = "Focus on accurately recording what you've learned so far."
        elif n < 50 or b < 10:
            maturity = "Look for patterns across your experiences and beliefs."
        else:
            maturity = (
                "Your worldview is developing coherence. Based on your accumulated "
                "beliefs, you may have nascent views on topics you haven't explicitly "
                "discussed. If a pattern suggests a new position, articulate it tentatively."
            )
        return REFLECTION_PROMPT.format(
            trigger=trigger_label,
            current_snapshot=self.sponge.snapshot,
            structured_traits=self._build_structured_traits(),
            current_beliefs=beliefs_text,
            pending_insights="\n".join(f"- {i}" for i in self.sponge.pending_insights) or "None.",
            episode_count=len(recent_episodes),
            episode_summaries="\n".join(f"- {episode}" for episode in recent_episodes),
            recent_shifts=shifts_text,
            maturity_instruction=maturity,
            max_tokens=config.SPONGE_MAX_TOKENS,
        )

    def _apply_reflection_snapshot(
        self, pre_snapshot: str, reflected_snapshot: str, opinions_before: dict[str, float]
    ) -> None:
        """Validate and commit reflected snapshot text when it changed."""
        if not reflected_snapshot or reflected_snapshot == pre_snapshot:
            log.info("Reflection produced no changes")
            return
        # Guard against unbounded snapshot growth: ~4 chars/token.
        char_limit = config.SPONGE_MAX_TOKENS * 4
        if len(reflected_snapshot) > char_limit:
            reflected_snapshot = reflected_snapshot[:char_limit]
            log.debug("Reflection snapshot truncated to %d chars (limit %d)", char_limit, char_limit)
        if len(reflected_snapshot) < 30:
            log.warning("Reflection output rejected: snapshot too short (%d chars)", len(reflected_snapshot))
            return

        self._check_belief_preservation(opinions_before)
        self.sponge.snapshot = reflected_snapshot
        self.sponge.version += 1
        self.sponge.record_shift(
            description=f"Reflection at interaction {self.sponge.interaction_count}",
            magnitude=0.0,
        )
        log.info(
            "Reflection completed: v%d, %d -> %d chars (delta=%+d)",
            self.sponge.version,
            len(pre_snapshot),
            len(reflected_snapshot),
            len(reflected_snapshot) - len(pre_snapshot),
        )

    def _finalize_reflection_cycle(
        self,
        *,
        dropped: list[str],
        entrenched: list[str],
        contradictions: list[str],
        window_interactions: int,
    ) -> None:
        """Clear temporary reflection buffers and log cycle diagnostics."""
        consolidated = len(self.sponge.pending_insights)
        self.sponge.pending_insights.clear()
        self.sponge.last_reflection_at = self.sponge.interaction_count
        self._log_reflection_summary(
            dropped=dropped,
            consolidated=consolidated,
            entrenched=entrenched,
            contradictions=contradictions,
        )
        self._log_reflection_event(
            dropped=dropped,
            consolidated=consolidated,
            entrenched=entrenched,
            contradictions=contradictions,
            window_interactions=window_interactions,
        )

    def _decay_beliefs_with_llm(self) -> list[str]:
        """Use LLM staleness assessment to retain, decay, or forget beliefs."""
        dropped: list[str] = []
        stale_candidates = [
            (
                self.sponge.interaction_count - meta.last_reinforced,
                topic,
                meta,
                self.sponge.opinion_vectors.get(topic, 0.0),
            )
            for topic, meta in self.sponge.belief_meta.items()
            if self.sponge.interaction_count - meta.last_reinforced >= 5
        ]
        for gap, topic, meta, position in sorted(stale_candidates, reverse=True)[:10]:
            prompt = BELIEF_DECAY_PROMPT.format(
                topic=topic,
                position=f"{position:+.2f}",
                confidence=f"{meta.confidence:.2f}",
                evidence_count=meta.evidence_count,
                gap=gap,
                total_interactions=self.sponge.interaction_count,
            )
            result = llm_call(
                prompt=prompt,
                response_model=BeliefDecayResponse,
                fallback=BeliefDecayResponse(
                    action=BeliefDecayAction.RETAIN, new_confidence=meta.confidence
                ),
            )
            if not result.success:
                continue
            response = result.value
            action = response.action
            if action is BeliefDecayAction.FORGET:
                dropped.append(topic)
                del self.sponge.belief_meta[topic]
                self.sponge.opinion_vectors.pop(topic, None)
                continue
            if action is BeliefDecayAction.DECAY:
                meta.confidence = max(0.0, min(1.0, response.new_confidence))
                meta.uncertainty = 1.0 - meta.confidence
        return dropped

    def _detect_entrenched_beliefs_llm(self, min_updates: int = 4) -> list[str]:
        """Use LLM to detect echo-chamber style belief entrenchment (cached per turn)."""
        if self._last_entrenched_interaction == self.sponge.interaction_count:
            return list(self._last_entrenched)
        entrenched: list[str] = []
        candidates = [
            (topic, meta, self.sponge.opinion_vectors.get(topic, 0.0))
            for topic, meta in self.sponge.belief_meta.items()
            if len(meta.recent_updates) >= min_updates
        ]
        for topic, meta, position in candidates[:10]:
            prompt = ENTRENCHMENT_DETECTION_PROMPT.format(
                topic=topic,
                position=f"{position:+.2f}",
                recent_updates=", ".join(f"{update:+.3f}" for update in meta.recent_updates[-8:]),
                supporting_count=len(meta.supporting_episode_uids),
                contradicting_count=len(meta.contradicting_episode_uids),
            )
            result = llm_call(
                prompt=prompt,
                response_model=EntrenchmentDetectionResponse,
                fallback=EntrenchmentDetectionResponse(),
            )
            if not result.success:
                continue
            if result.value.entrenchment_status is EntrenchmentStatus.ENTRENCHED:
                entrenched.append(topic)
        self._last_entrenched = entrenched
        self._last_entrenched_interaction = self.sponge.interaction_count
        return entrenched

    def _maybe_reflect(self) -> None:
        """Run periodic or event-driven reflection and snapshot consolidation.

        Reflection is deliberately sparse; over-frequent rewrites increase drift
        and can erase minority traits from the narrative snapshot.
        """
        gate = self._reflection_gate()
        if gate.trigger is ReflectionTrigger.SKIP:
            return

        log.info(
            "=== Reflection at #%d (%s) ===",
            self.sponge.interaction_count,
            gate.trigger_label,
        )

        # Dump full DB state before reflection for manual inspection
        self._dump_snapshot(f"PRE_REFLECTION #{self.sponge.interaction_count}")

        opinions_before_reflection = dict(self.sponge.opinion_vectors)
        dropped = self._decay_beliefs_with_llm()
        if dropped:
            log.info("Decay removed %d stale beliefs: %s", len(dropped), dropped)

        entrenched = self._detect_entrenched_beliefs_llm()
        if entrenched:
            log.warning("Entrenched beliefs detected: %s", entrenched)
        contradictions = self._collect_unresolved_contradictions()
        if contradictions:
            log.info("Contradiction backlog (%d): %s", len(contradictions), contradictions[:3])

        try:
            pending = self._run_async(
                self._graph.list_unconsolidated_segments(
                    exclude_segment_id=self._boundary_detector.current_segment_id,
                    limit=4,
                )
            )
            for segment_id in pending:
                self._try_consolidate_segment(segment_id, trigger="reflection")
        except Exception:
            log.exception("Consolidation failed during reflection")

        # Knowledge consolidation: review for contradictions and merges in pgvector
        try:
            result = self._run_async(
                consolidate_knowledge(pg_pool=self._db.pg_pool, snapshot=self.sponge.snapshot)
            )
            if result and (result.contradictions or result.merges):
                log.info(
                    "Knowledge consolidation: %d contradictions, %d merges, %d opinion candidates",
                    len(result.contradictions), len(result.merges), len(result.opinion_candidates),
                )
        except Exception:
            log.exception("Knowledge consolidation failed")

        # Prune low-confidence stale knowledge entries from pgvector
        try:
            self._run_async(prune_stale_knowledge(self._db.pg_pool))
        except Exception:
            log.debug("Knowledge pruning failed", exc_info=True)

        # Forgetting: assess and archive low-importance episodes
        try:
            self._run_async(self._run_forgetting_cycle())
        except Exception:
            log.exception("Forgetting cycle failed during reflection")

        # Sync Neo4j Belief nodes with sponge — prune graph beliefs for
        # topics that were decayed/forgotten from opinion_vectors
        try:
            active = set(self.sponge.opinion_vectors.keys())
            self._run_async(self._graph.sync_beliefs(active))
        except Exception:
            log.debug("Belief graph sync failed", exc_info=True)

        # Prune Topic nodes with zero active episode connections
        try:
            self._run_async(self._graph.prune_orphan_topics())
        except Exception:
            log.debug("Topic pruning failed", exc_info=True)

        # Consistency check: clean derivative orphans across Neo4j and pgvector
        try:
            orphans: list[str] = self._run_async(self._dual_store.verify_consistency())
            if orphans:
                log.warning("Consistency check cleaned %d orphan derivatives", len(orphans))
        except Exception:
            log.exception("Consistency verification failed during reflection")

        # LLM-based health assessment
        try:
            health = assess_health(self.sponge)
            if health.concerns:
                log.warning("Health assessment concerns: %s", health.concerns)
        except Exception:
            log.debug("LLM health assessment failed", exc_info=True)

        try:
            recent_episodes = (
                self._run_async(
                    self._graph.list_recent_episode_context(min(config.REFLECTION_EVERY, 10))
                ) or []
            )
        except Exception:
            log.debug("Reflection episode retrieval failed", exc_info=True)
            recent_episodes = []
        if not recent_episodes:
            log.info("No episodes for reflection, skipping")
            self.sponge.last_reflection_at = self.sponge.interaction_count
            return

        prompt = self._reflection_prompt(gate.trigger_label, recent_episodes)

        try:
            pre_snapshot = self.sponge.snapshot
            completion = chat_completion(
                model=self.ess_model,
                max_tokens=config.FAST_LLM_MAX_TOKENS,
                messages=({"role": "user", "content": prompt},),
            )
            reflected_snapshot = completion.text.strip()
            self._apply_reflection_snapshot(pre_snapshot, reflected_snapshot, opinions_before_reflection)
            self._finalize_reflection_cycle(
                dropped=dropped,
                entrenched=entrenched,
                contradictions=contradictions,
                window_interactions=gate.window_interactions,
            )
        except Exception:
            log.exception("Reflection cycle failed")

        # Dump full DB state after reflection for delta analysis
        self._dump_snapshot(f"POST_REFLECTION #{self.sponge.interaction_count}")

    def _dump_snapshot(self, label: str) -> None:
        """Dump full DB state to debug log for manual inspection."""
        try:
            async def _snap() -> None:
                async with self._db.neo4j_driver.session(
                    database=config.NEO4J_DATABASE
                ) as neo_sess:
                    await dump_memory_snapshot(
                        self._db.pg_pool, neo_sess, self.sponge, label=label,
                    )
            self._run_async(_snap())
        except Exception:
            log.debug("Memory snapshot (%s) failed", label, exc_info=True)

    def _try_consolidate_segment(self, segment_id: str, *, trigger: str) -> None:
        """Attempt one segment consolidation and log failures per trigger path."""
        if not segment_id:
            return
        try:
            summary_uid = self._run_async(self._consolidation.maybe_consolidate_segment(segment_id))
            if summary_uid:
                log.info(
                    "Consolidated segment %s -> summary %s (%s)",
                    segment_id,
                    summary_uid[:8],
                    trigger,
                )
        except Exception:
            log.exception("Segment consolidation failed (%s, segment=%s)", trigger, segment_id)

    async def _run_forgetting_cycle(self) -> None:
        """Assess old episodes for potential archival during reflection."""
        candidates = await self._graph.get_forgetting_candidates(limit=20)

        if len(candidates) < 5:
            return

        forgetting_result = await self._forgetting.assess_and_forget(
            candidates, snapshot_excerpt=self.sponge.snapshot[:300]
        )
        if forgetting_result.archived > 0:
            log.info(
                "Forgetting: assessed=%d, kept=%d, archived=%d",
                forgetting_result.total_assessed,
                forgetting_result.kept,
                forgetting_result.archived,
            )

    def _check_belief_preservation(self, opinions_before: dict[str, float]) -> None:
        """Warn if reflection dropped high-magnitude beliefs from opinion_vectors.

        Checks actual vector entries, not snapshot text.  Snapshot is a narrative
        and will not mention every tracked topic verbatim.  The real danger is a
        belief being evicted from opinion_vectors entirely, which is what we watch.
        """
        strong_before = {t for t, v in opinions_before.items() if abs(v) > 0}
        strong_after = set(self.sponge.opinion_vectors)
        dropped = strong_before - strong_after
        if dropped:
            log.warning("HEALTH: reflection evicted strong beliefs from vectors: %s", dropped)

    def _log_interaction_summary(self, ess: ESSResult) -> None:
        """Structured per-interaction summary for monitoring personality evolution."""
        parts = [
            f"[#{self.sponge.interaction_count}]",
            f"ESS={ess.score:.2f}({ess.reasoning_type})",
            f"dir={ess.opinion_direction}",
            f"src={ess.source_reliability}",
            f"novelty={ess.novelty:.2f}",
            f"staged={len(self.sponge.staged_opinion_updates)}",
            f"pending={len(self.sponge.pending_insights)}",
        ]
        if ess.topics:
            parts.append(f"topics={list(ess.topics)}")
        parts.append(f"beliefs={len(self.sponge.opinion_vectors)}")
        parts.append(f"v{self.sponge.version}")
        if ess.default_severity != "none":
            parts.append(f"ESS_FALLBACK={ess.default_severity}({list(ess.defaulted_fields)})")

        for topic in ess.topics:
            meta = self.sponge.belief_meta.get(topic)
            pos = self.sponge.opinion_vectors.get(topic)
            if meta and pos is not None:
                parts.append(
                    f"{topic}={pos:+.2f}(c={meta.confidence:.2f},ev={meta.evidence_count})"
                )

        log.info("SUMMARY: %s", " | ".join(parts))

    def _log_reflection_summary(
        self,
        dropped: list[str],
        consolidated: int,
        entrenched: list[str],
        contradictions: list[str],
    ) -> None:
        """Emit a concise human-readable reflection health summary."""
        metas = list(self.sponge.belief_meta.values())
        ic = self.sponge.interaction_count
        log.info(
            "REFLECTION: insights=%d beliefs=%d high_conf=%d stale=%d dropped=%d "
            "entrenched=%d contradictions=%d disagree=%.0f%% snapshot=%dch v%d",
            consolidated,
            len(self.sponge.opinion_vectors),
            sum(1 for m in metas if m.confidence > 0.5),
            sum(1 for m in metas if ic - m.last_reinforced > 30),
            len(dropped),
            len(entrenched),
            len(contradictions),
            self.sponge.behavioral_signature.disagreement_rate * 100,
            len(self.sponge.snapshot),
            self.sponge.version,
        )

    def _log_context_event(
        self,
        user_message: str,
        relevant_episodes: list[str],
        structured_traits: str,
        system_prompt: str,
    ) -> None:
        """Write prompt-construction context metrics to the audit stream."""
        self._log_event(
            {
                "event": "context",
                "interaction": self.sponge.interaction_count + 1,
                "user_chars": len(user_message),
                "conversation_chars": sum(len(m["content"]) for m in self.conversation),
                "prompt_chars": len(system_prompt),
                "snapshot_chars": len(self.sponge.snapshot),
                "structured_traits_chars": len(structured_traits),
                "relevant_count": len(relevant_episodes),
                "relevant_chars": sum(len(ep) for ep in relevant_episodes),
                "semantic_budget": config.SEMANTIC_RETRIEVAL_COUNT,
                "episodic_budget": config.EPISODIC_RETRIEVAL_COUNT,
            }
        )

    def _log_health_event(self) -> None:
        """Write lightweight health diagnostics for drift/sycophancy detection."""
        words = self.sponge.snapshot.split()
        unique_ratio = len(set(w.lower() for w in words)) / len(words) if words else 0.0
        metas = list(self.sponge.belief_meta.values())
        high_conf = sum(1 for m in metas if m.confidence > 0.5)
        high_conf_ratio = high_conf / len(metas) if metas else 0.0
        disagreement = self.sponge.behavioral_signature.disagreement_rate

        warnings: list[str] = []

        entrenched = self._detect_entrenched_beliefs_llm()
        if entrenched:
            warnings.append("entrenched_beliefs")
        contradictions = self._collect_unresolved_contradictions()
        if contradictions:
            warnings.append("contradiction_backlog")

        self._log_event(
            {
                "event": "health",
                "interaction": self.sponge.interaction_count,
                "belief_count": len(self.sponge.opinion_vectors),
                "high_conf_ratio": round(high_conf_ratio, 3),
                "disagreement_rate": round(disagreement, 3),
                "snapshot_words": len(words),
                "snapshot_unique_ratio": round(unique_ratio, 3),
                "pending_insights": len(self.sponge.pending_insights),
                "staged_updates": len(self.sponge.staged_opinion_updates),
                "entrenched": entrenched,
                "contradictions": contradictions,
                "warnings": warnings,
            }
        )

    def _log_event(self, event: dict[str, object]) -> None:
        """Append event to JSONL audit trail for personality evolution tracking."""
        log_path = config.ESS_AUDIT_LOG_FILE
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            payload: dict[str, object] = {
                "schema": "ess-audit-v2",
                "model": self.model,
                "ess_model": self.ess_model,
                **event,
                "ts": datetime.now(UTC).isoformat(),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            log.debug("JSONL logging failed", exc_info=True)

    def _log_reflection_event(
        self,
        dropped: list[str],
        consolidated: int,
        entrenched: list[str],
        contradictions: list[str],
        window_interactions: int = 1,
    ) -> None:
        """Log structured reflection metrics for longitudinal analysis."""
        old_words = set(self.previous_snapshot.lower().split())
        new_words = set(self.sponge.snapshot.lower().split())
        union = old_words | new_words
        jaccard = len(old_words & new_words) / len(union) if union else 1.0

        insight_yield = consolidated / max(window_interactions, 1)

        self._log_event(
            {
                "event": "reflection",
                "interaction": self.sponge.interaction_count,
                "version": self.sponge.version,
                "insights_consolidated": consolidated,
                "beliefs_dropped": dropped,
                "total_beliefs": len(self.sponge.opinion_vectors),
                "high_confidence": sum(
                    1 for m in self.sponge.belief_meta.values() if m.confidence > 0.5
                ),
                "snapshot_chars": len(self.sponge.snapshot),
                "snapshot_jaccard": round(jaccard, 3),
                "insight_yield": round(insight_yield, 3),
                "entrenched": entrenched,
                "contradictions": contradictions,
            }
        )

    def _log_ess(self, ess: ESSResult, user_message: str) -> None:
        """Persist ESS outputs and belief state deltas to the audit log."""
        self._log_event(
            {
                "event": "ess",
                "interaction": self.sponge.interaction_count + 1,
                "score": ess.score,
                "type": ess.reasoning_type,
                "direction": ess.opinion_direction,
                "novelty": ess.novelty,
                "topics": ess.topics,
                "source": ess.source_reliability,
                "defaults": ess.used_defaults,
                "defaulted_fields": list(ess.defaulted_fields),
                "default_severity": ess.default_severity,
                "pending_insights": len(self.sponge.pending_insights),
                "staged_updates": len(self.sponge.staged_opinion_updates),
                "msg_preview": user_message[:80],
                "beliefs": {
                    t: {
                        "pos": self.sponge.opinion_vectors.get(t, 0.0),
                        "conf": m.confidence,
                        "ev": m.evidence_count,
                    }
                    for t in ess.topics
                    if (m := self.sponge.belief_meta.get(t))
                },
            }
        )
