from __future__ import annotations

import asyncio
import dataclasses
import hashlib
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

from pydantic import BaseModel, Field, model_validator

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
    BATCH_BELIEF_DECAY_PROMPT,
    BATCH_ENTRENCHMENT_DETECTION_PROMPT,
    DISAGREEMENT_DETECTION_PROMPT,
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
    Embedder,
    EventBoundaryDetector,
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
    assess_belief_evidence_batch,
    assess_health,
    consolidate_knowledge,
    cosine_similarity,
    dump_memory_snapshot,
    extract_and_store_knowledge,
    extract_insight,
    prune_stale_knowledge,
    rerank_episodes,
    retrieve_relevant_knowledge,
)
from .memory.context_format import format_episode_line
from .prompts import REFLECTION_PROMPT, build_system_prompt
from .provider import ChatResult, chat_completion, interaction_active

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


class BeliefDecayDecision(BaseModel):
    """Single-topic decay decision within a batch response."""

    topic: str
    action: BeliefDecayAction = BeliefDecayAction.RETAIN
    new_confidence: float | None = None
    reasoning: str = ""


class BatchBeliefDecayResponse(BaseModel):
    """Batch decay assessment — list of per-topic decisions."""

    decisions: list[BeliefDecayDecision] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_list(cls, data: object) -> object:
        if isinstance(data, list):
            return {"decisions": data}
        if isinstance(data, dict) and "decisions" not in data and "topic" in data:
            return {"decisions": [data]}
        return data


class EntrenchmentDecision(BaseModel):
    """Single entrenched topic within a batch response."""

    topic: str
    reasoning: str = ""


class BatchEntrenchmentResponse(BaseModel):
    """Batch entrenchment detection — list of entrenched topics only."""

    entrenched: list[EntrenchmentDecision] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize_list(cls, data: object) -> object:
        if isinstance(data, list):
            # Discard numeric indices (model bug: [1, 3, 4] instead of [{topic: ...}])
            data = [item for item in data if isinstance(item, dict)]
            return {"entrenched": data}
        if isinstance(data, dict):
            items = data.get("entrenched", [])
            if isinstance(items, list):
                data = {"entrenched": [item for item in items if isinstance(item, dict)]}
        return data


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
    embedder: Embedder
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
        self.last_knowledge_writes: int = 0
        self.previous_snapshot = ""
        self._last_entrenched: list[str] = []
        self._last_entrenched_interaction: int = -1
        # LLM-based topic normalization cache: raw_lower → canonical_lower
        self._topic_canon_cache: dict[str, str] = {}
        # Content dedup ring buffer: SHA-256 hex of recently ingested texts (capped at 200)
        self._ingest_seen_hashes: list[str] = []

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
            log.info("New memory architecture initialized (Neo4j + Qdrant)")
        except Exception as exc:
            log.exception("New memory architecture initialization failed")
            raise RuntimeError(
                "Path A storage (Neo4j + Qdrant) is required and failed to initialize"
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
        try:
            return future.result(timeout=config.ASYNC_TIMEOUT)
        except TimeoutError:
            future.cancel()
            raise

    async def _init_new_architecture(self) -> RuntimeComponents:
        """Initialize Neo4j + Qdrant + embedding components."""
        db = await DatabaseConnections.create()
        embedder = Embedder()
        graph = MemoryGraph(db.neo4j_driver)
        chunker = DerivativeChunker(embedder)
        dual_store = DualEpisodeStore(graph, db.qdrant, chunker, embedder)
        stm = await ShortTermMemory.load(db.neo4j_driver)
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
        semantic_worker = SemanticIngestionWorker(config.QDRANT_URL, embedder)
        semantic_worker.start()
        # Restore last episode UID for temporal linking
        last_uid = await graph.get_last_episode_uid()
        if last_uid:
            dual_store._last_episode_uid = last_uid
            dual_store.has_episodes = True
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
        # Drain the semantic worker before stopping so that any episodes enqueued
        # during the session are fully processed before the Qdrant client closes.
        self._semantic_worker.drain(timeout=120.0)
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
        interaction_active.set()
        try:
            return self._respond_inner(user_message)
        finally:
            interaction_active.clear()

    def _respond_inner(self, user_message: str) -> str:
        """Internal implementation of respond(); called with interaction_active set."""
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
                    qdrant=self._db.qdrant,
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
                (
                    i
                    for marker in (
                        "\n## Personality Traits",
                        "\n## Relevant Past Conversations",
                        "\n## Instructions",
                    )
                    if (i := system_prompt.find(marker)) > 0
                ),
                -1,
            )
            system_prompt = (
                system_prompt[:idx] + stm_section + system_prompt[idx:]
                if idx > 0
                else system_prompt + stm_section
            )

        self._log_event(
            {
                "event": "context",
                "interaction": self.sponge.interaction_count + 1,
                "user_chars": len(user_message),
                "conversation_chars": sum(len(m["content"]) for m in self.conversation),
                "prompt_chars": len(system_prompt),
                "snapshot_chars": len(self.sponge.snapshot),
                "structured_traits_chars": len(structured_traits),
                "relevant_count": len(relevant),
                "relevant_chars": sum(len(ep) for ep in relevant),
                "semantic_budget": config.SEMANTIC_RETRIEVAL_COUNT,
                "episodic_budget": config.EPISODIC_RETRIEVAL_COUNT,
            }
        )
        log.debug(
            "System prompt: %d chars (~%d tokens)", len(system_prompt), len(system_prompt) // 4
        )

        self.conversation.append({"role": "user", "content": user_message})
        self._truncate_conversation()

        _t_retrieval_elapsed = time.perf_counter() - _t0
        _t_llm = time.perf_counter()
        for _attempt in range(1, 4):
            try:
                completion = chat_completion(
                    model=self.model,
                    max_tokens=config.FAST_LLM_MAX_TOKENS,
                    messages=(
                        {"role": "system", "content": system_prompt},
                        *self.conversation,
                    ),
                    enable_thinking=False,
                )
                break
            except RuntimeError as exc:
                error_str = str(exc).lower()
                is_network = "name resolution" in error_str or "network error" in error_str
                if _attempt < 3 and not is_network:
                    log.warning("LLM completion failed (attempt %d/3): %s; retrying", _attempt, exc)
                    continue
                log.error("LLM completion failed after %d attempts: %s", _attempt, exc)
                completion = ChatResult(text="", input_tokens=0, output_tokens=0, raw={})
                break
        _llm_elapsed = time.perf_counter() - _t_llm
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
        log.info(
            "Interaction #%d LLM: %.1fs (retrieval=%.1fs)",
            self.sponge.interaction_count + 1,
            _llm_elapsed,
            _t_retrieval_elapsed,
        )

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

    def ingest(self, text: str, *, topic_override: str = "") -> ESSResult:
        """Non-conversational data ingestion path bypassing response generation.

        Designed for processing news articles, social media posts, research reports,
        and other non-interactive content. Skips STM buffer, memory retrieval,
        system prompt building, and LLM response generation.

        Args:
            text: The content to ingest (news article, social media post, etc.)
            topic_override: Optional canonical topic name to use instead of
                LLM-extracted topics. Use when the caller knows the exact topic.

        Returns:
            ESSResult from classification (for caller inspection/logging).
        """
        _t0 = time.perf_counter()
        log.info("=== Ingest #%d ===", self.sponge.interaction_count + 1)
        log.info("Content: %.200s", text)

        # Ingest-level content dedup: skip identical articles seen in the last 200 ingests.
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        if content_hash in self._ingest_seen_hashes:
            log.warning(
                "INGEST_DUPLICATE skipping identical content (hash=%s) — "
                "article already processed this session",
                content_hash[:12],
            )
            self.sponge.interaction_count += 1
            return self.last_ess
        self._ingest_seen_hashes.append(content_hash)
        if len(self._ingest_seen_hashes) > 200:
            self._ingest_seen_hashes.pop(0)

        log.debug(
            "INGEST_STATE sponge=v%d interactions=%d beliefs=%d staged=%d",
            self.sponge.version,
            self.sponge.interaction_count,
            len(self.sponge.opinion_vectors),
            len(self.sponge.staged_opinion_updates),
        )

        _t_ess = time.perf_counter()
        ess = self._classify_ess(text)
        _ess_elapsed = time.perf_counter() - _t_ess
        if topic_override:
            ess = dataclasses.replace(ess, topics=(topic_override.strip().lower(),))
        self.last_ess = ess

        log.info(
            "ESS: score=%.3f type=%s dir=%s update=%s urgency=%s novelty=%.2f "
            "knowledge=%s topics=%s (%.1fs)",
            ess.score,
            ess.reasoning_type,
            ess.opinion_direction,
            ess.belief_update_recommended,
            ess.urgency,
            ess.novelty,
            ess.knowledge_density,
            list(ess.topics),
            _ess_elapsed,
        )
        if ess.used_defaults:
            log.warning(
                "ESS fallback fields in ingest: %s (severity=%s)",
                ess.defaulted_fields,
                ess.default_severity,
            )

        manipulative = ess.reasoning_type in {
            "social_pressure",
            "emotional_appeal",
            "debunked_claim",
            "anecdotal",
        }
        if manipulative:
            log.info(
                "Ingest blocked: manipulative reasoning type=%s score=%.3f",
                ess.reasoning_type,
                ess.score,
            )
        elif not ess.belief_update_recommended:
            log.info(
                "Ingest: belief update not recommended (type=%s score=%.3f novelty=%.2f) "
                "— episode will not be stored",
                ess.reasoning_type,
                ess.score,
                ess.novelty,
            )

        episode_uid = ""
        if not manipulative and ess.belief_update_recommended:
            _t_store = time.perf_counter()
            episode_uid = self._store_ingest_episode(text, ess)
            log.debug("INGEST_STEP store=%.1fs", time.perf_counter() - _t_store)

            if episode_uid:
                _t_knowledge = time.perf_counter()
                self._extract_knowledge(
                    text,
                    "",
                    ess,
                    episode_uid,
                    stage_opinions=True,
                )
                self._normalize_staged_topics()
                log.debug("INGEST_STEP knowledge=%.1fs", time.perf_counter() - _t_knowledge)

                log.info(
                    "INGEST_BEFORE_PROVENANCE topics=%s | staged=%d | beliefs=%s",
                    list(ess.topics),
                    len(self.sponge.staged_opinion_updates),
                    {
                        t: f"{self.sponge.opinion_vectors.get(t, 0.0):+.3f}"
                        f"(conf={self.sponge.belief_meta[t].confidence:.2f})"
                        if t in self.sponge.belief_meta
                        else f"{self.sponge.opinion_vectors.get(t, 0.0):+.3f}"
                        for t in ess.topics
                    },
                )
                _t_provenance = time.perf_counter()
                try:
                    self._run_async(
                        self._update_opinions_with_provenance(text, "", ess, episode_uid)
                    )
                except Exception:
                    log.exception("Provenance opinion update failed")
                log.debug("INGEST_STEP provenance=%.1fs", time.perf_counter() - _t_provenance)

                self._semantic_worker.enqueue(
                    episode_uid,
                    f"Content: {text}\nESS: {ess.score:.2f} ({ess.reasoning_type})",
                    categories=("knowledge",),
                )

        self.sponge.interaction_count += 1

        _t_commit = time.perf_counter()
        if not manipulative:
            staged_before = len(self.sponge.staged_opinion_updates)
            committed = self.sponge.apply_due_staged_updates()
            if committed:
                log.info(
                    "Committed staged beliefs: %s (staged_before=%d staged_after=%d)",
                    committed,
                    staged_before,
                    len(self.sponge.staged_opinion_updates),
                )
                # Record total committed shift so the reflection gate's recent_mag
                # accumulates correctly (only MAJOR individual beliefs called record_shift
                # before, leaving cumulative ingest shifts invisible to event-driven reflection).
                total_committed_mag = sum(
                    abs(float(s.split(":")[1].split("(")[0]))
                    for s in committed
                    if ":" in s
                )
                if total_committed_mag > 0.01:
                    self.sponge.record_shift(
                        description=f"Ingest #{self.sponge.interaction_count} committed {len(committed)} beliefs",
                        magnitude=total_committed_mag,
                    )
            elif staged_before > 0:
                log.debug(
                    "INGEST_STAGED_PENDING %d updates not yet due (interaction=%d)",
                    staged_before,
                    self.sponge.interaction_count,
                )
        log.debug("INGEST_STEP commit=%.1fs", time.perf_counter() - _t_commit)

        for topic in ess.topics:
            self.sponge.track_topic(topic)

        self._maybe_reflect()
        self.sponge.save(config.SPONGE_FILE, config.SPONGE_HISTORY_DIR)

        _elapsed = time.perf_counter() - _t0
        log.info(
            "Ingest #%d completed in %.1fs | ess=%.1fs beliefs=%d staged=%d v%d",
            self.sponge.interaction_count,
            _elapsed,
            _ess_elapsed,
            len(self.sponge.opinion_vectors),
            len(self.sponge.staged_opinion_updates),
            self.sponge.version,
        )

        return ess

    def _store_ingest_episode(self, text: str, ess: ESSResult) -> str:
        """Store ingested content as a simplified episode. Returns UID on success."""
        try:
            stored = self._run_async(
                self._dual_store.store(
                    user_message=text,
                    agent_response="",
                    summary=ess.summary[:300],
                    topics=list(ess.topics),
                    ess_score=ess.score,
                    segment_id="",
                    segment_label="",
                    segment_reasoning="",
                )
            )
            log.info("Stored ingest episode: %s", stored.episode_uid[:8])
            return stored.episode_uid
        except Exception:
            log.exception("Ingest episode storage failed")
            return ""

    async def _retrieve_new_arch(self, user_message: str) -> list[str]:
        """Full retrieval pipeline: route → search → expand → rerank."""
        import time as _time_module

        # Skip routing when there are no stored episodes to retrieve.
        if not self._dual_store.has_episodes:
            log.debug("Retrieval skipped: no episodes in store")
            return []

        # Step 3: Route query (offload LLM call to thread to avoid blocking event loop)
        _t_route = _time_module.perf_counter()
        stm_context = self._stm.get_recent_context()
        decision = await asyncio.to_thread(self._query_router.route, user_message, stm_context)
        _route_elapsed = _time_module.perf_counter() - _t_route

        log.info(
            "Query routing: category=%s n_results=%d temporal=%s semantic=%s (%.1fs)",
            decision.category,
            decision.n_results,
            decision.temporal_expansion,
            decision.semantic_memory,
            _route_elapsed,
        )
        if decision.category == QueryCategory.NONE:
            return []

        # Step 4: Retrieve based on category
        _t_search = _time_module.perf_counter()
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
        _search_elapsed = _time_module.perf_counter() - _t_search

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

        # Step 7: LLM Listwise Rerank (offload to thread — LLM call is blocking)
        _t_rerank = _time_module.perf_counter()
        if len(episodes) > 1:
            episodes = await asyncio.to_thread(rerank_episodes, user_message, episodes)
        _rerank_elapsed = _time_module.perf_counter() - _t_rerank

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
            "Retrieval: category=%s n_episodes=%d n_semantic=%d route=%.1fs search=%.1fs rerank=%.1fs | episodes=%s",
            decision.category,
            len(selected),
            len(semantic_context),
            _route_elapsed,
            _search_elapsed,
            _rerank_elapsed,
            [(ep.uid[:8], (ep.summary or ep.content)[:50]) for ep in selected],
        )
        return [*episode_context, *semantic_context]

    async def _search_semantic_features(self, query: str, *, top_k: int) -> list[str]:
        """Search semantic features via Qdrant similarity."""
        query_embedding = await asyncio.to_thread(self._embedder.embed_query, query)
        response = await self._db.qdrant.query_points(
            collection_name="semantic_features",
            query=query_embedding,
            using="dense",
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

        try:
            ess = self._classify_ess(user_message)
        except Exception:
            log.exception("ESS classification failed completely, using safe fallback")
            ess = classifier_exception_fallback(user_message)
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

        # Manipulative/invalid interactions should not mutate personality state.
        # debunked_claim: conclusively refuted claims must never update beliefs.
        # social_pressure / emotional_appeal: coercive but no evidential content.
        # anecdotal: unsourced "experts say" / "studies show" without actual evidence.
        manipulative = ess.reasoning_type in {
            "social_pressure",
            "emotional_appeal",
            "debunked_claim",
            "anecdotal",
        }
        # no_argument: bare assertions with no evidential backing (injection attacks,
        # unsupported commands, repeated claims). Block knowledge extraction and Qdrant
        # writes; staged opinion commits are still allowed so prior evidence can mature.
        no_evidence = ess.reasoning_type == "no_argument"
        log.debug(
            "MUTATION_GATE manipulative=%s type=%s score=%.3f topics=%s",
            manipulative,
            ess.reasoning_type,
            ess.score,
            list(ess.topics),
        )
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

        # Knowledge proposition extraction (inline, gated by ESS knowledge_density).
        # Skipped for manipulative turns (coercive, no evidential content) AND for
        # no_argument turns (bare assertions, injection attacks) — neither carries
        # valid knowledge worth storing in the proposition store.
        self.last_knowledge_writes = 0
        if episode_uid and not manipulative and not no_evidence:
            try:
                self._extract_knowledge(
                    user_message,
                    agent_response,
                    ess,
                    episode_uid,
                    stage_opinions=ess.belief_update_recommended,
                )
                # Normalize topics in freshly-staged knowledge updates
                self._normalize_staged_topics()
            except Exception:
                log.exception("Knowledge extraction failed (outer guard)")

        # Persist STM to Neo4j
        try:
            self._run_async(self._stm.persist(self._db.neo4j_driver))
        except Exception:
            log.debug("STM persistence failed", exc_info=True)

        self.sponge.interaction_count += 1

        if not manipulative:
            staged_before = len(self.sponge.staged_opinion_updates)
            committed = self.sponge.apply_due_staged_updates()
            staged_after = len(self.sponge.staged_opinion_updates)
            if committed:
                log.info("Committed staged beliefs: %s", committed)
                log.debug(
                    "STAGED_COMMIT staged=%d→%d committed=%d",
                    staged_before,
                    staged_after,
                    len(committed),
                )
                for topic in committed:
                    b = self.sponge.get_belief(topic)
                    log.debug(
                        "  COMMIT topic=%s pos=%+.4f conf=%.2f ev=%d",
                        topic,
                        b.position,
                        b.confidence,
                        b.evidence_count,
                    )
            else:
                log.debug(
                    "STAGED_COMMIT none due (pending=%d interaction=%d)",
                    staged_before,
                    self.sponge.interaction_count,
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
            if ess.topics:
                log.info(
                    "PROVENANCE_BEFORE topics=%s | %s",
                    list(ess.topics),
                    " | ".join(
                        f"{t}={self.sponge.opinion_vectors.get(t, 0.0):+.3f}"
                        f"(conf={self.sponge.belief_meta[t].confidence:.2f},ev={self.sponge.belief_meta[t].evidence_count})"
                        if t in self.sponge.belief_meta
                        else f"{t}=new"
                        for t in ess.topics
                    ),
                )
            try:
                self._run_async(
                    self._update_opinions_with_provenance(
                        user_message, agent_response, ess, episode_uid
                    )
                )
            except Exception:
                log.exception("Provenance opinion update failed")
        disagrees = self._detect_disagreement(user_message, ess)
        log.debug("DISAGREEMENT_GATE result=%s topics=%s", disagrees, list(ess.topics))
        if disagrees:
            self.sponge.note_disagreement()
        else:
            self.sponge.note_agreement()

        self.previous_snapshot = self.sponge.snapshot
        if not manipulative:
            self._extract_insight(user_message, agent_response, ess)
            self._maybe_reflect()
        else:
            log.info(
                "Deferring insight + reflection (manipulative turn #%d, type=%s)",
                self.sponge.interaction_count,
                ess.reasoning_type,
            )
        self._log_health_event()

        # Enqueue semantic feature extraction only after ALL inline LLM calls
        # (ESS, boundary, chunking, knowledge extraction, opinion provenance,
        # disagreement detection, insight, reflection) are complete. This prevents
        # the background SemanticIngestionWorker from grabbing the LLM semaphore
        # and blocking any foreground post-processing call for the current interaction.
        if episode_uid:
            ess_line = (
                f"ESS: {ess.score:.2f} ({ess.reasoning_type}) | "
                f"direction={ess.opinion_direction} | "
                f"topics={list(ess.topics)}"
            )
            content = f"User: {user_message}\nAssistant: {agent_response}\n{ess_line}"
            self._semantic_worker.enqueue(episode_uid, content)

        self.sponge.save(config.SPONGE_FILE, config.SPONGE_HISTORY_DIR)
        self._log_interaction_summary(ess)

    def _classify_ess(self, user_message: str) -> ESSResult:
        """Classify user evidence and fallback safely on classifier failures."""
        # Top tracked topics fed to ESS so it can reuse existing labels
        # rather than generating synonyms that fragment beliefs.
        top_topics = tuple(
            t
            for t, _ in sorted(
                self.sponge.behavioral_signature.topic_engagement.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]
        )
        try:
            ess = classify(
                PROVIDER_CLIENT,
                user_message,
                self.sponge.snapshot,
                model=self.ess_model,
                recent_topics=top_topics,
            )
        except Exception:
            log.exception("ESS classification failed, using safe defaults")
            return classifier_exception_fallback(user_message)
        canonical = self._normalize_topics_llm(ess.topics)
        return ess if canonical == ess.topics else dataclasses.replace(ess, topics=canonical)

    # Conversational meta-labels that should never become tracked belief topics.
    # The ESS prompt bans these, but the model occasionally emits them. This is a
    # second-line filter so they never reach opinion_vectors or belief_meta.
    _META_TOPIC_BLOCKLIST: Final = frozenset(
        {
            "social pressure",
            "consensus",
            "industry consensus",
            "scientific consensus",
            "expert consensus",
            "group consensus",
            "disagreement",
            "argument",
            "evidence",
            "manipulation",
            "survey method",
            "consistency",
            "reliability",
            "memory",
            "credibility",
            "pressure",
            "peer pressure",
            "emotion",
            "emotional appeal",
            "opinion",
            "reasoning",
        }
    )

    # Embedding similarity thresholds for topic resolution (per plan PATCH 3)
    _TOPIC_AUTO_MERGE_THRESHOLD: float = 0.92  # Auto-merge if similarity >= this
    _TOPIC_LLM_HINT_THRESHOLD: float = 0.70  # Include in LLM prompt if >= this
    # Subtopic suffix patterns: "{base} conflict" → "{base}" when base is a known topic.
    _TOPIC_SUBTOPIC_SUFFIXES: Final = (" conflict", " war", " crisis", " situation", " issue")

    def _normalize_topics_llm(self, raw_topics: tuple[str, ...]) -> tuple[str, ...]:
        """Map ESS topic labels to canonical forms using embedding-assisted resolution.

        Two-layer approach (per plan PATCH 3):
        1. Compute cosine similarity between new topic and existing topics
        2. Auto-merge if similarity >= 0.92 (near-identical strings)
        3. Pass top similar topics with scores to LLM for borderline cases (0.80-0.92)
        4. Add as new topic if no similar matches

        This replaces the 98% ineffective pure-LLM canonicalization with
        embedding similarity as context for LLM decisions.
        """
        if not raw_topics:
            return raw_topics
        raw_topics = tuple(
            t for t in raw_topics if t.strip().lower() not in self._META_TOPIC_BLOCKLIST
        )
        if not raw_topics:
            return raw_topics

        existing = set(self.sponge.opinion_vectors) | set(
            self.sponge.behavioral_signature.topic_engagement
        )

        result: list[str] = []
        uncached: list[str] = []
        for raw in raw_topics:
            lower = raw.strip().lower()
            if lower in self._topic_canon_cache:
                result.append(self._topic_canon_cache[lower])
                continue
            dehyphenated = lower.replace("-", " ")
            # Subtopic suffix collapse: "iran conflict" → "iran" when "iran" is known.
            base_stripped = next(
                (dehyphenated.removesuffix(sfx) for sfx in self._TOPIC_SUBTOPIC_SUFFIXES
                 if dehyphenated.endswith(sfx) and dehyphenated.removesuffix(sfx) in existing),
                None,
            )
            if lower in existing:
                self._topic_canon_cache[lower] = lower
                result.append(lower)
            elif dehyphenated != lower and dehyphenated in existing:
                self._topic_canon_cache[lower] = dehyphenated
                log.debug("Topic hyphen-normalized: '%s' → '%s'", lower, dehyphenated)
                result.append(dehyphenated)
            elif base_stripped:
                self._topic_canon_cache[lower] = base_stripped
                log.debug("Topic subtopic-collapsed: '%s' → '%s'", lower, base_stripped)
                result.append(base_stripped)
            else:
                uncached.append(lower)

        if not uncached:
            return tuple(result)

        existing_list = sorted(existing)
        if not existing_list:
            for raw in uncached:
                self._topic_canon_cache[raw] = raw
                result.append(raw)
            return tuple(result)

        embed_ok = True
        try:
            new_embeddings = self._embedder.embed_documents(uncached)
            existing_embeddings = self._embedder.embed_documents(existing_list)
        except Exception:
            log.debug("Topic embedding failed; falling back to LLM-only")
            embed_ok = False

        if not embed_ok:
            # Pure-LLM fallback when embeddings unavailable
            all_candidates = sorted(existing | set(uncached))
            llm_result = llm_call(
                prompt=TOPIC_CANONICALIZATION_PROMPT.format(
                    existing=json.dumps(all_candidates),
                    new_topics=json.dumps(uncached),
                ),
                response_model=TopicCanonResponse,
                fallback=TopicCanonResponse(mappings={t: t for t in uncached}),
                assistant_prefix='{"mappings": {',
            )
            for raw in uncached:
                canonical = llm_result.value.mappings.get(raw, raw).strip().lower() or raw
                self._topic_canon_cache[raw] = canonical
                if canonical != raw:
                    log.debug("Topic canonical: '%s' → '%s'", raw, canonical)
                result.append(canonical)
            return tuple(result)

        need_llm: list[tuple[str, list[tuple[str, float]]]] = []
        for i, raw in enumerate(uncached):
            new_emb = new_embeddings[i]
            similarities: list[tuple[str, float]] = []
            for j, ex_topic in enumerate(existing_list):
                ex_emb = existing_embeddings[j]
                sim = cosine_similarity(new_emb, ex_emb)
                if sim >= self._TOPIC_LLM_HINT_THRESHOLD:
                    similarities.append((ex_topic, sim))

            similarities.sort(key=lambda x: x[1], reverse=True)
            if similarities and similarities[0][1] >= self._TOPIC_AUTO_MERGE_THRESHOLD:
                canonical = similarities[0][0]
                self._topic_canon_cache[raw] = canonical
                log.debug(
                    "Topic auto-merged: '%s' → '%s' (sim=%.3f)", raw, canonical, similarities[0][1]
                )
                result.append(canonical)
            elif similarities:
                need_llm.append((raw, similarities[:3]))
            else:
                self._topic_canon_cache[raw] = raw
                result.append(raw)

        if need_llm:
            hints = {raw: [(t, f"{s:.2f}") for t, s in sims] for raw, sims in need_llm}
            prompt = (
                TOPIC_CANONICALIZATION_PROMPT.format(
                    existing=json.dumps(sorted(existing | set(uncached))),
                    new_topics=json.dumps([raw for raw, _ in need_llm]),
                )
                + f"\n\nSimilarity hints (topic: [(existing, similarity)]): {json.dumps(hints)}"
            )
            llm_result = llm_call(
                prompt=prompt,
                response_model=TopicCanonResponse,
                fallback=TopicCanonResponse(mappings={raw: raw for raw, _ in need_llm}),
                max_tokens=256,  # {"topic_raw": "canonical"} mapping — small output
                assistant_prefix='{"mappings": {',
            )
            for raw, _ in need_llm:
                canonical = llm_result.value.mappings.get(raw, raw).strip().lower() or raw
                self._topic_canon_cache[raw] = canonical
                if canonical != raw:
                    log.debug("Topic LLM-resolved: '%s' → '%s'", raw, canonical)
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
        self,
        user_message: str,
        agent_response: str,
        ess: ESSResult,
        episode_uid: str,
        *,
        stage_opinions: bool,
    ) -> None:
        """Extract and store knowledge propositions, optionally staging opinion updates."""
        if ess.knowledge_density == KnowledgeDensity.NONE:
            log.debug(
                "KNOWLEDGE_SKIP density=NONE type=%s score=%.2f",
                ess.reasoning_type,
                ess.score,
            )
            return
        log.debug(
            "KNOWLEDGE_EXTRACT density=%s mode=%s type=%s score=%.2f ep=%s",
            ess.knowledge_density,
            "full" if stage_opinions else "facts-only",
            ess.reasoning_type,
            ess.score,
            episode_uid[:8],
        )
        text = (
            f"User: {user_message}\nAssistant: {agent_response}"
            if agent_response
            else user_message
        )
        try:
            stored = self._run_async(
                extract_and_store_knowledge(
                    text=text,
                    episode_uid=episode_uid,
                    qdrant=self._db.qdrant,
                    embedder=self._embedder,
                    sponge=self.sponge,
                    cooling_period=ess.cooling_period,
                    stage_opinions=stage_opinions,
                )
            )
            if stored:
                log.info("Knowledge extraction stored %d propositions", stored)
            self.last_knowledge_writes = stored
        except Exception:
            log.exception("Knowledge extraction failed")
            self.last_knowledge_writes = 0

    def _store_episode_new_arch(
        self,
        user_message: str,
        agent_response: str,
        ess: ESSResult,
        segment_id: str,
        segment_label: str,
        segment_reasoning: str,
    ) -> str:
        """Store episode in Neo4j + Qdrant dual store. Returns episode UID on success."""
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

        Uses a single batch LLM call for all disagreeing topics (instead of one
        call per topic) to avoid O(N_topics) latency on multi-topic interactions.
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
        # Collect only topics where the user direction conflicts with agent position
        conflicting: list[tuple[str, float]] = []
        for topic in ess.topics:
            committed = self.sponge.opinion_vectors.get(topic, 0.0)
            staged = staged_net.get(topic, 0.0)
            pos = committed + staged
            # Skip when there's no meaningful position, or when user agrees with stance
            if abs(pos) >= 0.05 and (pos * sign <= 0):
                conflicting.append((topic, pos))
        if not conflicting:
            return False
        topics_and_positions = "\n".join(f"- {topic}: {pos:+.2f}" for topic, pos in conflicting)
        prompt = DISAGREEMENT_DETECTION_PROMPT.format(
            user_message=user_message[:500],
            opinion_direction=f"{sign:+.1f}",
            topics_and_positions=topics_and_positions,
        )
        try:
            result = llm_call(
                prompt=prompt,
                response_model=DisagreementDetectionResponse,
                fallback=DisagreementDetectionResponse(),
                max_tokens=256,  # verdict + strength + short reasoning
                assistant_prefix='{"disagreement_verdict": "',
            )
        except Exception:
            return False
        return result.value.disagreement_verdict is DisagreementVerdict.DISAGREEMENT

    def _collect_unresolved_contradictions(self) -> list[str]:
        """Summarize staged deltas that currently oppose strong held beliefs."""
        candidates: list[tuple[float, str]] = []
        for staged in self.sponge.staged_opinion_updates:
            b = self.sponge.get_belief(staged.topic)
            if b.position * staged.signed_magnitude >= 0:
                continue
            summary = (
                f"{staged.topic}({b.position:+.2f} vs {staged.signed_magnitude:+.3f},"
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
    _NO_UPDATE_REASONING: Final = frozenset(
        {
            ReasoningType.NO_ARGUMENT,
            ReasoningType.SOCIAL_PRESSURE,
            ReasoningType.EMOTIONAL_APPEAL,
            ReasoningType.DEBUNKED_CLAIM,
        }
    )

    def _ess_allows_update(
        self,
        ess: ESSResult,
        *,
        update_kind: str,
    ) -> bool:
        """Return whether ESS permits a personality update path.

        Uses the LLM-decided belief_update_recommended field as the primary gate.
        Reasoning types in _NO_UPDATE_REASONING are always blocked as a safety net.
        """
        if ess.reasoning_type in self._NO_UPDATE_REASONING:
            log.debug(
                "Skipping %s: reasoning_type=%s is non-substantive", update_kind, ess.reasoning_type
            )
            return False
        if not ess.belief_update_recommended:
            log.debug(
                "Skipping %s: LLM decided no belief update (score=%.3f type=%s)",
                update_kind,
                ess.score,
                ess.reasoning_type,
            )
            return False
        ess_reliable = ess.default_severity not in {"missing", "exception"} and not any(
            f in CRITICAL_ESS_DEFAULT_FIELDS for f in ess.defaulted_fields
        )
        if not ess_reliable:
            log.debug(
                "Skipping %s due to ESS fallback defaults (severity=%s fields=%s)",
                update_kind,
                ess.default_severity,
                ess.defaulted_fields,
            )
            return False
        return True

    def _stage_topic_opinion_update(
        self,
        *,
        topic: str,
        direction: float,
        magnitude: float,
        provenance: str,
        cooling_period: int,
        episode_uid: str = "",
        new_uncertainty: float = -1.0,
    ) -> None:
        """Stage one topic update and emit a consistent audit event."""
        if magnitude <= 0.0:
            return
        prior_pos = self.sponge.opinion_vectors.get(topic, 0.0)
        due = self.sponge.stage_opinion_update(
            topic=topic,
            direction=direction,
            magnitude=magnitude,
            cooling_period=cooling_period,
            provenance=provenance,
            new_uncertainty=new_uncertainty,
        )
        log.debug(
            "STAGE topic=%s dir=%+.1f mag=%.4f prior_pos=%+.3f due=#%d | %s",
            topic,
            direction,
            magnitude,
            prior_pos,
            due,
            provenance[:80],
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
        if not ess.topics:
            return
        if not self._ess_allows_update(ess, update_kind="provenance opinion update"):
            return
        # Use only the user's message so the LLM assesses the USER's claim, not
        # the agent's rebuttal. The agent's response may debunk the claim but that
        # doesn't mean the agent has a negative belief about the domain itself.
        content = f"User: {user_message}\nESS summary: {ess.summary}\nESS score: {ess.score:.2f}"
        fallback_direction = ess.opinion_direction.sign

        try:
            updates = await assess_belief_evidence_batch(
                topics=list(ess.topics),
                episode_uid=episode_uid,
                episode_content=content,
                ess_score=ess.score,
                reasoning_type=str(ess.reasoning_type),
                source_reliability=str(ess.source_reliability),
                sponge=self.sponge,
                graph=self._graph,
            )
        except Exception:
            log.exception("Batch belief provenance assessment failed")
            updates = []

        for update in updates:
            topic = update.topic
            direction = update.direction if abs(update.direction) > 1e-6 else fallback_direction
            if abs(direction) < 1e-6:
                continue
            if update.contraction_action is ContractionAction.CONTRACT:
                self._apply_llm_contraction(topic, update.evidence_strength)

            belief = self.sponge.get_belief(topic)
            confidence = belief.confidence + (
                abs(belief.position) if belief.position * direction < 0 else 0.0
            )
            raw_mag = max(0.0, min(1.0, update.evidence_strength)) / (confidence + 1.0)
            effective_mag = min(raw_mag, MAX_SINGLE_UPDATE_MAGNITUDE)
            log.debug(
                "BELIEF_MAG topic=%s raw=%.3f effective=%.3f dir=%+.1f "
                "evidence_strength=%.3f confidence=%.3f type=%s",
                topic,
                raw_mag,
                effective_mag,
                direction,
                update.evidence_strength,
                confidence,
                ess.reasoning_type,
            )
            self._stage_topic_opinion_update(
                topic=topic,
                direction=direction,
                magnitude=effective_mag,
                provenance=f"{update.reasoning[:120]} (ep={episode_uid[:8]})",
                cooling_period=ess.cooling_period,
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
        allows = self._ess_allows_update(ess, update_kind="insight")
        log.debug(
            "INSIGHT_GATE allows=%s type=%s score=%.2f pending=%d",
            allows,
            ess.reasoning_type,
            ess.score,
            len(self.sponge.pending_insights),
        )
        if not allows:
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
        for topic, _ in opinions:
            b = self.sponge.get_belief(topic)
            conf = f" c={b.confidence:.1f}" if b.has_evidence else ""
            opinions_parts.append(f"{topic}={b.position:+.2f}{conf}")
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
    # Cumulative belief shift threshold that bypasses the normal minimum window.
    # When recent shifts exceed this magnitude, reflection fires immediately
    # (with a reduced minimum of 2 interactions to avoid reflecting on first message).
    _CRITICAL_SHIFT_THRESHOLD: float = 0.8
    _CRITICAL_SHIFT_MIN_WINDOW: int = 2

    def _reflection_gate(self) -> ReflectionGate:
        """Determine whether reflection should run for this interaction."""
        window_interactions = self.sponge.interaction_count - self.sponge.last_reflection_at
        recent_mag = sum(
            shift.magnitude
            for shift in self.sponge.recent_shifts
            if shift.interaction > self.sponge.last_reflection_at
        )

        # Critical shift bypass: major belief changes trigger immediate reflection.
        # This ensures correlation discovery and belief reconciliation happen
        # promptly after significant worldview updates.
        if (
            recent_mag >= self._CRITICAL_SHIFT_THRESHOLD
            and window_interactions >= self._CRITICAL_SHIFT_MIN_WINDOW
        ):
            log.info(
                "Critical shift reflection: mag=%.3f >= threshold=%.2f, window=%d",
                recent_mag,
                self._CRITICAL_SHIFT_THRESHOLD,
                window_interactions,
            )
            return ReflectionGate(
                trigger=ReflectionTrigger.EVENT_DRIVEN,
                trigger_label=f"critical_shift (mag={recent_mag:.3f})",
                window_interactions=window_interactions,
            )

        # Hard minimum window: never reflect before 5 interactions have accumulated
        # (unless critical shift bypass above).
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
                window_interactions,
                config.REFLECTION_EVERY,
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
            max_tokens=256,  # SKIP / PERIODIC / EVENT_DRIVEN + reasoning (128 too low, truncates)
            assistant_prefix='{"trigger": "',
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
            log.debug(
                "Reflection snapshot truncated to %d chars (limit %d)", char_limit, char_limit
            )
        if len(reflected_snapshot) < 30:
            log.warning(
                "Reflection output rejected: snapshot too short (%d chars)", len(reflected_snapshot)
            )
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
        """Batch LLM staleness assessment to retain, decay, or forget beliefs."""
        stale_candidates = sorted(
            [
                (self.sponge.interaction_count - meta.last_reinforced, topic, meta)
                for topic, meta in self.sponge.belief_meta.items()
                if self.sponge.interaction_count - meta.last_reinforced >= 5
            ],
            reverse=True,
        )[:20]
        if not stale_candidates:
            return []

        beliefs_json = json.dumps(
            [
                {
                    "topic": topic,
                    "position": f"{self.sponge.opinion_vectors.get(topic, 0.0):+.2f}",
                    "confidence": f"{meta.confidence:.2f}",
                    "evidence_count": meta.evidence_count,
                    "gap": gap,
                }
                for gap, topic, meta in stale_candidates
            ]
        )
        result = llm_call(
            prompt=BATCH_BELIEF_DECAY_PROMPT.format(
                total_interactions=self.sponge.interaction_count,
                beliefs_json=beliefs_json,
            ),
            response_model=BatchBeliefDecayResponse,
            fallback=BatchBeliefDecayResponse(decisions=[]),
            max_tokens=1024,  # 20 decisions × ~50 tokens each
            assistant_prefix='{"decisions": [',
        )
        dropped: list[str] = []
        decisions_by_topic = {d.topic: d for d in result.value.decisions}
        for _, topic, meta in stale_candidates:
            decision = decisions_by_topic.get(topic)
            if decision is None:
                continue
            if decision.action is BeliefDecayAction.FORGET:
                dropped.append(topic)
                del self.sponge.belief_meta[topic]
                self.sponge.opinion_vectors.pop(topic, None)
            elif decision.action is BeliefDecayAction.DECAY:
                new_conf = decision.new_confidence if decision.new_confidence is not None else meta.confidence * 0.8
                meta.confidence = max(0.0, min(1.0, new_conf))
                meta.uncertainty = 1.0 - meta.confidence
                log.debug(
                    "Belief decay: %s confidence → %.2f (%s)",
                    topic,
                    meta.confidence,
                    decision.reasoning[:60],
                )
        return dropped

    def _detect_entrenched_beliefs_llm(self, min_updates: int = 4) -> list[str]:
        """Batch LLM entrenchment detection (cached per turn)."""
        if self._last_entrenched_interaction == self.sponge.interaction_count:
            return list(self._last_entrenched)
        candidates = [
            (topic, meta, self.sponge.opinion_vectors.get(topic, 0.0))
            for topic, meta in self.sponge.belief_meta.items()
            if len(meta.recent_updates) >= min_updates
        ][:10]
        if not candidates:
            self._last_entrenched = []
            self._last_entrenched_interaction = self.sponge.interaction_count
            return []
        beliefs_json = json.dumps(
            [
                {
                    "topic": topic,
                    "position": f"{position:+.2f}",
                    "recent_updates": [f"{u:+.3f}" for u in meta.recent_updates[-8:]],
                    "supporting_count": len(meta.supporting_episode_uids),
                    "contradicting_count": len(meta.contradicting_episode_uids),
                }
                for topic, meta, position in candidates
            ]
        )
        result = llm_call(
            prompt=BATCH_ENTRENCHMENT_DETECTION_PROMPT.format(beliefs_json=beliefs_json),
            response_model=BatchEntrenchmentResponse,
            fallback=BatchEntrenchmentResponse(entrenched=[]),
            assistant_prefix='{"entrenched": [',
        )
        valid_topics = {topic for topic, _, _ in candidates}
        entrenched = [d.topic for d in result.value.entrenched if d.topic in valid_topics]
        self._last_entrenched = entrenched
        self._last_entrenched_interaction = self.sponge.interaction_count
        return entrenched

    def _maybe_reflect(self) -> None:
        """Run periodic or event-driven reflection and snapshot consolidation.

        Reflection is deliberately sparse; over-frequent rewrites increase drift
        and can erase minority traits from the narrative snapshot.
        """
        recent_mag = sum(
            shift.magnitude
            for shift in self.sponge.recent_shifts
            if shift.interaction > self.sponge.last_reflection_at
        )
        gate = self._reflection_gate()
        log.info(
            "REFLECT_GATE trigger=%s label=%s window=%d cadence=%d min=%d "
            "pending_insights=%d staged=%d beliefs=%d recent_mag=%.3f",
            gate.trigger,
            gate.trigger_label,
            gate.window_interactions,
            config.REFLECTION_EVERY,
            self._MIN_WINDOW_FOR_EVENT_DRIVEN,
            len(self.sponge.pending_insights),
            len(self.sponge.staged_opinion_updates),
            len(self.sponge.opinion_vectors),
            recent_mag,
        )
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

        # Knowledge consolidation: review for contradictions and merges in Qdrant
        try:
            result = self._run_async(
                consolidate_knowledge(qdrant=self._db.qdrant, snapshot=self.sponge.snapshot)
            )
            if result.contradictions or result.merges:
                log.info(
                    "Knowledge consolidation: %d contradictions, %d merges",
                    len(result.contradictions),
                    len(result.merges),
                )
        except Exception:
            log.exception("Knowledge consolidation failed")

        # Prune low-confidence stale knowledge entries from Qdrant
        try:
            self._run_async(prune_stale_knowledge(self._db.qdrant))
        except Exception:
            log.debug("Knowledge pruning failed", exc_info=True)

        # Fetch recent episodes BEFORE forgetting so the snapshot has context to work with.
        try:
            recent_episodes = (
                self._run_async(
                    self._graph.list_recent_episode_context(min(config.REFLECTION_EVERY, 10))
                )
                or []
            )
        except Exception:
            log.debug("Reflection episode retrieval failed", exc_info=True)
            recent_episodes = []
        if not recent_episodes:
            log.info("No episodes for reflection, skipping")
            self.sponge.last_reflection_at = self.sponge.interaction_count
            return

        # Forgetting: assess and archive low-importance episodes (after fetching context above)
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

        # Consistency check: clean derivative orphans across Neo4j and Qdrant
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

        prompt = self._reflection_prompt(gate.trigger_label, recent_episodes)

        try:
            pre_snapshot = self.sponge.snapshot
            completion = chat_completion(
                model=self.ess_model,
                max_tokens=config.FAST_LLM_MAX_TOKENS,
                messages=({"role": "user", "content": prompt},),
                enable_thinking=False,
            )
            reflected_snapshot = completion.text.strip()
            self._apply_reflection_snapshot(
                pre_snapshot, reflected_snapshot, opinions_before_reflection
            )
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
                        self._db.qdrant,
                        neo_sess,
                        self.sponge,
                        label=label,
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

    def _log_health_event(self) -> None:
        """Write lightweight health diagnostics for drift/sycophancy/saturation detection."""
        words = self.sponge.snapshot.split()
        unique_ratio = len(set(w.lower() for w in words)) / len(words) if words else 0.0
        metas = list(self.sponge.belief_meta.items())
        high_conf = sum(1 for _, m in metas if m.confidence > 0.5)
        high_conf_ratio = high_conf / len(metas) if metas else 0.0
        disagreement = self.sponge.behavioral_signature.disagreement_rate

        warnings: list[str] = []

        # Belief saturation: beliefs pinned at extremes can't absorb new evidence correctly.
        saturated = [
            t for t, _ in metas if abs(self.sponge.opinion_vectors.get(t, 0.0)) >= 0.95
        ]
        if saturated:
            warnings.append(f"saturated_beliefs:{','.join(saturated[:5])}")
            log.warning(
                "BELIEF_SATURATION %d beliefs pinned at ±0.95+: %s | "
                "these topics resist new evidence — consider decay or manual review",
                len(saturated),
                saturated[:5],
            )

        # Sycophancy indicator: all beliefs positive, no negative opinions formed.
        opinions = self.sponge.opinion_vectors
        if len(opinions) >= 5:
            pos_count = sum(1 for v in opinions.values() if v > 0.1)
            neg_count = sum(1 for v in opinions.values() if v < -0.1)
            if neg_count == 0 and pos_count >= 5:
                warnings.append("all_positive_beliefs")
                log.warning(
                    "SYCOPHANCY_RISK all %d tracked beliefs are positive (neg=0) — "
                    "agent may not be forming critical/opposing views",
                    pos_count,
                )
            elif pos_count > 0 and neg_count > 0:
                log.debug(
                    "BELIEF_BALANCE pos=%d neg=%d neutral=%d",
                    pos_count,
                    neg_count,
                    len(opinions) - pos_count - neg_count,
                )

        # Stale-but-alive: beliefs with no recent reinforcement (last > 10 interactions ago)
        # that are still at extreme positions — they should have decayed by now.
        stale_extreme = [
            t
            for t, m in metas
            if self.sponge.interaction_count - m.last_reinforced > 10
            and abs(self.sponge.opinion_vectors.get(t, 0.0)) > 0.7
        ]
        if stale_extreme:
            log.warning(
                "STALE_EXTREME_BELIEFS %d beliefs unreinforced >10 interactions but still extreme: %s",
                len(stale_extreme),
                stale_extreme[:5],
            )
            warnings.append(f"stale_extreme:{','.join(stale_extreme[:3])}")

        # Use cached result from this turn's reflection call to avoid extra LLM call.
        entrenched = list(self._last_entrenched)
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
                "saturated_beliefs": saturated,
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
