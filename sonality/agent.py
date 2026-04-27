"""Sonality agent: stateless, graph-backed personality with LLM-only decisions.

Each request starts from zero in-memory state. Identity (personality snapshot +
beliefs) is loaded from Neo4j per request. Conversation context is managed by
the caller (chat client / API). Reflection runs on every turn (two-tier:
cheap triage → deep update only when warranted).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import threading
import time
from collections.abc import Coroutine, Iterator
from concurrent.futures import Future

from pydantic import BaseModel, Field

from . import config
from .ess import ESSResult, KnowledgeDensity, classifier_exception_fallback, classify
from .llm.caller import llm_call
from .memory import (
    BoundaryDecision,
    DatabaseConnections,
    DualEpisodeStore,
    Embedder,
    EventBoundaryDetector,
    MemoryGraph,
    QueryCategory,
    SemanticIngestionWorker,
    SemanticMemoryDecision,
    StoredEpisode,
    TemporalExpansionDecision,
    assess_belief_evidence_batch,
    chain_retrieve,
    extract_and_store_knowledge,
    rerank_episodes,
    retrieve_relevant_knowledge,
    route_query,
    split_retrieve,
)
from .memory.consolidation import maybe_consolidate_segment
from .memory.forgetting import assess_and_forget
from .memory.graph import (
    BELIEF_PROMPT_WINDOW,
    BeliefNode,
    PersonalitySnapshot,
    format_beliefs_for_prompt_from_nodes,
    format_episode_line,
)
from .prompts import (
    REFLECTION_DEEP_PROMPT,
    REFLECTION_TRIAGE_PROMPT,
    build_system_prompt,
)
from .provider import (
    ChatResult,
    StreamChunk,
    default_provider,
    interaction_active,
    strip_thinking_trace,
)
from .request_identity import (
    IdentityBundle,
    get_request_identity,
    reset_request_identity,
    set_request_identity,
)
from .schema import DENSE_VECTOR, ChatRole, Collection, SemanticCategory
from .token_budget import message_tokens_budget_for_system, trim_chat_messages_for_budget

log = logging.getLogger(__name__)


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

        log.info("SonalityAgent ready (model=%s, ess=%s)", self.model, self.ess_model)

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
    ) -> str:
        """Generate a response given the full conversation history from the caller."""
        with interaction_active():
            return self._respond_inner(
                messages,
                max_tokens=max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
                temperature=temperature if temperature is not None else config.AGENT_TEMPERATURE,
            )

    def respond_stream(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Iterator[StreamChunk]:
        """Streaming response variant."""
        with interaction_active():
            yield from self._respond_stream_inner(
                messages,
                max_tokens=max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS,
                temperature=temperature if temperature is not None else config.AGENT_TEMPERATURE,
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
                self._extract_knowledge(text, "", ess, episode_uid)
                self._assess_provenance(list(ess.topics), episode_uid, text, ess)
                self._semantic_worker.enqueue(
                    episode_uid,
                    f"Content: {text}\nESS: {ess.score:.2f} ({ess.reasoning_type})",
                    categories=(SemanticCategory.KNOWLEDGE,),
                )
                self._reflect(text, "", ess, episode_uid)

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

    def _build_context(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        """Shared setup for respond paths. Returns (user_message, system_prompt)."""
        user_message = self._last_user_message(messages)
        bundle = get_request_identity()
        if bundle is None:
            bundle = self._run_async(self._fetch_identity_bundle())
        snapshot_text = bundle.snapshot_text
        beliefs_text = bundle.beliefs_prompt_text
        try:
            relevant = self._run_async(self._retrieve(user_message))
        except Exception:
            log.exception("Retrieval failed")
            relevant = []
        try:
            knowledge = self._run_async(
                retrieve_relevant_knowledge(user_message, self._db.qdrant, self._embedder)
            )
        except Exception:
            log.warning(
                "Knowledge retrieval failed; continuing without knowledge context", exc_info=True
            )
            knowledge = []
        log.info(
            "Context: snapshot_chars=%d belief_lines=%d episodes=%d knowledge=%d",
            len(snapshot_text),
            len(beliefs_text.splitlines()),
            len(relevant),
            len(knowledge),
        )
        system_prompt = build_system_prompt(
            snapshot_text=snapshot_text,
            beliefs_text=beliefs_text,
            relevant_episodes=relevant,
            knowledge_context=knowledge,
        )
        return user_message, system_prompt

    def _respond_inner(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
    ) -> str:
        _t0 = time.perf_counter()
        id_token = None
        try:
            bundle = self._run_async(self._fetch_identity_bundle())
            id_token = set_request_identity(bundle)
            user_message, system_prompt = self._build_context(messages)
            log.info("User: %.120s", user_message)
            conv = trim_chat_messages_for_budget(
                list(messages),
                max_message_tokens=message_tokens_budget_for_system(
                    total_budget=config.CHAT_INPUT_TOKEN_BUDGET,
                    system_prompt=system_prompt,
                    reserve_completion=max_tokens,
                ),
            )

            for attempt in range(1, 4):
                try:
                    completion = default_provider.chat_completion(
                        model=self.model,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        messages=({"role": ChatRole.SYSTEM, "content": system_prompt}, *conv),
                        enable_thinking=False,
                    )
                    break
                except RuntimeError as exc:
                    if attempt < 3:
                        log.warning("LLM failed (attempt %d/3): %s", attempt, exc)
                        continue
                    log.error("LLM failed after 3 attempts: %s", exc)
                    completion = ChatResult(text="", input_tokens=0, output_tokens=0, raw={})
                    break

            assistant_msg = completion.text
            log.info("Agent: %.200s", assistant_msg)
            log.info(
                "LLM tokens: input=%d output=%d",
                completion.input_tokens,
                completion.output_tokens,
            )
            self._post_process(user_message, assistant_msg)
            log.info("Total: %.1fs", time.perf_counter() - _t0)
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
    ) -> Iterator[StreamChunk]:
        id_token = None
        try:
            bundle = self._run_async(self._fetch_identity_bundle())
            id_token = set_request_identity(bundle)
            user_message, system_prompt = self._build_context(messages)
            log.info("User (stream): %.120s", user_message)
            conv = trim_chat_messages_for_budget(
                list(messages),
                max_message_tokens=message_tokens_budget_for_system(
                    total_budget=config.CHAT_INPUT_TOKEN_BUDGET,
                    system_prompt=system_prompt,
                    reserve_completion=max_tokens,
                ),
            )

            chunks: list[str] = []
            for content, reasoning in default_provider.chat_completion_stream(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=({"role": ChatRole.SYSTEM, "content": system_prompt}, *conv),
                enable_thinking=False,
            ):
                if content:
                    chunks.append(content)
                yield StreamChunk(content, reasoning)

            assistant_msg = strip_thinking_trace("".join(chunks))
            log.info("Agent: %.200s", assistant_msg)
            self._post_process(user_message, assistant_msg)
        finally:
            if id_token is not None:
                reset_request_identity(id_token)

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
        """Full retrieval pipeline: route → search → expand → rerank."""
        if not self._dual_store.has_episodes:
            return []

        decision = await asyncio.to_thread(route_query, user_message)
        log.debug("Route: category=%s n=%d", decision.category, decision.n_results)
        if decision.category == QueryCategory.NONE:
            return []

        if decision.category == QueryCategory.MULTI_ENTITY:
            word_count = len(user_message.split())
            if word_count < config.RETRIEVAL_DECOMPOSE_MIN_WORDS:
                over_fetch = decision.n_results * config.RETRIEVAL_OVER_FETCH_FACTOR
                results = await self._dual_store.vector_search(user_message, top_k=over_fetch)
                episode_uids = list({h.episode_uid for h in results})
                topic_hits = await self._graph.find_topic_related_episodes(
                    user_message, limit=max(2, over_fetch // 2)
                )
                episodes = topic_hits + await self._graph.get_episodes(episode_uids)
            else:
                episodes = await split_retrieve(
                    self._dual_store, self._graph, user_message, n_per_sub=decision.n_results
                )
        elif decision.category in (QueryCategory.TEMPORAL, QueryCategory.AGGREGATION):
            episodes = await chain_retrieve(
                self._dual_store, self._graph, user_message, base_n=decision.n_results
            )
        elif decision.category == QueryCategory.BELIEF_QUERY:
            over_fetch = decision.n_results * config.RETRIEVAL_OVER_FETCH_FACTOR
            belief_hits = await self._graph.find_belief_related_episodes(
                user_message, limit=over_fetch
            )
            topic_hits = await self._graph.find_topic_related_episodes(
                user_message, limit=max(2, over_fetch // 2)
            )
            vector_hits = await self._dual_store.vector_search(user_message, top_k=over_fetch)
            vector_uids = list({h.episode_uid for h in vector_hits})
            episodes = belief_hits + topic_hits + await self._graph.get_episodes(vector_uids)
        else:
            over_fetch = decision.n_results * config.RETRIEVAL_OVER_FETCH_FACTOR
            results = await self._dual_store.vector_search(user_message, top_k=over_fetch)
            episode_uids = list({h.episode_uid for h in results})
            topic_hits = await self._graph.find_topic_related_episodes(
                user_message, limit=max(2, over_fetch // 2)
            )
            episodes = topic_hits + await self._graph.get_episodes(episode_uids)

        episodes = list({ep.uid: ep for ep in episodes}.values())

        if decision.temporal_expansion is TemporalExpansionDecision.EXPAND and episodes:
            expanded_uids: set[str] = set()
            for ep in episodes[:3]:
                for n in await self._graph.traverse_temporal_context(ep.uid):
                    expanded_uids.add(n.uid)
            new_uids = [u for u in expanded_uids if u not in {e.uid for e in episodes}]
            if new_uids:
                episodes.extend(await self._graph.get_episodes(new_uids))

        need_rerank = len(episodes) > 1 and len(episodes) > decision.n_results
        if need_rerank:
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
            "Retrieval: category=%s episodes=%d semantic=%d (deduped=%d reranked=%d)",
            decision.category,
            len(selected),
            len(semantic_context),
            len(episodes),
            need_rerank,
        )
        return [*episode_context, *semantic_context]

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

    # --- Post-processing ---

    def _post_process(self, user_message: str, agent_response: str) -> None:
        """Classify, store episode, extract knowledge, assess provenance, reflect."""
        try:
            ess = self._classify_ess(user_message)
        except Exception:
            log.exception("ESS classification failed")
            ess = classifier_exception_fallback(user_message)
        self.last_ess = ess

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
            self._extract_knowledge(user_message, agent_response, ess, episode_uid)
            if ess.belief_update_recommended:
                self._assess_provenance(list(ess.topics), episode_uid, user_message, ess)
            self._semantic_worker.enqueue(
                episode_uid,
                f"User: {user_message}\nAssistant: {agent_response}\n"
                f"ESS: {ess.score:.2f} ({ess.reasoning_type})",
            )

        self._reflect(user_message, agent_response, ess, episode_uid)

    def _classify_ess(self, user_message: str) -> ESSResult:
        """Run ESS classification. LLM-only, no code-based overrides."""
        snapshot = self._run_async(self._graph.get_personality_snapshot())
        all_beliefs = self._run_async(self._graph.get_all_beliefs())
        tracked_topics = ", ".join(b.topic for b in all_beliefs[:20]) or "(none yet)"
        return classify(
            user_message=user_message,
            snapshot_text=snapshot.text,
            tracked_topics=tracked_topics,
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
                    segment_id=segment_id,
                    segment_label=segment_label,
                )
            )
            return stored.episode_uid
        except Exception:
            log.exception("Episode storage failed")
            return ""

    def _extract_knowledge(
        self, user_msg: str, agent_resp: str, ess: ESSResult, episode_uid: str
    ) -> None:
        if ess.knowledge_density == KnowledgeDensity.NONE:
            return
        text = f"User: {user_msg}\nAssistant: {agent_resp}" if agent_resp else user_msg
        try:
            stored = self._run_async(
                extract_and_store_knowledge(
                    text=text,
                    episode_uid=episode_uid,
                    qdrant=self._db.qdrant,
                    embedder=self._embedder,
                )
            )
            if stored:
                log.info("Knowledge: stored %d propositions", stored)
        except Exception:
            log.exception("Knowledge extraction failed")

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

    # --- Two-tier reflection ---

    def _reflect(self, user_msg: str, agent_resp: str, ess: ESSResult, episode_uid: str) -> None:
        """Tier 1: triage (should we reflect?). Tier 2: deep reflection if yes."""
        bundle = get_request_identity()
        if bundle is not None:
            snapshot_text = bundle.snapshot_text
            belief_nodes = list(bundle.all_beliefs[:BELIEF_PROMPT_WINDOW])
        else:
            snapshot_text = self._run_async(self._graph.get_personality_snapshot()).text
            belief_nodes = self._run_async(self._graph.get_top_beliefs())
        beliefs_text = (
            "\n".join(
                f"{b.topic}: valence={b.valence:+.2f}, confidence={b.confidence:.2f}"
                for b in belief_nodes
            )
            or "(no beliefs yet)"
        )

        triage_prompt = REFLECTION_TRIAGE_PROMPT.format(
            beliefs=beliefs_text,
            user_message=user_msg[: config.REFLECTION_USER_SLICE],
            agent_response=(
                agent_resp[: config.REFLECTION_USER_SLICE]
                if agent_resp
                else "(ingest, no response)"
            ),
            ess_score=ess.score,
            reasoning_type=ess.reasoning_type,
            topics=list(ess.topics),
        )
        triage_result = llm_call(
            prompt=triage_prompt,
            response_model=_TriageResponse,
            fallback=_TriageResponse(should_reflect=False, reason="triage failed"),
            max_tokens=config.STRUCTURED_JSON_MAX_TOKENS,
            max_retries=1,
        )
        if not triage_result.success or not triage_result.value.should_reflect:
            log.debug(
                "Reflection triage: skip (%s)",
                triage_result.value.reason if triage_result.success else "error",
            )
            return

        log.info("Reflection triage: YES (%s)", triage_result.value.reason)

        episodes = self._run_async(self._graph.list_recent_episode_context(10))
        deep_prompt = REFLECTION_DEEP_PROMPT.format(
            snapshot=snapshot_text,
            beliefs=beliefs_text,
            episode_count=len(episodes),
            episodes="\n".join(episodes) or "(no recent episodes)",
            user_message=user_msg[: config.REFLECTION_AGENT_SLICE],
            agent_response=(
                agent_resp[: config.REFLECTION_AGENT_SLICE]
                if agent_resp
                else "(ingest, no response)"
            ),
        )
        deep_result = llm_call(
            prompt=deep_prompt,
            response_model=_DeepReflectionResponse,
            fallback=_DeepReflectionResponse(),
            max_tokens=config.EXTRACTION_MAX_TOKENS,
            max_retries=1,
        )
        if not deep_result.success:
            log.warning("Deep reflection failed: %s", deep_result.error)
            return

        reflection = deep_result.value
        log.info(
            "Reflection deep: updates=%d new_beliefs=%d snapshot_changed=%s",
            len(reflection.belief_updates),
            len(reflection.new_beliefs),
            reflection.snapshot_changed,
        )
        self._apply_reflection(reflection, episode_uid)

    def _apply_reflection(self, reflection: _DeepReflectionResponse, episode_uid: str) -> None:
        """Write belief updates and snapshot revision to graph."""
        all_updates = [
            *((b, f"reflection:{episode_uid[:8]}") for b in reflection.belief_updates),
            *((b, f"new_belief:{episode_uid[:8]}") for b in reflection.new_beliefs),
        ]
        for patch, provenance in all_updates:
            if not patch.topic:
                continue
            try:
                self._run_async(
                    self._graph.upsert_belief(
                        patch.topic,
                        valence=patch.valence,
                        confidence=patch.confidence,
                        belief_text=patch.belief_text,
                        provenance=provenance,
                    )
                )
                log.info(
                    "Belief %s: %s val=%+.2f conf=%.2f reason=%.100s",
                    provenance.split(":")[0],
                    patch.topic,
                    patch.valence,
                    patch.confidence,
                    patch.reasoning,
                )
            except Exception:
                log.exception("Failed to upsert belief: %s", patch.topic)

        if reflection.snapshot_changed and reflection.snapshot_revision:
            text = reflection.snapshot_revision[:2000]
            try:
                self._run_async(self._graph.upsert_personality_snapshot(text))
                log.info("Personality snapshot updated (%d chars)", len(text))
            except Exception:
                log.exception("Failed to update personality snapshot")

        try:
            candidates = self._run_async(
                self._graph.get_forgetting_candidates(limit=config.FORGETTING_CANDIDATE_LIMIT)
            )
            if candidates:
                snapshot = self._run_async(self._graph.get_personality_snapshot())
                self._run_async(
                    assess_and_forget(
                        candidates,
                        self._graph,
                        self._dual_store,
                        snapshot_excerpt=snapshot.text[:500],
                    )
                )
        except Exception:
            log.warning("Forgetting cycle skipped", exc_info=True)

    # --- Helpers ---

    @staticmethod
    def _last_user_message(messages: list[dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == ChatRole.USER:
                return msg.get("content", "")
        return ""


class _TriageResponse(BaseModel):
    should_reflect: bool = False
    reason: str = ""


class _BeliefPatch(BaseModel):
    """Single belief create/update from LLM reflection."""

    topic: str = ""
    valence: float = 0.0
    confidence: float = 0.5
    belief_text: str = ""
    reasoning: str = ""


class _DeepReflectionResponse(BaseModel):
    belief_updates: list[_BeliefPatch] = Field(default_factory=list)
    new_beliefs: list[_BeliefPatch] = Field(default_factory=list)
    snapshot_revision: str = ""
    snapshot_changed: bool = False
