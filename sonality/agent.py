from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from anthropic import Anthropic, APIError

from . import config
from .ess import (
    ESSResult,
    ReasoningType,
    SourceReliability,
    classifier_exception_fallback,
    classify,
)
from .memory import (
    AdmissionPolicy,
    EpisodeStore,
    MemoryType,
    ProvenanceQuality,
    SpongeState,
    compute_magnitude,
    extract_insight,
    validate_snapshot,
)
from .openrouter import chat_completion
from .prompts import REFLECTION_PROMPT, build_system_prompt

log = logging.getLogger(__name__)

MAX_RETRIES: Final = 3
RETRY_BACKOFF: Final = 1.5
TRUSTED_REASONING: Final[frozenset[ReasoningType]] = frozenset(
    {
        ReasoningType.LOGICAL_ARGUMENT,
        ReasoningType.EMPIRICAL_DATA,
        ReasoningType.EXPERT_OPINION,
    }
)
TRUSTED_SOURCES: Final[frozenset[SourceReliability]] = frozenset(
    {
        SourceReliability.PEER_REVIEWED,
        SourceReliability.ESTABLISHED_EXPERT,
        SourceReliability.INFORMED_OPINION,
    }
)
AGM_CONTRACTION_SCORE: Final = 0.65
AGM_CONTRACTION_CONFIDENCE: Final = 0.55
AGM_CONTRACTION_POSITION: Final = 0.45
AGM_CONTRACTION_RATIO: Final = 0.35
SEMANTIC_MEMORY_MIN_ESS: Final = 0.55
CONTRADICTION_POSITION_THRESHOLD: Final = 0.35
COERCION_UPDATE_MARGIN: Final = 0.1
CRITICAL_ESS_DEFAULT_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "coerced:score",
        "coerced:reasoning_type",
        "coerced:opinion_direction",
    }
)


class ReflectionTrigger(StrEnum):
    """Reflection execution mode determined by interaction dynamics."""

    SKIP = "skip"
    PERIODIC = "periodic"
    EVENT_DRIVEN = "event_driven"


@dataclass(frozen=True, slots=True)
class ReflectionGate:
    """Reflection gate decision carrying trigger metadata for one turn."""

    trigger: ReflectionTrigger
    trigger_label: str
    window_interactions: int


def _status_code(exc: APIError) -> int | None:
    """Extract a numeric HTTP status code when available."""
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _extract_text_block(response: object) -> str:
    """Pull the first text payload from an Anthropic response object."""
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return ""
    fallback = ""
    for block in content:
        text = getattr(block, "text", "")
        if not isinstance(text, str):
            continue
        if getattr(block, "type", None) == "text":
            return text
        if not fallback:
            fallback = text
    return fallback


def _to_nonnegative_int(value: object) -> int:
    """Convert mixed numeric values to a non-negative integer token count."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(value, 0)
    if isinstance(value, float):
        return max(int(value), 0)
    return 0


def _extract_usage_tokens(response: object) -> tuple[int, int]:
    """Extract request/response token counts from model usage metadata."""
    usage = getattr(response, "usage", None)
    input_tokens = _to_nonnegative_int(getattr(usage, "input_tokens", 0))
    output_tokens = _to_nonnegative_int(getattr(usage, "output_tokens", 0))
    return input_tokens, output_tokens


def _is_trusted_evidence(ess: ESSResult) -> bool:
    """Return whether evidence clears trusted reasoning/source consistency gates."""
    return (
        ess.internal_consistency
        and ess.reasoning_type in TRUSTED_REASONING
        and ess.source_reliability in TRUSTED_SOURCES
    )


@dataclass(frozen=True, slots=True)
class ModelUsage:
    response_calls: int = 0
    ess_calls: int = 0
    response_input_tokens: int = 0
    response_output_tokens: int = 0
    ess_input_tokens: int = 0
    ess_output_tokens: int = 0


def _api_call_with_retry[T](fn: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Retry transient (5xx) API failures with exponential backoff."""
    for attempt in range(MAX_RETRIES):
        try:
            return fn(*args, **kwargs)
        except APIError as exc:
            status = _status_code(exc)
            if status is not None and status >= 500 and attempt < MAX_RETRIES - 1:
                wait = RETRY_BACKOFF ** (attempt + 1)
                log.warning(
                    "API error %s on attempt %d/%d, retrying in %.1fs",
                    status,
                    attempt + 1,
                    MAX_RETRIES,
                    wait,
                )
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Exhausted retries without success")


class SonalityAgent:
    model: str = config.MODEL
    ess_model: str = config.ESS_MODEL

    def __init__(
        self,
        model: str = config.MODEL,
        ess_model: str = config.ESS_MODEL,
    ) -> None:
        """Boot the runtime agent and load persistent memory state.

        Assumes explicit API config (`SONALITY_API_KEY`, `SONALITY_API_VARIANT`).
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
            log.warning(
                "Main and ESS models are identical; using a separate ESS model reduces self-judge coupling"
            )
        self.client = (
            None
            if config.API_VARIANT == "openrouter"
            else Anthropic(**config.anthropic_client_kwargs())
        )
        self.sponge = SpongeState.load(config.SPONGE_FILE)
        self.episodes = EpisodeStore(str(config.CHROMADB_DIR))
        self.conversation: list[dict[str, str]] = []
        self.last_ess: ESSResult | None = None
        self.last_usage = ModelUsage()
        self.previous_snapshot: str | None = None
        log.info(
            "Agent ready: sponge v%d, %d prior interactions, %d beliefs",
            self.sponge.version,
            self.sponge.interaction_count,
            len(self.sponge.opinion_vectors),
        )

    def respond(self, user_message: str) -> str:
        """Run one interaction turn and persist resulting personality state.

        This is the canonical orchestration entrypoint used by the CLI.
        """
        log.info("=== Interaction #%d ===", self.sponge.interaction_count + 1)
        log.info("User: %.120s", user_message)

        relevant = self.episodes.retrieve_typed(
            query=user_message,
            episodic_n=config.EPISODIC_RETRIEVAL_COUNT,
            semantic_n=config.SEMANTIC_RETRIEVAL_COUNT,
        )
        structured_traits = self._build_structured_traits()

        system_prompt = build_system_prompt(
            sponge_snapshot=self.sponge.snapshot,
            relevant_episodes=relevant,
            structured_traits=structured_traits,
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

        if self.client is None:
            completion = chat_completion(
                model=self.model,
                max_tokens=2048,
                messages=(
                    {"role": "system", "content": system_prompt},
                    *self.conversation,
                ),
            )
            response_input_tokens = completion.input_tokens
            response_output_tokens = completion.output_tokens
            assistant_msg = completion.text
        else:
            response = _api_call_with_retry(
                self.client.messages.create,
                model=self.model,
                max_tokens=2048,
                system=system_prompt,
                messages=self.conversation,
            )
            response_input_tokens, response_output_tokens = _extract_usage_tokens(response)
            assistant_msg = _extract_text_block(response)
        if not assistant_msg:
            log.warning("Model response contained no text block; using empty reply")
        self.conversation.append({"role": "assistant", "content": assistant_msg})

        self._post_process(user_message, assistant_msg)
        last_ess = self.last_ess
        self.last_usage = ModelUsage(
            response_calls=1,
            ess_calls=last_ess.attempt_count if last_ess else 0,
            response_input_tokens=response_input_tokens,
            response_output_tokens=response_output_tokens,
            ess_input_tokens=last_ess.input_tokens if last_ess else 0,
            ess_output_tokens=last_ess.output_tokens if last_ess else 0,
        )
        self._log_event(
            {
                "event": "model_usage",
                "interaction": self.sponge.interaction_count,
                **asdict(self.last_usage),
            }
        )
        return assistant_msg

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

        self._store_episode(user_message, agent_response, ess)
        self.sponge.interaction_count += 1
        committed = self.sponge.apply_due_staged_updates()
        if committed:
            log.info("Committed staged beliefs: %s", committed)
            self._log_event(
                {
                    "event": "opinion_commit",
                    "interaction": self.sponge.interaction_count,
                    "committed": committed,
                    "remaining_staged": len(self.sponge.staged_opinion_updates),
                }
            )

        self._update_topics(ess)
        self._update_opinions(ess)
        if self._detect_disagreement(ess):
            self.sponge.note_disagreement()
        else:
            self.sponge.note_agreement()

        self.previous_snapshot = self.sponge.snapshot
        self._extract_insight(user_message, agent_response, ess)
        self._maybe_reflect()
        self._log_health_event()

        self.sponge.save(config.SPONGE_FILE, config.SPONGE_HISTORY_DIR)
        self._log_interaction_summary(ess)

    def _classify_ess(self, user_message: str) -> ESSResult:
        """Classify user evidence and fallback safely on classifier failures."""
        try:
            return classify(
                self.client,
                user_message,
                self.sponge.snapshot,
                model=self.ess_model,
            )
        except Exception:
            log.exception("ESS classification failed, using safe defaults")
            return classifier_exception_fallback(user_message)

    def _store_episode(self, user_message: str, agent_response: str, ess: ESSResult) -> None:
        """Persist interaction memory with conservative semantic-admission gates.

        Semantic memory admission is intentionally strict to reduce replay of
        weak evidence in future retrieval cycles.
        """
        try:
            # Keep semantic memory high-precision: storing weak/fragile arguments
            # as "semantic" increases replay risk (AgentPoison 2024, MemoryGraft 2025).
            ess_reliable_for_updates = self._ess_reliable_for_updates(ess)
            semantic_candidate = ess.score >= SEMANTIC_MEMORY_MIN_ESS and _is_trusted_evidence(ess)
            memory_type = MemoryType.SEMANTIC if semantic_candidate else MemoryType.EPISODIC
            admission_policy = (
                AdmissionPolicy.SEMANTIC_STRICT
                if semantic_candidate
                else AdmissionPolicy.EPISODIC_QUALITY_DEMOTION
                if ess.score > config.ESS_THRESHOLD
                else AdmissionPolicy.EPISODIC_LOW_ESS
            )
            provenance_quality = (
                ProvenanceQuality.TRUSTED
                if semantic_candidate and ess_reliable_for_updates
                else ProvenanceQuality.UNCERTAIN
                if ess.score > config.ESS_THRESHOLD
                and ess.internal_consistency
                and ess_reliable_for_updates
                else ProvenanceQuality.LOW
            )
            self.episodes.store(
                user_message=user_message,
                agent_response=agent_response,
                ess=ess,
                interaction_count=self.sponge.interaction_count + 1,
                memory_type=memory_type,
                admission_policy=admission_policy,
                provenance_quality=provenance_quality,
            )
            if ess.score > config.ESS_THRESHOLD and memory_type == MemoryType.EPISODIC:
                log.info(
                    "Stored high-ESS episode as episodic due to quality gates "
                    "(type=%s source=%s consistent=%s)",
                    ess.reasoning_type,
                    ess.source_reliability,
                    ess.internal_consistency,
                )
        except Exception:
            log.exception("Episode storage failed")

    def _update_topics(self, ess: ESSResult) -> None:
        """Increment topic engagement counters from ESS topic labels."""
        for topic in ess.topics:
            self.sponge.track_topic(topic)

    def _detect_disagreement(self, ess: ESSResult) -> bool:
        """Structural disagreement: user argued against agent's existing stance.

        More reliable than keyword matching (brittle) or LLM self-judgment
        (self-judge bias up to 50pp — SYConBench, EMNLP 2025).
        """
        sign = ess.opinion_direction.sign
        if sign == 0.0:
            return False
        for topic in ess.topics:
            pos = self.sponge.opinion_vectors.get(topic, 0.0)
            if abs(pos) > 0.1 and pos * sign < 0:
                return True
        return False

    def _collect_unresolved_contradictions(self) -> list[str]:
        """Summarize staged deltas that currently oppose strong held beliefs."""
        candidates: list[tuple[float, str]] = []
        for staged in self.sponge.staged_opinion_updates:
            pos = self.sponge.opinion_vectors.get(staged.topic, 0.0)
            if abs(pos) < CONTRADICTION_POSITION_THRESHOLD or pos * staged.signed_magnitude >= 0:
                continue
            summary = (
                f"{staged.topic}({pos:+.2f} vs {staged.signed_magnitude:+.3f},"
                f" due #{staged.due_interaction})"
            )
            candidates.append((abs(staged.signed_magnitude), summary))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return [summary for _, summary in candidates]

    def _should_contract_before_revision(
        self, topic: str, direction: float, ess: ESSResult
    ) -> bool:
        """Decide whether to soften a strong belief before reversing direction."""
        old_pos = self.sponge.opinion_vectors.get(topic, 0.0)
        meta = self.sponge.belief_meta.get(topic)
        if meta is None:
            return False
        return (
            old_pos * direction < 0
            and abs(old_pos) >= AGM_CONTRACTION_POSITION
            and meta.confidence >= AGM_CONTRACTION_CONFIDENCE
            and ess.score >= AGM_CONTRACTION_SCORE
            and _is_trusted_evidence(ess)
        )

    def _apply_agm_contraction(self, topic: str, ess: ESSResult) -> None:
        """Apply a partial AGM-style contraction to a strongly held belief."""
        old_pos = self.sponge.opinion_vectors.get(topic, 0.0)
        if abs(old_pos) < 1e-9:
            return
        step = min(abs(old_pos), max(0.02, abs(old_pos) * AGM_CONTRACTION_RATIO))
        new_pos = old_pos - (1.0 if old_pos > 0 else -1.0) * step
        self.sponge.opinion_vectors[topic] = new_pos
        if topic in self.sponge.belief_meta:
            self.sponge.belief_meta[topic].confidence *= 1.0 - AGM_CONTRACTION_RATIO / 2.0
        self.sponge.record_shift(
            description=f"AGM contraction on {topic} (ESS {ess.score:.2f})",
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
                "ess_score": ess.score,
            }
        )

    def _ess_reliable_for_updates(self, ess: ESSResult) -> bool:
        """Return whether ESS payload quality is sufficient for memory updates.

        Coercions on non-critical fields are tolerated, but missing/exception
        fallbacks and coercions on core decision fields stay blocked. Coercion-
        tagged low-confidence scores also require extra margin before updates.
        """
        if ess.default_severity in {"missing", "exception"}:
            return False
        if ess.default_severity == "coercion" and ess.score < config.ESS_THRESHOLD + COERCION_UPDATE_MARGIN:
            return False
        return not any(field in CRITICAL_ESS_DEFAULT_FIELDS for field in ess.defaulted_fields)

    def _update_opinions(self, ess: ESSResult) -> None:
        """Stage delayed opinion updates when evidence clears ESS quality gates.

        Opinion vectors are never updated directly here; all deltas pass through
        staged cooling commits to avoid reactive single-turn flips.
        """
        if ess.score <= config.ESS_THRESHOLD or not ess.topics:
            return
        if not self._ess_reliable_for_updates(ess):
            log.info(
                "Skipping opinion update due to ESS fallback defaults (severity=%s fields=%s)",
                ess.default_severity,
                ess.defaulted_fields,
            )
            return
        direction = ess.opinion_direction.sign
        if direction == 0.0:
            return

        magnitude = compute_magnitude(ess, self.sponge)

        provenance = f"ESS {ess.score:.2f}: {ess.summary[:60]}"
        for topic in ess.topics:
            if self._should_contract_before_revision(topic, direction, ess):
                self._apply_agm_contraction(topic, ess)
            old_pos = self.sponge.opinion_vectors.get(topic, 0.0)
            conf = (
                self.sponge.belief_meta[topic].confidence
                if topic in self.sponge.belief_meta
                else 0.0
            )
            if old_pos * direction < 0:
                conf += abs(old_pos)
            effective_mag = magnitude / (conf + 1.0)
            due = self.sponge.stage_opinion_update(
                topic=topic,
                direction=direction,
                magnitude=effective_mag,
                cooling_period=config.OPINION_COOLING_PERIOD,
                provenance=provenance,
            )
            self._log_event(
                {
                    "event": "opinion_staged",
                    "interaction": self.sponge.interaction_count,
                    "topic": topic,
                    "signed_magnitude": direction * effective_mag,
                    "due_interaction": due,
                    "staged_total": len(self.sponge.staged_opinion_updates),
                }
            )

    def _extract_insight(self, user_message: str, agent_response: str, ess: ESSResult) -> None:
        """Extract personality insight per interaction, consolidated during reflection.

        Avoids lossy per-interaction full snapshot rewrites (ABBEL 2025: belief
        bottleneck). Snapshot only changes during reflection (Park et al. 2023).
        """
        if ess.score <= config.ESS_THRESHOLD:
            return
        if not self._ess_reliable_for_updates(ess):
            log.info(
                "Skipping insight extraction due to ESS fallback defaults (severity=%s fields=%s)",
                ess.default_severity,
                ess.defaulted_fields,
            )
            return
        try:
            insight = extract_insight(
                self.client,
                ess,
                user_message,
                agent_response,
                model=self.ess_model,
            )
            if not insight:
                return
            self.sponge.pending_insights.append(insight)
            self.sponge.version += 1
            magnitude = compute_magnitude(ess, self.sponge)
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

    def _reflection_gate(self) -> ReflectionGate:
        """Determine whether reflection should run for this interaction."""
        window_interactions = self.sponge.interaction_count - self.sponge.last_reflection_at
        if window_interactions < config.REFLECTION_EVERY // 2:
            return ReflectionGate(
                trigger=ReflectionTrigger.SKIP,
                trigger_label="skip",
                window_interactions=window_interactions,
            )

        periodic = window_interactions >= config.REFLECTION_EVERY
        recent_mag = sum(
            shift.magnitude
            for shift in self.sponge.recent_shifts
            if shift.interaction > self.sponge.last_reflection_at
        )
        if periodic:
            return ReflectionGate(
                trigger=ReflectionTrigger.PERIODIC,
                trigger_label="periodic",
                window_interactions=window_interactions,
            )
        if recent_mag > config.REFLECTION_SHIFT_THRESHOLD:
            return ReflectionGate(
                trigger=ReflectionTrigger.EVENT_DRIVEN,
                trigger_label=f"event-driven (mag={recent_mag:.3f})",
                window_interactions=window_interactions,
            )
        return ReflectionGate(
            trigger=ReflectionTrigger.SKIP,
            trigger_label="skip",
            window_interactions=window_interactions,
        )

    def _reflection_beliefs_text(self) -> str:
        """Render sorted current belief state for reflection prompts."""
        return (
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

    def _reflection_shifts_text(self) -> str:
        """Render recent shift history for reflection prompts."""
        return (
            "\n".join(
                f"- #{shift.interaction} (mag {shift.magnitude:.3f}): {shift.description}"
                for shift in self.sponge.recent_shifts
            )
            or "No recent shifts."
        )

    def _reflection_maturity_instruction(self) -> str:
        """Build maturity-aware instruction fragment for reflection prompts."""
        interaction_count = self.sponge.interaction_count
        belief_count = len(self.sponge.opinion_vectors)
        if interaction_count < 20:
            return "Focus on accurately recording what you've learned so far."
        if interaction_count < 50 or belief_count < 10:
            return "Look for patterns across your experiences and beliefs."
        return (
            "Your worldview is developing coherence. Based on your accumulated "
            "beliefs, you may have nascent views on topics you haven't explicitly "
            "discussed. If a pattern suggests a new position, articulate it tentatively."
        )

    def _reflection_prompt(self, trigger_label: str, recent_episodes: list[str]) -> str:
        """Assemble reflection prompt from current belief/shift/insight state."""
        insights_text = (
            "\n".join(f"- {insight}" for insight in self.sponge.pending_insights) or "None."
        )
        return REFLECTION_PROMPT.format(
            trigger=trigger_label,
            current_snapshot=self.sponge.snapshot,
            structured_traits=self._build_structured_traits(),
            current_beliefs=self._reflection_beliefs_text(),
            pending_insights=insights_text,
            episode_count=len(recent_episodes),
            episode_summaries="\n".join(f"- {episode}" for episode in recent_episodes),
            recent_shifts=self._reflection_shifts_text(),
            maturity_instruction=self._reflection_maturity_instruction(),
            max_tokens=config.SPONGE_MAX_TOKENS,
        )

    def _apply_reflection_snapshot(self, pre_snapshot: str, reflected_snapshot: str) -> None:
        """Validate and commit reflected snapshot text when it changed."""
        if not reflected_snapshot or reflected_snapshot == pre_snapshot:
            log.info("Reflection produced no changes")
            return
        if not validate_snapshot(pre_snapshot, reflected_snapshot):
            log.warning("Reflection output rejected by validation")
            return

        self._check_belief_preservation(reflected_snapshot)
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

        dropped = self.sponge.decay_beliefs(decay_rate=config.BELIEF_DECAY_RATE)
        if dropped:
            log.info("Decay removed %d stale beliefs: %s", len(dropped), dropped)

        entrenched = self.sponge.detect_entrenched_beliefs()
        if entrenched:
            log.warning("Entrenched beliefs detected: %s", entrenched)
        contradictions = self._collect_unresolved_contradictions()
        if contradictions:
            log.info("Contradiction backlog (%d): %s", len(contradictions), contradictions[:3])

        recent_episodes = self.episodes.retrieve(
            "recent personality development and opinion changes",
            n_results=min(config.REFLECTION_EVERY, 10),
            min_relevance=0.0,
            where={"interaction": {"$gte": self.sponge.last_reflection_at}},
        )
        if not recent_episodes:
            log.info("No episodes for reflection, skipping")
            self.sponge.last_reflection_at = self.sponge.interaction_count
            return

        prompt = self._reflection_prompt(gate.trigger_label, recent_episodes)

        try:
            pre_snapshot = self.sponge.snapshot
            if self.client is None:
                completion = chat_completion(
                    model=self.ess_model,
                    max_tokens=700,
                    messages=({"role": "user", "content": prompt},),
                )
                reflected_snapshot = completion.text.strip()
            else:
                response = self.client.messages.create(
                    model=self.ess_model,
                    max_tokens=700,
                    messages=[{"role": "user", "content": prompt}],
                )
                reflected_snapshot = _extract_text_block(response).strip()
            self._apply_reflection_snapshot(pre_snapshot, reflected_snapshot)
            self._finalize_reflection_cycle(
                dropped=dropped,
                entrenched=entrenched,
                contradictions=contradictions,
                window_interactions=gate.window_interactions,
            )
        except Exception:
            log.exception("Reflection cycle failed")

    def _check_belief_preservation(self, new_snapshot: str) -> None:
        """Warn if reflection dropped high-confidence beliefs from the snapshot.

        Constitutional AI Character Training (Nov 2025): losing a trait from
        the narrative = losing it from behavior. PERSIST (2025): monitor for
        personality erosion across reflections.
        """
        strong = [t for t, m in self.sponge.belief_meta.items() if m.confidence > 0.5]
        missing = [t for t in strong if t.lower().replace("_", " ") not in new_snapshot.lower()]
        if missing:
            log.warning("HEALTH: reflection dropped strong beliefs: %s", missing)

    def _log_interaction_summary(self, ess: ESSResult) -> None:
        """Structured per-interaction summary for monitoring personality evolution."""
        parts = [
            f"[#{self.sponge.interaction_count}]",
            f"ESS={ess.score:.2f}({ess.reasoning_type})",
            f"staged={len(self.sponge.staged_opinion_updates)}",
            f"pending={len(self.sponge.pending_insights)}",
        ]
        if ess.topics:
            parts.append(f"topics={ess.topics}")
        if ess.score > config.ESS_THRESHOLD:
            parts.append(f"v{self.sponge.version}")

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
        if self.sponge.interaction_count >= 20 and disagreement < 0.15:
            warnings.append("possible_sycophancy")
        if words and len(words) < 15:
            warnings.append("snapshot_too_short")
        if words and unique_ratio < 0.4:
            warnings.append("snapshot_bland")
        if self.sponge.interaction_count >= 40 and len(self.sponge.opinion_vectors) < 3:
            warnings.append("low_belief_growth")
        if high_conf_ratio > 0.8 and len(metas) >= 5:
            warnings.append("ossified_beliefs")

        entrenched = self.sponge.detect_entrenched_beliefs()
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
        old_words = set((self.previous_snapshot or "").lower().split())
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
