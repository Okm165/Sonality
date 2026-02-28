from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Final

from anthropic import Anthropic, APIError

from . import config
from .ess import ESSResult, ReasoningType, SourceReliability, classify
from .memory import EpisodeStore, SpongeState, compute_magnitude, extract_insight, validate_snapshot
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


def _status_code(exc: APIError) -> int | None:
    code = getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def _extract_text_block(response: object) -> str:
    content = getattr(response, "content", None)
    if not isinstance(content, list):
        return ""
    for block in content:
        if getattr(block, "type", None) != "text":
            continue
        text = getattr(block, "text", "")
        if isinstance(text, str):
            return text
    for block in content:
        text = getattr(block, "text", "")
        if isinstance(text, str):
            return text
    return ""


def _api_call_with_retry[T](fn: Callable[..., T], *args: object, **kwargs: object) -> T:
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
    def __init__(self) -> None:
        log.info(
            "Initializing SonalityAgent (model=%s, ess_model=%s)", config.MODEL, config.ESS_MODEL
        )
        if config.MODEL == config.ESS_MODEL:
            log.warning(
                "Main and ESS models are identical; using a separate ESS model reduces self-judge coupling"
            )
        self.client = Anthropic(api_key=config.API_KEY)
        self.sponge = SpongeState.load(config.SPONGE_FILE)
        self.episodes = EpisodeStore(str(config.CHROMADB_DIR))
        self.conversation: list[dict[str, str]] = []
        self.last_ess: ESSResult | None = None
        self.previous_snapshot: str | None = None
        log.info(
            "Agent ready: sponge v%d, %d prior interactions, %d beliefs",
            self.sponge.version,
            self.sponge.interaction_count,
            len(self.sponge.opinion_vectors),
        )

    def respond(self, user_message: str) -> str:
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

        response = _api_call_with_retry(
            self.client.messages.create,
            model=config.MODEL,
            max_tokens=2048,
            system=system_prompt,
            messages=self.conversation,
        )
        assistant_msg = _extract_text_block(response)
        if not assistant_msg:
            log.warning("Model response contained no text block; using empty reply")
        self.conversation.append({"role": "assistant", "content": assistant_msg})

        self._post_process(user_message, assistant_msg)
        return assistant_msg

    def _truncate_conversation(self) -> None:
        total = sum(len(m["content"]) for m in self.conversation)
        removed_count = 0
        while total > config.MAX_CONVERSATION_CHARS and len(self.conversation) > 2:
            removed = self.conversation.pop(0)
            total -= len(removed["content"])
            removed_count += 1
        if removed_count:
            log.info("Truncated %d old messages (conversation now %d chars)", removed_count, total)

    def _post_process(self, user_message: str, agent_response: str) -> None:
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
        self.sponge.track_disagreement(self._detect_disagreement(ess))

        self.previous_snapshot = self.sponge.snapshot
        self._extract_insight(user_message, agent_response, ess)
        self._maybe_reflect()
        self._log_health_event()

        self.sponge.save(config.SPONGE_FILE, config.SPONGE_HISTORY_DIR)
        self._log_interaction_summary(ess)

    def _classify_ess(self, user_message: str) -> ESSResult:
        try:
            return classify(self.client, user_message, self.sponge.snapshot)
        except Exception:
            log.exception("ESS classification failed, using safe defaults")
            return ESSResult(
                score=0.0,
                reasoning_type=ReasoningType.NO_ARGUMENT,
                source_reliability=SourceReliability.NOT_APPLICABLE,
                internal_consistency=True,
                novelty=0.0,
                topics=(),
                summary=user_message[:120],
            )

    def _store_episode(self, user_message: str, agent_response: str, ess: ESSResult) -> None:
        try:
            # Keep semantic memory high-precision: storing weak/fragile arguments
            # as "semantic" increases replay risk (AgentPoison 2024, MemoryGraft 2025).
            semantic_candidate = (
                ess.score >= 0.55
                and ess.internal_consistency
                and ess.reasoning_type in TRUSTED_REASONING
                and ess.source_reliability in TRUSTED_SOURCES
            )
            memory_type = "semantic" if semantic_candidate else "episodic"
            admission_policy = (
                "semantic_strict"
                if semantic_candidate
                else "episodic_quality_demotion"
                if ess.score > config.ESS_THRESHOLD
                else "episodic_low_ess"
            )
            provenance_quality = (
                "trusted"
                if semantic_candidate and not ess.used_defaults
                else "uncertain"
                if ess.score > config.ESS_THRESHOLD
                and ess.internal_consistency
                and not ess.used_defaults
                else "low"
            )
            self.episodes.store(
                user_message=user_message,
                agent_response=agent_response,
                ess_score=ess.score,
                topics=ess.topics,
                summary=ess.summary,
                interaction_count=self.sponge.interaction_count + 1,
                memory_type=memory_type,
                reasoning_type=ess.reasoning_type,
                source_reliability=ess.source_reliability,
                internal_consistency=ess.internal_consistency,
                admission_policy=admission_policy,
                provenance_quality=provenance_quality,
            )
            if ess.score > config.ESS_THRESHOLD and memory_type == "episodic":
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
            if abs(pos) < 0.35 or pos * staged.signed_magnitude >= 0:
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
        old_pos = self.sponge.opinion_vectors.get(topic, 0.0)
        meta = self.sponge.belief_meta.get(topic)
        if meta is None:
            return False
        return (
            old_pos * direction < 0
            and abs(old_pos) >= AGM_CONTRACTION_POSITION
            and meta.confidence >= AGM_CONTRACTION_CONFIDENCE
            and ess.score >= AGM_CONTRACTION_SCORE
            and ess.internal_consistency
            and ess.reasoning_type in TRUSTED_REASONING
            and ess.source_reliability in TRUSTED_SOURCES
        )

    def _apply_agm_contraction(self, topic: str, ess: ESSResult) -> None:
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

    def _update_opinions(self, ess: ESSResult) -> None:
        if ess.score <= config.ESS_THRESHOLD or not ess.topics:
            return
        if ess.used_defaults:
            log.info("Skipping opinion update due to ESS fallback defaults")
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
        if ess.used_defaults:
            log.info("Skipping insight extraction due to ESS fallback defaults")
            return
        try:
            insight = extract_insight(self.client, ess, user_message, agent_response)
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

    def _maybe_reflect(self) -> None:
        window_interactions = self.sponge.interaction_count - self.sponge.last_reflection_at
        if window_interactions < config.REFLECTION_EVERY // 2:
            return

        periodic = window_interactions >= config.REFLECTION_EVERY
        recent_mag = sum(
            s.magnitude
            for s in self.sponge.recent_shifts
            if s.interaction > self.sponge.last_reflection_at
        )
        event_driven = recent_mag > config.REFLECTION_SHIFT_THRESHOLD

        if not (periodic or event_driven):
            return

        trigger = "periodic" if periodic else f"event-driven (mag={recent_mag:.3f})"
        log.info("=== Reflection at #%d (%s) ===", self.sponge.interaction_count, trigger)

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

        insights_text = "\n".join(f"- {i}" for i in self.sponge.pending_insights) or "None."

        shifts_text = (
            "\n".join(
                f"- #{s.interaction} (mag {s.magnitude:.3f}): {s.description}"
                for s in self.sponge.recent_shifts
            )
            or "No recent shifts."
        )

        beliefs_text = (
            "\n".join(
                f"- {t}: {self.sponge.opinion_vectors.get(t, 0):+.2f} "
                f"(conf={m.confidence:.2f}, ev={m.evidence_count}, last=#{m.last_reinforced})"
                for t, m in sorted(
                    self.sponge.belief_meta.items(),
                    key=lambda x: -abs(self.sponge.opinion_vectors.get(x[0], 0)),
                )
            )
            or "No beliefs formed yet."
        )

        ic = self.sponge.interaction_count
        nb = len(self.sponge.opinion_vectors)
        if ic < 20:
            maturity = "Focus on accurately recording what you've learned so far."
        elif ic < 50 or nb < 10:
            maturity = "Look for patterns across your experiences and beliefs."
        else:
            maturity = (
                "Your worldview is developing coherence. Based on your accumulated "
                "beliefs, you may have nascent views on topics you haven't explicitly "
                "discussed. If a pattern suggests a new position, articulate it tentatively."
            )

        prompt = REFLECTION_PROMPT.format(
            trigger=trigger,
            current_snapshot=self.sponge.snapshot,
            structured_traits=self._build_structured_traits(),
            current_beliefs=beliefs_text,
            pending_insights=insights_text,
            episode_count=len(recent_episodes),
            episode_summaries="\n".join(f"- {ep}" for ep in recent_episodes),
            recent_shifts=shifts_text,
            maturity_instruction=maturity,
            max_tokens=config.SPONGE_MAX_TOKENS,
        )

        try:
            pre_snapshot = self.sponge.snapshot
            response = self.client.messages.create(
                model=config.ESS_MODEL,
                max_tokens=700,
                messages=[{"role": "user", "content": prompt}],
            )
            reflected = _extract_text_block(response).strip()
            if reflected and reflected != pre_snapshot:
                if not validate_snapshot(pre_snapshot, reflected):
                    log.warning("Reflection output rejected by validation")
                else:
                    self._check_belief_preservation(reflected)
                    self.sponge.snapshot = reflected
                    self.sponge.version += 1
                    self.sponge.record_shift(
                        description=f"Reflection at interaction {self.sponge.interaction_count}",
                        magnitude=0.0,
                    )
                    log.info(
                        "Reflection completed: v%d, %d -> %d chars (delta=%+d)",
                        self.sponge.version,
                        len(pre_snapshot),
                        len(reflected),
                        len(reflected) - len(pre_snapshot),
                    )
            else:
                log.info("Reflection produced no changes")

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
        log_path = config.DATA_DIR / "ess_log.jsonl"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            event["ts"] = datetime.now(UTC).isoformat()
            with open(log_path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
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
