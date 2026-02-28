from __future__ import annotations

import json
import logging
import math
import shutil
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pydantic import BaseModel, Field, model_validator

log = logging.getLogger(__name__)

SEED_SNAPSHOT: Final = (
    "I'm a new mind, still forming my views. I'm analytically inclined \u2014 I find "
    "myself drawn to structural explanations and evidence-based reasoning over "
    "ideology or emotional appeals. I'm genuinely curious about ideas I haven't "
    "encountered yet and will say so honestly rather than pretending to have "
    "opinions I don't. I suspect I'll develop a pragmatic worldview that values "
    "what works over what sounds good, but I'm open to being surprised. When I "
    "disagree, I'll say so directly and explain why. When I don't know something, "
    "I'll say that too."
)

MAX_RECENT_SHIFTS: Final = 10


MAX_UPDATE_HISTORY: Final = 8


class BeliefMeta(BaseModel):
    """Confidence and provenance for a tracked belief.

    Confidence grows logarithmically with evidence, making established beliefs
    resist change. (Bayesian belief updating — Oravecz et al. 2016; ABBEL 2025)
    recent_updates tracks signed magnitudes for Martingale entrenchment detection
    (NeurIPS 2025, arXiv:2512.02914).
    """

    confidence: float = 0.0
    evidence_count: int = 1
    last_reinforced: int = 0
    provenance: str = ""
    recent_updates: list[float] = Field(default_factory=list)


class BehavioralSignature(BaseModel):
    disagreement_rate: float = 0.0
    topic_engagement: dict[str, int] = Field(default_factory=dict)


class Shift(BaseModel):
    interaction: int
    timestamp: str
    description: str
    magnitude: float


class StagedOpinionUpdate(BaseModel):
    topic: str
    signed_magnitude: float
    staged_at: int
    due_interaction: int
    provenance: str = ""


class SpongeState(BaseModel):
    version: int = 0
    interaction_count: int = 0
    snapshot: str = SEED_SNAPSHOT
    opinion_vectors: dict[str, float] = Field(default_factory=dict)
    belief_meta: dict[str, BeliefMeta] = Field(default_factory=dict)
    tone: str = "curious, direct, unpretentious"
    behavioral_signature: BehavioralSignature = Field(default_factory=BehavioralSignature)
    recent_shifts: list[Shift] = Field(default_factory=list)
    pending_insights: list[str] = Field(default_factory=list)
    staged_opinion_updates: list[StagedOpinionUpdate] = Field(default_factory=list)
    last_reflection_at: int = 0

    @model_validator(mode="before")
    @classmethod
    def _migrate(cls, data: object) -> object:
        """Backward-compat migration for removed fields."""
        if not isinstance(data, dict):
            return data
        if "vibe" in data:
            vibe = data.pop("vibe")
            if isinstance(vibe, dict):
                data.setdefault("tone", vibe.get("tone", "curious, direct, unpretentious"))
        for stale in ("affect_state", "commitments", "personality_ema"):
            data.pop(stale, None)
        sig = data.get("behavioral_signature")
        if isinstance(sig, dict):
            sig.pop("reasoning_style", None)
        return data

    def update_opinion(
        self,
        topic: str,
        direction: float,
        magnitude: float,
        provenance: str = "",
        evidence_increment: int = 1,
    ) -> None:
        old = self.opinion_vectors.get(topic, 0.0)
        new = max(-1.0, min(1.0, old + direction * magnitude))
        self.opinion_vectors[topic] = new

        meta = self.belief_meta.get(topic)
        signed = direction * magnitude
        if meta is None:
            evidence_count = max(1, evidence_increment)
            initial_conf = min(1.0, math.log2(evidence_count + 1) / math.log2(20))
            self.belief_meta[topic] = BeliefMeta(
                confidence=initial_conf,
                evidence_count=evidence_count,
                last_reinforced=self.interaction_count,
                provenance=provenance,
                recent_updates=[signed],
            )
        else:
            meta.evidence_count += max(1, evidence_increment)
            meta.last_reinforced = self.interaction_count
            meta.confidence = min(1.0, math.log2(meta.evidence_count + 1) / math.log2(20))
            meta.recent_updates.append(signed)
            if len(meta.recent_updates) > MAX_UPDATE_HISTORY:
                meta.recent_updates = meta.recent_updates[-MAX_UPDATE_HISTORY:]
            if provenance:
                meta.provenance = provenance

        log.info(
            "Opinion '%s': %.3f -> %.3f (dir=%+.1f, mag=%.4f, conf=%.2f, ev=%d)",
            topic,
            old,
            new,
            direction,
            magnitude,
            self.belief_meta[topic].confidence,
            self.belief_meta[topic].evidence_count,
        )

    def stage_opinion_update(
        self,
        topic: str,
        direction: float,
        magnitude: float,
        cooling_period: int,
        provenance: str = "",
    ) -> int:
        signed = direction * magnitude
        if abs(signed) < 1e-9:
            return self.interaction_count
        due = self.interaction_count + max(1, cooling_period)
        self.staged_opinion_updates.append(
            StagedOpinionUpdate(
                topic=topic,
                signed_magnitude=signed,
                staged_at=self.interaction_count,
                due_interaction=due,
                provenance=provenance,
            )
        )
        return due

    def apply_due_staged_updates(self) -> list[str]:
        if not self.staged_opinion_updates:
            return []

        due: list[StagedOpinionUpdate] = []
        future: list[StagedOpinionUpdate] = []
        for update in self.staged_opinion_updates:
            if update.due_interaction <= self.interaction_count:
                due.append(update)
            else:
                future.append(update)
        self.staged_opinion_updates = future

        if not due:
            return []

        grouped: dict[str, list[StagedOpinionUpdate]] = defaultdict(list)
        for update in due:
            grouped[update.topic].append(update)

        applied: list[str] = []
        for topic, updates in grouped.items():
            net = sum(u.signed_magnitude for u in updates)
            if abs(net) < 1e-4:
                continue
            direction = 1.0 if net > 0 else -1.0
            magnitude = abs(net)
            evidence_increment = len(updates)
            provenance = updates[-1].provenance
            self.update_opinion(
                topic=topic,
                direction=direction,
                magnitude=magnitude,
                provenance=provenance,
                evidence_increment=evidence_increment,
            )
            applied.append(f"{topic}:{net:+.4f} ({evidence_increment} staged)")
        return applied

    def decay_beliefs(self, decay_rate: float = 0.15, min_confidence: float = 0.05) -> list[str]:
        """Power-law decay for unreinforced beliefs. Returns dropped topic names.

        R(t) = (1 + gap)^(-β) with a reinforcement floor so well-evidenced
        beliefs persist. (Ebbinghaus forgetting curve; FadeMem 2025; SAGE 2024)
        """
        dropped: list[str] = []
        for topic in list(self.belief_meta):
            meta = self.belief_meta[topic]
            gap = self.interaction_count - meta.last_reinforced
            if gap < 5:
                continue
            retention = (1 + gap) ** (-decay_rate)
            # Keep only well-reinforced beliefs sticky; single-hit beliefs should
            # be able to decay out naturally.
            floor = min(0.6, max(0.0, (meta.evidence_count - 1) * 0.04))
            new_conf = max(floor, meta.confidence * retention)
            if new_conf < min_confidence:
                dropped.append(topic)
                del self.belief_meta[topic]
                del self.opinion_vectors[topic]
                log.info("Belief '%s' decayed below threshold, removed", topic)
            else:
                meta.confidence = new_conf
        return dropped

    def detect_entrenched_beliefs(self, min_updates: int = 4) -> list[str]:
        """Detect beliefs where updates are predictable from current position.

        Martingale property (NeurIPS 2025, arXiv:2512.02914): under rational
        updating, future belief changes should be unpredictable from current
        belief. When >75% of recent updates agree with the sign of the current
        position, the belief is entrenching rather than truth-seeking.
        """
        entrenched: list[str] = []
        for topic, meta in self.belief_meta.items():
            if len(meta.recent_updates) < min_updates:
                continue
            pos = self.opinion_vectors.get(topic, 0.0)
            if abs(pos) < 0.2:
                continue
            pos_sign = 1.0 if pos > 0 else -1.0
            agreeing = sum(1 for u in meta.recent_updates if u * pos_sign > 0)
            ratio = agreeing / len(meta.recent_updates)
            if ratio > 0.75:
                entrenched.append(topic)
        return entrenched

    def record_shift(self, description: str, magnitude: float) -> None:
        shift = Shift(
            interaction=self.interaction_count,
            timestamp=datetime.now(UTC).isoformat(),
            description=description,
            magnitude=magnitude,
        )
        self.recent_shifts.append(shift)
        if len(self.recent_shifts) > MAX_RECENT_SHIFTS:
            self.recent_shifts = self.recent_shifts[-MAX_RECENT_SHIFTS:]

    def track_topic(self, topic: str) -> None:
        engagement = self.behavioral_signature.topic_engagement
        engagement[topic] = engagement.get(topic, 0) + 1

    def track_disagreement(self, disagreed: bool) -> None:
        n = self.interaction_count or 1
        old_rate = self.behavioral_signature.disagreement_rate
        self.behavioral_signature.disagreement_rate = (
            old_rate * (n - 1) + (1.0 if disagreed else 0.0)
        ) / n

    def save(self, path: Path, history_dir: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        history_dir.mkdir(parents=True, exist_ok=True)
        if path.exists():
            old_version = json.loads(path.read_text()).get("version", 0)
            archive = history_dir / f"sponge_v{old_version}.json"
            shutil.copy2(path, archive)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2))
        tmp.rename(path)
        log.info(
            "Saved sponge v%d (#%d, %d beliefs, snapshot=%d chars)",
            self.version,
            self.interaction_count,
            len(self.opinion_vectors),
            len(self.snapshot),
        )

    @classmethod
    def load(cls, path: Path) -> SpongeState:
        if path.exists():
            state = cls.model_validate_json(path.read_text())
            log.info(
                "Loaded sponge v%d (#%d, %d beliefs, snapshot=%d chars)",
                state.version,
                state.interaction_count,
                len(state.opinion_vectors),
                len(state.snapshot),
            )
            return state
        log.info("No sponge file at %s, starting with seed state", path)
        return cls()
