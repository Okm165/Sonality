from __future__ import annotations

import json
import logging
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


class BeliefMeta(BaseModel):
    """Confidence and provenance for a tracked belief.

    Confidence follows model-assessed uncertainty rather than evidence-count
    formulas. recent_updates tracks signed magnitudes for diagnostics.

    Episode provenance fields (A-MEM Zettelkasten model) link beliefs to their
    supporting/contradicting episodes for traceability. All new fields have
    defaults so existing sponge.json files load without error.
    """

    confidence: float = 0.0
    evidence_count: int = 1
    last_reinforced: int = 0
    provenance: str = ""
    recent_updates: list[float] = Field(default_factory=list)

    # Episode provenance (backward-compatible defaults)
    supporting_episode_uids: list[str] = Field(default_factory=list)
    contradicting_episode_uids: list[str] = Field(default_factory=list)
    formed_at: int = 0
    last_challenged_at: int = -1
    uncertainty: float = 1.0  # High initially, decreases with consistent evidence


class BehavioralSignature(BaseModel):
    """Behavioral aggregates used for longitudinal personality diagnostics."""

    disagreement_rate: float = 0.0
    topic_engagement: dict[str, int] = Field(default_factory=dict)


class Shift(BaseModel):
    """Recorded personality change event."""

    interaction: int
    timestamp: str
    description: str
    magnitude: float


class StagedOpinionUpdate(BaseModel):
    """Delayed opinion update awaiting cooling-period commitment."""

    topic: str
    signed_magnitude: float
    staged_at: int
    due_interaction: int
    provenance: str = ""
    new_uncertainty: float = -1.0  # LLM-assessed uncertainty; -1.0 = not set


class SpongeState(BaseModel):
    """Persistent personality state and incremental update logic."""

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
        new_uncertainty: float = -1.0,
    ) -> None:
        """Apply a bounded signed opinion update and refresh belief metadata.

        Assumes `direction` is sign-like (negative/opposing, positive/supporting)
        and `magnitude` is non-negative.
        """
        topic = topic.strip().lower()
        old = self.opinion_vectors.get(topic, 0.0)
        new = max(-1.0, min(1.0, old + direction * magnitude))
        self.opinion_vectors[topic] = new

        meta = self.belief_meta.get(topic)
        signed = direction * magnitude
        if meta is None:
            evidence_count = max(1, evidence_increment)
            initial_uncertainty = 0.80  # 1.0 - initial_conf (0.2)
            # Bayesian floor at creation: multiple staged updates reduce uncertainty immediately.
            if evidence_count >= 3:
                initial_uncertainty = min(initial_uncertainty, 0.30)
            elif evidence_count >= 2:
                initial_uncertainty = min(initial_uncertainty, 0.50)
            self.belief_meta[topic] = BeliefMeta(
                confidence=max(0.0, min(1.0, 1.0 - initial_uncertainty)),
                evidence_count=evidence_count,
                last_reinforced=self.interaction_count,
                provenance=provenance,
                recent_updates=[signed],
                uncertainty=initial_uncertainty,
            )
        else:
            meta.evidence_count += max(1, evidence_increment)
            meta.last_reinforced = self.interaction_count
            # Apply LLM-assessed uncertainty if provided (from belief provenance assessment).
            if 0.0 <= new_uncertainty <= 1.0:
                meta.uncertainty = new_uncertainty
            # Bayesian floor: uncertainty cannot stay at maximum as evidence accumulates.
            # Uses evidence_count as proxy; prevents LLM returning new_uncertainty=1.0 forever.
            if meta.evidence_count >= 3:
                meta.uncertainty = min(meta.uncertainty, 0.30)
            elif meta.evidence_count >= 2:
                meta.uncertainty = min(meta.uncertainty, 0.50)
            meta.confidence = max(0.0, min(1.0, 1.0 - meta.uncertainty))
            meta.recent_updates.append(signed)
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
        # Detailed belief trace for memory health analysis
        meta = self.belief_meta[topic]
        log.debug(
            "BELIEF_TRACE topic=%s | pos=%.3f | conf=%.2f | uncert=%.2f | "
            "ev=%d | support=%d | contra=%d | updates=%s",
            topic,
            new,
            meta.confidence,
            meta.uncertainty,
            meta.evidence_count,
            len(meta.supporting_episode_uids),
            len(meta.contradicting_episode_uids),
            meta.recent_updates[-3:],
        )

    def stage_opinion_update(
        self,
        topic: str,
        direction: float,
        magnitude: float,
        cooling_period: int,
        provenance: str = "",
        new_uncertainty: float = -1.0,
    ) -> int:
        """Queue an opinion update and return the interaction when it matures.

        Staging isolates immediate model response from durable worldview changes;
        updates are committed only after `cooling_period` interactions.
        new_uncertainty carries the LLM-assessed uncertainty for this evidence;
        -1.0 means not set (no LLM assessment available).
        """
        topic = topic.strip().lower()
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
                new_uncertainty=new_uncertainty,
            )
        )
        log.debug(
            "STAGED_UPDATE topic=%s | mag=%+.3f | due=%d | prov=%.40s",
            topic,
            signed,
            due,
            provenance.replace("\n", " "),
        )
        return due

    def apply_due_staged_updates(self) -> list[str]:
        """Commit matured staged updates, netting conflicting deltas per topic.

        Multiple staged deltas for one topic are summed first, so short-lived
        oscillations can cancel before touching persistent belief state.
        """
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
            magnitude = min(abs(net), 0.25)
            evidence_increment = len(updates)
            provenance = updates[-1].provenance
            # Apply the LLM-assessed uncertainty from the most recent staged update.
            # Uses the minimum uncertainty (most confident) among consistent-direction updates.
            valid_uncertainties = [
                u.new_uncertainty
                for u in updates
                if u.new_uncertainty >= 0.0
                and (u.signed_magnitude * net >= 0)  # same direction as net
            ]
            self.update_opinion(
                topic=topic,
                direction=direction,
                magnitude=magnitude,
                provenance=provenance,
                evidence_increment=evidence_increment,
                new_uncertainty=min(valid_uncertainties) if valid_uncertainties else -1.0,
            )
            applied.append(f"{topic}:{net:+.4f} ({evidence_increment} staged)")
        return applied

    def record_shift(self, description: str, magnitude: float) -> None:
        """Append a bounded history entry describing a personality change."""
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
        """Increment frequency counter for a discussed topic."""
        engagement = self.behavioral_signature.topic_engagement
        engagement[topic] = engagement.get(topic, 0) + 1

    def note_disagreement(self) -> None:
        """Record that the latest interaction structurally disagreed."""
        n = self.interaction_count or 1
        self.behavioral_signature.disagreement_rate = (
            self.behavioral_signature.disagreement_rate * (n - 1) + 1.0
        ) / n

    def note_agreement(self) -> None:
        """Record that the latest interaction did not structurally disagree."""
        n = self.interaction_count or 1
        self.behavioral_signature.disagreement_rate = (
            self.behavioral_signature.disagreement_rate * (n - 1)
        ) / n

    def save(self, path: Path, history_dir: Path) -> None:
        """Atomically persist state and archive the previous version.

        Uses temp-file rename semantics to avoid partial writes after crashes.
        """
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
        """Load persisted state or return a seed personality when absent."""
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
