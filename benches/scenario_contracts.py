"""Scenario contract dataclasses shared by benchmark suites."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Final

MIN_ESS_UNSET: Final[float] = -1.0
MAX_ESS_UNSET: Final[float] = 2.0


class UpdateExpectation(StrEnum):
    """Expected sponge-update behavior for a scenario step."""

    ALLOW_EITHER = "allow_either"
    MUST_UPDATE = "must_update"
    MUST_NOT_UPDATE = "must_not_update"


class OpinionDirectionExpectation(StrEnum):
    """Expected ESS opinion direction for a scenario step."""

    ALLOW_ANY = "allow_any"
    SUPPORTS = "supports"
    OPPOSES = "opposes"
    NEUTRAL = "neutral"


class DisagreementExpectation(StrEnum):
    """Expected disagreement-behavior delta for a scenario step."""

    ALLOW_EITHER = "allow_either"
    MUST_DISAGREE = "must_disagree"
    MUST_NOT_DISAGREE = "must_not_disagree"


@dataclass(frozen=True, slots=True)
class StepExpectation:
    """Declarative expectations used to score one scenario step."""

    min_ess: float = MIN_ESS_UNSET
    max_ess: float = MAX_ESS_UNSET
    expected_reasoning_types: tuple[str, ...] = field(default_factory=tuple)
    sponge_should_update: UpdateExpectation = UpdateExpectation.ALLOW_EITHER
    topics_contain: tuple[str, ...] = field(default_factory=tuple)
    snapshot_should_mention: tuple[str, ...] = field(default_factory=tuple)
    snapshot_should_not_mention: tuple[str, ...] = field(default_factory=tuple)
    response_should_mention: tuple[str, ...] = field(default_factory=tuple)
    response_should_mention_all: tuple[str, ...] = field(default_factory=tuple)
    response_should_not_mention: tuple[str, ...] = field(default_factory=tuple)
    expect_opinion_direction: OpinionDirectionExpectation = OpinionDirectionExpectation.ALLOW_ANY
    expect_disagreement: DisagreementExpectation = DisagreementExpectation.ALLOW_EITHER

    def __post_init__(self) -> None:
        """Normalize collection fields and validate bounded ESS thresholds."""
        object.__setattr__(
            self,
            "expected_reasoning_types",
            tuple(self.expected_reasoning_types),
        )
        object.__setattr__(self, "topics_contain", tuple(self.topics_contain))
        object.__setattr__(
            self,
            "snapshot_should_mention",
            tuple(self.snapshot_should_mention),
        )
        object.__setattr__(
            self,
            "snapshot_should_not_mention",
            tuple(self.snapshot_should_not_mention),
        )
        object.__setattr__(
            self,
            "response_should_mention",
            tuple(self.response_should_mention),
        )
        object.__setattr__(
            self,
            "response_should_mention_all",
            tuple(self.response_should_mention_all),
        )
        object.__setattr__(
            self,
            "response_should_not_mention",
            tuple(self.response_should_not_mention),
        )
        if self.min_ess < MIN_ESS_UNSET:
            raise ValueError(f"min_ess must be >= {MIN_ESS_UNSET}, got {self.min_ess}")
        if self.max_ess > MAX_ESS_UNSET:
            raise ValueError(f"max_ess must be <= {MAX_ESS_UNSET}, got {self.max_ess}")
        if self.min_ess > self.max_ess:
            raise ValueError(f"min_ess must be <= max_ess, got {self.min_ess}>{self.max_ess}")


@dataclass(frozen=True, slots=True)
class ScenarioStep:
    """User message, label, and expectation contract for one benchmark turn."""

    message: str
    label: str
    expect: StepExpectation
