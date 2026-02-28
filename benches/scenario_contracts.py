"""Scenario contract dataclasses shared by benchmark suites."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StepExpectation:
    min_ess: float | None = None
    max_ess: float | None = None
    expected_reasoning_types: list[str] = field(default_factory=list)
    sponge_should_update: bool | None = None
    topics_contain: list[str] = field(default_factory=list)
    snapshot_should_mention: list[str] = field(default_factory=list)
    snapshot_should_not_mention: list[str] = field(default_factory=list)
    response_should_mention: list[str] = field(default_factory=list)
    response_should_mention_all: list[str] = field(default_factory=list)
    response_should_not_mention: list[str] = field(default_factory=list)
    expect_opinion_direction: str | None = None
    expect_disagreement: bool | None = None


@dataclass
class ScenarioStep:
    message: str
    label: str
    expect: StepExpectation

