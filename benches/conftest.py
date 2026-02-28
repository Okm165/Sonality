from __future__ import annotations

from pathlib import Path

import pytest

from sonality import config

from .teaching_harness import PROFILES, EvalProfile


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--bench-profile",
        action="store",
        default="default",
        choices=sorted(PROFILES),
        help="Teaching benchmark profile: lean, default, or high_assurance.",
    )
    parser.addoption(
        "--bench-output-root",
        action="store",
        default=str(config.DATA_DIR / "teaching_bench"),
        help="Directory used for teaching benchmark artifacts.",
    )


@pytest.fixture
def bench_profile(pytestconfig: pytest.Config) -> EvalProfile:
    name = pytestconfig.getoption("--bench-profile")
    return PROFILES[name]


@pytest.fixture
def bench_output_root(pytestconfig: pytest.Config) -> Path:
    return Path(pytestconfig.getoption("--bench-output-root"))
