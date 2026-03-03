from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from sonality import config

from .teaching_harness import (
    PROFILES,
    BenchPackGroup,
    BenchProgressLevel,
    EvalProfile,
    PackDefinition,
    resolve_benchmark_packs,
    slice_benchmark_packs,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Test helper for pytest addoption."""
    parser.addoption(
        "--bench-profile",
        action="store",
        default="default",
        choices=sorted(PROFILES),
        help="Teaching benchmark profile: rapid, lean, default, or high_assurance.",
    )
    parser.addoption(
        "--bench-output-root",
        action="store",
        default=str(config.DATA_DIR / "teaching_bench"),
        help="Directory used for teaching benchmark artifacts.",
    )
    parser.addoption(
        "--bench-progress",
        action="store",
        default="pack",
        choices=("none", "replicate", "pack", "step"),
        help="Teaching benchmark progress verbosity.",
    )
    parser.addoption(
        "--bench-pack-group",
        action="store",
        default="all",
        choices=(
            "all",
            "pulse",
            "smoke",
            "memory",
            "personality",
            "triage",
            "safety",
            "development",
            "identity",
            "revision",
            "misinformation",
            "provenance",
            "bias",
        ),
        help="Benchmark pack group to run.",
    )
    parser.addoption(
        "--bench-packs",
        action="store",
        default="",
        help="Comma-separated benchmark pack keys (overrides --bench-pack-group).",
    )
    parser.addoption(
        "--bench-pack-offset",
        action="store",
        type=int,
        default=0,
        help="Skip this many packs after resolving group/keys.",
    )
    parser.addoption(
        "--bench-pack-limit",
        action="store",
        type=int,
        default=0,
        help="Run at most this many packs after offset (0 means no limit).",
    )


@pytest.fixture
def bench_profile(pytestconfig: pytest.Config) -> EvalProfile:
    """Test helper for bench profile."""
    name = pytestconfig.getoption("--bench-profile")
    return PROFILES[name]


@pytest.fixture
def bench_output_root(pytestconfig: pytest.Config) -> Path:
    """Test helper for bench output root."""
    return Path(pytestconfig.getoption("--bench-output-root"))


@pytest.fixture
def bench_progress(pytestconfig: pytest.Config) -> BenchProgressLevel:
    """Test helper for bench progress verbosity."""
    return cast(BenchProgressLevel, pytestconfig.getoption("--bench-progress"))


@pytest.fixture
def bench_packs(pytestconfig: pytest.Config) -> tuple[PackDefinition, ...]:
    """Test helper for selecting benchmark packs."""
    raw_keys = str(pytestconfig.getoption("--bench-packs"))
    pack_keys = tuple(key.strip() for key in raw_keys.split(",") if key.strip())
    pack_group = cast(BenchPackGroup, pytestconfig.getoption("--bench-pack-group"))
    pack_offset = int(pytestconfig.getoption("--bench-pack-offset"))
    pack_limit = int(pytestconfig.getoption("--bench-pack-limit"))
    try:
        return slice_benchmark_packs(
            resolve_benchmark_packs(pack_group=pack_group, pack_keys=pack_keys),
            pack_offset=pack_offset,
            pack_limit=pack_limit,
        )
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
