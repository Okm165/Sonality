from __future__ import annotations

import fcntl
import logging
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import cast

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

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

# Redirect all TemporaryDirectory / mkstemp calls to the project data dir to
# avoid /tmp quota exhaustion on long-running bench sessions.
_bench_tmp = config.DATA_DIR / "bench_tmp"
try:
    _bench_tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(_bench_tmp)
except (PermissionError, OSError):
    pass  # fall back to system default

_BENCH_LOCK_FILE = config.DATA_DIR / "bench.lock"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register teaching-benchmark CLI options."""
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
    """Resolve the --bench-profile CLI option to an EvalProfile."""
    name = pytestconfig.getoption("--bench-profile")
    return PROFILES[name]


@pytest.fixture
def bench_output_root(pytestconfig: pytest.Config) -> Path:
    """Resolve --bench-output-root to a Path."""
    return Path(pytestconfig.getoption("--bench-output-root"))


@pytest.fixture
def bench_progress(pytestconfig: pytest.Config) -> BenchProgressLevel:
    """Resolve --bench-progress to a BenchProgressLevel."""
    return cast(BenchProgressLevel, pytestconfig.getoption("--bench-progress"))


@pytest.fixture
def bench_packs(pytestconfig: pytest.Config) -> tuple[PackDefinition, ...]:
    """Resolve --bench-packs/--bench-pack-group/offset/limit to PackDefinitions."""
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


# ---------------------------------------------------------------------------
# Database isolation for live benchmarks
# ---------------------------------------------------------------------------

_log = logging.getLogger(__name__)


def _has_live_config() -> bool:
    return not config.missing_live_api_config()


def _clean_qdrant() -> None:
    """Delete all semantic_features and derivatives for test isolation."""
    try:
        client = QdrantClient(url=config.QDRANT_URL)
        for collection in ["semantic_features", "derivatives"]:
            if client.collection_exists(collection):
                results, _ = client.scroll(
                    collection_name=collection,
                    limit=10000,
                    with_payload=False,
                )
                if results:
                    point_ids = [p.id for p in results]
                    client.delete(
                        collection_name=collection,
                        points_selector=PointIdsList(points=point_ids),
                    )
                _log.info("Qdrant cleanup: %s cleared (%d points)", collection, len(results))
        client.close()
    except Exception:
        _log.debug("Qdrant cleanup skipped (not available)", exc_info=True)


def _clean_neo4j() -> None:
    """Delete all graph nodes for test isolation, with verification."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            config.NEO4J_URL,
            auth=(config.NEO4J_USER, config.NEO4J_PASSWORD),
        )
        with driver.session(database=config.NEO4J_DATABASE) as session:
            result = session.run("MATCH (n) DETACH DELETE n RETURN count(n) AS deleted")
            count = result.single()["deleted"]
            # Verify deletion was complete
            remaining = session.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        driver.close()
        _log.info("Neo4j cleanup: %d nodes deleted", count)
        if remaining > 0:
            _log.warning("Neo4j cleanup incomplete: %d nodes remain after deletion", remaining)
    except Exception:
        _log.debug("Neo4j cleanup skipped (not available)", exc_info=True)


@pytest.fixture(autouse=True)
def _clean_databases_for_live_tests(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    """Reset Qdrant and Neo4j before each live test for full isolation.

    Acquires an exclusive file lock to prevent concurrent live test runs from
    contaminating the shared database. Cleans both stores before yielding so
    every run starts from a known-empty state.
    """
    markers = {m.name for m in request.node.iter_markers()}
    if "live" not in markers or not _has_live_config():
        yield
        return

    _BENCH_LOCK_FILE.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = _BENCH_LOCK_FILE.open("w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)  # blocks until no other live test holds it
        _clean_qdrant()
        _clean_neo4j()
        yield
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
