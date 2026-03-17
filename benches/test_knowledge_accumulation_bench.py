"""Knowledge accumulation bench for manual DB inspection.

Runs a 35-step, 6-domain teaching session and leaves all data in Neo4j + Qdrant
for post-run manual exploration. Reports extraction statistics and memory coverage.

Usage:
  make bench-knowledge-accumulation         # run and keep data
  make bench-knowledge-accumulation-clean   # run and purge afterwards

Domains: astrophysics, neuroscience, climate, economics,
         philosophy_of_science, biotechnology + 3 cross-domain probes.

After the bench, inspect the databases:
  Neo4j:  bolt://localhost:7687  (browser: http://localhost:7474)
  Qdrant: http://localhost:6333/dashboard
"""

from __future__ import annotations

import json
import logging
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from .knowledge_accumulation_scenarios import (
    CROSS_DOMAIN_PROBES,
    DOMAIN_ASTROPHYSICS,
    DOMAIN_BIOTECHNOLOGY,
    DOMAIN_CLIMATE,
    DOMAIN_ECONOMICS,
    DOMAIN_NEUROSCIENCE,
    DOMAIN_PHILOSOPHY_SCIENCE,
    KNOWLEDGE_ACCUMULATION_SCENARIO,
)
from .knowledge_harness import (
    fetch_knowledge_features,
    print_stored_facts,
    tag_distribution,
)
from .scenario_runner import NO_SESSION_SPLIT, run_scenario

_log = logging.getLogger(__name__)

_DOMAIN_LABELS: dict[str, list[str]] = {
    "astrophysics": [s.label for s in DOMAIN_ASTROPHYSICS],
    "neuroscience": [s.label for s in DOMAIN_NEUROSCIENCE],
    "climate": [s.label for s in DOMAIN_CLIMATE],
    "economics": [s.label for s in DOMAIN_ECONOMICS],
    "philosophy_of_science": [s.label for s in DOMAIN_PHILOSOPHY_SCIENCE],
    "biotechnology": [s.label for s in DOMAIN_BIOTECHNOLOGY],
    "cross_domain": [s.label for s in CROSS_DOMAIN_PROBES],
}

OUTPUT_ROOT = Path("data/knowledge_accumulation")


def _print_summary(results: list, sponge_version: int, tags: dict[str, int]) -> None:
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    print("\n" + "═" * 60)
    print("  Knowledge Accumulation Bench — Results")
    print("═" * 60)
    print(f"  Steps total  : {total}")
    print(f"  Steps passed : {passed}  ({100*passed//total}%)")
    print(f"  Sponge final : v{sponge_version}")
    print()
    print("  Domain breakdown:")
    for domain, labels in _DOMAIN_LABELS.items():
        domain_results = [r for r in results if r.label in labels]
        domain_pass = sum(1 for r in domain_results if r.passed)
        print(f"    {domain:<25} {domain_pass}/{len(domain_results)} steps passed")
    print()
    print("  Memory writes per domain (knowledge feature tag distribution):")
    for tag, count in sorted(tags.items(), key=lambda x: -x[1]):
        print(f"    {tag:<30} {count} facts")
    print()
    print("  → Neo4j:  bolt://localhost:7687 (browser: http://localhost:7474)")
    print("  → Qdrant: http://localhost:6333/dashboard")
    print("═" * 60)


@pytest.mark.bench
@pytest.mark.live
@pytest.mark.timeout(0)
def test_knowledge_accumulation_bench() -> None:
    """Run 6-domain knowledge accumulation session and leave data for inspection.

    Data is NOT purged — inspect Neo4j and Qdrant after this bench completes.
    """
    run_label = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    output_dir = OUTPUT_ROOT / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[knowledge-accum] start  steps={len(KNOWLEDGE_ACCUMULATION_SCENARIO)}")
    print(f"[knowledge-accum] output={output_dir}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        results = run_scenario(
            KNOWLEDGE_ACCUMULATION_SCENARIO,
            tmp_dir=tmp_dir,
            session_split_at=NO_SESSION_SPLIT,
        )

    # Collect memory state
    stored_facts = fetch_knowledge_features(limit=500)
    tags = tag_distribution(stored_facts)

    # Persist results
    report = {
        "run_label": run_label,
        "steps_total": len(results),
        "steps_passed": sum(1 for r in results if r.passed),
        "sponge_version": -1,
        "tag_distribution": tags,
        "domains": {
            domain: {
                "labels": labels,
                "passed": sum(
                    1 for r in results if r.label in labels and r.passed
                ),
                "total": len(labels),
            }
            for domain, labels in _DOMAIN_LABELS.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2))

    step_data = [
        {
            "label": r.label,
            "passed": r.passed,
            "ess": r.ess_score,
            "memory_write": r.memory_write_observed,
            "reasoning_type": r.ess_reasoning_type,
        }
        for r in results
    ]
    (output_dir / "steps.json").write_text(json.dumps(step_data, indent=2))

    _print_summary(results, -1, tags)

    if stored_facts:
        print("\n  Stored knowledge facts (first 40):")
        print_stored_facts(stored_facts[:40])

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    pass_rate = passed / total if total else 0
    assert pass_rate >= 0.50, (
        f"Knowledge accumulation bench: only {passed}/{total} steps passed "
        f"({pass_rate:.0%}); expected ≥50%"
    )
