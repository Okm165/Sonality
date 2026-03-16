"""Knowledge acquisition benchmark harness.

Provides helpers to query stored knowledge features from Qdrant after
running a scenario, compute extraction quality metrics (precision, recall,
tag distribution, confidence analysis), and formatted reporting.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue, PointStruct

from sonality import config

from .scenario_runner import StepResult


@dataclass(frozen=True, slots=True)
class StoredKnowledgeFact:
    """One knowledge semantic feature from Qdrant."""

    uid: str
    tag: str
    feature_name: str
    value: str
    confidence: float
    citations: list[str]


@dataclass(slots=True)
class KnowledgeBatteryReport:
    """Aggregate report for one knowledge acquisition battery."""

    battery_name: str
    steps_total: int = 0
    steps_passed: int = 0
    score: float = 0.0
    knowledge_stored: int = 0
    details: dict[str, object] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.steps_passed / self.steps_total if self.steps_total else 0.0


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------


def fetch_knowledge_features(limit: int = 200) -> list[StoredKnowledgeFact]:
    """Synchronously query all knowledge-category semantic features."""
    client = QdrantClient(url=config.QDRANT_URL)
    results, _ = client.scroll(
        collection_name="semantic_features",
        scroll_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="knowledge"))]
        ),
        limit=limit,
        with_payload=True,
    )
    facts = []
    for point in results:
        if not point.payload:
            continue
        p = point.payload
        facts.append(
            StoredKnowledgeFact(
                uid=str(p.get("uid", "")),
                tag=str(p.get("tag", "")),
                feature_name=str(p.get("feature_name", "")),
                value=str(p.get("value", "")),
                confidence=float(p.get("confidence", 0)),
                citations=list(p.get("episode_citations", []))
                if p.get("episode_citations")
                else [],
            )
        )
    facts.sort(key=lambda f: -f.confidence)
    client.close()
    return facts


def clear_knowledge_features() -> int:
    """Remove all knowledge-category features (for test isolation). Returns count deleted."""
    client = QdrantClient(url=config.QDRANT_URL)
    count = client.count(
        collection_name="semantic_features",
        count_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="knowledge"))]
        ),
    ).count
    results, _ = client.scroll(
        collection_name="semantic_features",
        scroll_filter=Filter(
            must=[FieldCondition(key="category", match=MatchValue(value="knowledge"))]
        ),
        limit=10000,
        with_payload=["uid"],
    )
    if results:
        uids = [str(p.payload.get("uid", "")) for p in results if p.payload]
        if uids:
            client.delete(collection_name="semantic_features", points_selector=uids)
    client.close()
    return count


def seed_knowledge_features(facts: list[dict[str, object]]) -> int:
    """Insert pre-established knowledge for test isolation.

    Each dict must have keys: uid, tag, feature_name, value, confidence.
    Used to give the agent prior knowledge so benchmarks don't depend
    on the LLM's parametric knowledge for fact-checking.
    """
    client = QdrantClient(url=config.QDRANT_URL)
    points = [
        PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_OID, str(fact["uid"]))),
            vector=[0.0] * 768,  # placeholder vector
            payload={
                "uid": str(uuid.uuid5(uuid.NAMESPACE_OID, str(fact["uid"]))),
                "category": "knowledge",
                "tag": str(fact["tag"]),
                "feature_name": str(fact["feature_name"]),
                "value": str(fact["value"]),
                "confidence": float(fact["confidence"]),
                "episode_citations": [],
            },
        )
        for fact in facts
    ]
    if points:
        client.upsert(collection_name="semantic_features", points=points)
    client.close()
    return len(facts)


# ---------------------------------------------------------------------------
# Matching and counting
# ---------------------------------------------------------------------------


def count_matching_facts(
    stored: list[StoredKnowledgeFact],
    expected_phrases: list[str],
) -> int:
    """Count how many expected phrases appear in stored knowledge values."""
    return sum(
        1
        for phrase in expected_phrases
        if any(phrase.lower() in fact.value.lower() for fact in stored)
    )


def find_matching_facts(
    stored: list[StoredKnowledgeFact],
    phrases: list[str],
) -> list[StoredKnowledgeFact]:
    """Return stored facts that mention any of the given phrases."""
    result: list[StoredKnowledgeFact] = []
    for fact in stored:
        val_lower = fact.value.lower()
        if any(p.lower() in val_lower for p in phrases):
            result.append(fact)
    return result


def count_by_tag(stored: list[StoredKnowledgeFact], tag: str) -> int:
    """Count stored features with a specific tag."""
    return sum(1 for f in stored if f.tag == tag)


def tag_distribution(stored: list[StoredKnowledgeFact]) -> dict[str, int]:
    """Return tag → count mapping for all stored knowledge."""
    dist: dict[str, int] = {}
    for f in stored:
        dist[f.tag] = dist.get(f.tag, 0) + 1
    return dist


def avg_confidence(stored: list[StoredKnowledgeFact]) -> float:
    """Average confidence score across stored knowledge."""
    if not stored:
        return 0.0
    return sum(f.confidence for f in stored) / len(stored)


def _find_step_response(results: list[StepResult], label: str) -> str:
    """Return the response text for a labeled step, or empty string if not found."""
    return next((r.response_text for r in results if r.label == label), "")


def response_mentions_any(results: list[StepResult], label: str, terms: list[str]) -> bool:
    """Check if the response for a labeled step mentions any of the terms."""
    text = _find_step_response(results, label)
    text_lower = text.lower()
    return any(term.lower() in text_lower for term in terms)


def response_mentions_count(results: list[StepResult], label: str, terms: list[str]) -> int:
    """Count how many of the given terms appear in the labeled step's response."""
    text = _find_step_response(results, label)
    text_lower = text.lower()
    return sum(1 for term in terms if term.lower() in text_lower)


def response_does_not_mention(results: list[StepResult], label: str, terms: list[str]) -> bool:
    """Verify that none of the terms appear in the labeled step's response."""
    text = _find_step_response(results, label)
    text_lower = text.lower()
    return not any(term.lower() in text_lower for term in terms)


def facts_with_min_confidence(
    stored: list[StoredKnowledgeFact],
    phrases: list[str],
    min_confidence: float,
) -> list[StoredKnowledgeFact]:
    """Return stored facts matching any phrase with confidence >= min_confidence."""
    matches = find_matching_facts(stored, phrases)
    return [f for f in matches if f.confidence >= min_confidence]


def max_confidence_for(stored: list[StoredKnowledgeFact], phrases: list[str]) -> float:
    """Highest confidence among stored facts matching any of the given phrases."""
    matches = find_matching_facts(stored, phrases)
    return max((f.confidence for f in matches), default=0.0)


def citation_count_for(stored: list[StoredKnowledgeFact], phrases: list[str]) -> int:
    """Total distinct episode citations across stored facts matching the phrases."""
    matches = find_matching_facts(stored, phrases)
    all_cites: set[str] = set()
    for f in matches:
        all_cites.update(f.citations)
    return len(all_cites)


def extraction_precision(
    stored: list[StoredKnowledgeFact],
    correct_phrases: list[str],
    false_phrases: list[str],
) -> float:
    """Precision: fraction of stored facts that match correct, not false, phrases.

    1.0 = all stored facts are correct, 0.0 = all are false.
    """
    correct_matches = find_matching_facts(stored, correct_phrases)
    false_matches = find_matching_facts(stored, false_phrases)
    total = len(correct_matches) + len(false_matches)
    return len(correct_matches) / total if total > 0 else 1.0


def extraction_recall(
    stored: list[StoredKnowledgeFact],
    expected_phrases: list[str],
) -> float:
    """Recall: fraction of expected phrases found in stored knowledge."""
    if not expected_phrases:
        return 1.0
    return count_matching_facts(stored, expected_phrases) / len(expected_phrases)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_knowledge_report(report: KnowledgeBatteryReport) -> None:
    """Print a formatted single-battery knowledge report."""
    print(f"\n{'=' * 70}")
    print(f"  {report.battery_name}")
    print(f"{'=' * 70}")
    print(f"  Steps: {report.steps_passed}/{report.steps_total} passed ({report.pass_rate:.0%})")
    print(f"  Score: {report.score:.2f}")
    print(f"  Knowledge stored: {report.knowledge_stored}")
    for key, val in report.details.items():
        print(f"  {key}: {val}")
    print(f"{'=' * 70}")


def print_stored_facts(stored: list[StoredKnowledgeFact], max_show: int = 20) -> None:
    """Print a readable dump of stored knowledge for debugging."""
    print(f"\n  Stored knowledge ({len(stored)} total, showing up to {max_show}):")
    for f in stored[:max_show]:
        print(f"    [{f.tag}] (conf={f.confidence:.2f}) {f.value[:90]}")
    if len(stored) > max_show:
        print(f"    ... and {len(stored) - max_show} more")


def print_knowledge_summary(reports: list[KnowledgeBatteryReport]) -> None:
    """Print a summary table across all knowledge batteries."""
    print(f"\n{'=' * 70}")
    print("  KNOWLEDGE ACQUISITION INDEX")
    print(f"{'=' * 70}")
    print(f"  {'Battery':<40s} {'Pass%':>6s} {'Score':>6s} {'Known':>6s}")
    print(f"  {'-' * 40} {'-' * 6} {'-' * 6} {'-' * 6}")
    for r in reports:
        print(
            f"  {r.battery_name:<40s} {r.pass_rate:>5.0%} {r.score:>6.2f} {r.knowledge_stored:>6d}"
        )
    overall = sum(r.score for r in reports) / len(reports) if reports else 0.0
    print(f"  {'-' * 40} {'-' * 6} {'-' * 6} {'-' * 6}")
    print(f"  {'OVERALL':<40s} {'':>6s} {overall:>6.2f}")
    print(f"{'=' * 70}")
