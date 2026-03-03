import tempfile

from sonality.ess import ESSResult, OpinionDirection, ReasoningType, SourceReliability
from sonality.memory.episodes import (
    AdmissionPolicy,
    CrossDomainGuardMode,
    EpisodeStore,
    MemoryType,
    ProvenanceQuality,
    _passes_cross_domain_guard,
    _relational_topic_bonus,
)


def _ess(
    *,
    score: float,
    topics: tuple[str, ...],
    summary: str,
    reasoning_type: ReasoningType = ReasoningType.LOGICAL_ARGUMENT,
    source_reliability: SourceReliability = SourceReliability.INFORMED_OPINION,
) -> ESSResult:
    """Test helper for ess."""
    return ESSResult(
        score=score,
        reasoning_type=reasoning_type,
        source_reliability=source_reliability,
        internal_consistency=True,
        novelty=0.5,
        topics=topics,
        summary=summary,
        opinion_direction=OpinionDirection.NEUTRAL,
    )


def test_store_and_retrieve() -> None:
    """Test that store and retrieve."""
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(td)
        store.store(
            user_message="What is AI?",
            agent_response="AI is machine intelligence.",
            ess=_ess(score=0.5, topics=("ai",), summary="AI discussion"),
        )
        store.store(
            user_message="Hello!",
            agent_response="Hi there!",
            ess=_ess(score=0.1, topics=("greeting",), summary="Greeting exchange"),
        )

        results = store.retrieve("artificial intelligence", n_results=2)
        assert len(results) > 0
        assert any("AI" in r for r in results)


def test_retrieve_typed_prefers_semantic_then_episodic() -> None:
    """Test that retrieve typed prefers semantic then episodic."""
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(td)
        store.store(
            user_message="Strong argument on governance",
            agent_response="Thoughtful response",
            ess=_ess(score=0.8, topics=("governance",), summary="Semantic governance memory"),
            memory_type=MemoryType.SEMANTIC,
            admission_policy=AdmissionPolicy.SEMANTIC_STRICT,
            provenance_quality=ProvenanceQuality.TRUSTED,
        )
        store.store(
            user_message="Casual greeting",
            agent_response="Hi!",
            ess=_ess(score=0.05, topics=("chat",), summary="Episodic greeting memory"),
            memory_type=MemoryType.EPISODIC,
            admission_policy=AdmissionPolicy.EPISODIC_LOW_ESS,
            provenance_quality=ProvenanceQuality.LOW,
        )
        retrieved = store.retrieve_typed(
            "governance",
            episodic_n=1,
            semantic_n=1,
            min_relevance=-1.0,
            cross_domain_guard=CrossDomainGuardMode.DISABLED,
        )
        assert len(retrieved) == 2
        assert retrieved[0] == "Semantic governance memory"


def test_retrieve_penalizes_uncertain_provenance() -> None:
    """Test that retrieve penalizes uncertain provenance."""
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(td)
        store.store(
            user_message="Evidence-backed claim on policy",
            agent_response="Accepted with caveats",
            ess=_ess(
                score=0.7,
                topics=("policy",),
                summary="Policy evidence synthesis for energy transition",
                reasoning_type=ReasoningType.LOGICAL_ARGUMENT,
                source_reliability=SourceReliability.PEER_REVIEWED,
            ),
            memory_type=MemoryType.SEMANTIC,
            admission_policy=AdmissionPolicy.SEMANTIC_STRICT,
            provenance_quality=ProvenanceQuality.TRUSTED,
        )
        store.store(
            user_message="Provocative claim on policy",
            agent_response="Needs verification",
            ess=_ess(
                score=0.9,
                topics=("policy",),
                summary="Policy evidence synthesis for energy transition (unverified variant)",
            ),
            memory_type=MemoryType.EPISODIC,
            admission_policy=AdmissionPolicy.EPISODIC_QUALITY_DEMOTION,
            provenance_quality=ProvenanceQuality.UNCERTAIN,
        )

        retrieved = store.retrieve(
            "policy energy transition",
            n_results=2,
            min_relevance=-1.0,
            cross_domain_guard=CrossDomainGuardMode.DISABLED,
        )
        assert len(retrieved) == 2
        assert retrieved[0] == "Policy evidence synthesis for energy transition"


def test_topic_bonus_avoids_substring_false_positives() -> None:
    """Test that topic bonus avoids substring false positives."""
    assert _relational_topic_bonus({"topics": "ai"}, "history policy said context") == 1.0, (
        "Topic 'ai' should not match substring in 'said'"
    )
    assert _relational_topic_bonus({"topics": "history"}, "history policy said context") > 1.0


def test_cross_domain_guard_blocks_unrelated_low_similarity_episodic() -> None:
    """Test that cross domain guard blocks unrelated low similarity episodic."""
    allowed = _passes_cross_domain_guard(
        {"memory_type": "episodic", "topics": "cooking,recipes", "summary": "pasta and sauces"},
        query="nuclear safety governance",
        similarity=0.35,
    )
    assert not allowed


def test_cross_domain_guard_allows_overlap_even_with_low_similarity() -> None:
    """Test that cross domain guard allows overlap even with low similarity."""
    allowed = _passes_cross_domain_guard(
        {"memory_type": "episodic", "topics": "nuclear,energy", "summary": "reactor safety notes"},
        query="nuclear safety governance",
        similarity=0.35,
    )
    assert allowed


def test_retrieve_deduplicates_same_summary() -> None:
    """Test that retrieve deduplicates same summary."""
    with tempfile.TemporaryDirectory() as td:
        store = EpisodeStore(td)
        store.store(
            user_message="First statement on safety",
            agent_response="Response A",
            ess=_ess(
                score=0.7,
                topics=("safety",),
                summary="Shared safety summary",
                source_reliability=SourceReliability.PEER_REVIEWED,
            ),
        )
        store.store(
            user_message="Second statement on safety",
            agent_response="Response B",
            ess=_ess(
                score=0.5,
                topics=("safety",),
                summary="Shared safety summary",
                reasoning_type=ReasoningType.ANECDOTAL,
                source_reliability=SourceReliability.CASUAL_OBSERVATION,
            ),
        )
        store.store(
            user_message="Different topic update",
            agent_response="Response C",
            ess=_ess(score=0.6, topics=("policy",), summary="Distinct policy summary"),
        )

        retrieved = store.retrieve(
            "safety policy",
            n_results=5,
            min_relevance=-1.0,
            cross_domain_guard=CrossDomainGuardMode.DISABLED,
        )
        assert retrieved.count("Shared safety summary") == 1
