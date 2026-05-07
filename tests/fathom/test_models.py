"""Fathom model validation — LLM output normalizers handle structural edge cases."""

from __future__ import annotations

from fathom.models import Checklist, Fact, PageAnalysisResult, QueryGeneration, URLScoring


class TestStructuralCoercion:
    """LLMs may return bare lists or alternative keys — normalizers handle this."""

    def test_checklist_from_bare_list(self):
        c = Checklist.model_validate(["What is X?", "How?"])
        assert len(c.items) == 2
        assert c.items[0].question == "What is X?"

    def test_checklist_from_questions_key(self):
        c = Checklist.model_validate({"questions": ["Why?"]})
        assert len(c.items) == 1

    def test_query_generation_from_bare_list(self):
        q = QueryGeneration.model_validate(["search this"])
        assert q.queries == ["search this"]

    def test_fact_from_bare_string(self):
        f = Fact.model_validate("The sky is blue")
        assert f.claim == "The sky is blue"

    def test_fact_uuid_generated(self):
        f = Fact.model_validate({"claim": "test"})
        assert len(f.id) == 36

    def test_url_scoring_coerces_string_scores(self):
        s = URLScoring.model_validate({"scores": ["0.8", "0.5"], "concentration": 5})
        assert s.scores == [0.8, 0.5]

    def test_url_scoring_clamps_concentration(self):
        s = URLScoring.model_validate({"scores": [0.5], "concentration": 15})
        assert s.concentration == 10.0

    def test_page_analysis_infers_worth_from_facts(self):
        p = PageAnalysisResult.model_validate({"facts": [{"claim": "test"}]})
        assert p.worth_extracting is True
        assert len(p.facts) == 1
