"""All Pydantic types. Single source of truth for every data shape in Fathom.

Validators handle only structural coercion (bare list → dict wrapping).
Field-level correctness is the LLM's responsibility via explicit JSON schemas in prompts.
"""

from __future__ import annotations

from urllib.parse import urlparse

from pydantic import BaseModel, Field, model_validator

from shared.types import new_id


def extract_domain(url: str) -> str:
    """Extract lowercase domain from URL, fallback to truncated URL."""
    try:
        return urlparse(url).netloc.lower() or url[:50].lower()
    except Exception:
        return url[:50].lower()


class Fact(BaseModel, frozen=True):
    id: str = Field(default_factory=new_id)
    claim: str = Field(max_length=2000)
    confidence: float = Field(ge=0, le=1, default=0.5)
    source_quality: float = Field(ge=0, le=1, default=0.5)
    topic: str | None = Field(default=None, max_length=200)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if isinstance(data, str):
            return {"claim": data}
        if isinstance(data, dict) and "claim" not in data:
            for alt in ("text", "statement", "finding"):
                if alt in data:
                    data["claim"] = data.pop(alt)
                    break
        if isinstance(data, dict):
            for k in ("confidence", "source_quality"):
                v = data.get(k)
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    data[k] = max(0.0, min(1.0, float(v)))
            if isinstance(data.get("claim"), str) and len(data["claim"]) > 2000:
                data["claim"] = data["claim"][:2000]
            if isinstance(data.get("topic"), str) and len(data["topic"]) > 200:
                data["topic"] = data["topic"][:200]
        return data


class Link(BaseModel, frozen=True):
    url: str
    anchor_text: str = ""
    context: str = ""


class ExtractedPage(BaseModel, frozen=True):
    markdown: str
    links: list[Link]
    title: str
    has_content: bool


class PageAnalysisResult(BaseModel, frozen=True):
    summary: str = Field(default="", max_length=3000)
    facts: list[Fact] = Field(default_factory=list, max_length=50)
    follow_links: list[int] = Field(default_factory=list, max_length=20)
    new_questions: list[str] = Field(default_factory=list, max_length=10)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if isinstance(data, list):
            return {"facts": data}
        if not isinstance(data, dict):
            return data
        if isinstance(data.get("summary"), str) and len(data["summary"]) > 3000:
            data["summary"] = data["summary"][:3000]
        fl = data.get("follow_links")
        if isinstance(fl, list):
            data["follow_links"] = [
                int(x)
                for x in fl[:20]
                if isinstance(x, int) or (isinstance(x, str) and x.isdigit())
            ]
        facts = data.get("facts")
        if isinstance(facts, list) and len(facts) > 50:
            data["facts"] = facts[:50]
        nq = data.get("new_questions")
        if isinstance(nq, list) and len(nq) > 10:
            data["new_questions"] = nq[:10]
        return data


class ChecklistItem(BaseModel):
    question: str = Field(max_length=500)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if isinstance(data, str):
            return {"question": data[:500]}
        if isinstance(data, dict) and isinstance(data.get("question"), str):
            data["question"] = data["question"][:500]
        return data


class Checklist(BaseModel):
    items: list[ChecklistItem] = Field(default_factory=list, max_length=12)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if isinstance(data, list):
            return {"items": data[:12]}
        if isinstance(data, dict) and "items" not in data:
            for alt in ("questions", "checklist", "sub_questions"):
                if alt in data:
                    data["items"] = data.pop(alt)
                    break
            else:
                if "question" in data:
                    return {"items": [data]}
        if isinstance(data, dict):
            items = data.get("items")
            if isinstance(items, list) and len(items) > 12:
                data["items"] = items[:12]
        return data


class DomainStats(BaseModel, frozen=True):
    """Continuous quality stats for a single domain."""

    visit_count: int = 0
    quality_sum: float = 0.0
    total_facts: int = 0

    @property
    def quality_rate(self) -> float:
        """Laplace-smoothed quality estimate: posterior mean of Beta(1,1) prior."""
        return (self.quality_sum + 1.0) / (self.visit_count + 2.0)


class SessionMemory(BaseModel):
    productive_urls: list[str] = Field(default_factory=list)
    unproductive_urls: list[str] = Field(default_factory=list)
    facts_per_round: list[int] = Field(default_factory=list)
    domain_stats: dict[str, DomainStats] = Field(default_factory=dict)

    def record_domain(self, url: str, *, page_quality: float, fact_count: int) -> None:
        domain = extract_domain(url)
        prev = self.domain_stats.get(domain, DomainStats())
        self.domain_stats[domain] = DomainStats(
            visit_count=prev.visit_count + 1,
            quality_sum=prev.quality_sum + page_quality,
            total_facts=prev.total_facts + fact_count,
        )


class QueryGeneration(BaseModel, frozen=True):
    queries: list[str] = Field(default_factory=list, max_length=20)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if isinstance(data, list):
            return {"queries": data[:20]}
        if isinstance(data, dict) and "queries" not in data:
            for alt in ("search_queries", "search", "query_list"):
                if alt in data:
                    data["queries"] = data.pop(alt)
                    break
        if isinstance(data, dict):
            q = data.get("queries")
            if isinstance(q, list):
                data["queries"] = [str(s)[:500] for s in q[:20] if s]
        return data


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


class SourceEntry(BaseModel, frozen=True):
    url: str
    title: str
    page_quality: float
    facts_extracted: int
    summary: str


# ---------------------------------------------------------------------------
# API types
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    goal: str = Field(min_length=5, max_length=2000)
    max_pages: int | None = Field(default=None, ge=1, le=500)
    n: int | None = Field(default=None, ge=1, le=20)
    seeds: list[str] = Field(default_factory=list, max_length=20)
    depth: str = Field(
        default="standard", pattern="^(glance|quick|focused|standard|thorough|deep|exhaustive)$"
    )


class ResearchResponse(BaseModel):
    id: str
    status: str


class FactWithSource(BaseModel, frozen=True):
    """Fact claim with its source URL — returned in session status."""

    claim: str
    confidence: float
    source_quality: float = 0.5
    source_url: str
    topic: str = ""


class SessionStatus(BaseModel):
    id: str
    status: str
    goal: str
    pages_scraped: int
    facts: list[FactWithSource]
    checklist: list[ChecklistItem]
