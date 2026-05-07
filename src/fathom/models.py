"""All Pydantic types. Single source of truth for every data shape in Fathom.

Validators handle only structural coercion (bare list → dict wrapping).
Field-level correctness is the LLM's responsibility via explicit JSON schemas in prompts.
"""

from __future__ import annotations

import contextlib

from pydantic import BaseModel, Field, model_validator

from shared.types import new_id


class Fact(BaseModel, frozen=True):
    id: str = Field(default_factory=new_id)
    claim: str
    confidence: float = Field(ge=0, le=1, default=0.5)
    has_evidence: bool = True
    topic: str | None = None
    contradicts: list[str] | None = None

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
    worth_extracting: bool = False
    summary: str = ""
    topics: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    follow_links: list[int] = Field(default_factory=list)
    new_questions: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if isinstance(data, list):
            return {"facts": data, "worth_extracting": bool(data)}
        if not isinstance(data, dict):
            return data
        if "worth_extracting" not in data:
            data["worth_extracting"] = bool(data.get("facts", []))
        fl = data.get("follow_links")
        if isinstance(fl, list):
            data["follow_links"] = [
                int(x) for x in fl if isinstance(x, int) or (isinstance(x, str) and x.isdigit())
            ]
        return data


class ChecklistItem(BaseModel):
    id: str = Field(default_factory=new_id)
    question: str
    answered: bool = False
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if isinstance(data, str):
            return {"question": data}
        return data


class Checklist(BaseModel):
    items: list[ChecklistItem]

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if isinstance(data, list):
            return {"items": data}
        if isinstance(data, dict) and "items" not in data:
            for alt in ("questions", "checklist", "sub_questions"):
                if alt in data:
                    data["items"] = data.pop(alt)
                    break
            else:
                if "question" in data:
                    return {"items": [data]}
        return data


class URLScoring(BaseModel, frozen=True):
    scores: list[float] = Field(default_factory=list)
    concentration: float = Field(default=5.0, ge=1.0, le=10.0)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw = data.get("scores")
        if isinstance(raw, list):
            coerced: list[float] = []
            for s in raw:
                try:
                    coerced.append(float(s))
                except (ValueError, TypeError):
                    coerced.append(0.5)
            data = {**data, "scores": coerced}
        conc = data.get("concentration")
        if isinstance(conc, (int, float, str)):
            with contextlib.suppress(ValueError, TypeError):
                data = {**data, "concentration": max(1.0, min(10.0, float(conc)))}
        return data


class Contradiction(BaseModel, frozen=True):
    id: str = Field(default_factory=new_id)
    claim_a: str
    source_a: str
    claim_b: str
    source_b: str
    topic: str = ""


class SessionMemory(BaseModel):
    productive_urls: list[str] = Field(default_factory=list)
    unproductive_urls: list[str] = Field(default_factory=list)
    facts_per_round: list[int] = Field(default_factory=list)
    contradictions: list[Contradiction] = Field(default_factory=list)
    stall_rounds: int = 0


class QueryGeneration(BaseModel, frozen=True):
    queries: list[str]

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: object) -> object:
        if isinstance(data, list):
            return {"queries": data}
        if isinstance(data, dict) and "queries" not in data:
            for alt in ("search_queries", "search", "query_list"):
                if alt in data:
                    data["queries"] = data.pop(alt)
                    break
        return data


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------


class SourceEntry(BaseModel, frozen=True):
    id: str = Field(default_factory=new_id)
    url: str
    title: str
    productive: bool
    facts_extracted: int
    summary: str


class ResearchOutput(BaseModel):
    document: str
    sources: list[SourceEntry]
    facts: list[Fact]
    checklist: list[ChecklistItem]
    contradictions: list[Contradiction] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# API types
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    goal: str = Field(min_length=5, max_length=2000)
    max_pages: int | None = Field(default=None, ge=1, le=500)
    n: int | None = Field(default=None, ge=1, le=20)
    seeds: list[str] = Field(default_factory=list)
    depth: str = Field(default="standard", pattern="^(shallow|standard|deep|exhaustive)$")


class ResearchResponse(BaseModel):
    id: str
    status: str


class SessionStatus(BaseModel):
    id: str
    status: str
    goal: str
    document: str
    pages_scraped: int
    facts_gathered: int
    checklist: list[ChecklistItem]


# ---------------------------------------------------------------------------
# Lightweight search / extract API (used by sonality web tools)
# ---------------------------------------------------------------------------


class WebSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=400)
    max_results: int = Field(default=8, ge=1, le=20)


class WebSearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    content: str = ""


class WebSearchResponse(BaseModel):
    results: list[WebSearchResult]
    query: str


class WebExtractRequest(BaseModel):
    url: str = Field(min_length=1)


class WebExtractResponse(BaseModel):
    url: str
    title: str
    content: str
