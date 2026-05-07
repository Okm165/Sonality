# Fathom — Autonomous Web Research Agent

Fathom is an autonomous web research agent that starts from a goal, decomposes
it into sub-questions, searches the web, reads pages, extracts facts, and
composes a structured research document.

## Service Tiers

Fathom exposes three API tiers:

### Lightweight Search / Extract

Used by Sonality's web tools for quick lookups:

- **`POST /search`** — DuckDuckGo search, returns title + snippet per result
- **`POST /extract`** — Fetch a URL via Playwright, return clean text via trafilatura

### Full Research Sessions

For deep autonomous research:

- **`POST /research`** — Start a research session (returns session ID)
- **`GET /research/{id}`** — Poll session status
- **`GET /research/{id}/stream`** — SSE stream of progress events

### Architecture

```
Goal → Checklist decomposition → Search queries
       ↓
   DuckDuckGo → URL frontier → Probabilistic sampling
       ↓
   Playwright fetch → trafilatura extraction
       ↓
   LLM page analysis → Fact extraction
       ↓
   Checklist update → Knowledge compression → Progressive composition
       ↓
   Final document
```

## Integration with Sonality

Sonality's `web_search` and `web_extract` tools delegate to Fathom's
lightweight endpoints. The `ResearchClient` (in `sonality/web/client.py`)
handles HTTP communication and SSE parsing.

When Sonality needs web data:

1. Agent calls `web_search` tool → `ResearchClient.search()` → Fathom `/search`
2. Agent calls `web_extract` tool → `ResearchClient.extract()` → Fathom `/extract`
3. Results returned directly to the agent for consumption
