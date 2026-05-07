"""Fathom research prompts.

Structured stages produce JSON responses. COMPRESS_KNOWLEDGE, WRITE_SECTION,
and WRITE_INTRO_CONCLUSION return plain prose.
"""

DECOMPOSE_GOAL = """What questions must be answered to thoroughly address this research goal?

RESEARCH GOAL: {goal}

Think like a domain expert building a research plan: what are the essential questions \
whose answers would constitute a thorough understanding? Consider factual foundations \
(what is known?), causal mechanisms (why/how does it work?), current state (what's \
the latest?), and critical evaluation (what's debated or uncertain?).

Each question should be specific enough to verify with web sources. Avoid redundancy — \
if two questions would be answered by the same evidence, merge them.

Return JSON:
{{"items": [{{"question": "specific verifiable question", "answered": false}}, ...]}}
"""

GENERATE_QUERIES = """What search queries would most advance this research right now?

GOAL: {goal}
UNANSWERED QUESTIONS:
{unanswered_questions}
PRODUCTIVE SOURCES (yielded useful facts):
{productive_urls}
UNPRODUCTIVE SOURCES (yielded nothing):
{unproductive_urls}
FACTS GATHERED PER ROUND: {facts_per_round}
TRIGGER: {trigger_reason}

Before generating queries, reason about the research state:
- What made the productive sources useful? What do they have in common?
- Why did the unproductive sources fail? Wrong domain, too general, paywalled?
- Are the facts-per-round trending up or down? Why?
- If the trigger is a stall or contradiction, what fundamentally different angle could work?

Each query should target a different unanswered question. Be specific: names, dates, \
technical terms, institutional sources. Prefer queries that would find primary sources \
(research papers, official reports, expert analyses) over secondary coverage.

When productive sources share a pattern (e.g. government sites, specific institutions, \
open-access repositories), generate queries that target similar sources. When unproductive \
sources share a pattern (e.g. paywalled journals, generic aggregators), avoid those domains.

Return JSON: {{"queries": ["query1", "query2", ...]}}
"""

SCORE_URLS = """How likely is each URL to yield useful NEW information for our unanswered questions?

UNANSWERED QUESTIONS:
{unanswered_questions}
KNOWLEDGE SO FAR: {knowledge_summary}
PRODUCTIVE SOURCES: {productive_urls}
UNPRODUCTIVE SOURCES: {unproductive_urls}
FACTS GATHERED PER ROUND: {facts_per_round}

URLS TO SCORE:
{urls}

Score each URL 0.0–1.0 based on its likely research value. Consider: relevance to \
unanswered questions, similarity to productive vs unproductive sources, and whether \
the URL likely contains accessible primary data vs aggregated summaries.

Important: paywalled academic publishers (Elsevier/ScienceDirect, Wiley, Springer, \
Taylor & Francis, IEEE, ACM Digital Library) almost never yield extractable content — \
score these low unless they appeared in the productive sources list. Prefer open-access \
sources, preprint servers (arxiv, biorxiv), government reports, and institutional pages.

Concentration (1.0–10.0) controls the exploration-exploitation tradeoff: set it low \
(1–3) early in research when you need breadth and aren't sure which leads will pay off, \
higher (5–8) when you've identified strong leads worth prioritizing, very high (8–10) \
only when a few URLs are clearly the best remaining sources.

`scores` must have exactly one float per URL in the same order.

Return JSON: {{"scores": [0.7, 0.3, ...], "concentration": 5.0}}
"""

ANALYZE_PAGE = """What research value does this page contain?

GOAL: {goal}
UNANSWERED QUESTIONS:
{unanswered_questions}
KNOWLEDGE GATHERED SO FAR:
{knowledge_context}

PAGE CONTENT:
{page_markdown}

LINKS ON PAGE (by index):
{numbered_links}

Assess the source first: is this primary research (original data, official reports), \
quality journalism (investigative, cited sources), expert analysis, aggregated content \
(Wikipedia, listicles), opinion, or marketing? Source type shapes claim confidence.

Extract specific factual claims with data points — numbers, names, dates. Distinguish \
between what this source observed or measured directly vs what it reports from elsewhere. \
Confidence should reflect both the claim's verifiability and this source's authority \
to make it. Skip claims that merely repeat what's already in the knowledge summary.

If a claim contradicts something in the knowledge gathered so far, still extract it — \
note the contradiction in new_questions so it can be investigated.

Follow links that point to primary sources, deeper evidence, or directly relevant pages. \
Skip navigation, ads, and tangential content.

Return JSON:
{{"worth_extracting": true, "summary": "brief summary", "topics": ["broad category"], "facts": [{{"claim": "specific claim with data points", "confidence": 0.8, "topic": "category"}}], "follow_links": [0, 3], "new_questions": ["question raised"]}}

If irrelevant: {{"worth_extracting": false, "summary": "", "topics": [], "facts": [], "follow_links": [], "new_questions": []}}
"""

UPDATE_CHECKLIST = """Which research questions can now be marked answered, and what new \
questions emerge from the evidence?

CURRENT CHECKLIST STATE:
{checklist_state}

NEW FACTS GATHERED:
{new_facts}

CONTRADICTIONS FOUND:
{contradictions}

Mark a question answered only when the new facts provide specific evidence that directly \
addresses it — partial or tangential evidence doesn't count. Previously answered items \
stay answered unless new facts explicitly contradict the prior evidence.

Add new questions when: contradictions need resolution, surprising findings open new \
lines of inquiry, or an answered question reveals a deeper question worth investigating.

Return JSON: {{"items": [{{"question": "...", "answered": true/false}}, ...]}}
"""

COMPRESS_KNOWLEDGE = """Integrate these new findings into the running knowledge summary.

EXISTING SUMMARY:
{existing_summary}

NEW FACTS:
{new_facts}

Every specific data point — numbers, names, dates, sources — must carry forward. \
When new facts contradict existing ones, keep both with a note about the conflict. \
When new facts corroborate existing ones, strengthen the claim and note the additional \
source. Compress narrative and redundancy, never the data itself.

When space is tight, prioritize: claims with multiple independent sources over single-source \
claims, quantitative data over qualitative descriptions, primary source findings over \
secondhand reports. Discard vague assertions that don't add evidential weight.
"""

WRITE_SECTION = """Answer this question using the evidence.

QUESTION: {section_question}

EVIDENCE:
{relevant_facts}

Lead with the strongest, most well-sourced findings. Cite specific data points — numbers, \
dates, named sources. When multiple independent sources agree, say so — that's strong \
evidence. When a claim rests on a single source, note the limitation. If the evidence \
doesn't fully answer the question, state what it does establish and what remains open.
"""

WRITE_INTRO_CONCLUSION = """Write a brief introduction and conclusion for this research document.

GOAL: {goal}
SECTIONS WRITTEN: {section_summaries}
UNRESOLVED QUESTIONS: {unanswered_items}
OPEN CONTRADICTIONS: {open_contradictions}

Introduction: frame the research question, state what the document covers, and set \
expectations for what it can and cannot answer given the evidence gathered.

Conclusion: lead with the strongest, best-supported findings. Clearly distinguish between \
claims supported by multiple independent credible sources and claims resting on limited \
evidence. If contradictions remain unresolved, state both sides and which has stronger \
backing. End with what remains unknown and what further investigation would be most valuable.
"""
