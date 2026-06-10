"""Fathom research prompts — single templates with data placeholders.

Call sites use: format_prompt(PROMPT, key=value, ...) which compresses individual
values proportionally when total content exceeds the context budget.
"""

# ---------------------------------------------------------------------------
# Goal Decomposition
# ---------------------------------------------------------------------------

DECOMPOSE_GOAL_PROMPT = """\
Break this research goal into specific questions to investigate. Match the \
number of questions to the goal's actual scope — don't inflate simple lookups \
into broad agendas.

CRITICAL: Every question will be used as a web search query. Each question \
MUST contain the full name of the subject — never use pronouns like "he", \
"his", "she", "they". Phrase questions as concise keyword-rich queries, \
not verbose conversational sentences.

JSON (at most 12): {{"items": [{{"question": "specific verifiable question"}}, ...]}}

RESEARCH GOAL: {goal}"""

# ---------------------------------------------------------------------------
# Query Generation
# ---------------------------------------------------------------------------

GENERATE_QUERIES_PROMPT = """\
Generate search queries that will advance this research most effectively.

Look at what's worked so far and what hasn't. Learn from the productive sources — \
what made them useful? Avoid the patterns of unproductive ones. Each query should \
target a different unanswered question. Be specific: names, dates, technical terms. \
Include both institutional sources (data, reports) and human perspectives (news, \
discourse, expert commentary) — what people are saying matters as much as official data.

If research is stalling, try a fundamentally different angle rather than \
rephrasing the same approach.

Return JSON: {{"queries": ["query1", "query2", ...]}}

GOAL: {goal}
UNANSWERED QUESTIONS:
{unanswered_questions}
PRODUCTIVE SOURCES (yielded useful facts):
{productive_urls}
UNPRODUCTIVE SOURCES (yielded nothing):
{unproductive_urls}
FACTS PER ROUND: {facts_per_round}
TRIGGER: {trigger_reason}"""

# ---------------------------------------------------------------------------
# Page Analysis
# ---------------------------------------------------------------------------

ANALYZE_PAGE_PROMPT = """\
Extract research value from this page.

Pull out specific factual claims — numbers, names, dates, direct quotes. \
Calibrate confidence by how well-supported each claim is within the page. \
Assess source_quality by the page's overall credibility — consider the type \
of source, whether claims are attributed, and whether the content is \
peer-reviewed, journalistic, user-generated, or anonymous. \
Skip anything already covered in the knowledge summary.

If the page links to promising deeper sources, include their indices. If the \
page raises new questions worth investigating, note them.

Return JSON:
{{"summary": "1-2 sentences", "facts": [{{"claim": "specific claim", "confidence": 0.75, "source_quality": 0.7, "topic": "category"}}], "follow_links": [0, 3], "new_questions": ["question raised"]}}
confidence and source_quality: 0.0–1.0.

If irrelevant: {{"summary": "", "facts": [], "follow_links": [], "new_questions": []}}

GOAL: {goal}
UNANSWERED QUESTIONS:
{unanswered_questions}
KNOWLEDGE GATHERED SO FAR:
{knowledge_context}

PAGE CONTENT:
{page_markdown}

LINKS ON PAGE (by index):
{numbered_links}"""
