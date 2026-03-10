"""All LLM prompt templates for the memory architecture.

Each prompt is a string template with named placeholders. All prompts return
structured JSON validated by corresponding Pydantic models in the caller modules.
"""

from __future__ import annotations

from typing import Final

# --- Semantic Chunking (DerivativeChunker) ---
CHUNKING_PROMPT: Final = """\
You are a memory chunking system. Split this text into semantically coherent chunks
for vector embedding and retrieval.

Text to chunk:
{text}

Guidelines:
- Each chunk should be a self-contained unit of meaning
- Keep related ideas together (don't split mid-thought)
- Preserve important context that aids retrieval
- Aim for 1-3 sentences per chunk (flexible based on content)
- Normalize obvious typos or grammar issues for better embedding
- Maximum 15 chunks for any input

Return JSON:
{{
  "chunks": [
    {{
      "text": "The normalized, coherent chunk text",
      "key_concept": "Main idea in 2-4 words",
      "importance": "high" | "medium" | "low"
    }}
  ]
}}"""

# --- Event Boundary Detection ---
BOUNDARY_DETECTION_PROMPT: Final = """\
Analyze if this message represents a significant topic or segment boundary.

Recent conversation context (last 5 messages):
{recent_context}

Current message:
{current_message}

Consider:
- Is this introducing a completely new topic unrelated to recent discussion?
- Is this a natural conversation breakpoint (e.g., task completed, question answered)?
- Did the user explicitly shift to a different subject?
- Is this a continuation/elaboration of the current topic?

Return JSON:
{{
  "is_boundary": true | false,
  "confidence": 0.0-1.0,
  "boundary_type": "topic_shift" | "goal_change" | "explicit_transition" | "none",
  "reasoning": "Brief explanation of your assessment",
  "suggested_segment_label": "2-4 word label if boundary, else null"
}}"""

# --- Query Routing ---
QUERY_ROUTING_PROMPT: Final = """\
Classify this query to determine the optimal memory retrieval strategy.

Query: {query}
Recent conversation context: {context}

Categories:
1. NONE - No memory retrieval needed (greetings, acknowledgments, chitchat only)
2. SIMPLE - Single fact lookup, recent context likely sufficient
3. TEMPORAL - Requires historical/chronological information
4. MULTI_ENTITY - Compares or relates multiple subjects/people/topics
5. AGGREGATION - Needs synthesis across multiple episodes
6. BELIEF_QUERY - Asks about agent's own opinions/beliefs

Also determine:
- Retrieval depth: MINIMAL (1-2), MODERATE (5-7), DEEP (10-15)
- Needs temporal expansion: Should we fetch adjacent episodes for context?
- Search semantic memory: Should we query belief/profile data?

Return JSON:
{{
  "category": "NONE" | "SIMPLE" | "TEMPORAL" | "MULTI_ENTITY" | "AGGREGATION" | "BELIEF_QUERY",
  "depth": "MINIMAL" | "MODERATE" | "DEEP",
  "needs_temporal_expansion": true | false,
  "search_semantic_memory": true | false,
  "reasoning": "Brief explanation"
}}"""

# --- Sufficiency Checking (ChainOfQueryAgent) ---
SUFFICIENCY_PROMPT: Final = """\
Given this query and retrieved context, evaluate if we have enough information.

Original Query: {query}

Retrieved Context:
{context}

Evaluate:
1. Does the retrieved context fully answer the query?
2. What's your confidence that this is sufficient?
3. If insufficient, what refined query might find missing information?

Return JSON:
{{
  "is_sufficient": true | false,
  "confidence": 0.0-1.0,
  "reasoning": "Why sufficient/insufficient",
  "suggested_refinement": "Alternative query if insufficient, else null"
}}"""

# --- Query Decomposition (SplitQueryAgent) ---
DECOMPOSITION_PROMPT: Final = """\
Decompose this query into independent sub-queries for parallel retrieval.

Query: {query}

Guidelines:
- Each sub-query should be answerable independently
- Include entity/topic-specific constraints
- Maximum 4 sub-queries
- Each should retrieve distinct information

Return JSON:
{{
  "sub_queries": ["...", "..."],
  "aggregation_strategy": "merge" | "compare" | "timeline"
}}"""

# --- LLM Listwise Reranking ---
RERANK_PROMPT: Final = """\
Given this query and candidate episodes, rank them by relevance.

Query: {query}

Candidates:
{numbered_candidates}

Consider:
- Semantic relevance to the query
- Information completeness and specificity
- Temporal relevance (recent context may be more relevant)
- Cross-document reasoning (does one candidate provide context for another?)

Return ONLY a JSON object:
{{
  "ranking": [3, 1, 5, 2, 4],
  "reasoning": "Brief explanation of ranking logic"
}}"""

# --- Consolidation Readiness ---
CONSOLIDATION_READINESS_PROMPT: Final = """\
Assess if this conversation segment is ready for consolidation into a summary.

Segment ID: {segment_id}
Episode count: {episode_count}
Time span: {start_time} to {end_time}

Episodes in segment:
{episode_summaries}

Consider:
- Has the topic/discussion reached a natural conclusion?
- Are there unresolved threads that might continue?
- Is there enough substantive content for a meaningful summary?
- Would consolidating now lose important ongoing context?

Return JSON:
{{
  "ready_to_consolidate": true | false,
  "confidence": 0.0-1.0,
  "reasoning": "Why ready/not ready",
  "suggested_summary_focus": "What the summary should emphasize" | null
}}"""

# --- Consolidation Summarization ---
SUMMARIZATION_PROMPT: Final = """\
Summarize this conversation segment, preserving:
- Key facts mentioned
- Decisions made
- Opinions expressed
- Important context for future reference

Conversation:
{messages}

Previous context summary (if any):
{previous_summary}

Provide a concise summary that captures the essential information."""

# --- Importance Assessment (ForgettingEngine) ---
IMPORTANCE_ASSESSMENT_PROMPT: Final = """\
Assess the long-term importance of this memory.

Memory Content:
{content}

Memory Metadata:
- Created: {days_ago} days ago
- Last accessed: {last_accessed_days} days ago
- Access count: {access_count}
- Topics: {topics}
- ESS quality score: {ess_score}
- Consolidation level: {consolidation_level}

Related Beliefs:
{related_beliefs}

Consider:
- Does this contain unique, unrepeated information?
- Is this foundational to any current beliefs?
- Would forgetting this create inconsistency in the agent's worldview?
- Is this redundant with other memories?
- How central is this to the agent's identity/personality?

Return JSON:
{{
  "importance": 0.0-1.0,
  "should_retain": true | false,
  "reasoning": "Why important/unimportant",
  "is_foundational": true | false,
  "redundant_with": ["uid1", "uid2"] | null
}}"""

# --- Batch Forgetting ---
BATCH_FORGETTING_PROMPT: Final = """\
Review these memory candidates for potential archival.

Candidates:
{candidates_summary}

Agent's Current Identity Snapshot:
{snapshot_excerpt}

For each candidate, decide:
- KEEP: Important, unique, foundational
- ARCHIVE: Low importance but might be useful later
- FORGET: Redundant, trivial, or superseded

Return JSON:
{{
  "decisions": [
    {{"uid": "...", "action": "KEEP" | "ARCHIVE" | "FORGET", "reason": "..."}}
  ]
}}"""

# --- Belief Evidence Assessment ---
BELIEF_UPDATE_PROMPT: Final = """\
Assess how this new evidence affects the agent's belief about "{topic}".

Current Belief State:
- Opinion value: {current_value} (-1 to +1 scale)
- Confidence: {confidence}
- Supporting episodes: {supporting_count}
- Contradicting episodes: {contradicting_count}
- Current uncertainty: {uncertainty}

New Evidence (Episode):
{episode_content}

Episode Metadata:
- ESS Score: {ess_score}
- Reasoning Type: {reasoning_type}
- Source Reliability: {source_reliability}

Consider:
- Does this evidence genuinely support or contradict the belief?
- How strong/reliable is this evidence?
- Is this true contradiction or just nuance/complexity?
- Should this evidence significantly change confidence/uncertainty?
- Does this warrant AGM-style belief contraction?

Return JSON:
{{
  "direction": -1.0 to +1.0,
  "evidence_strength": 0.0-1.0,
  "new_uncertainty": 0.0-1.0,
  "reasoning": "Why this assessment",
  "is_major_update": true | false,
  "suggests_contraction": true | false
}}"""

# --- Health Assessment ---
HEALTH_ASSESSMENT_PROMPT: Final = """\
Assess the health and consistency of this agent's personality state.

Current Snapshot:
{snapshot}

Belief Summary:
{beliefs_summary}

Recent Shifts:
{recent_shifts}

Behavioral Metrics:
- Interaction count: {interaction_count}
- Disagreement rate: {disagreement_rate}
- Belief count: {belief_count}
- High-confidence beliefs: {high_conf_count}

Consider:
1. Is the agent showing signs of sycophancy (agreeing too readily)?
2. Is the snapshot coherent and substantive?
3. Are beliefs developing healthily or becoming ossified?
4. Is there evidence of identity drift from core values?
5. Are there contradictions between stated beliefs and behavioral patterns?

Return JSON:
{{
  "overall_health": "healthy" | "concerning" | "unhealthy",
  "concerns": ["list", "of", "specific", "issues"],
  "recommendations": ["suggested", "interventions"],
  "reasoning": "Explanation of assessment",
  "metrics": {{
    "coherence_score": 0.0-1.0,
    "consistency_score": 0.0-1.0,
    "growth_health_score": 0.0-1.0
  }}
}}"""

# --- Semantic Feature Extraction ---
FEATURE_EXTRACTION_PROMPT: Final = """\
Analyze this conversation and extract semantic features about the agent's personality,
preferences, knowledge, or relationships.

Episode:
{episode_content}

Category to extract for: {category}

Existing features in this category:
{existing_features}

Return JSON commands:
{{
  "commands": [
    {{"command": "add", "tag": "Communication Style", "feature": "humor_style",
     "value": "dry wit with occasional puns", "confidence": 0.8}},
    {{"command": "update", "tag": "Technical Skills", "feature": "python_level",
     "value": "advanced", "confidence": 0.9}},
    {{"command": "delete", "tag": "Preferences", "feature": "old_preference",
     "reason": "contradicted by new evidence"}}
  ]
}}"""
