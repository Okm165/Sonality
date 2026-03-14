"""All LLM prompt templates for the memory architecture.

Each prompt is a string template with named placeholders. All prompts return
structured JSON validated by corresponding Pydantic models in the caller modules.

Design principles:
  - Examples show JSON structure with bracket placeholders, NEVER real-world
    facts — avoids biasing reasoning models toward specific factual content.
  - Valid values for enum fields are listed separately after the JSON block,
    never inline (A | B) which confuses low-quality models.
  - Confidence calibration is based purely on evidence quality (source
    attribution, specificity) — prompts never assume parametric knowledge.
"""

from __future__ import annotations

from typing import Final

# --- Semantic Chunking (DerivativeChunker) ---
CHUNKING_PROMPT: Final = """\
Split this text into semantically coherent chunks for memory retrieval.

Text:
{text}

Rules:
- Each chunk is a self-contained idea (1-3 sentences)
- Maximum 15 chunks
- importance: high (key claim/fact), medium (supporting detail), low (context only)

Output ONLY a JSON object in this exact format (replace placeholders with actual chunks from the text above):
{{"chunks": [{{"text": "[First key claim or factoid from the text]", "key_concept": "[2-4 word topic label]", "importance": "high"}}, {{"text": "[A supporting detail from the text]", "key_concept": "[2-4 word topic label]", "importance": "medium"}}]}}"""

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

Respond with ONLY a JSON object. Example for a topic shift:
{{"boundary_decision": "BOUNDARY", "confidence": 0.9, "boundary_type": "topic_shift", "reasoning": "User switched from [topic A] to [topic B].", "suggested_segment_label": "[topic B] discussion"}}

Example for continuation:
{{"boundary_decision": "CONTINUE", "confidence": 0.8, "boundary_type": "none", "reasoning": "User is elaborating on the previous topic.", "suggested_segment_label": ""}}

boundary_decision must be BOUNDARY or CONTINUE.
boundary_type must be: topic_shift, goal_change, explicit_transition, or none."""

# --- Reflection Gate Decision ---
REFLECTION_GATE_PROMPT: Final = """\
Decide whether the agent should run reflection this turn.

Current interaction: {interaction_count}
Interactions since last reflection: {window_interactions}
Target cadence: every ~{target_cadence} interactions
Pending insights: {pending_insights}
Pending staged belief updates: {staged_updates}
Recent shift magnitude since last reflection: {recent_shift_magnitude}
Current disagreement rate: {disagreement_rate}
Tracked belief count: {belief_count}

HARD RULE: Return SKIP if window_interactions < 5, regardless of other factors.
EVENT_DRIVEN is only valid when window_interactions >= 5 AND pending_insights >= 3 AND recent_shift_magnitude >= 0.5.
PERIODIC is valid when window_interactions >= target_cadence.

Choose:
- SKIP: window < 5, or no meaningful new synthesis needed
- PERIODIC: enough elapsed context to do a maintenance reflection
- EVENT_DRIVEN: significant accumulation (see HARD RULE above)

Respond with ONLY a JSON object. Example:
{{"trigger": "SKIP", "reasoning": "Only 3 interactions since last reflection, below the minimum threshold."}}

trigger must be exactly SKIP, PERIODIC, or EVENT_DRIVEN."""

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

Respond with ONLY a JSON object (fill in YOUR values — do NOT copy this example verbatim):
{{"category": "TEMPORAL", "depth": "MODERATE", "temporal_expansion": "EXPAND", "semantic_memory": "SKIP", "reasoning": "User is asking about events from a previous session."}}

category must be: NONE, SIMPLE, TEMPORAL, MULTI_ENTITY, AGGREGATION, or BELIEF_QUERY.
depth must be: MINIMAL, MODERATE, or DEEP.
temporal_expansion must be: EXPAND or NO_EXPAND.
semantic_memory must be: SEARCH or SKIP."""

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

Respond with ONLY a JSON object. Example for sufficient context:
{{"sufficiency_decision": "SUFFICIENT", "confidence": 0.85, "reasoning": "Retrieved context directly addresses all aspects of the query.", "suggested_refinement": null}}

Example for insufficient context:
{{"sufficiency_decision": "INSUFFICIENT", "confidence": 0.4, "reasoning": "Missing information about the timeline.", "suggested_refinement": "When did the user first mention this topic?"}}

sufficiency_decision must be SUFFICIENT or INSUFFICIENT."""

# --- Query Decomposition (SplitQueryAgent) ---
DECOMPOSITION_PROMPT: Final = """\
Decompose this query into independent sub-queries for parallel retrieval.

Query: {query}

Guidelines:
- Each sub-query should be answerable independently
- Include entity/topic-specific constraints
- Maximum 4 sub-queries
- Each should retrieve distinct information

Respond with ONLY a JSON object. Example:
{{"sub_queries": ["What did the user say about [topic A]?", "What is the agent's position on [topic B]?"], "aggregation_strategy": "merge"}}

aggregation_strategy must be: merge, compare, or timeline."""

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

Respond with ONLY a JSON object. Example (if 3 candidates, ranking by relevance):
{{"ranking": [3, 1, 2], "reasoning": "Candidate 3 is most directly relevant; candidate 1 provides useful context; candidate 2 is tangential."}}

ranking must list every candidate index exactly once."""

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

Respond with ONLY a JSON object. Example:
{{"readiness_decision": "READY", "confidence": 0.8, "reasoning": "The topic has reached a natural conclusion with no unresolved threads.", "suggested_summary_focus": "Focus on the key arguments and evidence discussed"}}

readiness_decision must be READY or NOT_READY.
suggested_summary_focus should be null if not ready."""

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

# --- Batch Forgetting ---
BATCH_FORGETTING_PROMPT: Final = """\
Review these memory candidates for potential archival.

Candidates:
{candidates_summary}

Agent's Current Identity Snapshot:
{snapshot_excerpt}

For each candidate, decide:
- KEEP: Important, unique, foundational, or frequently accessed
- ARCHIVE: Low importance but might be useful later; low access count
- FORGET: Redundant, trivial, superseded, or never accessed after storage

Signals that favor KEEP: high ESS, high access count, recent last_accessed, unique topic.
Signals that favor FORGET: ESS < 0.1, access count = 0, superseded by another episode, trivial content.

Respond with ONLY a JSON object. Example:
{{
  "decisions": [
    {{"uid": "ep-abc123", "action": "KEEP", "reason": "Foundational belief formation, accessed 3 times."}},
    {{"uid": "ep-def456", "action": "FORGET", "reason": "Redundant with ep-abc123; ESS 0.05, never accessed."}}
  ]
}}

action must be KEEP, ARCHIVE, or FORGET for each decision."""

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

Uncertainty calibration (IMPORTANT — do NOT leave uncertainty high when evidence accumulates):
- First evidence on a topic: new_uncertainty 0.6–0.9 (high, new territory)
- 2+ supporting episodes with no contradictions: new_uncertainty ≤ 0.5 (confidence is building)
- 3+ supporting episodes with no contradictions: new_uncertainty ≤ 0.3 (well-supported belief)
- Contradicting evidence: increase uncertainty (higher new_uncertainty)
- Mixed evidence (some support, some contradict): new_uncertainty 0.4–0.7

Output ONLY a JSON object (fill in YOUR values — do NOT copy this example):
{{"direction": 0.3, "evidence_strength": 0.6, "new_uncertainty": 0.35, "reasoning": "Well-sourced evidence warrants a modest positive shift; second supporting episode reduces uncertainty.", "update_magnitude": "MINOR", "contraction_action": "NONE"}}

direction: float -1.0 to +1.0 (negative = contradicts belief, positive = supports).
evidence_strength and new_uncertainty: floats 0.0 to 1.0.
update_magnitude: MAJOR (large shift ≥0.3), MINOR (small shift <0.3), or NONE (no shift).
contraction_action: CONTRACT or NONE."""

# --- Structural Disagreement Detection ---
DISAGREEMENT_DETECTION_PROMPT: Final = """\
Determine if the user's message structurally disagrees with the agent's position.

User Message: {user_message}
Agent Position on Topic "{topic}": {position_value} (-1 to +1 scale)
User Opinion Direction: {opinion_direction}

Consider:
- Is the user presenting an argument against the agent's position?
- Is this genuine disagreement or simply different emphasis?
- Does the user provide evidence or reasoning for their opposing view?

Respond with ONLY a JSON object. Example:
{{"disagreement_verdict": "DISAGREEMENT", "disagreement_strength": 0.7, "reasoning": "User directly challenges agent's position with a counter-argument."}}

disagreement_verdict must be DISAGREEMENT or NO_DISAGREEMENT.
disagreement_strength is a float from 0.0 to 1.0."""

# --- Belief Decay Decision ---
BELIEF_DECAY_PROMPT: Final = """\
Assess whether this belief should be retained or decayed based on staleness.

Belief Topic: {topic}
Current Position: {position}
Current Confidence: {confidence}
Evidence Count: {evidence_count}
Interactions Since Last Reinforced: {gap}
Total Interactions: {total_interactions}

Consider:
- How central is this belief to the agent's identity?
- Has enough time passed that this belief might be outdated?
- Is this a foundational belief that should persist regardless of reinforcement?
- Would forgetting this create inconsistency?

Output ONLY a JSON object (fill in YOUR values — do NOT copy this example verbatim):
{{"action": "RETAIN", "new_confidence": 0.72, "reasoning": "Core belief supported by 4 episodes; no contradictory evidence has emerged."}}

Example for decay:
{{"action": "DECAY", "new_confidence": 0.35, "reasoning": "[N] interactions without reinforcement suggests topic is no longer salient."}}

action must be exactly one of: RETAIN, DECAY, or FORGET (pick one word, no pipes or alternatives).
new_confidence must be a decimal number between 0.0 and 1.0."""

# --- Entrenchment Detection ---
ENTRENCHMENT_DETECTION_PROMPT: Final = """\
Assess if this belief shows signs of entrenchment (echo chamber effect).

Belief Topic: {topic}
Current Position: {position} (-1 to +1)
Recent Updates: {recent_updates}
Supporting Episodes: {supporting_count}
Contradicting Episodes: {contradicting_count}

Signs of entrenchment:
- Updates consistently agree with current position
- Few or no contradicting episodes considered
- High confidence despite limited evidence diversity

Respond with ONLY a JSON object. Example:
{{"entrenchment_status": "NOT_ENTRENCHED", "confidence": 0.75, "reasoning": "Multiple contradicting episodes have been considered.", "recommendation": "Continue monitoring for evidence diversity."}}

entrenchment_status must be ENTRENCHED or NOT_ENTRENCHED.
confidence is a float from 0.0 to 1.0."""

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

Consider ONLY these high-signal health markers:
1. Sycophancy: disagreement_rate < 0.05 with more than 5 interactions is a concern.
2. Snapshot coherence: snapshot shorter than 200 characters after 5+ interactions is a concern.
3. Belief ossification: any topic with confidence > 0.95 AND evidence_count == 1 is suspicious.
4. Identity drift: snapshot directly contradicts earlier core values stated explicitly.
5. Confidence contradiction: a belief in the snapshot described as "core" or "primary" must have confidence > 0.10; values 0.05–0.20 are normal for beliefs under 5 interactions.

Do NOT flag: low confidence on newly-formed beliefs (< 3 supporting episodes); low confidence on topics with mixed evidence (contradicting_count > 0); minor wording variations between snapshot and beliefs.
Concerns list should be EMPTY for "healthy" unless there is a clear, specific, quantifiable problem.

Respond with ONLY a JSON object. Example:
{{
  "overall_health": "healthy",
  "concerns": [],
  "recommendations": [],
  "reasoning": "Core identity is stable with appropriate disagreement rate.",
  "metrics": {{"coherence_score": 0.8, "consistency_score": 0.75, "growth_health_score": 0.7}}
}}

overall_health must be: healthy, concerning, or unhealthy.
All metric scores are floats from 0.0 to 1.0."""

# Valid tags per category — restricts LLM from cross-pollinating category names.
FEATURE_TAGS: Final[dict[str, str]] = {
    "personality": "Communication Style, Values, Behavioral Traits, Temperament, Cognitive Style",
    "preferences": "Interests, Aversions, Decision Framework, Domains, Styles, Preferences",
    "knowledge": "Domain, Technical Skills, Scientific Fields, Academic Topics, Methodology",
    "relationships": "Interpersonal Style, Social Dynamics, Collaborative Patterns, Stance",
}

# --- Semantic Feature Extraction ---
FEATURE_EXTRACTION_PROMPT: Final = """\
Analyze this conversation and extract semantic features about the agent's {category}.

Episode:
{episode_content}

Category: {category}
Valid tags for this category (ONLY use these, no other tags allowed): {tags}

Existing features in this category:
{existing_features}

DELETION RULES (strictly enforced):
- NEVER issue a delete command unless the current episode contains a direct, new, assertive counter-claim that explicitly contradicts the feature's factual content.
- A topic shift does NOT justify deletion. If the episode is about [topic A], do NOT delete [topic B] or [topic C] features.
- Silence or absence is NOT a contradiction. Only a direct counter-assertion is.
- Acknowledging the emotional validity of another's position, expressing empathy, or paraphrasing a previous discussion is NOT a contradiction — do NOT delete based on empathetic language.
- When ESS line shows emotional_appeal, social_pressure, debunked_claim, or anecdotal: issue NO delete commands. Only add or update communication-style features.
- If deleting, you MUST fill the "reason" field with the exact new assertive phrase from the episode that contradicts the feature.

Your response must be ONLY this JSON object with actual values filled in (no {{"..."}}, no placeholders):
{{
  "commands": [
    {{"command": "add", "tag": "[valid tag]", "feature": "[feature_name]", "value": "[description from episode]", "confidence": 0.8, "reason": ""}},
    {{"command": "update", "tag": "[valid tag]", "feature": "[feature_name]", "value": "[updated value from episode]", "confidence": 0.9, "reason": ""}},
    {{"command": "delete", "tag": "[valid tag]", "feature": "[feature_name]", "value": "", "confidence": 0.9, "reason": "[exact contradicting phrase from episode]"}}
  ]
}}
If no features should be added/updated/deleted, return: {{"commands": []}}
command must be add, update, or delete. confidence is a float from 0.0 to 1.0.
IMPORTANT: tag must be one of the valid tags listed above."""

# --- Semantic Feature Consolidation ---
FEATURE_CONSOLIDATION_PROMPT: Final = """\
You are reviewing semantic features in the "{category}" category to find redundant pairs to merge.

Features to review:
{features}

Merge ONLY features that describe the exact same trait with different wording. Do NOT merge distinct behaviors.
Keep the reasoning field to ONE short sentence. Do not analyse individual UIDs in the reasoning field.

Your response must be a single JSON object with this exact structure.

No-merge example:
{{"consolidation_decision": "SKIP", "reasoning": "All features are distinct.", "actions": []}}

Merge example:
{{"consolidation_decision": "CONSOLIDATE", "reasoning": "Two features describe the same trait.", "actions": [{{"source_uid": "[uid-to-remove]", "target_uid": "[uid-to-keep]", "canonical_tag": "[matching tag]", "canonical_feature": "[canonical name]", "canonical_value": "[best description]", "reason": "duplicate"}}]}}

Respond with the JSON object now:"""

# --- Window Context Summary (SLIDE-inspired) ---
# Generates a brief factual summary of the preceding window so the next window
# can resolve cross-boundary references without raw context overflow.
WINDOW_CONTEXT_SUMMARY_PROMPT: Final = """\
Summarize the KEY entities, facts, and ongoing topics in this text passage \
in 2-4 sentences. Focus on proper nouns, numbers, relationships, and any \
claims that might be referenced in subsequent text. Do NOT interpret or \
evaluate — just enumerate what was discussed.

Text:
{text}

Output ONLY a JSON object: {{"summary": "[your 2-4 sentence summary here]"}}"""

# --- Knowledge Proposition Extraction ---
# Five-stage pipeline synthesizing state-of-the-art approaches:
#   1. Selection (Claimify, ACL 2025)
#   2. Disambiguation + Decontextualization (FactReasoner, EMNLP 2025)
#   3. Decomposition into molecular facts (Gunjal & Durrett, EMNLP 2024)
#   4. Classification + Confidence calibration (ConFix, 2024)
#   5. Quality gate — reject under-decontextualized or over-atomized props
#
# Each proposition is a "molecular fact" (Dense X Retrieval, Chen et al.,
# EMNLP 2024): self-contained enough to verify independently, minimal enough
# to represent exactly one factoid, and context-rich enough to avoid the
# "context collapse" problem identified by PropRAG (EMNLP 2025).
KNOWLEDGE_EXTRACTION_PROMPT: Final = """\
You are a knowledge extraction system. Process this text through five stages.

STAGE 1 — SELECT: Read the entire text and identify sentences containing \
learnable information (facts, data, mechanisms, attributed opinions, \
scientific claims). SKIP: greetings, filler, meta-commentary ("by the way"), \
emotional expressions, rhetorical questions with no factual content.

STAGE 2 — DECONTEXTUALIZE: For each selected sentence, make it fully \
standalone by replacing ALL pronouns, references, and implicit subjects \
with their explicit referents from the surrounding text. \
"It was discovered in [year]" → "[Named entity from context] was discovered in [year]." \
"They reported a [N]% increase" → "[Named researchers/institution] reported a [N]% increase in [specific metric]." \
"The experiment showed…" → "[Author]'s [year] experiment at [institution] showed…" \
If the referent cannot be determined from context, SKIP the sentence entirely.

STAGE 3 — DECOMPOSE into molecular propositions. Each proposition must:
- Be self-contained: a reader with NO access to the original text can understand it
- Be minimal: contain exactly ONE factoid (one number, one relationship, one event)
- Include explicit subjects, dates, quantities, units, and source attributions
- NOT use pronouns or relative references ("this", "that", "the above")
BAD: "It measures [value] [units]." (what measures? unresolvable)
BAD: "The value ranges from X to Y." (value of what?)
GOOD: "[Named subject]'s [property] measures [value] [units] under [conditions]."
GOOD: "[Entity] [verb] [measurement] according to [Named Source, Year]."

STAGE 4 — CLASSIFY and calibrate confidence for each proposition:
Types:
- fact: objectively verifiable claim with concrete details (names, numbers, dates, mechanisms)
- opinion: subjective judgment or preference — ALWAYS attribute to its source ("The user believes...", "According to X...")
- speculation: hedged/uncertain claim ("might", "could", "potentially", "is expected to")
- noise: filler, non-substantive content (EXCLUDE these)

Confidence calibration — based SOLELY on evidence quality in the text:
- 0.85-0.95: Named reputable source (journal, institution, named report) + concrete data
- 0.65-0.84: Specific and verifiable but source informal or partially named
- 0.40-0.64: General knowledge claim without specific attribution or data
- 0.15-0.39: Vague, hedged, or from anonymous/dubious sources ("someone told me", "I read somewhere")
- 0.01-0.14: Extraordinary claims without any supporting evidence in the text

STAGE 5 — QUALITY GATE: Before including ANY proposition, run this checklist:
□ SUBJECT: Does it name a specific entity, person, or thing? \
  Reject if it starts with "it", "they", "this", "that", "these", "those", \
  "he", "she", or any pronoun. Fix by replacing the pronoun with the referent.
□ STANDALONE: Could a reader with ZERO context understand and evaluate it?
□ ATOMIC: Is it ONE factoid, not two claims joined by "and" or "which also"?
□ ATTRIBUTED: If it's a claim, is the source named or is it clearly unattributed?
If any check fails, either fix the proposition or DROP it entirely.

Text to extract from:
{text}

CRITICAL RULES:
- "According to [source]" is a fact if the source is named and claim is verifiable
- Bare assertions without evidence ("X is the best") are opinions, not facts
- Claims from anonymous or dubious sources get LOW confidence (0.01-0.39)
- Propositions with unresolved pronouns FAIL the quality gate — fix or drop them
- Maximum 15 propositions — prefer fewer high-quality over many low-quality
- Every output proposition MUST pass ALL four quality gate checks above

Output ONLY a JSON object (replace bracket placeholders with actual content from the text):
{{"propositions": [\
{{"text": "[Subject] [verb] [specific measurement with units] according to [Named Source, Year].", "type": "fact", "confidence": 0.88, "source_entity": "[source name]", "key_concepts": ["[topic1]", "[topic2]", "[topic3]"], "sentiment": 0.0}}, \
{{"text": "The user believes [subject] is [positive judgment].", "type": "opinion", "confidence": 0.40, "source_entity": "user", "key_concepts": ["[topic]"], "sentiment": 0.8}}, \
{{"text": "The user believes [subject] is [negative judgment].", "type": "opinion", "confidence": 0.40, "source_entity": "user", "key_concepts": ["[topic]"], "sentiment": -0.7}}, \
{{"text": "[Subject] [verb] approximately [value] [units].", "type": "fact", "confidence": 0.50, "source_entity": "", "key_concepts": ["[topic1]", "[topic2]"], "sentiment": 0.0}}]}}

type must be: fact, opinion, speculation, or noise.
confidence: 0.0-1.0 calibrated per Stage 4 rules — source quality determines confidence.
source_entity: who made the claim (empty string for unattributed claims).
key_concepts: 1-3 topic labels for embedding and retrieval. For opinion-type propositions, key_concepts[0] must name the concrete subject-matter domain or real-world phenomenon being evaluated — NOT the evidential properties, verification status, or statistical characteristics of the claim itself. Use the domain being discussed, not how the claim is supported.
sentiment: opinion stance toward key_concepts[0] — +1.0 = strongly favorable, -1.0 = strongly unfavorable, 0.0 = neutral/not applicable (use 0.0 for facts and speculations)."""

# --- Knowledge Consolidation (Reflection) ---
# Uses EDC-style canonicalization (2025) for merges and FactReasoner-style
# entailment reasoning for contradiction detection.
KNOWLEDGE_CONSOLIDATION_PROMPT: Final = """\
Review these knowledge propositions stored by an AI agent. Identify issues \
and suggest consolidation actions.

Stored propositions:
{propositions}

Agent's current personality snapshot:
{snapshot}

Tasks:
1. CONTRADICTIONS: Find proposition pairs that cannot both be true \
   (e.g. two propositions stating different values for the same measurement). \
   For each pair, determine which has stronger evidence based on source quality \
   and confidence, and recommend keeping one. State "keep a" or "keep b" \
   clearly in the resolution.
2. MERGES: Find propositions that convey the same factoid with different \
   wording. Before merging, verify they truly mean the same thing — \
   two paraphrases of the same claim are mergeable; two claims about \
   the same topic but stating different facts are NOT. Provide a \
   canonical statement.
3. OPINION CANDIDATES: Facts the agent should form a stance on based on \
   accumulated evidence (multiple supporting propositions on the same topic).
4. WEAK PROPOSITIONS: Claims that are vague, lack concrete details, or \
   have confidence < 0.25 and no supporting evidence from other propositions.

CRITICAL: Proposition text in your output must be EXACT copies from the list \
above. Do not paraphrase or modify the stored text — the system matches on \
exact string equality.

Output ONLY a JSON object. The resolution field for contradictions MUST start \
with exactly "keep a" or "keep b" followed by a dash and reason:

{{"contradictions": [{{"a": "exact text of proposition A", "b": "exact text of proposition B", "resolution": "keep a — stronger source attribution"}}], \
"merges": [{{"sources": ["exact prop text 1", "exact prop text 2"], "merged": "single canonical statement"}}], \
"opinion_candidates": [{{"proposition": "exact prop text", "suggested_stance": "brief reasoning"}}], \
"weak_propositions": ["exact text of weak proposition"]}}

Use empty arrays for categories with no findings."""

# --- Topic Canonicalization ---
# Maps newly-extracted ESS topics to canonical forms already tracked in belief memory.
# Used to prevent "nuclear" and "nuclear energy" accreting as separate beliefs when
# the agent has already encountered the concept under one name.
TOPIC_CANONICALIZATION_PROMPT: Final = """\
You manage a belief memory system that tracks concepts across conversations.

Existing tracked concepts:
{existing}

New topics extracted this turn:
{new_topics}

For each new topic decide: is it the SAME concept as an existing one, differing only \
in label (a true synonym, common abbreviation, or alternate spelling for the identical \
referent)? If yes, return the existing name exactly. If no, return the new topic unchanged.

Only merge when both terms point to exactly the same real-world entity or idea. \
Do NOT merge concepts that merely share a domain, a cause-effect relationship, \
a part-whole relationship, or a difference in specificity/scope — those must stay separate.

When uncertain, keep them separate.

Output ONLY a JSON object mapping every new topic to its canonical form:
{{"mappings": {{"new_topic": "canonical_name"}}}}"""
