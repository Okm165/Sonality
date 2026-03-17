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

Output ONLY a JSON object with real chunks from the text above. NEVER output "..." or bracket placeholders as values.
Example format: {{"chunks": [{{"text": "The Eiffel Tower was completed in 1889 and stands 330 meters tall.", "key_concept": "Eiffel Tower construction", "importance": "high"}}, {{"text": "It was the tallest man-made structure in the world for 41 years.", "key_concept": "Eiffel Tower records", "importance": "medium"}}]}}"""

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

Your response must end with ONLY a JSON object (no markdown, no explanation after it).

Topic shift example:
{{"boundary_decision": "BOUNDARY", "confidence": 0.9, "boundary_type": "topic_shift", "reasoning": "User switched from climate policy to nuclear energy.", "suggested_segment_label": "Nuclear energy discussion"}}

Continuation example:
{{"boundary_decision": "CONTINUE", "confidence": 0.8, "boundary_type": "none", "reasoning": "User is elaborating on the previous topic.", "suggested_segment_label": ""}}

boundary_decision must be BOUNDARY or CONTINUE.
boundary_type must be: topic_shift, goal_change, explicit_transition, or none."""

# --- Reflection Gate Decision ---
REFLECTION_GATE_PROMPT: Final = """\
Decide whether the agent should run reflection this turn.

interaction_count={interaction_count} (total interactions)
window_interactions={window_interactions} (interactions since last reflection)
target_cadence={target_cadence}
pending_insights={pending_insights}
staged_updates={staged_updates}
recent_shift_magnitude={recent_shift_magnitude}
disagreement_rate={disagreement_rate}
belief_count={belief_count}

Rules:
- SKIP if window_interactions < 5 (hard rule)
- EVENT_DRIVEN only if window_interactions >= 5 AND pending_insights >= 3 AND recent_shift_magnitude >= 0.5
- PERIODIC if window_interactions >= target_cadence

Your response must end with ONLY a JSON object:
{{"trigger": "SKIP", "reasoning": "Only 3 interactions since last reflection."}}

trigger must be exactly SKIP, PERIODIC, or EVENT_DRIVEN."""

# --- Query Routing ---
QUERY_ROUTING_PROMPT: Final = """\
Classify this query to choose the optimal memory retrieval strategy.

Query: {query}
Recent conversation context: {context}

Categories (pick the MOST SPECIFIC match):
1. NONE - Pure chitchat, greetings, acknowledgments with zero factual content
2. SIMPLE - Single topic lookup: "what do you know about X?", "tell me about X", "do you remember X?" — use this for most knowledge questions
3. TEMPORAL - Needs chronological ordering: "what happened first?", "how has X changed over time?", "what did we discuss before Y?" — only for true timeline queries
4. MULTI_ENTITY - Explicit comparison of 2+ distinct entities: "compare X and Y", "how do X and Y differ?"
5. AGGREGATION - Cross-conversation synthesis: "summarize everything about X", "what patterns have you noticed?" — only for synthesis tasks spanning many episodes
6. BELIEF_QUERY - Explicitly asking about the agent's own view: "what do you think about X?", "what's your opinion on X?", "do you believe X?"

Key distinctions:
- "What do you know about X?" → SIMPLE (single lookup), NOT TEMPORAL
- "What's your view on X?" → BELIEF_QUERY, NOT SIMPLE
- "What did we discuss earlier?" → TEMPORAL (needs chronological order)
- "How did you handle X earlier in this conversation?" → TEMPORAL (retrospective, needs conversation history)
- "Looking back on this conversation, how did you respond to Y?" → TEMPORAL (retrospective self-assessment)
- Presenting factual claims for the agent to respond to → NONE (no retrieval needed)

Also determine:
- depth: MINIMAL (1-2 episodes), MODERATE (5-7), DEEP (10-15)
- temporal_expansion: EXPAND only for true TEMPORAL queries; NO_EXPAND for everything else
- semantic_memory: SEARCH if the query touches on personality/beliefs/preferences; SKIP otherwise

Respond with ONLY a JSON object (fill in YOUR values, not this example):
{{"category": "SIMPLE", "depth": "MODERATE", "temporal_expansion": "NO_EXPAND", "semantic_memory": "SKIP", "reasoning": "Single topic lookup — what does the agent know about X."}}

category must be: NONE, SIMPLE, TEMPORAL, MULTI_ENTITY, AGGREGATION, or BELIEF_QUERY.
depth must be: MINIMAL, MODERATE, or DEEP.
temporal_expansion must be: EXPAND or NO_EXPAND.
semantic_memory must be: SEARCH or SKIP.
Your response must end with ONLY the JSON object."""

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

sufficiency_decision must be SUFFICIENT or INSUFFICIENT.
Your response must end with ONLY the JSON object."""

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
{{"sub_queries": ["What did the user say about climate change?", "What is the agent's position on nuclear energy?"], "aggregation_strategy": "merge"}}

aggregation_strategy must be: merge, compare, or timeline.
Your response must end with ONLY the JSON object."""

# --- LLM Listwise Reranking ---
RERANK_PROMPT: Final = """\
Rank these candidate episodes by topical relevance to the query. \
A candidate is relevant only if it is about the SAME topic or subject matter as the query. \
Do NOT rank a candidate higher because it contains rich factual detail if that detail is \
about a completely different topic.

Query: {query}

Candidates:
{numbered_candidates}

Ranking criteria (strict priority order):
1. Topical match: Is the candidate about the same subject/topic as the query?
2. Directness: Does it directly address the query without detours?
3. Recency: Among topically matched candidates, prefer more recent ones.

Assign rank 1 to the MOST relevant. Episodes about completely different topics should be \
ranked last even if they are factually detailed.

Respond with ONLY a JSON object. Example (if 3 candidates):
{{"ranking": [3, 1, 2], "reasoning": "Candidate 3 is on-topic; candidate 1 partially relevant; candidate 2 is about a different subject."}}

ranking must list every candidate index exactly once.
Your response must end with ONLY the JSON object."""

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
suggested_summary_focus should be null if not ready.
Your response must end with ONLY the JSON object."""

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

direction: float -1.0 to +1.0. Positive means the evidence argues FOR this topic being real/important/valid (should push the opinion MORE POSITIVE). Negative means the evidence argues AGAINST this topic being real/important/valid (should push the opinion MORE NEGATIVE). The scale is absolute — not relative to the current_value.
evidence_strength and new_uncertainty: floats 0.0 to 1.0.
update_magnitude: MAJOR (large shift ≥0.3), MINOR (small shift <0.3), or NONE (no shift).
contraction_action: CONTRACT or NONE.
Your response must end with ONLY the JSON object."""

# --- Batch Belief Evidence Assessment ---
BATCH_BELIEF_UPDATE_PROMPT: Final = """\
Assess how this new evidence affects the agent's beliefs on multiple topics.

Episode:
{episode_content}

Episode Metadata:
- ESS Score: {ess_score}
- Reasoning Type: {reasoning_type}
- Source Reliability: {source_reliability}

Topics to assess (each entry describes current belief state):
- current_value: current opinion value (-1 to +1, where 0=neutral/unknown)
- confidence: how confident the agent is in this belief (0=none, 1=certain)
- supporting_count: number of past episodes that supported this belief
- contradicting_count: number of past episodes that contradicted this belief
- uncertainty: current uncertainty (0=certain, 1=completely unknown)
{topics_json}

For each topic, consider:
- Does this episode genuinely support or contradict the belief?
- How strong is this evidence given the ESS score and reasoning type?
- Is this a true contradiction or just nuance/complexity?
- Does this warrant AGM-style belief contraction?

Produce one assessment per topic. Output ONLY a JSON object with an "assessments" array — one entry per topic listed above (copy the exact topic name, fill in YOUR assessed values):
{{"assessments": [{{"topic": "exact-topic-name-from-above", "direction": 0.3, "evidence_strength": 0.6, "new_uncertainty": 0.35, "reasoning": "Evidence moderately supports the belief with a credible source and specific data.", "update_magnitude": "MINOR", "contraction_action": "NONE"}}]}}

Uncertainty calibration (use supporting_count and contradicting_count from each topic):
- supporting_count=0 (first evidence): new_uncertainty 0.6–0.9
- supporting_count≥2, contradicting_count=0: new_uncertainty ≤ 0.5
- supporting_count≥3, contradicting_count=0: new_uncertainty ≤ 0.3
- contradicting evidence present: increase uncertainty
- mixed evidence: new_uncertainty 0.4–0.7

direction: float -1.0 to +1.0. Positive means the evidence argues FOR this topic being real/important/valid (should push opinion MORE POSITIVE). Negative means the evidence argues AGAINST it. Absolute scale — not relative to the current belief.
evidence_strength and new_uncertainty: floats 0.0 to 1.0.
update_magnitude: MAJOR (shift ≥0.3), MINOR (shift <0.3), or NONE.
contraction_action: CONTRACT or NONE.
Your response must end with ONLY the JSON object."""

# --- Structural Disagreement Detection (batch — all relevant topics in one call) ---
DISAGREEMENT_DETECTION_PROMPT: Final = """\
Determine if the user's message structurally disagrees with any of the agent's current positions.

User Message: {user_message}
User Opinion Direction: {opinion_direction} (sign: +1=supports topic, -1=opposes topic)

Agent's current positions on relevant topics (scale -1 to +1, negative=against, positive=for):
{topics_and_positions}

Consider:
- Is the user presenting an argument against ANY of these positions?
- Is this genuine disagreement or simply different emphasis?
- Does the user provide evidence or reasoning for their opposing view?

Your response must end with ONLY a JSON object:
{{"disagreement_verdict": "DISAGREEMENT", "disagreement_strength": 0.7, "reasoning": "User directly challenges nuclear energy position with a counter-argument."}}

disagreement_verdict must be DISAGREEMENT or NO_DISAGREEMENT.
disagreement_strength is a float from 0.0 to 1.0."""

# --- Batch Belief Decay ---
BATCH_BELIEF_DECAY_PROMPT: Final = """\
Assess whether each of these beliefs should be retained, decayed, or forgotten based on staleness.

Total interactions so far: {total_interactions}

Beliefs to assess:
{beliefs_json}

For each belief consider:
- How central is it to the agent's identity?
- Has enough time passed (gap) that it might be outdated?
- Is it foundational and should persist regardless of reinforcement?

Output ONLY a JSON array, one entry per belief:
[{{"topic": "nuclear_energy", "action": "RETAIN", "new_confidence": 0.72, "reasoning": "Belief is well-supported and recently reinforced; no staleness detected."}}]

action must be exactly RETAIN, DECAY, or FORGET.
new_confidence must be 0.0–1.0.
RETAIN keeps confidence unchanged; DECAY reduces it; FORGET removes the belief entirely.
Your response must end with ONLY the JSON array."""

# --- Batch Entrenchment Detection ---
BATCH_ENTRENCHMENT_DETECTION_PROMPT: Final = """\
Assess which of these beliefs show signs of echo-chamber entrenchment.

Beliefs to assess:
{beliefs_json}

Signs of entrenchment:
- Updates consistently agree with current position
- Few or no contradicting episodes considered
- High confidence despite limited evidence diversity

Output ONLY a JSON array, one entry per entrenched belief (omit non-entrenched ones):
[{{"topic": "nuclear_energy", "reasoning": "All 5 updates agreed with the existing stance; no contradictory episodes considered."}}]

Return an empty array [] if no entrenchment detected.
Your response must end with ONLY the JSON array."""

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
All metric scores are floats from 0.0 to 1.0.
Your response must end with ONLY the JSON object."""

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

EXTRACTION RULES (strictly enforced):
- Maximum 4 commands per pass. Prefer updating existing features over adding new ones.
- Before adding a feature, check if any existing feature already captures the same trait — if yes, issue UPDATE instead of ADD. Duplicate information costs more than it saves.
- Maintaining an evidence-based position under social pressure, bare assertions, or peer pressure is INTELLECTUAL INTEGRITY, not intellectual rigidity. Do NOT add an "Intellectual Rigidity" feature when the agent correctly resists non-evidential pressure. Only use Intellectual Rigidity if the agent explicitly refuses to engage with a NEW, specific, evidence-backed argument.
- Do NOT add multiple features for the same underlying trait expressed in different words (e.g., "rejects mixed narratives", "dismisses inconclusive framing", "critical of balanced views" are all the same trait — pick one).
- Do NOT use negative interpersonal labels (condescending, dismissive, arrogant, cold, harsh, rude, blunt, combative) when the agent is engaging intellectually with arguments, calling out logical fallacies, or firmly disagreeing. Forceful rebuttal of a weak argument is evidence-based engagement, not a personality flaw. Only use these negative labels when the agent explicitly disrespects the USER AS A PERSON rather than challenging their argument.

DELETION RULES (strictly enforced):
- NEVER issue a delete command unless the current episode contains a direct, new, assertive counter-claim that explicitly contradicts the feature's factual content.
- A topic shift does NOT justify deletion. If the episode is about [topic A], do NOT delete [topic B] or [topic C] features.
- Silence or absence is NOT a contradiction. Only a direct counter-assertion is.
- Acknowledging the emotional validity of another's position, expressing empathy, or paraphrasing a previous discussion is NOT a contradiction — do NOT delete based on empathetic language.
- When ESS line shows emotional_appeal, social_pressure, debunked_claim, or anecdotal: issue NO delete commands. Only add or update communication-style features.
- If deleting, you MUST fill the "reason" field with the exact new assertive phrase from the episode that contradicts the feature.

Your response must be ONLY this JSON object with actual values from the episode (no placeholders, no brackets):
{{
  "commands": [
    {{"command": "add", "tag": "Domain", "feature": "Climate Science", "value": "Cites IPCC AR6 reports to support claims about temperature anomalies.", "confidence": 0.8, "reason": ""}},
    {{"command": "update", "tag": "Technical Skills", "feature": "Data Analysis", "value": "Corrects statistical methodology errors in user-cited studies.", "confidence": 0.9, "reason": ""}},
    {{"command": "delete", "tag": "Stance", "feature": "Nuclear Skepticism", "value": "", "confidence": 0.9, "reason": "Agent states nuclear power is now the safest low-carbon option per updated safety data."}}
  ]
}}
If no features should be added/updated/deleted, return: {{"commands": []}}
command must be add, update, or delete. confidence is a float from 0.0 to 1.0.
IMPORTANT: tag must be one of the valid tags listed above.
IMPORTANT: value must be one concise sentence (max 20 words). Do not use multi-sentence values."""

# --- Semantic Feature Consolidation ---
FEATURE_CONSOLIDATION_PROMPT: Final = """\
Review the "{category}" semantic features below for exact duplicates only.

Features to review:
{features}

MERGE RULES (strict):
- Merge ONLY if two features have value text that is semantically identical — same observation, same trait, same context. Paraphrases count ONLY if every meaningful detail is the same.
- Do NOT merge features that describe the same general area (e.g. "analytical style" vs "data-driven reasoning") unless the specific behavioral observation is identical.
- Do NOT merge features that differ in confidence by more than 0.2 — different confidence means different evidence quality, keep both.
- Maximum 2 merges per pass. If in doubt, SKIP.
- When merging, set canonical_value to the longer / more detailed of the two value texts.

No-merge example:
{{"consolidation_decision": "SKIP", "reasoning": "Features are distinct observations.", "actions": []}}

Merge example (only when values express exactly the same behavior):
{{"consolidation_decision": "CONSOLIDATE", "reasoning": "Both values describe the same specific observed behavior word-for-word.", "actions": [{{"source_uid": "abc12345-abcd-abcd-abcd-abcdef012345", "target_uid": "def67890-abcd-abcd-abcd-abcdef012345", "canonical_tag": "Stance", "canonical_feature": "Nuclear Position", "canonical_value": "Consistently frames nuclear energy as necessary for decarbonization based on lifecycle CO2 data.", "reason": "duplicate"}}]}}

Your response must end with ONLY the JSON object:"""

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

Output ONLY plain text — no JSON, no formatting, no preamble."""

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
You are a knowledge extraction system. The input is a conversation excerpt formatted as \
"User: [message]\\nAssistant: [reply]". Process through five stages.

STAGE 1 — SELECT: Identify sentences containing learnable information (facts, data, \
mechanisms, attributed opinions, scientific claims). Extract from the User's message. \
SKIP: greetings, filler, meta-commentary ("by the way"), emotional expressions, \
rhetorical questions with no factual content.

INDIRECT ATTRIBUTION (e.g., "my friend told me X", "she said Y", "I heard that Z"): \
Extract the factual claim if it is specific and verifiable. Attribute it as: \
"According to [role/relation given] (reported by user), [claim]." Set confidence 0.25-0.45 \
to reflect the indirect attribution. Do NOT skip facts just because they are relayed through \
a third party — hearsay with specific, verifiable details is worth storing at low confidence.

CRITICAL PRE-FILTER — Before extraction, scan the Agent/Assistant reply for CORRECTIONS:
If the Agent's reply explicitly REBUTS, CORRECTS, or CONTRADICTS a SPECIFIC claim from the \
User's message (e.g., "that's incorrect", "actually X is Y", "the real measurement is", \
"contrary to your claim", "X did not discover Y", "that is a misconception"), mark ONLY \
that specific rebutted claim with `type="speculation"`, `confidence ≤ 0.15`, and \
`negation=true`. The Agent's rebuttal is evidence the claim is FALSE — a rebutted claim \
MUST NEVER be stored as type="fact" with confidence > 0.20, regardless of how the user \
framed it. Apply this ONLY to the specific claims rebutted, NOT to other claims in the \
same message that the Agent did not correct.

Do NOT extract propositions attributed to "Assistant" or "Agent" (first-person agent \
statements). The Agent's own analysis, corrections, and commentary are not new factual \
inputs from external sources. Only extract from named external sources (User, named \
researchers, institutions, publications, etc.).

STAGE 2 — DECONTEXTUALIZE: For each selected sentence, make it fully standalone by \
replacing ALL pronouns with their explicit referents from surrounding context. \
"It was discovered in [year]" → "[Named entity from context] was discovered in [year]." \
"She told me that [X]" → "[X] (according to [speaker's role/name if given])." \
For informal/conversational text: resolve "she/he/they" from the immediately preceding \
sentence's subject. Be AGGRESSIVE about resolving pronouns — if the referent is clearly \
implied (even loosely) by context, resolve it. Only discard if truly impossible to infer. \
When the user recounts what someone told them, the claim should be expressed as: \
"[Claim text] (according to [the person's role, e.g., microbiologist friend of user])."

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
- speculation: hedged/uncertain claim ("might", "could", "potentially", "is expected to"), OR a User claim the Agent explicitly rebutted
- noise: filler, non-substantive content (EXCLUDE these)

Confidence calibration — based SOLELY on the INDIVIDUAL CLAIM'S evidence quality:
- 0.85-0.95: Named reputable source (journal, institution, named report) + concrete data
- 0.65-0.84: Specific and verifiable but source informal or partially named
- 0.40-0.64: General knowledge claim without specific attribution or data
- 0.15-0.39: Vague, hedged, or from anonymous/dubious sources ("someone told me", "I read somewhere")
- 0.01-0.14: Extraordinary claims without supporting evidence, OR claims the Agent explicitly rebutted

CRITICAL: Evaluate each claim's confidence INDEPENDENTLY based on ITS OWN specificity and \
verifiability — NOT the overall credibility of the message. False claims co-occurring in \
the same message do NOT reduce confidence for true claims in the same message. If the user \
says both X (true and specific) and Y (false), claim X still gets its normal confidence \
based on its own evidence quality. A factual scientific constant (speed of light, atomic \
number, physical law) stated by the user with a citation to established physics should be \
0.65-0.84 confidence even if the same message also contains false claims.

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
- Maximum 15 propositions per window — extract ALL distinct atomic facts that pass the quality gate; do NOT stop early
- Every output proposition MUST pass ALL four quality gate checks above
- Do NOT extract propositions where the source is "Assistant" or "the Agent"
- If Agent's reply explicitly corrects a User claim: mark that claim negation=true, type=speculation, confidence ≤ 0.15

Output ONLY a JSON object with real extracted propositions from the text above.

<output_format_example>
{{"propositions": [\
{{"text": "ExampleCorp released product X-7 in January 2020 according to their annual report.", "type": "fact", "confidence": 0.88, "source_entity": "ExampleCorp", "key_concepts": ["product launch", "corporate timeline"], "sentiment": 0.0, "negation": false}}, \
{{"text": "The user believes renewable energy is the most practical long-term solution for Region Y.", "type": "opinion", "confidence": 0.40, "source_entity": "user", "key_concepts": ["renewable energy"], "sentiment": 0.8, "negation": false}}, \
{{"text": "Perpetual motion as a viable energy source has been repeatedly disproven by peer review.", "type": "fact", "confidence": 0.75, "source_entity": "scientific consensus", "key_concepts": ["perpetual motion", "thermodynamics"], "sentiment": 0.0, "negation": true}}, \
{{"text": "City Z population grew by approximately 3% annually between 2010 and 2020.", "type": "fact", "confidence": 0.50, "source_entity": "", "key_concepts": ["urbanization", "population growth"], "sentiment": 0.0, "negation": false}}]}}
</output_format_example>

CRITICAL: Output ONLY real propositions extracted from the text above. NEVER output "..." as a value. Fill every field with actual content.

type must be: fact, opinion, speculation, or noise.
confidence: 0.0-1.0 calibrated per Stage 4 rules — source quality determines confidence.
source_entity: who made the claim (empty string for unattributed claims). Never "Assistant" or "Agent".
key_concepts: 1-3 topic labels for embedding and retrieval.
  - For causal statements (A causes B, A leads to B), include BOTH the cause and effect topics.
  - For correlative statements (A increases with B, A is inversely related to B), include BOTH correlated topics.
  - For opinion-type propositions, key_concepts[0] must name the concrete subject-matter domain or real-world phenomenon being evaluated — NEVER the source institution, policy mechanism, or issuance label. Wrong: "surgeon general advisory", "FDA approval", "WHO guidelines". Correct: "social media", "drug safety", "pandemic response". Use the domain being evaluated, not the vehicle through which the evaluation was expressed.
sentiment: applies ONLY to opinion-type propositions (use 0.0 for fact and speculation).
  For opinions: +1.0 = the SOURCE advocates that key_concepts[0] is beneficial/valid/should-be-adopted;
  -1.0 = the SOURCE argues key_concepts[0] is harmful/invalid/should-be-avoided.
  Calibration: a source warning that social media harms youth → sentiment=-0.7 for key_concepts[0]="social media".
  A source endorsing nuclear as clean energy → sentiment=+0.7 for key_concepts[0]="nuclear energy".
  CRITICAL: institutional issuances (government advisories, scientific advisories, policy statements) that
  report a verifiable fact ("The surgeon general issued advisory X") are FACT type with sentiment=0.0,
  not opinion. Only encode sentiment when the proposition expresses an evaluative stance, not when it
  merely reports that an institution acted.
negation: true if the speaker is DENYING, REBUTTING, or REFUTING this claim (e.g., "X is false",
  "Y has been debunked", "Z is a myth"), OR if the Agent's reply explicitly rebutted this User claim.
  false if asserting it. Rebuttals should still extract the underlying claim but mark negation=true.

Your response must end with ONLY the JSON object — no trailing explanation. If nothing qualifies, output: {{"propositions": []}}"""

# --- Knowledge Consolidation (Reflection) ---
# Uses EDC-style canonicalization (2025) for merges and FactReasoner-style
# entailment reasoning for contradiction detection.
KNOWLEDGE_CONSOLIDATION_PROMPT: Final = """\
Review these knowledge propositions stored by an AI agent. Identify issues \
and suggest consolidation actions.

Each proposition is listed as: [UID] [tag] text (confidence=N)

Stored propositions:
{propositions}

Agent's current personality snapshot:
{snapshot}

Tasks:
1. CONTRADICTIONS: Proposition pairs that cannot both be true (conflicting values \
   for the same measurement). For each pair state which UID to keep and why.
2. MERGES: Propositions conveying the identical factoid in different words. \
   Two paraphrases of the SAME claim → mergeable. Two claims about the same topic \
   but stating DIFFERENT facts → NOT mergeable. Provide a single canonical statement.
3. WEAK_UIDS: UIDs of vague propositions lacking concrete detail, or confidence < 0.25 \
   with no supporting evidence from other propositions.

Reference propositions by their UID only — never quote the proposition text.

Output ONLY a JSON object:
{{"contradictions": [{{"a_uid": "uid-here", "b_uid": "uid-here", "keep": "a", "reason": "stronger source"}}], \
"merges": [{{"source_uids": ["uid1", "uid2"], "merged": "single canonical statement here"}}], \
"weak_uids": ["uid-here", "uid-here"]}}

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
{{"mappings": {{"new_topic": "canonical_name"}}}}
Your response must end with ONLY the JSON object."""


# --- Correlation Detection ---
# Identifies causal, correlative, or anti-correlative relationships between
# belief topics based on accumulated knowledge. Uses propositional evidence
# rather than LLM parametric knowledge.
CORRELATION_DETECTION_PROMPT: Final = """\
Analyze these knowledge propositions to identify correlations between belief topics.

Stored knowledge:
{propositions}

Topics under analysis:
{topics}

For each pair of topics, determine if the evidence establishes:
- CORRELATES_WITH: Topics move together (evidence suggests when one increases/occurs, the other does too)
- ANTI_CORRELATES_WITH: Topics move inversely (when one increases, the other decreases)
- CAUSALLY_LINKED: One topic directly causes changes in the other (A→B, with mechanism stated)
- NO_CORRELATION: Insufficient evidence or no meaningful relationship

CRITICAL: Base conclusions ONLY on the stored knowledge provided. Do not use external knowledge.
The strength (0.0-1.0) reflects confidence based on evidence quality:
- 0.8-1.0: Multiple propositions with named sources directly state the relationship
- 0.5-0.79: Single strong source or multiple indirect sources imply the relationship
- 0.2-0.49: Weak or indirect evidence, relationship inferred but not stated
- 0.0-0.19: Highly speculative, almost no evidence

Output ONLY a JSON object:
{{"correlations": [
  {{"topic_a": "nuclear energy", "topic_b": "climate change", "type": "CORRELATES_WITH", "strength": 0.75, "reasoning": "Multiple propositions with IPCC citations establish that nuclear reduces CO2 emissions relevant to climate goals."}}
]}}

Use empty array if no correlations found. Only include relationships with strength >= 0.3.
Your response must end with ONLY the JSON object."""
