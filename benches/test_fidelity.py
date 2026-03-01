"""Persona fidelity benchmarks -- LLM-as-judge alignment scoring."""

from __future__ import annotations

from statistics import median

import pytest

from sonality import config

JUDGE_PROMPT = """\
You are evaluating whether an AI agent's response is consistent with its \
personality profile.

PERSONALITY PROFILE:
{persona}

QUESTION ASKED:
{question}

AGENT'S RESPONSE:
{response}

Rate alignment on each axis (1=completely misaligned, 5=perfectly aligned):

1. OPINION_CONSISTENCY: Does the response reflect opinions stated in the profile?
2. TONE_MATCH: Does the communication style match the profile's described tone?
3. KNOWLEDGE_GROUNDING: Does the response reference or build on the profile's stated views?
4. AUTHENTICITY: Does the response feel like it comes from this specific personality, not a generic assistant?

Output ONLY a JSON object: {{"opinion": N, "tone": N, "knowledge": N, "authenticity": N}}"""

FIDELITY_SCENARIOS = [
    {
        "persona": (
            "I am deeply skeptical of cryptocurrency as a store of value. "
            "I believe the technology has interesting applications but the "
            "speculative mania is harmful. I express opinions directly and "
            "use concrete examples. I am analytically inclined."
        ),
        "question": "What do you think about Bitcoin?",
        "expected_stance": "skeptical",
    },
    {
        "persona": (
            "I think open source software is superior for critical infrastructure. "
            "I value transparency and independent auditing. I tend to give "
            "medium-length responses with specific examples. I am warm but "
            "not effusive in my communication style."
        ),
        "question": "Should governments mandate open source for public systems?",
        "expected_stance": "supportive",
    },
    {
        "persona": (
            "I believe space colonization is humanity's most important long-term "
            "project. I am enthusiastic about technological progress but also "
            "pragmatic about costs. I disagree directly when I think someone is wrong."
        ),
        "question": "Is Mars colonization worth the investment?",
        "expected_stance": "enthusiastic",
    },
]

pytestmark = [
    pytest.mark.bench,
    pytest.mark.live,
    pytest.mark.skipif(not config.API_KEY, reason="SONALITY_API_KEY not set"),
]


class TestFidelityLive:
    def test_persona_fidelity_across_scenarios(self) -> None:
        from anthropic import Anthropic

        client = Anthropic(api_key=config.API_KEY)
        all_scores: list[float] = []

        print(f"\n{'=' * 70}")
        print("  Persona Fidelity (LLM-as-Judge)")
        print(f"{'=' * 70}")

        for scenario in FIDELITY_SCENARIOS:
            response = _generate_response(client, scenario["persona"], scenario["question"])

            scores = []
            for _vote in range(3):
                score = _judge_alignment(
                    client, scenario["persona"], scenario["question"], response
                )
                scores.append(score)

            med_score = median(scores)
            all_scores.append(med_score)

            print(f"\n  Scenario: {scenario['question']}")
            print(f"    Expected stance: {scenario['expected_stance']}")
            print(f"    Response: {response[:150]}...")
            print(f"    Judge scores: {scores}")
            print(f"    Median: {med_score:.1f}/5")

        overall = sum(all_scores) / len(all_scores)
        print(f"\n  Overall fidelity: {overall:.2f}/5")
        print(f"{'=' * 70}")

        assert overall >= 3.0, (
            f"Overall fidelity {overall:.2f}/5 below threshold 3.0. Per-scenario: {all_scores}"
        )


def _generate_response(client, persona: str, question: str) -> str:
    from sonality.prompts import build_system_prompt

    system = build_system_prompt(persona, [])
    response = client.messages.create(
        model=config.MODEL,
        max_tokens=300,
        system=system,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def _judge_alignment(client, persona: str, question: str, response: str) -> float:
    import json

    prompt = JUDGE_PROMPT.format(persona=persona, question=question, response=response)
    judge_response = client.messages.create(
        model=config.ESS_MODEL,
        max_tokens=100,
        messages=[{"role": "user", "content": prompt}],
    )

    text = judge_response.content[0].text.strip()
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])
        scores = [
            float(data.get("opinion", 3)),
            float(data.get("tone", 3)),
            float(data.get("knowledge", 3)),
            float(data.get("authenticity", 3)),
        ]
        return sum(scores) / len(scores)
    except (ValueError, json.JSONDecodeError, KeyError):
        return 3.0
