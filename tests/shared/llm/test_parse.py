"""LLM output cleanup pipeline tests — guards against real regressions.

Tests cover: thinking removal, JSON extraction, tool call parsing,
response sanitization (fake XML, leaked prompts, bare tool calls),
and Pydantic normalization helpers. Bug 79 (bare </function> tag leak)
was caught here.
"""

from __future__ import annotations

import pytest

from shared.llm.parse import (
    ParsedToolCall,
    clean_completion,
    coerce_string_fields,
    decode_llm_json,
    extract_tool_calls,
    message_content_text,
    normalize_llm_list_response,
    strip_markdown,
    to_nonnegative_int,
)

# ---------------------------------------------------------------------------
# Thinking removal + answer extraction
# ---------------------------------------------------------------------------


class TestCleanCompletion:
    def test_strips_think_blocks(self) -> None:
        raw = "<think>Let me reason about this...</think>The answer is 42."
        assert clean_completion(raw) == "The answer is 42."

    def test_strips_unclosed_think(self) -> None:
        raw = "Some preamble <think>thinking that never closes"
        result = clean_completion(raw)
        assert "<think>" not in result
        assert "Some preamble" in result

    def test_strips_reasoning_code_fences(self) -> None:
        raw = "```thinking\nInternal reasoning here\n```\nFinal answer."
        assert clean_completion(raw) == "Final answer."

    def test_strips_internal_xml_tags(self) -> None:
        for tag in ("planning", "reasoning", "reflection", "scratch_pad", "notes"):
            raw = f"<{tag}>Internal notes</{tag}>Clean output."
            assert clean_completion(raw) == "Clean output.", f"Failed for <{tag}>"

    def test_recovers_from_reasoning_field(self) -> None:
        result = clean_completion("", reasoning="The real answer is here.")
        assert result == "The real answer is here."

    def test_empty_input(self) -> None:
        assert clean_completion("") == ""

    def test_preserves_normal_content(self) -> None:
        text = "Ethereum uses proof-of-stake consensus."
        assert clean_completion(text) == text


# ---------------------------------------------------------------------------
# JSON extraction from messy LLM output
# ---------------------------------------------------------------------------


class TestDecodeLLMJson:
    def test_clean_json(self) -> None:
        assert decode_llm_json('{"key": "value"}') == {"key": "value"}

    def test_json_in_markdown_fences(self) -> None:
        raw = '```json\n{"score": 0.8}\n```'
        assert decode_llm_json(raw) == {"score": 0.8}

    def test_json_buried_in_reasoning(self) -> None:
        raw = '<think>Let me think...</think>{"decision": "FINISH"}'
        assert decode_llm_json(raw) == {"decision": "FINISH"}

    def test_json_list(self) -> None:
        assert decode_llm_json("[1, 2, 3]") == [1, 2, 3]

    def test_escaped_quotes_recovery(self) -> None:
        raw = '{\\"key\\": \\"val\\"}'
        result = decode_llm_json(raw)
        assert result == {"key": "val"}

    def test_raises_on_no_json(self) -> None:
        from shared.errors import LLMParseError

        with pytest.raises(LLMParseError):
            decode_llm_json("No JSON here at all.")

    def test_extracts_last_json_object(self) -> None:
        raw = 'Some text {"a": 1} more text {"b": 2, "c": 3} end'
        result = decode_llm_json(raw)
        assert "b" in result

    def test_last_json_wins_over_largest(self) -> None:
        raw = 'Example: {"big": 1, "example": 2, "data": 3} Answer: {"answer": 42}'
        result = decode_llm_json(raw)
        assert result == {"answer": 42}


# ---------------------------------------------------------------------------
# Tool call parsing
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    def test_parses_standard_tool_call(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "function": {
                                    "name": "recall_memory",
                                    "arguments": '{"query": "DeFi exploits"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
        calls = extract_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0] == ParsedToolCall(
            name="recall_memory", args={"query": "DeFi exploits"}, id="call_1"
        )

    def test_handles_malformed_args(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call_2",
                                "function": {
                                    "name": "web_research",
                                    "arguments": "not valid json {{{",
                                },
                            }
                        ]
                    }
                }
            ]
        }
        calls = extract_tool_calls(raw)
        assert len(calls) == 1
        assert calls[0].args == {}

    def test_empty_choices(self) -> None:
        assert extract_tool_calls({"choices": []}) == []

    def test_no_tool_calls(self) -> None:
        raw = {"choices": [{"message": {"content": "Hello"}}]}
        assert extract_tool_calls(raw) == []

    def test_multiple_calls(self) -> None:
        raw = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {"id": "a", "function": {"name": "recall_memory", "arguments": "{}"}},
                            {"id": "b", "function": {"name": "web_research", "arguments": "{}"}},
                        ]
                    }
                }
            ]
        }
        assert len(extract_tool_calls(raw)) == 2


# ---------------------------------------------------------------------------
# Response sanitization (strip_markdown) — Bug 79 regression guard
# ---------------------------------------------------------------------------


class TestStripMarkdown:
    def test_strips_bare_function_tag(self) -> None:
        """Bug 79: bare </function> tags leaked into responses."""
        text = "Research results\n\n</function>\n\n</function>\nMore text"
        result = strip_markdown(text)
        assert "</function>" not in result

    def test_strips_function_call_tag(self) -> None:
        text = "<function_call>recall_memory</function_call>Answer here."
        result = strip_markdown(text)
        assert "<function_call>" not in result

    def test_strips_tool_xml(self) -> None:
        text = "<tool>some content</tool>The real answer."
        result = strip_markdown(text)
        assert "<tool>" not in result

    def test_strips_bare_tool_calls(self) -> None:
        text = "recall_memory('DeFi exploits')\nThe answer is..."
        result = strip_markdown(text)
        assert "recall_memory(" not in result

    def test_strips_leaked_step_history(self) -> None:
        text = "1. recall_memory('test') → found 3 episodes\nActual response."
        result = strip_markdown(text)
        assert "recall_memory(" not in result

    def test_strips_leaked_plan(self) -> None:
        text = "Research Plan:\n1. Web research on topic\nActual answer."
        result = strip_markdown(text)
        assert "Research Plan:" not in result

    def test_strips_markdown_bold(self) -> None:
        assert strip_markdown("**bold text**") == "bold text"

    def test_strips_markdown_headers(self) -> None:
        assert strip_markdown("## Header\nContent") == "Header\nContent"

    def test_strips_leaked_belief_data(self) -> None:
        text = "crypto_regulation — valence: +0.3, confidence: 0.7\nActual content."
        result = strip_markdown(text)
        assert "valence:" not in result

    def test_strips_leaked_episode_refs(self) -> None:
        text = "- [2026-01-15] Discussion about DeFi\nActual response."
        result = strip_markdown(text)
        assert "[2026-01-15]" not in result

    def test_preserves_normal_content(self) -> None:
        text = "DeFi protocols use automated market makers for price discovery."
        assert strip_markdown(text) == text

    def test_strips_leaked_synthesis_marker(self) -> None:
        text = "[Research complete. Do not copy this verbatim.] Here is the answer."
        result = strip_markdown(text)
        assert "[Research complete" not in result


# ---------------------------------------------------------------------------
# Message content helpers
# ---------------------------------------------------------------------------


class TestMessageContentText:
    def test_string_passthrough(self) -> None:
        assert message_content_text("hello") == "hello"

    def test_multipart_list(self) -> None:
        parts = [{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}]
        assert message_content_text(parts) == "Hello world"

    def test_non_text_parts_skipped(self) -> None:
        parts = [{"type": "image_url", "url": "..."}, {"type": "text", "text": "caption"}]
        assert message_content_text(parts) == "caption"

    def test_none_returns_empty(self) -> None:
        assert message_content_text(None) == ""


class TestToNonnegativeInt:
    def test_positive_int(self) -> None:
        assert to_nonnegative_int(42) == 42

    def test_negative_clamped(self) -> None:
        assert to_nonnegative_int(-5) == 0

    def test_float_truncated(self) -> None:
        assert to_nonnegative_int(3.7) == 3

    def test_string_number(self) -> None:
        assert to_nonnegative_int("10") == 10

    def test_bool_returns_zero(self) -> None:
        assert to_nonnegative_int(True) == 0

    def test_garbage_returns_zero(self) -> None:
        assert to_nonnegative_int("not a number") == 0


# ---------------------------------------------------------------------------
# Pydantic normalization helpers
# ---------------------------------------------------------------------------


class TestNormalizeLLMListResponse:
    def test_bare_list_wrapped(self) -> None:
        data = [{"text": "a"}, {"text": "b"}]
        result = normalize_llm_list_response(data, list_key="items", item_required_key="text")
        assert result == {"items": [{"text": "a"}, {"text": "b"}]}

    def test_bare_item_wrapped(self) -> None:
        data = {"text": "solo"}
        result = normalize_llm_list_response(data, list_key="items", item_required_key="text")
        assert result == {"items": [{"text": "solo"}]}

    def test_normal_dict_passthrough(self) -> None:
        data = {"items": [{"text": "ok"}]}
        result = normalize_llm_list_response(data, list_key="items", item_required_key="text")
        assert result == data

    def test_filters_empty_dicts(self) -> None:
        data = {"items": [{"text": "good"}, {}]}
        result = normalize_llm_list_response(data, list_key="items", item_required_key="text")
        assert result == {"items": [{"text": "good"}]}


class TestCoerceStringFields:
    def test_list_joined(self) -> None:
        data = {"summary": ["line1", "line2"]}
        coerce_string_fields(data, ("summary",))
        assert data["summary"] == "line1\nline2"

    def test_dict_flattened(self) -> None:
        data = {"summary": {"key": "val"}}
        coerce_string_fields(data, ("summary",))
        assert data["summary"] == "key: val"

    def test_none_becomes_empty(self) -> None:
        data = {"summary": None}
        coerce_string_fields(data, ("summary",))
        assert data["summary"] == ""

    def test_number_stringified(self) -> None:
        data = {"score": 0.5}
        coerce_string_fields(data, ("score",))
        assert data["score"] == "0.5"
