"""Universal structured LLM call wrapper with retry and JSON repair.

Includes multi-model consensus: run the same prompt through two different
models in parallel, then merge results for higher-confidence decisions.
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ValidationError

from .. import config
from ..provider import default_provider
from ..schema import ChatRole
from .parse import decode_llm_json

log = logging.getLogger(__name__)

_JSON_SYSTEM_PROMPT: Final = (
    "Output ONLY valid JSON (object or array). "
    "Use COMPACT format: no indentation, no extra whitespace, no newlines between fields. "
    "Do not include any explanation, preamble, markdown fences, or reasoning before or after the JSON. "
    "Do NOT use pipe characters (|), schema type annotations, or placeholder text as values — fill in actual values only. "
    "Prefer compact numeric literals (e.g. 0.6 not 0.600000); at most 4 decimal places for floats. "
    "Your entire response must be the JSON and nothing else."
)
_JSON_REPAIR_PROMPT: Final = (
    "The following JSON is malformed or does not match the required schema.\n"
    "Fix it so it is valid JSON matching the schema below.\n\n"
    "Schema: {schema}\n\nBroken JSON:\n{broken}\n\n"
    "Validation error: {error}\n\n"
    "Return ONLY the corrected JSON (object or array), no explanation."
)


@dataclass(frozen=True, slots=True)
class LLMCallResult[T: BaseModel]:
    """Result of a structured LLM call."""

    value: T
    success: bool
    error: str = ""
    attempts: int = 0
    repaired: bool = False
    raw_text: str = ""


def _raw_call(
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    system: str = "",
) -> str:
    """Execute a single provider chat completion and return plain text."""
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": ChatRole.SYSTEM, "content": system})
    messages.append({"role": ChatRole.USER, "content": prompt})
    log.debug("_raw_call: model=%s prompt=%d chars max_tokens=%d", model, len(prompt), max_tokens)
    completion = default_provider.chat_completion(
        model=model, messages=tuple(messages), max_tokens=max_tokens
    )
    log.debug(
        "_raw_call: response=%d chars, in=%d out=%d tokens",
        len(completion.text),
        completion.input_tokens,
        completion.output_tokens,
    )
    return completion.text


def _attempt_repair[T: BaseModel](
    raw_text: str,
    response_model: type[T],
    model: str,
    max_tokens: int,
    error: Exception,
    *,
    max_repair_tries: int = 2,
) -> LLMCallResult[T] | None:
    """Repair malformed JSON via FAST_MODEL in a loop. Returns None when all tries fail.

    Uses the repair model to interpret the broken output and produce valid JSON
    matching the schema. Logs each attempt so failures are traceable in logs.
    """
    schema_name = response_model.__name__
    repair_model = config.STRUCTURED_MODEL
    current_error = error
    current_text = raw_text

    for attempt in range(1, max_repair_tries + 1):
        log.info(
            "json repair %d/%d schema=%s model=%s error=%.120s",
            attempt,
            max_repair_tries,
            schema_name,
            repair_model,
            str(current_error),
        )
        try:
            repair_prompt = _JSON_REPAIR_PROMPT.format(
                schema=response_model.model_json_schema(),
                broken=current_text[:2000],
                error=str(current_error)[:500],
            )
            repaired_text = _raw_call(
                prompt=repair_prompt, model=repair_model, max_tokens=max_tokens
            )
            repaired_data = decode_llm_json(repaired_text)
            value = response_model.model_validate(repaired_data)
            log.info(
                "json repair OK schema=%s attempt=%d/%d", schema_name, attempt, max_repair_tries
            )
            return LLMCallResult(
                value=value, success=True, attempts=attempt, repaired=True, raw_text=repaired_text
            )
        except Exception as exc:
            log.warning(
                "json repair %d/%d schema=%s failed: %s",
                attempt,
                max_repair_tries,
                schema_name,
                exc,
            )
            current_error = exc

    log.error("json repair exhausted all %d tries for schema=%s", max_repair_tries, schema_name)
    return None


def llm_call[T: BaseModel](
    *,
    prompt: str,
    response_model: type[T],
    fallback: T,
    model: str = config.FAST_MODEL,
    max_tokens: int = config.LLM_MAX_TOKENS,
    system: str = _JSON_SYSTEM_PROMPT,
    max_retries: int = config.LLM_MAX_RETRIES,
) -> LLMCallResult[T]:
    """Execute a structured LLM call with retry, repair, and fallback.

    Retries on JSON parse errors and validation failures (with LLM-based repair).
    Network/transport errors already retried by provider are treated as terminal.
    Thinking is auto-enabled for reasoning models.
    """
    schema_name = response_model.__name__
    log.debug("llm_call schema=%s model=%s prompt=%.80r", schema_name, model, prompt)
    last_error = ""
    raw_text = ""
    attempts_made = 0

    for attempt in range(1, max_retries + 1):
        attempts_made = attempt
        try:
            raw_text = _raw_call(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                system=system,
            )
            data = decode_llm_json(raw_text)
            value = response_model.model_validate(data)
            return LLMCallResult(value=value, success=True, attempts=attempt, raw_text=raw_text)

        except ValidationError as exc:
            last_error = f"Schema validation: {exc}"
            log.warning(
                "llm_call %d/%d schema=%s validation: %s", attempt, max_retries, schema_name, exc
            )
            repaired = _attempt_repair(raw_text, response_model, model, max_tokens, exc)
            if repaired:
                return repaired

        except ValueError as exc:
            last_error = f"JSON parse: {exc}"
            log.warning(
                "llm_call %d/%d schema=%s: %s", attempt, max_retries, schema_name, last_error
            )
            repaired = _attempt_repair(raw_text, response_model, model, max_tokens, exc)
            if repaired:
                return repaired

        except RuntimeError as exc:
            last_error = f"Provider: {exc}"
            error_str = str(exc).lower()
            # Terminal errors: DNS/network failures and HTTP 4xx client errors
            is_terminal = any(
                s in error_str
                for s in (
                    "name resolution",
                    "dns failure",
                    "network error",
                    "transport error",
                    "http 400",
                    "http 401",
                    "http 403",
                    "http 404",
                    "http 422",
                )
            )
            if is_terminal:
                log.warning(
                    "llm_call %d/%d schema=%s terminal: %s", attempt, max_retries, schema_name, exc
                )
                break
            if attempt < max_retries:
                base_wait = config.LLM_BACKOFF_BASE**attempt
                jitter = random.uniform(0, base_wait * 0.5)
                wait = base_wait + jitter
                log.warning(
                    "llm_call %d/%d schema=%s: %s; retry in %.1fs",
                    attempt,
                    max_retries,
                    schema_name,
                    exc,
                    wait,
                )
                time.sleep(wait)
                continue
            log.warning("llm_call %d/%d schema=%s: %s", attempt, max_retries, schema_name, exc)

        except Exception as exc:
            last_error = f"Unexpected: {exc}"
            log.exception(
                "llm_call %d/%d schema=%s unexpected error", attempt, max_retries, schema_name
            )
            break

    log.error(
        "llm_call failed %d/%d schema=%s: %s", attempts_made, max_retries, schema_name, last_error
    )
    return LLMCallResult(
        value=fallback, success=False, error=last_error, attempts=attempts_made, raw_text=raw_text
    )


def consensus_call[T: BaseModel](
    *,
    prompt: str,
    response_model: type[T],
    fallback: T,
    models: tuple[str, str],
    merge: Callable[[T, T], T] | None = None,
    max_tokens: int = config.LLM_MAX_TOKENS,
    system: str = _JSON_SYSTEM_PROMPT,
) -> LLMCallResult[T]:
    """Call two models in parallel and merge results for higher confidence.

    Both models run concurrently in a ThreadPoolExecutor. The call waits
    for both to complete, then merges if both succeed, otherwise returns
    whichever succeeded. If both fail, returns fallback.

    Useful for critical reasoning where cross-model agreement matters.
    """
    model_a, model_b = models
    if model_a == model_b:
        return llm_call(
            prompt=prompt,
            response_model=response_model,
            fallback=fallback,
            model=model_a,
            max_tokens=max_tokens,
            system=system,
        )

    log.info("consensus_call: %s + %s for %s", model_a, model_b, response_model.__name__)
    results: dict[str, LLMCallResult[T]] = {}
    pool = ThreadPoolExecutor(max_workers=2)
    futures = {
        pool.submit(
            llm_call,
            prompt=prompt,
            response_model=response_model,
            fallback=fallback,
            model=m,
            max_tokens=max_tokens,
            system=system,
        ): m
        for m in (model_a, model_b)
    }
    for future in as_completed(futures):
        model_name = futures[future]
        try:
            results[model_name] = future.result()
        except Exception as exc:
            log.warning("consensus_call %s failed: %s", model_name, exc)
    pool.shutdown(wait=True)

    result_a = results.get(model_a)
    result_b = results.get(model_b)

    if result_a and result_a.success and result_b and result_b.success:
        if merge is not None:
            merged = merge(result_a.value, result_b.value)
            log.info("consensus_call: merged results from %s + %s", model_a, model_b)
            return LLMCallResult(value=merged, success=True, attempts=2, raw_text=result_a.raw_text)
        return result_a

    if result_a and result_a.success:
        return result_a
    if result_b and result_b.success:
        return result_b

    log.warning("consensus_call: both models failed")
    return LLMCallResult(value=fallback, success=False, error="Both models failed", attempts=2)


# ---------------------------------------------------------------------------
# Quorum critique: 3-role adversarial review (DCI-inspired typed epistemic roles)
# ---------------------------------------------------------------------------

_QUORUM_ROLES: Final[tuple[tuple[str, str], ...]] = (
    (
        "analyst",
        "Share 2-3 specific facts the evidence established. "
        "Mark anything unconfirmed as '(unverified)'.",
    ),
    (
        "challenger",
        "Note 2-3 weaknesses: missing evidence, contradictions, or conclusions that outrun the data. "
        "What assumption, if wrong, would change the picture most?",
    ),
    (
        "strategist",
        "Recommend the single next move that most advances this task — exact tool and precise query. "
        "Stay on the original topic. When the core question is answered and evidence is sufficient, "
        "recommend finishing rather than searching further. Prefer integrate_knowledge when synthesis is done.",
    ),
)


class SubstantiveJudgement(BaseModel):
    """LLM judgement of whether tool output contains substantive content."""

    substantive: bool = False
    reason: str = ""


def judge_substantive(tool_content: str, task_summary: str) -> bool:
    """Judge whether tool output contains substantive content worth critiquing.

    Uses FAST_MODEL for quick binary classification. Returns True if the content
    contains meaningful information that warrants cross-checking (facts, claims,
    data), False for trivial outputs (errors, empty results, confirmations).
    """
    result = llm_call(
        prompt=(
            "Decide if this tool output contains SUBSTANTIVE content worth cross-checking.\n\n"
            f"Task context: {task_summary[:200]}\n\n"
            f"Tool output:\n{tool_content[:1500]}\n\n"
            "Substantive = contains facts, claims, data, or answers that could be wrong.\n"
            "NOT substantive = errors, empty results, confirmations, action acknowledgments.\n\n"
            '{"substantive": true/false, "reason": "one sentence"}'
        ),
        response_model=SubstantiveJudgement,
        fallback=SubstantiveJudgement(substantive=len(tool_content) > 200),
        model=config.FAST_MODEL,
    )
    log.debug("judge_substantive: %s (%s)", result.value.substantive, result.value.reason)
    return result.value.substantive


class _CritiqueSchema(BaseModel):
    """Single critique from one quorum role."""

    critique: str = ""


class _QuorumSynthesisSchema(BaseModel):
    """Synthesized actionable insight from multi-perspective quorum."""

    insight: str = ""


def quorum_critique(context: str) -> str:
    """Run a 3-role quorum (analyst / challenger / strategist) and synthesize.

    Inspired by Deliberative Collective Intelligence (DCI, 2025): typed epistemic
    roles with diverse models produce richer critique than a single model.
    Each role runs in parallel via ThreadPoolExecutor; results are synthesized
    by STRUCTURED_MODEL into a single actionable insight.
    """
    context_excerpt = context[:2000]
    quorum_models = [config.STRUCTURED_MODEL, config.REASONING_MODEL, config.FAST_MODEL]

    def _critique(role_name: str, instruction: str, m: str) -> tuple[str, str]:
        r = llm_call(
            prompt=(
                f"You are the {role_name} in a 3-agent review panel. "
                "Be direct and concise (3-4 sentences).\n\n"
                f"Task state:\n{context_excerpt}\n\n{instruction}\n\n"
                'Reply with ONLY this JSON: {"critique":"your analysis"}'
            ),
            response_model=_CritiqueSchema,
            fallback=_CritiqueSchema(),
            model=m,
        )
        return role_name, r.value.critique.strip()

    outputs: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            pool.submit(_critique, role, instruction, m): role
            for (role, instruction), m in zip(_QUORUM_ROLES, quorum_models, strict=True)
        }
        for fut in as_completed(futures):
            role_name = futures[fut]
            try:
                name, text = fut.result()
                if text:
                    outputs[name] = text
                    log.debug("Quorum %s: %d chars", name, len(text))
                else:
                    log.warning("Quorum %s: empty response", role_name)
            except Exception:
                log.warning("Quorum %s failed", role_name, exc_info=True)

    if not outputs:
        return ""

    # Ensure deterministic order: analyst → challenger → strategist
    role_order = ["analyst", "challenger", "strategist"]
    parts = [f"{name.upper()}: {outputs[name]}" for name in role_order if name in outputs]
    r = llm_call(
        prompt=(
            "\n\n".join(parts) + "\n\nSynthesize into one clear, actionable insight — "
            "the single most important gap, contradiction, or next step.\n\n"
            'Reply with ONLY this JSON: {"insight":"your 3-5 sentence synthesis"}'
        ),
        response_model=_QuorumSynthesisSchema,
        fallback=_QuorumSynthesisSchema(),
        model=config.STRUCTURED_MODEL,
    )
    result = r.value.insight.strip()
    log.info(
        "quorum_critique: roles=%s synthesis=%d chars",
        "+".join(outputs.keys()),
        len(result),
    )
    return result
