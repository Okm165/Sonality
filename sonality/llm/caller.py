"""Universal LLM call wrapper with retry, repair, and fallback.

All LLM-based assessment prompts in the memory architecture MUST use ``llm_call``
rather than calling the Anthropic SDK or OpenRouter directly. This ensures
consistent JSON parsing, schema validation, repair logic, and graceful
degradation across all 12+ prompt types.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Final

from anthropic import Anthropic, APIError, APITimeoutError, RateLimitError
from pydantic import BaseModel, ValidationError

from .. import config

log = logging.getLogger(__name__)

_MAX_RETRIES: Final = 2
_BACKOFF_BASE: Final = 1.5
_JSON_REPAIR_PROMPT: Final = (
    "The following JSON is malformed or does not match the required schema.\n"
    "Fix it so it is valid JSON matching the schema below.\n\n"
    "Schema: {schema}\n\nBroken JSON:\n{broken}\n\n"
    "Validation error: {error}\n\n"
    "Return ONLY the corrected JSON object, no explanation."
)


@dataclass(frozen=True, slots=True)
class LLMCallResult(BaseModel.__class__.__bases__[0] if False else object):  # type: ignore[misc]
    """Result of a structured LLM call."""

    value: BaseModel | None = None
    success: bool = False
    error: str = ""
    attempts: int = 0
    repaired: bool = False
    raw_text: str = ""


def _get_client() -> Anthropic:
    return Anthropic(**config.anthropic_client_kwargs())


def _raw_call(
    client: Anthropic,
    *,
    prompt: str,
    model: str,
    max_tokens: int,
    system: str = "",
) -> str:
    """Execute a single Anthropic chat completion and return the text."""
    messages = [{"role": "user", "content": prompt}]
    kwargs: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if system:
        kwargs["system"] = system
    response = client.messages.create(**kwargs)  # type: ignore[arg-type]
    blocks = response.content
    parts: list[str] = []
    for block in blocks:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "".join(parts)


def _parse_json(text: str) -> dict[str, object]:
    """Extract a JSON object from LLM response text, tolerating markdown fences."""
    stripped = text.strip()
    for candidate in (
        stripped,
        stripped.replace("```json", "").replace("```", "").strip(),
    ):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    # Try extracting first { ... } block
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end > start:
        try:
            parsed = json.loads(stripped[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No valid JSON object found in LLM response", text, 0)


def llm_call[T: BaseModel](
    *,
    prompt: str,
    response_model: type[T],
    model: str | None = None,
    max_tokens: int | None = None,
    system: str = "",
    fallback: T | None = None,
    max_retries: int = _MAX_RETRIES,
) -> LLMCallResult:
    """Execute a structured LLM call with retry, repair, and fallback.

    Parameters
    ----------
    prompt:
        The full prompt to send (including any formatted data).
    response_model:
        A Pydantic BaseModel subclass to validate the JSON response against.
    model:
        Model ID. Defaults to ``config.FAST_LLM_MODEL``.
    max_tokens:
        Max output tokens. Defaults to ``config.FAST_LLM_MAX_TOKENS``.
    system:
        Optional system message.
    fallback:
        Value to return when all retries are exhausted.
    max_retries:
        Total attempts before giving up.

    Returns
    -------
    LLMCallResult with ``.value`` set to the validated Pydantic model on success,
    or ``.value = fallback`` on failure.
    """
    resolved_model = model or config.FAST_LLM_MODEL
    resolved_max_tokens = max_tokens or config.FAST_LLM_MAX_TOKENS

    client = _get_client()
    last_error = ""
    raw_text = ""

    for attempt in range(1, max_retries + 1):
        try:
            raw_text = _raw_call(
                client,
                prompt=prompt,
                model=resolved_model,
                max_tokens=resolved_max_tokens,
                system=system,
            )
            data = _parse_json(raw_text)
            value = response_model.model_validate(data)
            return LLMCallResult(
                value=value,
                success=True,
                attempts=attempt,
                raw_text=raw_text,
            )

        except json.JSONDecodeError as exc:
            last_error = f"JSON parse error: {exc}"
            log.warning("LLM call attempt %d/%d: %s", attempt, max_retries, last_error)

        except ValidationError as exc:
            last_error = f"Schema validation: {exc}"
            log.warning("LLM call attempt %d/%d: %s", attempt, max_retries, last_error)
            # Try repair on validation error
            try:
                repair_prompt = _JSON_REPAIR_PROMPT.format(
                    schema=response_model.model_json_schema(),
                    broken=raw_text,
                    error=str(exc),
                )
                repaired_text = _raw_call(
                    client,
                    prompt=repair_prompt,
                    model=resolved_model,
                    max_tokens=resolved_max_tokens,
                )
                repaired_data = _parse_json(repaired_text)
                value = response_model.model_validate(repaired_data)
                return LLMCallResult(
                    value=value,
                    success=True,
                    attempts=attempt,
                    repaired=True,
                    raw_text=repaired_text,
                )
            except Exception as repair_exc:
                log.warning("Repair attempt failed: %s", repair_exc)

        except RateLimitError:
            wait = _BACKOFF_BASE**attempt
            log.warning("Rate limited on attempt %d/%d; backing off %.1fs", attempt, max_retries, wait)
            time.sleep(wait)
            last_error = "Rate limited"

        except APITimeoutError:
            last_error = "API timeout"
            log.warning("LLM timeout on attempt %d/%d", attempt, max_retries)

        except APIError as exc:
            last_error = f"API error: {exc}"
            log.warning("LLM API error on attempt %d/%d: %s", attempt, max_retries, exc)

        except Exception as exc:
            last_error = f"Unexpected: {exc}"
            log.error("Unexpected LLM call error on attempt %d/%d: %s", attempt, max_retries, exc)
            break  # Don't retry on unexpected errors

    log.error("LLM call exhausted %d retries. Last error: %s", max_retries, last_error)
    return LLMCallResult(
        value=fallback,
        success=False,
        error=last_error,
        attempts=max_retries,
        raw_text=raw_text,
    )
