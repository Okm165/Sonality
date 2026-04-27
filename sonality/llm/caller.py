"""Universal structured LLM call wrapper with retry and JSON repair."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ValidationError

from .. import config
from ..provider import decode_llm_json, default_provider
from ..schema import ChatRole

log = logging.getLogger(__name__)

_JSON_SYSTEM_PROMPT: Final = (
    "Output ONLY valid JSON (object or array). "
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
    assistant_prefix: str = "",
) -> str:
    """Execute a single provider chat completion and return plain text.

    assistant_prefix: if non-empty, added as the last message with role=assistant
    so the model continues from it (prefilling). Use to force JSON output by
    prefixing with the opening of the expected structure, e.g. ``{"propositions": [``.
    The prefix is prepended to the returned text so callers get a complete string.
    """
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": ChatRole.SYSTEM, "content": system})
    messages.append({"role": ChatRole.USER, "content": prompt})
    if assistant_prefix:
        messages.append({"role": ChatRole.ASSISTANT, "content": assistant_prefix})
    completion = default_provider.chat_completion(
        model=model,
        messages=tuple(messages),
        max_tokens=max_tokens,
        # Prefilled assistant turns are incompatible with enable_thinking on most servers.
        # Thinking mode also hurts JSON tasks: the model reasons in the thinking trace then
        # fails to produce valid JSON in content. Disable it for all structured llm_call uses.
        enable_thinking=False,
    )
    return (assistant_prefix + completion.text) if assistant_prefix else completion.text


def llm_call[T: BaseModel](
    *,
    prompt: str,
    response_model: type[T],
    fallback: T,
    model: str = config.FAST_LLM_MODEL,
    max_tokens: int = config.LLM_MAX_TOKENS,
    system: str = _JSON_SYSTEM_PROMPT,
    max_retries: int = config.LLM_MAX_RETRIES,
    assistant_prefix: str = "",
) -> LLMCallResult[T]:
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
        Max output tokens. Defaults to ``config.LLM_MAX_TOKENS``.
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
                assistant_prefix=assistant_prefix,
            )
            data = decode_llm_json(raw_text)
            value = response_model.model_validate(data)
            log.debug(
                "llm_call OK schema=%s attempt=%d raw=%.80r",
                schema_name,
                attempt,
                raw_text,
            )
            return LLMCallResult(
                value=value,
                success=True,
                attempts=attempt,
                raw_text=raw_text,
            )

        except ValidationError as exc:
            # Must be before `except ValueError` — pydantic v2 ValidationError IS a ValueError.
            last_error = f"Schema validation: {exc}"
            log.warning(
                "llm_call attempt %d/%d schema=%s validation failed: %s | raw=%.80r",
                attempt,
                max_retries,
                schema_name,
                exc,
                raw_text,
            )
            # Try repair on validation error
            try:
                repair_prompt = _JSON_REPAIR_PROMPT.format(
                    schema=response_model.model_json_schema(),
                    broken=raw_text,
                    error=str(exc),
                )
                repaired_text = _raw_call(
                    prompt=repair_prompt,
                    model=model,
                    max_tokens=max_tokens,
                )
                repaired_data = decode_llm_json(repaired_text)
                value = response_model.model_validate(repaired_data)
                log.info("llm_call repaired schema=%s attempt=%d", schema_name, attempt)
                return LLMCallResult(
                    value=value,
                    success=True,
                    attempts=attempt,
                    repaired=True,
                    raw_text=repaired_text,
                )
            except Exception as repair_exc:
                log.warning("llm_call repair failed schema=%s: %s", schema_name, repair_exc)

        except ValueError as exc:
            last_error = f"JSON parse error: {exc}"
            log.warning(
                "llm_call attempt %d/%d schema=%s: %s | raw=%.200r",
                attempt,
                max_retries,
                schema_name,
                last_error,
                raw_text,
            )
            log.debug("llm_call full raw (schema=%s): %s", schema_name, raw_text)

        except RuntimeError as exc:
            last_error = f"Provider error: {exc}"
            error_str = str(exc).lower()
            # Transport errors and network errors are already retried inside provider._post_json.
            # Re-retrying them here compounds the timeout — treat them as terminal.
            is_exhausted = (
                "name resolution" in error_str
                or "network error" in error_str
                or "transport error" in error_str
            )
            if attempt < max_retries and not is_exhausted:
                wait = config.LLM_BACKOFF_BASE**attempt
                log.warning(
                    "LLM provider error on attempt %d/%d schema=%s: %s; retrying in %.1fs",
                    attempt,
                    max_retries,
                    schema_name,
                    exc,
                    wait,
                )
                time.sleep(wait)
                continue
            log.warning(
                "LLM provider error on attempt %d/%d schema=%s: %s",
                attempt,
                max_retries,
                schema_name,
                exc,
            )
            if is_exhausted:
                break  # No point retrying — provider already exhausted its own retries

        except Exception as exc:
            last_error = f"Unexpected: {exc}"
            log.error(
                "Unexpected llm_call error attempt %d/%d schema=%s: %s",
                attempt,
                max_retries,
                schema_name,
                exc,
            )
            break  # Don't retry on unexpected errors

    log.error(
        "llm_call failed after %d/%d attempts schema=%s: %s",
        attempts_made,
        max_retries,
        schema_name,
        last_error,
    )
    return LLMCallResult(
        value=fallback,
        success=False,
        error=last_error,
        attempts=attempts_made,
        raw_text=raw_text,
    )
