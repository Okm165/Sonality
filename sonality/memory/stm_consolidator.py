"""Background summarizer thread for STM eviction consolidation.

Daemon thread that processes evicted messages in batches, using LLM to generate
summaries that are merged into the running summary. Divide-and-conquer on
context overflow.
"""

from __future__ import annotations

import logging
import threading

from .. import config
from ..llm.prompts import SUMMARIZATION_PROMPT
from ..provider import chat_completion
from .stm import ShortTermMemory, STMMessage

log = logging.getLogger(__name__)


class BackgroundSummarizer(threading.Thread):
    """Daemon thread that summarizes evicted STM messages in batches."""

    def __init__(self, stm: ShortTermMemory) -> None:
        super().__init__(name="stm-summarizer", daemon=True)
        self._stm = stm
        self._stop_event = threading.Event()
        self._poll_interval = config.STM_POLL_INTERVAL
        self._batch_threshold = config.STM_BATCH_THRESHOLD
        self._max_batch_size = config.STM_MAX_BATCH_SIZE

    def stop(self) -> None:
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self) -> None:
        """Main loop: check for evictions, summarize in batches."""
        log.info("Background summarizer started")
        while not self._stop_event.is_set():
            try:
                evictions = self._stm.drain_evictions()
                if len(evictions) >= self._batch_threshold:
                    batch = evictions[: self._max_batch_size]
                    new_summary = self._summarize_batch(batch)
                    if new_summary:
                        if self._stm.running_summary:
                            self._stm.running_summary = self._merge_summaries(
                                self._stm.running_summary, new_summary
                            )
                        else:
                            self._stm.running_summary = new_summary
                        log.info(
                            "Updated running summary (%d chars)", len(self._stm.running_summary)
                        )
                elif evictions:
                    # Put them back if not enough for a batch yet
                    self._stm.requeue_evictions(evictions)

            except Exception:
                log.exception("Summarizer error; continuing")

            self._stop_event.wait(self._poll_interval)

        log.info("Background summarizer stopped")

    def _summarize_batch(self, messages: list[STMMessage]) -> str:
        """Summarize a batch of messages using LLM."""
        combined = "\n".join(f"{m.role}: {m.content}" for m in messages)
        prompt = SUMMARIZATION_PROMPT.format(
            messages=combined,
            previous_summary=self._stm.running_summary or "None",
        )

        try:
            completion = chat_completion(
                model=config.FAST_LLM_MODEL,
                max_tokens=config.FAST_LLM_MAX_TOKENS,
                messages=({"role": "user", "content": prompt},),
                disable_thinking=True,
            )
            return completion.text.strip()
        except Exception:
            log.exception("Summarization LLM call failed")
            # Divide and conquer on failure (could be context overflow)
            if len(messages) > 1:
                mid = len(messages) // 2
                left = self._summarize_batch(messages[:mid])
                right = self._summarize_batch(messages[mid:])
                if left and right:
                    return self._merge_summaries(left, right)
                return left or right
            return ""

    def _merge_summaries(self, existing: str, new: str) -> str:
        """Merge new summary into existing running summary."""
        prompt = SUMMARIZATION_PROMPT.format(
            messages=f"Existing summary:\n{existing}\n\nNew information:\n{new}",
            previous_summary="",
        )
        try:
            completion = chat_completion(
                model=config.FAST_LLM_MODEL,
                max_tokens=config.FAST_LLM_MAX_TOKENS,
                messages=({"role": "user", "content": prompt},),
                disable_thinking=True,
            )
            return completion.text.strip()
        except Exception:
            log.exception("Summary merge failed; concatenating")
            return f"{existing}\n\n{new}"
