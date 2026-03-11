"""Short-Term Memory with PostgreSQL persistence and LLM summarization.

Bounded deque of recent messages with character-based capacity. Evicted messages
are queued for background LLM summarization into a running summary. PostgreSQL
persistence enables crash recovery.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from .. import config

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class STMMessage:
    role: str
    content: str
    timestamp: str


class ShortTermMemory:
    """Bounded message buffer with running summary and PostgreSQL persistence."""

    def __init__(self, capacity: int | None = None) -> None:
        self._capacity = capacity or config.STM_BUFFER_CAPACITY
        self._buffer: deque[STMMessage] = deque()
        self.running_summary: str = ""
        self._eviction_queue: deque[STMMessage] = deque()

    @property
    def messages(self) -> list[STMMessage]:
        return list(self._buffer)

    @property
    def pending_evictions(self) -> list[STMMessage]:
        return list(self._eviction_queue)

    def drain_evictions(self) -> list[STMMessage]:
        """Return and clear pending evictions (for background summarizer)."""
        evicted = list(self._eviction_queue)
        self._eviction_queue.clear()
        return evicted

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the buffer, evicting oldest if over capacity."""
        msg = STMMessage(
            role=role,
            content=content,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._buffer.append(msg)

        while self._total_chars() > self._capacity and self._buffer:
            evicted = self._buffer.popleft()
            self._eviction_queue.append(evicted)

    def get_recent_context(self, max_messages: int = 5) -> str:
        """Format recent messages as context string."""
        recent = list(self._buffer)[-max_messages:]
        return "\n".join(f"{m.role}: {m.content}" for m in recent)

    def to_dict(self) -> dict[str, object]:
        """Serialize for PostgreSQL persistence."""
        return {
            "running_summary": self.running_summary,
            "message_buffer": [asdict(m) for m in self._buffer],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object], capacity: int | None = None) -> ShortTermMemory:
        """Restore from PostgreSQL persistence."""
        stm = cls(capacity=capacity)
        stm.running_summary = str(data.get("running_summary", ""))
        buffer_data = data.get("message_buffer", [])
        if isinstance(buffer_data, list):
            for item in buffer_data:
                if isinstance(item, dict):
                    stm._buffer.append(
                        STMMessage(
                            role=str(item.get("role", "")),
                            content=str(item.get("content", "")),
                            timestamp=str(item.get("timestamp", "")),
                        )
                    )
        return stm

    async def persist(self, pg_pool: object) -> None:
        """Save STM state to PostgreSQL for crash recovery."""
        data = self.to_dict()
        async with pg_pool.connection() as conn, conn.cursor() as cur:  # type: ignore[union-attr]
            await cur.execute(
                """
                INSERT INTO stm_state (session_id, running_summary, message_buffer, last_updated)
                VALUES ('default', %s, %s::jsonb, NOW())
                ON CONFLICT (session_id) DO UPDATE SET
                    running_summary = EXCLUDED.running_summary,
                    message_buffer = EXCLUDED.message_buffer,
                    last_updated = NOW()
                """,
                (data["running_summary"], json.dumps(data["message_buffer"])),
            )

    @classmethod
    async def load(cls, pg_pool: object) -> ShortTermMemory:
        """Load STM state from PostgreSQL."""
        try:
            async with pg_pool.connection() as conn, conn.cursor() as cur:  # type: ignore[union-attr]
                await cur.execute(
                    "SELECT running_summary, message_buffer FROM stm_state WHERE session_id = 'default'"
                )
                row = await cur.fetchone()
                if row:
                    data = {
                        "running_summary": row[0],
                        "message_buffer": row[1] if isinstance(row[1], list) else json.loads(str(row[1])),
                    }
                    return cls.from_dict(data)
        except Exception:
            log.exception("Failed to load STM from PostgreSQL; starting fresh")
        return cls()

    def _total_chars(self) -> int:
        return sum(len(m.content) for m in self._buffer)
