"""Short-Term Memory with Neo4j persistence and LLM summarization.

Bounded deque of recent messages with character-based capacity. Evicted messages
are queued for background LLM summarization into a running summary. Neo4j
persistence enables crash recovery.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from neo4j import AsyncDriver

from .. import config

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class STMMessage:
    role: str
    content: str
    timestamp: str


class ShortTermMemory:
    """Bounded message buffer with running summary and Neo4j persistence."""

    def __init__(self, capacity: int = config.STM_BUFFER_CAPACITY) -> None:
        self._capacity = capacity
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

    def requeue_evictions(self, messages: list[STMMessage]) -> None:
        """Push messages back to eviction queue preserving order."""
        self._eviction_queue.extend(messages)

    def add_message(self, role: str, content: str) -> None:
        """Add a message to the buffer, evicting oldest if over capacity."""
        msg = STMMessage(
            role=role,
            content=content,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._buffer.append(msg)

        while sum(len(m.content) for m in self._buffer) > self._capacity and self._buffer:
            evicted = self._buffer.popleft()
            self._eviction_queue.append(evicted)

    def get_recent_context(self, max_messages: int = 5) -> str:
        """Format recent messages as context string."""
        recent = list(self._buffer)[-max_messages:]
        return "\n".join(f"{m.role}: {m.content}" for m in recent)

    def to_dict(self) -> dict[str, object]:
        """Serialize for Neo4j persistence."""
        return {
            "running_summary": self.running_summary,
            "message_buffer": [asdict(m) for m in self._buffer],
        }

    @classmethod
    def from_dict(
        cls, data: dict[str, object], capacity: int = config.STM_BUFFER_CAPACITY
    ) -> ShortTermMemory:
        """Restore from Neo4j persistence."""
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

    async def persist(self, neo4j_driver: AsyncDriver) -> None:
        """Save STM state to Neo4j for crash recovery."""
        data = self.to_dict()
        async with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
            await session.run(
                """
                MERGE (s:STMState {session_id: 'default'})
                SET s.running_summary = $summary,
                    s.message_buffer = $buffer,
                    s.last_updated = datetime()
                """,
                summary=data["running_summary"],
                buffer=json.dumps(data["message_buffer"]),
            )

    @classmethod
    async def load(cls, neo4j_driver: AsyncDriver) -> ShortTermMemory:
        """Load STM state from Neo4j."""
        try:
            async with neo4j_driver.session(database=config.NEO4J_DATABASE) as session:
                # MERGE ensures the node and its properties exist, preventing Neo4j
                # property-key-not-found notifications on first access after a fresh DB.
                result = await session.run(
                    """
                    MERGE (s:STMState {session_id: 'default'})
                    ON CREATE SET s.running_summary = '', s.message_buffer = '[]', s.last_updated = datetime()
                    RETURN s.running_summary, s.message_buffer
                    """
                )
                record = await result.single()
                if record:
                    raw_buffer = record[1]
                    parsed_buffer = (
                        json.loads(raw_buffer) if isinstance(raw_buffer, str) else raw_buffer or []
                    )
                    data = {"running_summary": record[0] or "", "message_buffer": parsed_buffer}
                    return cls.from_dict(data)
        except Exception:
            log.exception("Failed to load STM from Neo4j; starting fresh")
        return cls()
