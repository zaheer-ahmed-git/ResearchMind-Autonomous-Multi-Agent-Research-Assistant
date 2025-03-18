"""In-run short-term memory store with priority metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Any
from uuid import uuid4


@dataclass
class MemoryItem:
    content: str
    agent_id: str
    priority: float = 0.5
    context_tags: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time)
    id: str = field(default_factory=lambda: str(uuid4())[:8])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "agent_id": self.agent_id,
            "priority": self.priority,
            "context_tags": list(self.context_tags),
            "timestamp": self.timestamp,
        }


class MemoryStore:
    """Bounded short-term memory used across graph nodes."""

    def __init__(self, capacity: int = 24) -> None:
        self.capacity = max(4, capacity)
        self._items: list[MemoryItem] = []

    def add(
        self,
        content: str,
        *,
        agent_id: str,
        priority: float = 0.5,
        context_tags: list[str] | None = None,
    ) -> MemoryItem:
        item = MemoryItem(
            content=content.strip(),
            agent_id=agent_id,
            priority=max(0.0, min(1.0, priority)),
            context_tags=context_tags or [],
        )
        self._items.append(item)
        return item

    def extend_from_dicts(self, rows: list[dict[str, Any]]) -> None:
        for row in rows:
            self._items.append(
                MemoryItem(
                    id=str(row.get("id", str(uuid4())[:8])),
                    content=str(row.get("content", "")),
                    agent_id=str(row.get("agent_id", "unknown")),
                    priority=float(row.get("priority", 0.5)),
                    context_tags=list(row.get("context_tags") or []),
                    timestamp=float(row.get("timestamp", time())),
                )
            )

    def as_dicts(self) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._items]

    @property
    def items(self) -> list[MemoryItem]:
        return list(self._items)

    def replace(self, items: list[MemoryItem]) -> None:
        self._items = list(items)
