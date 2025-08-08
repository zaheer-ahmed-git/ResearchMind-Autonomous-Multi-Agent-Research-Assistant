"""Short-term memory used by every agent in the research pipeline.

Matches the original Memory/Stm.py contract: add, get, remove, clear,
with category and priority. Items can also be exported to the pruning store.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from research_agent.memory.store import MemoryItem, MemoryStore


class ShortTermMemory:
    """Bounded in-run memory log shared across coordinator, research, analysis, and QA."""

    def __init__(self, max_size: int = 24) -> None:
        self.max_size = max(4, max_size)
        self._items: list[dict[str, Any]] = []

    def add(
        self,
        content: Any,
        agent_id: str,
        category: str = "default",
        priority: int = 5,
    ) -> str:
        """Store one memory. Priority is an integer from 1 (low) to 10 (high)."""
        level = max(1, min(10, int(priority)))
        memory_id = str(uuid.uuid4())[:8]
        self._items.append(
            {
                "id": memory_id,
                "content": str(content),
                "agent_id": agent_id,
                "category": category,
                "priority": level,
                "timestamp": time.time(),
            }
        )
        if len(self._items) > self.max_size:
            self._items = self._items[-self.max_size :]
        return memory_id

    def get(
        self,
        category: str | None = None,
        max_age: float | None = None,
        min_priority: int = 1,
    ) -> list[dict[str, Any]]:
        now = time.time()
        found: list[dict[str, Any]] = []
        for item in self._items:
            if category and item["category"] != category:
                continue
            if max_age is not None and (now - float(item["timestamp"])) > max_age:
                continue
            if int(item["priority"]) < min_priority:
                continue
            found.append(dict(item))
        return found

    def remove(self, memory_id: str) -> bool:
        before = len(self._items)
        self._items = [item for item in self._items if item["id"] != memory_id]
        return len(self._items) < before

    def clear(self, category: str | None = None) -> None:
        if category is None:
            self._items = []
            return
        self._items = [item for item in self._items if item["category"] != category]

    def to_dicts(self) -> list[dict[str, Any]]:
        return [dict(item) for item in self._items]

    @classmethod
    def from_dicts(cls, rows: list[dict[str, Any]], max_size: int = 24) -> ShortTermMemory:
        memory = cls(max_size=max_size)
        for row in rows:
            priority = row.get("priority", 5)
            # Pruning store uses 0-1 floats. STM uses 1-10 integers.
            if isinstance(priority, float) and priority <= 1:
                priority = max(1, int(round(priority * 10)))
            memory._items.append(
                {
                    "id": str(row.get("id") or str(uuid.uuid4())[:8]),
                    "content": str(row.get("content", "")),
                    "agent_id": str(row.get("agent_id", "unknown")),
                    "category": str(row.get("category") or (row.get("context_tags") or ["default"])[0]),
                    "priority": max(1, min(10, int(priority))),
                    "timestamp": float(row.get("timestamp", time.time())),
                }
            )
        if len(memory._items) > memory.max_size:
            memory._items = memory._items[-memory.max_size :]
        return memory

    def to_memory_store(self) -> MemoryStore:
        """Export to the float-priority store used by the pruning step."""
        store = MemoryStore(capacity=self.max_size)
        items = [
            MemoryItem(
                id=str(item["id"]),
                content=str(item["content"]),
                agent_id=str(item["agent_id"]),
                priority=int(item["priority"]) / 10.0,
                context_tags=[str(item["category"])],
                timestamp=float(item["timestamp"]),
            )
            for item in self._items
        ]
        store.replace(items)
        return store

    def __len__(self) -> int:
        return len(self._items)
