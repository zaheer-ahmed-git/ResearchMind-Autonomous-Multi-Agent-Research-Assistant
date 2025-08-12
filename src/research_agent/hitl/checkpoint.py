"""Human checkpoint registry.

The graph pauses with LangGraph interrupt(). This registry keeps the pending
review so the API can list it and submit a decision.
"""

from __future__ import annotations

from typing import Any

_CHECKPOINTS: dict[str, dict[str, Any]] = {}


class HumanCheckpoint:
    """Create, list, and resolve human review checkpoints."""

    def create(
        self,
        *,
        task_id: str,
        checkpoint_type: str,
        data: dict[str, Any],
        priority: int = 1,
    ) -> str:
        for checkpoint_id, row in _CHECKPOINTS.items():
            if (
                row["task_id"] == task_id
                and row["checkpoint_type"] == checkpoint_type
                and row["status"] == "pending"
            ):
                row["data"] = data
                row["priority"] = priority
                return checkpoint_id

        checkpoint_id = f"cp_{task_id}_{len(_CHECKPOINTS) + 1}"
        _CHECKPOINTS[checkpoint_id] = {
            "checkpoint_id": checkpoint_id,
            "task_id": task_id,
            "checkpoint_type": checkpoint_type,
            "data": data,
            "priority": priority,
            "status": "pending",
            "response": None,
        }
        return checkpoint_id

    def get(self, checkpoint_id: str) -> dict[str, Any] | None:
        row = _CHECKPOINTS.get(checkpoint_id)
        return dict(row) if row else None

    def pending(self, min_priority: int | None = None) -> list[dict[str, Any]]:
        rows = [
            dict(row)
            for row in _CHECKPOINTS.values()
            if row["status"] == "pending" and (min_priority is None or row["priority"] >= min_priority)
        ]
        return sorted(rows, key=lambda item: (-int(item["priority"]), item["checkpoint_id"]))

    def for_task(self, task_id: str) -> dict[str, Any] | None:
        matches = [row for row in _CHECKPOINTS.values() if row["task_id"] == task_id]
        if not matches:
            return None
        matches.sort(key=lambda item: item["checkpoint_id"])
        return dict(matches[-1])

    def submit(
        self,
        checkpoint_id: str,
        *,
        approved: bool,
        feedback: str | None = None,
        reviewer: str | None = None,
        modifications: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        row = _CHECKPOINTS.get(checkpoint_id)
        if row is None:
            raise KeyError(checkpoint_id)
        response = {
            "approved": approved,
            "feedback": feedback,
            "reviewer": reviewer or "reviewer",
            "modifications": modifications or {},
        }
        row["status"] = "approved" if approved else "rejected"
        row["response"] = response
        return dict(row)

    def clear(self) -> None:
        _CHECKPOINTS.clear()
