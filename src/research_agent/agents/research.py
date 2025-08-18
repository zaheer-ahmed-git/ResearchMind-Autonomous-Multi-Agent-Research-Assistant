"""Research agent: evidence gathering, ranking, and memory writes."""

from __future__ import annotations

from typing import Any

from research_agent.config import get_settings
from research_agent.graph.nodes.gatherer import gatherer_node
from research_agent.memory.stm import ShortTermMemory
from research_agent.memory.vector_store import VectorStore


def _retag(traces: list[dict[str, Any]], node: str) -> list[dict[str, Any]]:
    renamed: list[dict[str, Any]] = []
    for event in traces:
        item = dict(event)
        item["node"] = node
        renamed.append(item)
    return renamed


class ResearchAgent:
    """Collects evidence for each open subtask and stores it in memory."""

    def gather(self, state: dict[str, Any]) -> dict[str, Any]:
        gathered = gatherer_node(state)
        evidence = list(gathered.get("evidence") or [])
        settings = get_settings()
        memory = ShortTermMemory(max_size=settings.memory_capacity)
        for row in list(gathered.get("memory_items") or []):
            raw_priority = row.get("priority", 5)
            if isinstance(raw_priority, float) and raw_priority <= 1:
                raw_priority = max(1, int(round(raw_priority * 10)))
            memory.add(
                row.get("content", ""),
                "research_agent",
                category="research",
                priority=int(raw_priority),
            )
        vector = VectorStore.from_dicts(list(state.get("vector_docs") or []))
        vector.add_evidence(evidence, task_id=state.get("run_id"))
        return {
            "subtasks": gathered.get("subtasks") or [],
            "evidence": evidence,
            "ranking_meta": gathered.get("ranking_meta") or {},
            "memory_items": gathered.get("memory_items") or [],
            "stm_items": memory.to_dicts(),
            "vector_docs": vector.to_dicts(),
            "stage": "research",
            "route": "analysis",
            "traces": _retag(list(gathered.get("traces") or []), "research"),
            "metrics": gathered.get("metrics") or {},
        }
