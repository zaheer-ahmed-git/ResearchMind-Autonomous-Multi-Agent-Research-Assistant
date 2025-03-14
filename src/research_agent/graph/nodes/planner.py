"""Planner node — task decomposition."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from research_agent.config import get_settings
from research_agent.llm.provider import get_llm_provider
from research_agent.models.state import ResearchState
from research_agent.traces.collector import TraceCollector


def planner_node(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    run_id = state.get("run_id") or str(uuid4())
    query = state.get("query") or ""
    provider = get_llm_provider(settings)
    tracer = TraceCollector(run_id)

    start = tracer.start("planner", "Decomposing research query into subtasks")
    raw = provider.decompose(query, settings.max_subtasks)
    subtasks = [
        {
            "id": str(uuid4())[:8],
            "question": item["question"],
            "priority": float(item.get("priority", 0.7)),
            "status": "pending",
        }
        for item in raw
    ]
    end = tracer.end(
        "planner",
        f"Created {len(subtasks)} subtasks",
        subtask_count=len(subtasks),
        mode=provider.mode,
    )
    return {
        "run_id": run_id,
        "subtasks": subtasks,
        "mode": provider.mode,
        "refine_loops": state.get("refine_loops", 0),
        "max_refine_loops": state.get("max_refine_loops", settings.max_refine_loops),
        "traces": [start, end],
        "metrics": {"planned_subtasks": len(subtasks)},
    }
