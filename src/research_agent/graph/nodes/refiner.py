"""Refiner node — recursive gap-driven re-planning."""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from research_agent.config import get_settings
from research_agent.llm.provider import get_llm_provider
from research_agent.models.state import ResearchState
from research_agent.traces.collector import TraceCollector


def refiner_node(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    run_id = state.get("run_id") or str(uuid4())
    query = state.get("query") or ""
    provider = get_llm_provider(settings)
    tracer = TraceCollector(run_id)

    loops = int(state.get("refine_loops") or 0) + 1
    start = tracer.start("refiner", f"Starting refine loop {loops}")
    gaps = list((state.get("analysis") or {}).get("gaps") or [])
    prior = list(state.get("subtasks") or [])
    refined = provider.refine_plan(query, gaps, prior)
    subtasks = [
        {
            "id": str(uuid4())[:8],
            "question": item["question"],
            "priority": float(item.get("priority", 0.8)),
            "status": "pending",
        }
        for item in refined
    ]
    end = tracer.end(
        "refiner",
        f"Refined into {len(subtasks)} follow-up subtasks",
        refine_loops=loops,
        gaps=gaps,
    )
    return {
        "subtasks": subtasks,
        "refine_loops": loops,
        "should_refine": False,
        "traces": [start, end],
        "metrics": {"refine_loops": loops},
    }


def should_refine(state: ResearchState) -> Literal["refiner", "synthesizer"]:
    if state.get("should_refine"):
        return "refiner"
    return "synthesizer"
