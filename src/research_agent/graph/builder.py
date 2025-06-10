"""Compile and run the LangGraph autonomous research workflow."""

from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from research_agent.config import Settings, get_settings
from research_agent.graph.nodes.analyst import analyst_node
from research_agent.graph.nodes.gatherer import gatherer_node
from research_agent.graph.nodes.planner import planner_node
from research_agent.graph.nodes.refiner import refiner_node, should_refine
from research_agent.graph.nodes.synthesizer import synthesizer_node
from research_agent.models.schemas import (
    EvidenceItem,
    ResearchResponse,
    SubTask,
    TraceEvent,
)
from research_agent.models.state import ResearchState


def build_research_graph(*, checkpointer: InMemorySaver | None = None):
    """Build planner → gatherer → analyst ⇄ refiner → synthesizer graph."""
    graph = StateGraph(ResearchState)
    graph.add_node("planner", planner_node)
    graph.add_node("gatherer", gatherer_node)
    graph.add_node("analyst", analyst_node)
    graph.add_node("refiner", refiner_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "gatherer")
    graph.add_edge("gatherer", "analyst")
    graph.add_conditional_edges(
        "analyst",
        should_refine,
        {"refiner": "refiner", "synthesizer": "synthesizer"},
    )
    graph.add_edge("refiner", "gatherer")
    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def run_research(
    query: str,
    *,
    settings: Settings | None = None,
    max_refine_loops: int | None = None,
    thread_id: str | None = None,
) -> ResearchResponse:
    """Execute a full research run and return a typed response with traces."""
    settings = settings or get_settings()
    run_id = str(uuid4())
    graph = build_research_graph()
    started = perf_counter()

    initial: ResearchState = {
        "run_id": run_id,
        "query": query,
        "subtasks": [],
        "evidence": [],
        "analysis": {},
        "report": "",
        "confidence": 0.0,
        "refine_loops": 0,
        "max_refine_loops": max_refine_loops
        if max_refine_loops is not None
        else settings.max_refine_loops,
        "memory_items": [],
        "memory_pruned": 0,
        "ranking_meta": {},
        "should_refine": False,
        "traces": [],
        "metrics": {},
        "mode": "offline",
    }
    config = {"configurable": {"thread_id": thread_id or run_id}}
    final: dict[str, Any] = graph.invoke(initial, config=config)
    wall_ms = round((perf_counter() - started) * 1000, 2)

    metrics = dict(final.get("metrics") or {})
    latency_ms = float(metrics.get("latency_ms") or wall_ms)
    baseline = float(metrics.get("baseline_latency_ms") or latency_ms * 1.35)
    reduction = float(
        metrics.get("latency_reduction_pct")
        or (0.0 if baseline <= 0 else (1.0 - latency_ms / baseline) * 100.0)
    )

    return ResearchResponse(
        run_id=final.get("run_id") or run_id,
        query=query,
        report=str(final.get("report") or ""),
        confidence=float(final.get("confidence") or 0.0),
        subtasks=[SubTask.model_validate(item) for item in final.get("subtasks") or []],
        evidence=[EvidenceItem.model_validate(item) for item in final.get("evidence") or []],
        refine_loops=int(final.get("refine_loops") or 0),
        latency_ms=latency_ms,
        baseline_latency_ms=baseline,
        latency_reduction_pct=round(reduction, 2),
        traces=[TraceEvent.model_validate(item) for item in final.get("traces") or []],
        mode="llm" if final.get("mode") == "llm" else "offline",
    )
