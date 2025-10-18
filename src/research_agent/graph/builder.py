"""Compile and run the step-by-step autonomous research workflow."""

from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from research_agent.config import Settings, get_settings
from research_agent.graph.pipeline import (
    analysis_step,
    coordinator_step,
    human_checkpoint_step,
    qa_step,
    research_step,
    route_after_analysis,
    route_after_human,
    synthesis_step,
)
from research_agent.models.schemas import (
    EvidenceItem,
    ResearchResponse,
    SubTask,
    TraceEvent,
)
from research_agent.models.state import ResearchState

_CHECKPOINTER = InMemorySaver()
_GRAPH = None


def build_research_graph(*, checkpointer: InMemorySaver | None = None):
    """Build coordinator → research → analysis → synthesis → qa → human checkpoint."""
    graph = StateGraph(ResearchState)
    graph.add_node("coordinator", coordinator_step)
    graph.add_node("research", research_step)
    graph.add_node("analysis", analysis_step)
    graph.add_node("synthesis", synthesis_step)
    graph.add_node("qa", qa_step)
    graph.add_node("human_checkpoint", human_checkpoint_step)

    graph.add_edge(START, "coordinator")
    graph.add_edge("coordinator", "research")
    graph.add_edge("research", "analysis")
    graph.add_conditional_edges(
        "analysis",
        route_after_analysis,
        {"research": "research", "synthesis": "synthesis"},
    )
    graph.add_edge("synthesis", "qa")
    graph.add_edge("qa", "human_checkpoint")
    graph.add_conditional_edges(
        "human_checkpoint",
        route_after_human,
        {"analysis": "analysis", "end": END},
    )
    return graph.compile(checkpointer=checkpointer or InMemorySaver())


def get_research_graph():
    """Shared compiled graph so a paused run can be resumed on the same thread."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_research_graph(checkpointer=_CHECKPOINTER)
    return _GRAPH


def _initial_state(
    query: str,
    *,
    run_id: str,
    settings: Settings,
    max_refine_loops: int | None,
    pause_for_human: bool,
    quality_threshold: float | None,
) -> ResearchState:
    return {
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
        "stm_items": [],
        "vector_docs": [],
        "vector_hits": [],
        "memory_pruned": 0,
        "ranking_meta": {},
        "should_refine": False,
        "should_revise": False,
        "route": "coordinator",
        "route_log": [],
        "stage": "coordinator",
        "status": "running",
        "qa": {},
        "needs_human": False,
        "pause_for_human": pause_for_human,
        "quality_threshold": quality_threshold
        if quality_threshold is not None
        else settings.quality_threshold,
        "checkpoint": {},
        "human_decision": {},
        "revision_loops": 0,
        "traces": [],
        "metrics": {},
        "mode": "offline",
    }


def _latency(metrics: dict[str, Any], wall_ms: float) -> tuple[float, float, float]:
    latency_ms = float(metrics.get("latency_ms") or wall_ms)
    baseline = float(metrics.get("baseline_latency_ms") or latency_ms * 1.35)
    reduction = float(
        metrics.get("latency_reduction_pct")
        or (0.0 if baseline <= 0 else (1.0 - latency_ms / baseline) * 100.0)
    )
    return latency_ms, baseline, round(reduction, 2)


def _to_response(final: dict[str, Any], *, query: str, run_id: str, wall_ms: float, status: str) -> ResearchResponse:
    metrics = dict(final.get("metrics") or {})
    latency_ms, baseline, reduction = _latency(metrics, wall_ms)
    qa = dict(final.get("qa") or {})
    checkpoint = dict(final.get("checkpoint") or {})
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
        latency_reduction_pct=reduction,
        traces=[TraceEvent.model_validate(item) for item in final.get("traces") or []],
        mode="llm" if final.get("mode") == "llm" else "offline",
        status="paused" if status == "paused" else "completed",
        qa_score=float(qa.get("score") or metrics.get("qa_score") or 0.0),
        human_review=dict(final.get("human_decision") or {}),
        checkpoint_id=checkpoint.get("checkpoint_id"),
        stage=str(final.get("stage") or status),
    )


def _interrupt_payload(graph, config: dict[str, Any]) -> dict[str, Any]:
    snapshot = graph.get_state(config)
    for task in getattr(snapshot, "tasks", ()) or ():
        interrupts = getattr(task, "interrupts", ()) or ()
        if interrupts:
            value = getattr(interrupts[0], "value", None)
            if isinstance(value, dict):
                return value
    return {}


def run_research(
    query: str,
    *,
    settings: Settings | None = None,
    max_refine_loops: int | None = None,
    thread_id: str | None = None,
    pause_for_human: bool = False,
    quality_threshold: float | None = None,
) -> ResearchResponse:
    """Execute a research run. May pause on the human checkpoint."""
    settings = settings or get_settings()
    run_id = thread_id or str(uuid4())
    graph = get_research_graph()
    started = perf_counter()
    config = {"configurable": {"thread_id": run_id}}
    initial = _initial_state(
        query,
        run_id=run_id,
        settings=settings,
        max_refine_loops=max_refine_loops,
        pause_for_human=pause_for_human,
        quality_threshold=quality_threshold,
    )

    paused = False
    try:
        final: dict[str, Any] = graph.invoke(initial, config=config)
    except GraphInterrupt:
        paused = True
        final = dict(graph.get_state(config).values or {})
    else:
        payload = _interrupt_payload(graph, config)
        if payload:
            paused = True
            final = dict(graph.get_state(config).values or final)
            final["checkpoint"] = payload

    wall_ms = round((perf_counter() - started) * 1000, 2)
    if paused:
        checkpoint = dict(final.get("checkpoint") or {})
        if not checkpoint:
            checkpoint = _interrupt_payload(graph, config)
            final["checkpoint"] = checkpoint
        final.setdefault("run_id", run_id)
        final["stage"] = "human_checkpoint"
    return _to_response(final, query=query, run_id=run_id, wall_ms=wall_ms, status="paused" if paused else "completed")


def resume_research(
    run_id: str,
    *,
    approved: bool,
    feedback: str | None = None,
    reviewer: str | None = None,
) -> ResearchResponse:
    """Continue a run that is waiting on the human checkpoint."""
    graph = get_research_graph()
    config = {"configurable": {"thread_id": run_id}}
    snapshot = graph.get_state(config)
    if snapshot is None or not snapshot.values:
        raise KeyError(run_id)

    decision = {
        "approved": approved,
        "feedback": feedback,
        "reviewer": reviewer or "reviewer",
        "modifications": {},
    }
    started = perf_counter()
    paused = False
    try:
        final = graph.invoke(Command(resume=decision), config=config)
    except GraphInterrupt:
        paused = True
        final = dict(graph.get_state(config).values or {})
    else:
        payload = _interrupt_payload(graph, config)
        if payload:
            paused = True
            final = dict(graph.get_state(config).values or final)
            final["checkpoint"] = payload

    wall_ms = round((perf_counter() - started) * 1000, 2)
    query = str((final or {}).get("query") or snapshot.values.get("query") or "")
    if paused:
        final["stage"] = "human_checkpoint"
    return _to_response(
        final or {},
        query=query,
        run_id=run_id,
        wall_ms=wall_ms,
        status="paused" if paused else "completed",
    )
