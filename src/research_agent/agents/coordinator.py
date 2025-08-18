"""Coordinator agent: task decomposition and stage routing."""

from __future__ import annotations

from typing import Any, Literal

from research_agent.graph.nodes.planner import planner_node
from research_agent.graph.nodes.refiner import refiner_node


def _retag(traces: list[dict[str, Any]], node: str) -> list[dict[str, Any]]:
    renamed: list[dict[str, Any]] = []
    for event in traces:
        item = dict(event)
        item["node"] = node
        renamed.append(item)
    return renamed


class CoordinatorAgent:
    """Owns the first step and the decision to research again or synthesize."""

    def start(self, state: dict[str, Any]) -> dict[str, Any]:
        planned = planner_node(state)
        route_log = list(state.get("route_log") or [])
        route_log.append("research")
        return {
            "run_id": planned.get("run_id"),
            "subtasks": planned.get("subtasks") or [],
            "mode": planned.get("mode") or "offline",
            "refine_loops": planned.get("refine_loops", state.get("refine_loops", 0)),
            "max_refine_loops": planned.get("max_refine_loops", state.get("max_refine_loops")),
            "route": "research",
            "route_log": route_log,
            "stage": "coordinator",
            "traces": _retag(list(planned.get("traces") or []), "coordinator"),
            "metrics": planned.get("metrics") or {},
        }

    def refine_for_research(self, state: dict[str, Any]) -> dict[str, Any]:
        """When analysis finds gaps, open a new research pass with follow-up subtasks."""
        if not state.get("should_refine"):
            return {}
        refined = refiner_node(state)
        route_log = list(state.get("route_log") or [])
        route_log.append("research")
        return {
            "subtasks": refined.get("subtasks") or [],
            "refine_loops": refined.get("refine_loops", 0),
            "should_refine": False,
            "route": "research",
            "route_log": route_log,
            "stage": "coordinator",
            "traces": _retag(list(refined.get("traces") or []), "coordinator"),
            "metrics": refined.get("metrics") or {},
        }

    @staticmethod
    def route_after_analysis(state: dict[str, Any]) -> Literal["research", "synthesis"]:
        if state.get("should_refine"):
            return "research"
        return "synthesis"

    @staticmethod
    def route_after_human(state: dict[str, Any]) -> Literal["analysis", "end"]:
        if state.get("should_revise"):
            return "analysis"
        return "end"
