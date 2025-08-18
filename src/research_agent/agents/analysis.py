"""Analysis agent: themes, gaps, confidence, and memory pruning."""

from __future__ import annotations

from typing import Any

from research_agent.graph.nodes.analyst import analyst_node
from research_agent.memory.vector_store import VectorStore


def _retag(traces: list[dict[str, Any]], node: str) -> list[dict[str, Any]]:
    renamed: list[dict[str, Any]] = []
    for event in traces:
        item = dict(event)
        item["node"] = node
        renamed.append(item)
    return renamed


class AnalysisAgent:
    """Reads research evidence and decides whether another research pass is needed."""

    def analyze(self, state: dict[str, Any]) -> dict[str, Any]:
        analyzed = analyst_node(state)
        vector = VectorStore.from_dicts(list(state.get("vector_docs") or []))
        hits = vector.similarity_search(str(state.get("query") or ""), k=4)
        analysis = dict(analyzed.get("analysis") or {})
        analysis["vector_hits"] = len(hits)
        return {
            "analysis": analysis,
            "confidence": analyzed.get("confidence") or 0.0,
            "evidence": analyzed.get("evidence") or state.get("evidence") or [],
            "ranking_meta": analyzed.get("ranking_meta") or {},
            "memory_items": analyzed.get("memory_items") or [],
            "memory_pruned": analyzed.get("memory_pruned") or 0,
            "vector_hits": hits,
            "should_refine": bool(analyzed.get("should_refine")),
            "stage": "analysis",
            "traces": _retag(list(analyzed.get("traces") or []), "analysis"),
            "metrics": analyzed.get("metrics") or {},
        }
