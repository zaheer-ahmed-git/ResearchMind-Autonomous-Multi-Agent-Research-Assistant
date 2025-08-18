"""Synthesis agent: grounded report from ranked evidence."""

from __future__ import annotations

from typing import Any

from research_agent.graph.nodes.synthesizer import synthesizer_node


def _retag(traces: list[dict[str, Any]], node: str) -> list[dict[str, Any]]:
    renamed: list[dict[str, Any]] = []
    for event in traces:
        item = dict(event)
        item["node"] = node
        renamed.append(item)
    return renamed


class SynthesisAgent:
    """Turns analyzed evidence into a cited research brief."""

    def synthesize(self, state: dict[str, Any]) -> dict[str, Any]:
        drafted = synthesizer_node(state)
        return {
            "report": drafted.get("report") or "",
            "mode": drafted.get("mode") or state.get("mode") or "offline",
            "should_refine": False,
            "stage": "synthesis",
            "traces": _retag(list(drafted.get("traces") or []), "synthesis"),
            "metrics": drafted.get("metrics") or {},
        }
