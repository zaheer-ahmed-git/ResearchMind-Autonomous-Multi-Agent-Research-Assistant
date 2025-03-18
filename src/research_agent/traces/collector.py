"""Helpers for recording node spans and latency reduction metrics."""

from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import uuid4

from research_agent.models.schemas import utc_now


class TraceCollector:
    """Builds TraceEvent-compatible dicts for LangGraph state.traces."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self._starts: dict[str, float] = {}

    def start(self, node: str, message: str, **payload: Any) -> dict[str, Any]:
        self._starts[node] = perf_counter()
        return {
            "id": str(uuid4())[:8],
            "run_id": self.run_id,
            "node": node,
            "event_type": "start",
            "message": message,
            "duration_ms": None,
            "payload": payload,
            "timestamp": utc_now().isoformat(),
        }

    def end(self, node: str, message: str, **payload: Any) -> dict[str, Any]:
        started = self._starts.pop(node, None)
        duration = None if started is None else round((perf_counter() - started) * 1000, 2)
        return {
            "id": str(uuid4())[:8],
            "run_id": self.run_id,
            "node": node,
            "event_type": "end",
            "message": message,
            "duration_ms": duration,
            "payload": payload,
            "timestamp": utc_now().isoformat(),
        }

    def info(self, node: str, message: str, **payload: Any) -> dict[str, Any]:
        return {
            "id": str(uuid4())[:8],
            "run_id": self.run_id,
            "node": node,
            "event_type": "info",
            "message": message,
            "duration_ms": None,
            "payload": payload,
            "timestamp": utc_now().isoformat(),
        }

    def decision(self, node: str, message: str, **payload: Any) -> dict[str, Any]:
        event = self.info(node, message, **payload)
        event["event_type"] = "decision"
        return event


def estimate_latency(
    *,
    evidence_count: int,
    memory_count: int,
    memory_pruned: int,
    refine_loops: int,
    demo_latency_ms: int = 12,
) -> dict[str, float]:
    """Estimate optimized vs baseline graph latency from state size factors.

    Baseline assumes unpruned memory and unranked full evidence payloads.
    Optimized path benefits from pruning + adaptive ranking (target ~35%).
    """
    baseline = demo_latency_ms * (
        8 + evidence_count * 1.8 + memory_count * 1.4 + refine_loops * 3.5
    )
    # Pruning and ranking shrink payload / re-check cost
    prune_factor = 1.0 - min(0.35, 0.08 * memory_pruned)
    rank_factor = 0.88 if evidence_count else 1.0
    optimized = baseline * prune_factor * rank_factor
    reduction = 0.0 if baseline <= 0 else (1.0 - optimized / baseline) * 100.0
    return {
        "baseline_latency_ms": round(baseline, 2),
        "latency_ms": round(optimized, 2),
        "latency_reduction_pct": round(reduction, 2),
    }
