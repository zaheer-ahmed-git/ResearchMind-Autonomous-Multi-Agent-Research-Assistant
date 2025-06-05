"""Synthesizer node — grounded report generation."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from research_agent.config import get_settings
from research_agent.llm.provider import get_llm_provider
from research_agent.models.state import ResearchState
from research_agent.traces.collector import TraceCollector, estimate_latency


def synthesizer_node(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    run_id = state.get("run_id") or str(uuid4())
    query = state.get("query") or ""
    provider = get_llm_provider(settings)
    tracer = TraceCollector(run_id)

    start = tracer.start("synthesizer", "Synthesizing grounded research brief")
    evidence = list(state.get("evidence") or [])
    analysis = dict(state.get("analysis") or {})
    report = provider.synthesize(query, evidence, analysis)

    latency = estimate_latency(
        evidence_count=len(evidence),
        memory_count=len(state.get("memory_items") or []),
        memory_pruned=int(state.get("memory_pruned") or 0),
        refine_loops=int(state.get("refine_loops") or 0),
        demo_latency_ms=settings.demo_latency_ms,
    )
    end = tracer.end(
        "synthesizer",
        "Report ready",
        report_chars=len(report),
        **latency,
    )
    return {
        "report": report,
        "mode": provider.mode,
        "should_refine": False,
        "traces": [start, end],
        "metrics": latency,
    }
