"""Analyst node — theme extraction, gap detection, memory pruning."""

from __future__ import annotations

from collections import Counter
from typing import Any
from uuid import uuid4

from research_agent.config import get_settings
from research_agent.memory.pruning import prune_memory
from research_agent.memory.store import MemoryStore
from research_agent.models.state import ResearchState
from research_agent.ranking.adaptive import rank_evidence
from research_agent.traces.collector import TraceCollector


def analyst_node(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    run_id = state.get("run_id") or str(uuid4())
    query = state.get("query") or ""
    tracer = TraceCollector(run_id)
    start = tracer.start("analyst", "Analyzing evidence and pruning memory")

    evidence = list(state.get("evidence") or [])
    store = MemoryStore(capacity=settings.memory_capacity)
    store.extend_from_dicts(list(state.get("memory_items") or []))

    tag_counter: Counter[str] = Counter()
    for item in evidence:
        for tag in item.get("tags") or []:
            tag_counter[str(tag).lower()] += 1
    themes = [tag for tag, _ in tag_counter.most_common(5)]
    if not themes:
        themes = [t for t in query.lower().split() if len(t) > 3][:3]

    covered = " ".join(f"{e.get('title', '')} {e.get('content', '')}".lower() for e in evidence)
    candidate_gaps = [
        "empirical benchmarks",
        "failure modes",
        "deployment constraints",
        "evaluation metrics",
        "safety and bias",
    ]
    gaps = [gap for gap in candidate_gaps if gap.split()[0] not in covered][:3]

    top_scores = [float(e.get("score", 0.0)) for e in evidence[:5]]
    avg_score = sum(top_scores) / max(1, len(top_scores))
    coverage = min(1.0, len(evidence) / 6.0)
    confidence = round(0.55 * avg_score + 0.45 * coverage - 0.05 * len(gaps), 4)
    confidence = max(0.05, min(0.98, confidence))

    keep_ratio = 1.0 - settings.memory_prune_ratio
    pruned_store, removed = prune_memory(
        store,
        query=query,
        keep_ratio=keep_ratio,
        hard_cap=settings.memory_capacity,
    )
    pruned_store.add(
        f"themes={themes}; gaps={gaps}; confidence={confidence}",
        agent_id="analyst",
        priority=0.9,
        context_tags=themes,
    )

    ranked, meta = rank_evidence(
        evidence,
        query=query,
        analysis_keywords=themes,
        recency_bias=settings.ranking_recency_bias,
    )

    analysis = {
        "themes": themes,
        "gaps": gaps,
        "avg_evidence_score": round(avg_score, 4),
        "coverage": round(coverage, 4),
    }
    should = (
        confidence < settings.quality_threshold
        and int(state.get("refine_loops") or 0) < int(state.get("max_refine_loops") or settings.max_refine_loops)
        and bool(gaps)
    )

    end = tracer.end(
        "analyst",
        f"confidence={confidence:.2f}; pruned={removed}; refine={should}",
        confidence=confidence,
        memory_pruned=removed,
        gaps=gaps,
    )
    decision = tracer.decision(
        "analyst",
        "Route to refiner" if should else "Route to synthesizer",
        should_refine=should,
    )
    return {
        "analysis": analysis,
        "confidence": confidence,
        "evidence": ranked,
        "ranking_meta": meta,
        "memory_items": pruned_store.as_dicts(),
        "memory_pruned": int(state.get("memory_pruned") or 0) + removed,
        "should_refine": should,
        "traces": [start, end, decision],
        "metrics": {
            "confidence": confidence,
            "memory_pruned": int(state.get("memory_pruned") or 0) + removed,
        },
    }
