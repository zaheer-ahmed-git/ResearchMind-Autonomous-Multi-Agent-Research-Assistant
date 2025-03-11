"""LangGraph shared state for the research workflow."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def _merge_dicts(left: dict[str, Any] | None, right: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class ResearchState(TypedDict, total=False):
    """Shared snapshot passed between LangGraph nodes."""

    run_id: str
    query: str
    subtasks: list[dict[str, Any]]
    evidence: list[dict[str, Any]]
    analysis: dict[str, Any]
    report: str
    confidence: float
    refine_loops: int
    max_refine_loops: int
    memory_items: list[dict[str, Any]]
    memory_pruned: int
    ranking_meta: dict[str, Any]
    should_refine: bool
    traces: Annotated[list[dict[str, Any]], operator.add]
    metrics: Annotated[dict[str, Any], _merge_dicts]
    mode: str
