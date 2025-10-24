"""API and domain schemas for the research agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class SubTask(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    question: str
    priority: float = 1.0
    status: Literal["pending", "gathering", "done", "skipped"] = "pending"


class EvidenceItem(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    subtask_id: str
    title: str
    content: str
    source: str
    score: float = 0.0
    rank: int = 0
    tags: list[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    run_id: str
    node: str
    event_type: Literal["start", "end", "info", "decision", "error"] = "info"
    message: str
    duration_ms: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=utc_now)


class ResearchRequest(BaseModel):
    query: str = Field(min_length=3, max_length=2000)
    max_refine_loops: int | None = None
    use_llm: bool | None = None
    pause_for_human: bool = False
    quality_threshold: float | None = None


class ResumeRequest(BaseModel):
    approved: bool
    feedback: str | None = None
    reviewer: str | None = None


class ResearchResponse(BaseModel):
    run_id: str
    query: str
    report: str
    confidence: float
    subtasks: list[SubTask]
    evidence: list[EvidenceItem]
    refine_loops: int
    latency_ms: float
    baseline_latency_ms: float
    latency_reduction_pct: float
    traces: list[TraceEvent]
    mode: Literal["offline", "llm"]
    status: Literal["completed", "paused"] = "completed"
    qa_score: float = 0.0
    human_review: dict[str, Any] = Field(default_factory=dict)
    checkpoint_id: str | None = None
    stage: str = "completed"
