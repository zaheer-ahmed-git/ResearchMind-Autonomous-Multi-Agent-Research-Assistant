"""Step-by-step research pipeline.

coordinator → research → analysis → (research | synthesis) → qa → human checkpoint
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from langgraph.types import interrupt

from research_agent.agents.analysis import AnalysisAgent
from research_agent.agents.coordinator import CoordinatorAgent
from research_agent.agents.qa import QAAgent
from research_agent.agents.research import ResearchAgent
from research_agent.agents.synthesis import SynthesisAgent
from research_agent.hitl.checkpoint import HumanCheckpoint
from research_agent.hitl.verification import automatic_decision, decision_from_resume
from research_agent.memory.vector_store import VectorStore
from research_agent.models.state import ResearchState
from research_agent.traces.collector import TraceCollector

_COORDINATOR = CoordinatorAgent()
_RESEARCH = ResearchAgent()
_ANALYSIS = AnalysisAgent()
_SYNTHESIS = SynthesisAgent()
_QA = QAAgent()
_CHECKPOINTS = HumanCheckpoint()


def coordinator_step(state: ResearchState) -> dict[str, Any]:
    return _COORDINATOR.start(state)


def research_step(state: ResearchState) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    working: dict[str, Any] = dict(state)
    if state.get("should_refine"):
        refined = _COORDINATOR.refine_for_research(state)
        traces.extend(list(refined.pop("traces", [])))
        working.update(refined)
    gathered = _RESEARCH.gather(working)
    traces.extend(list(gathered.pop("traces", [])))
    gathered["traces"] = traces
    gathered["refine_loops"] = int(working.get("refine_loops") or 0)
    gathered["should_refine"] = False
    if working.get("route_log"):
        gathered["route_log"] = list(working.get("route_log") or [])
    return gathered


def analysis_step(state: ResearchState) -> dict[str, Any]:
    return _ANALYSIS.analyze(state)


def synthesis_step(state: ResearchState) -> dict[str, Any]:
    drafted = _SYNTHESIS.synthesize(state)
    vector = VectorStore.from_dicts(list(state.get("vector_docs") or []))
    report = str(drafted.get("report") or "")
    if report:
        vector.add(
            report[:1200],
            source="synthesis_agent",
            doc_type="synthesis",
            task_id=state.get("run_id"),
            tags=["report"],
            confidence=float(state.get("confidence") or 0.5),
            author="synthesis_agent",
        )
    drafted["vector_docs"] = vector.to_dicts()
    return drafted


def qa_step(state: ResearchState) -> dict[str, Any]:
    return _QA.validate(state)


def human_checkpoint_step(state: ResearchState) -> dict[str, Any]:
    run_id = state.get("run_id") or str(uuid4())
    qa = dict(state.get("qa") or {})
    tracer = TraceCollector(run_id)
    payload = {
        "run_id": run_id,
        "checkpoint_type": "final_review",
        "query": state.get("query"),
        "confidence": state.get("confidence"),
        "qa_score": qa.get("score"),
        "weaknesses": qa.get("weaknesses") or [],
        "report_excerpt": str(state.get("report") or "")[:500],
    }
    checkpoint_id = _CHECKPOINTS.create(
        task_id=run_id,
        checkpoint_type="final_review",
        data=payload,
        priority=2 if state.get("needs_human") else 1,
    )
    payload["checkpoint_id"] = checkpoint_id
    start = tracer.start("human_checkpoint", "Opening human verification checkpoint", checkpoint_id=checkpoint_id)

    pause = bool(state.get("pause_for_human")) and bool(state.get("needs_human"))
    if pause:
        resumed = interrupt(payload)
        decision = decision_from_resume(resumed)
    else:
        decision = automatic_decision(qa)

    _CHECKPOINTS.submit(
        checkpoint_id,
        approved=bool(decision.get("approved")),
        feedback=decision.get("feedback"),
        reviewer=decision.get("reviewer"),
        modifications=decision.get("modifications"),
    )
    loops = int(state.get("revision_loops") or 0)
    should_revise = (not decision.get("approved")) and loops < 1
    end = tracer.end(
        "human_checkpoint",
        "Human verification accepted" if decision.get("approved") else "Human verification requested a revision",
        approved=bool(decision.get("approved")),
        reviewer=decision.get("reviewer"),
    )
    return {
        "checkpoint": {"checkpoint_id": checkpoint_id, **payload},
        "human_decision": decision,
        "should_revise": should_revise,
        "revision_loops": loops + (1 if should_revise else 0),
        "stage": "human_checkpoint",
        "status": "completed",
        "traces": [start, end],
    }


def route_after_analysis(state: ResearchState) -> str:
    return _COORDINATOR.route_after_analysis(state)


def route_after_human(state: ResearchState) -> str:
    return _COORDINATOR.route_after_human(state)
