"""HTTP routes for starting research runs and fetching execution traces."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from research_agent import __version__
from research_agent.api.deps import get_run, list_runs, save_run
from research_agent.config import get_settings
from research_agent.graph.builder import run_research
from research_agent.models.schemas import ResearchRequest, ResearchResponse

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str | bool]:
    settings = get_settings()
    return {
        "status": "ok",
        "version": __version__,
        "use_llm": settings.use_llm and bool(settings.openai_api_key),
    }


@router.post("/research/run", response_model=ResearchResponse)
def start_research(body: ResearchRequest) -> ResearchResponse:
    settings = get_settings()
    # Optional per-request override without mutating global cached settings permanently.
    if body.use_llm is not None:
        settings.use_llm = body.use_llm
    result = run_research(
        body.query,
        settings=settings,
        max_refine_loops=body.max_refine_loops,
    )
    save_run(result)
    return result


@router.get("/research/runs", response_model=list[ResearchResponse])
def get_runs() -> list[ResearchResponse]:
    return list_runs()


@router.get("/research/runs/{run_id}", response_model=ResearchResponse)
def get_research_run(run_id: str) -> ResearchResponse:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return run


@router.get("/research/runs/{run_id}/traces")
def get_research_traces(run_id: str) -> dict:
    run = get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run.run_id,
        "query": run.query,
        "latency_ms": run.latency_ms,
        "baseline_latency_ms": run.baseline_latency_ms,
        "latency_reduction_pct": run.latency_reduction_pct,
        "traces": [event.model_dump(mode="json") for event in run.traces],
    }
