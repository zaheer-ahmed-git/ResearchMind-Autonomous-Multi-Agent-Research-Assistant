"""Shared in-memory run registry for API handlers."""

from __future__ import annotations

from research_agent.models.schemas import ResearchResponse

_RUNS: dict[str, ResearchResponse] = {}


def save_run(response: ResearchResponse) -> None:
    _RUNS[response.run_id] = response


def get_run(run_id: str) -> ResearchResponse | None:
    return _RUNS.get(run_id)


def list_runs() -> list[ResearchResponse]:
    return sorted(_RUNS.values(), key=lambda item: item.run_id, reverse=True)


def clear_runs() -> None:
    _RUNS.clear()
