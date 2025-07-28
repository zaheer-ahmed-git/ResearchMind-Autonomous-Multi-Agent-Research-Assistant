"""Offline demo: run the research graph without an OpenAI API key."""

from __future__ import annotations

import json

from research_agent.config import get_settings
from research_agent.graph.builder import run_research


def main() -> None:
    settings = get_settings()
    settings.use_llm = False
    result = run_research(
        "How can LangGraph agents use memory pruning and adaptive ranking "
        "to reduce hallucinations during autonomous research?"
    )
    print("run_id:", result.run_id)
    print("mode:", result.mode)
    print("confidence:", result.confidence)
    print(
        f"latency: {result.latency_ms} ms "
        f"(baseline {result.baseline_latency_ms} ms, "
        f"-{result.latency_reduction_pct}%)"
    )
    print("subtasks:", len(result.subtasks))
    print("evidence:", len(result.evidence))
    print("traces:", len(result.traces))
    print("\n--- report ---\n")
    print(result.report)
    print("\n--- first traces ---\n")
    print(json.dumps([t.model_dump(mode="json") for t in result.traces[:4]], indent=2))


if __name__ == "__main__":
    main()
