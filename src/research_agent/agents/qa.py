"""Quality-assurance agent: score the report and flag human review."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from research_agent.config import get_settings
from research_agent.models.state import ResearchState
from research_agent.traces.collector import TraceCollector


class QAAgent:
    """Validates factual coverage, citations, and whether a person should review."""

    def validate(self, state: ResearchState) -> dict[str, Any]:
        settings = get_settings()
        run_id = state.get("run_id") or str(uuid4())
        tracer = TraceCollector(run_id)
        start = tracer.start("qa", "Validating report quality")

        threshold = float(state.get("quality_threshold") or settings.quality_threshold)
        confidence = float(state.get("confidence") or 0.0)
        evidence = list(state.get("evidence") or [])
        report = str(state.get("report") or "")
        gaps = list((state.get("analysis") or {}).get("gaps") or [])

        coverage = min(1.0, len(evidence) / 4.0)
        has_citations = "[" in report and "]" in report
        score = min(1.0, 0.65 * confidence + 0.25 * coverage + (0.10 if has_citations else 0.0))
        score = round(score, 4)

        weaknesses: list[str] = []
        if not has_citations:
            weaknesses.append("Report is missing rank citations")
        if len(evidence) < 3:
            weaknesses.append("Evidence set is thin")
        if gaps:
            weaknesses.append("Open gaps remain: " + ", ".join(gaps[:3]))
        if confidence < threshold:
            weaknesses.append("Confidence is below the quality threshold")

        is_valid = score >= threshold and not gaps
        needs_human = score < threshold or bool(gaps)
        recommendations = [
            "Cite only ranked evidence",
            "Close remaining gaps before publication" if gaps else "Keep the bounded refine loop",
        ]

        end = tracer.end(
            "qa",
            f"score={score:.2f}; valid={is_valid}; needs_human={needs_human}",
            qa_score=score,
            is_valid=is_valid,
            needs_human=needs_human,
        )
        return {
            "qa": {
                "score": score,
                "is_valid": is_valid,
                "needs_human": needs_human,
                "weaknesses": weaknesses,
                "recommendations": recommendations,
                "threshold": threshold,
            },
            "needs_human": needs_human,
            "stage": "qa",
            "traces": [start, end],
            "metrics": {"qa_score": score},
        }
