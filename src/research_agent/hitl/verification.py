"""Human verification decision for the final QA checkpoint."""

from __future__ import annotations

from typing import Any


def automatic_decision(qa: dict[str, Any]) -> dict[str, Any]:
    """Offline default: record the checkpoint and continue without waiting."""
    if qa.get("needs_human"):
        feedback = "Automatic verification recorded a low QA score for later review."
    else:
        feedback = "Automatic verification accepted the QA result."
    return {
        "approved": True,
        "feedback": feedback,
        "reviewer": "auto",
        "modifications": {},
    }


def decision_from_resume(value: Any) -> dict[str, Any]:
    """Normalize the value passed back through LangGraph interrupt resume."""
    if isinstance(value, dict):
        return {
            "approved": bool(value.get("approved", False)),
            "feedback": value.get("feedback"),
            "reviewer": value.get("reviewer") or "reviewer",
            "modifications": dict(value.get("modifications") or {}),
        }
    return {
        "approved": bool(value),
        "feedback": None,
        "reviewer": "reviewer",
        "modifications": {},
    }
