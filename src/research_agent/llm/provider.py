"""OpenAI-backed and offline heuristic providers for research nodes."""

from __future__ import annotations

import json
import re
from typing import Any

from research_agent.config import Settings, get_settings


class LLMProvider:
    """Thin adapter: ChatOpenAI when enabled, otherwise deterministic heuristics."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client = None
        if self.settings.use_llm and self.settings.openai_api_key:
            try:
                from langchain_openai import ChatOpenAI

                self._client = ChatOpenAI(
                    model=self.settings.openai_model,
                    api_key=self.settings.openai_api_key,
                    temperature=0.2,
                )
            except Exception:  # noqa: BLE001
                self._client = None

    @property
    def mode(self) -> str:
        return "llm" if self._client is not None else "offline"

    def decompose(self, query: str, max_subtasks: int) -> list[dict[str, Any]]:
        if self._client is not None:
            prompt = (
                "Break the research query into focused sub-questions. "
                f"Return JSON list of objects with keys question, priority (0-1). "
                f"At most {max_subtasks} items. Query: {query}"
            )
            raw = self._invoke(prompt)
            parsed = self._parse_json_list(raw)
            if parsed:
                return parsed[:max_subtasks]
        return self._heuristic_decompose(query, max_subtasks)

    def synthesize(self, query: str, evidence: list[dict[str, Any]], analysis: dict[str, Any]) -> str:
        bullets = "\n".join(
            f"- [{item.get('rank', i+1)}] {item.get('title')}: {item.get('content')[:220]}"
            for i, item in enumerate(evidence[:6])
        )
        themes = ", ".join(analysis.get("themes", [])[:5]) or "general findings"
        if self._client is not None:
            prompt = (
                "Write a concise research brief grounded ONLY in the evidence. "
                "Cite ranks like [1]. Avoid unsupported claims.\n"
                f"Query: {query}\nThemes: {themes}\nEvidence:\n{bullets}"
            )
            return self._invoke(prompt).strip()
        lines = [
            f"# Research brief: {query}",
            "",
            f"Dominant themes: {themes}.",
            "",
            "## Grounded findings",
        ]
        for item in evidence[:5]:
            lines.append(
                f"- [{item.get('rank', '?')}] {item.get('title')} "
                f"(score={item.get('score', 0):.2f}) — {item.get('content')[:180]}"
            )
        gaps = analysis.get("gaps") or []
        if gaps:
            lines.extend(["", "## Open gaps", *[f"- {gap}" for gap in gaps]])
        lines.extend(
            [
                "",
                "## Confidence note",
                "Claims are limited to ranked evidence to reduce hallucinations.",
            ]
        )
        return "\n".join(lines)

    def refine_plan(self, query: str, gaps: list[str], prior: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self._client is not None and gaps:
            prompt = (
                "Propose refined sub-questions to close these research gaps. "
                "Return JSON list with question, priority.\n"
                f"Query: {query}\nGaps: {gaps}\nPrior: {[p.get('question') for p in prior]}"
            )
            parsed = self._parse_json_list(self._invoke(prompt))
            if parsed:
                return parsed
        return [
            {
                "question": gap if gap.endswith("?") else f"What evidence addresses: {gap}?",
                "priority": 0.85,
            }
            for gap in gaps[:3]
        ] or self._heuristic_decompose(f"{query} deeper analysis", 2)

    def _invoke(self, prompt: str) -> str:
        assert self._client is not None
        message = self._client.invoke(prompt)
        content = getattr(message, "content", message)
        return str(content)

    @staticmethod
    def _parse_json_list(raw: str) -> list[dict[str, Any]]:
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        rows: list[dict[str, Any]] = []
        for item in data:
            if isinstance(item, str):
                rows.append({"question": item, "priority": 0.7})
            elif isinstance(item, dict) and item.get("question"):
                rows.append(
                    {
                        "question": str(item["question"]),
                        "priority": float(item.get("priority", 0.7)),
                    }
                )
        return rows

    @staticmethod
    def _heuristic_decompose(query: str, max_subtasks: int) -> list[dict[str, Any]]:
        cleaned = re.sub(r"\s+", " ", query.strip())
        angles = [
            f"What is the core definition and scope of: {cleaned}?",
            f"What recent developments matter for: {cleaned}?",
            f"What risks, limits, or open debates surround: {cleaned}?",
            f"What practical methods or architectures are used for: {cleaned}?",
            f"What evidence quality and sources support claims about: {cleaned}?",
        ]
        return [
            {"question": question, "priority": round(1.0 - i * 0.1, 2)}
            for i, question in enumerate(angles[:max_subtasks])
        ]


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    return LLMProvider(settings=settings)
