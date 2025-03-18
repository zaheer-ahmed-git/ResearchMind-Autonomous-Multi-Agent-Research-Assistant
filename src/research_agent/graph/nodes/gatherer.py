"""Gatherer node — collect evidence per subtask (offline corpus + optional LLM)."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from research_agent.config import get_settings
from research_agent.memory.store import MemoryStore
from research_agent.models.state import ResearchState
from research_agent.ranking.adaptive import rank_evidence
from research_agent.traces.collector import TraceCollector

# Lightweight offline knowledge snippets used when no live search key is set.
_OFFLINE_CORPUS: list[dict[str, str]] = [
    {
        "title": "LangGraph state machines",
        "content": "LangGraph models agents as graphs with typed state, nodes, and conditional edges for loops such as refine-until-quality.",
        "source": "docs.langchain.com/langgraph",
        "tags": "langgraph agents orchestration",
    },
    {
        "title": "Task decomposition for research agents",
        "content": "Decomposing a query into ranked subtasks improves coverage and makes retrieval failures local instead of global hallucinations.",
        "source": "research-patterns/official",
        "tags": "decomposition planning research",
    },
    {
        "title": "Memory pruning in agent systems",
        "content": "Pruning short-term memory by relevance and priority reduces checkpoint size and graph latency without dropping high-signal context.",
        "source": "docs.langchain.com/memory",
        "tags": "memory pruning latency",
    },
    {
        "title": "Adaptive ranking of evidence",
        "content": "Adaptive ranking combines lexical overlap, source trust, and analysis hints so synthesizers cite stronger evidence and invent less.",
        "source": "arxiv-style/methods",
        "tags": "ranking evidence hallucination",
    },
    {
        "title": "Recursive refinement loops",
        "content": "Bounded refine loops re-plan around detected gaps until confidence clears a threshold or max iterations stop the cycle.",
        "source": "nature-style/agents",
        "tags": "refinement confidence quality",
    },
    {
        "title": "OpenAI tool-calling agents",
        "content": "ChatOpenAI with structured prompts can power planner and synthesizer nodes while LangGraph owns control flow and checkpoints.",
        "source": "docs.langchain.com/openai",
        "tags": "openai llm agents",
    },
    {
        "title": "Execution-trace observability",
        "content": "Recording per-node start/end spans enables Next.js visualizers to show timelines, decisions, and latency reduction metrics.",
        "source": "frontend/observability",
        "tags": "traces nextjs visualizer",
    },
]


def _match_corpus(question: str) -> list[dict[str, Any]]:
    tokens = {t for t in question.lower().split() if len(t) > 2}
    hits: list[dict[str, Any]] = []
    for row in _OFFLINE_CORPUS:
        hay = f"{row['title']} {row['content']} {row['tags']}".lower()
        overlap = sum(1 for t in tokens if t in hay)
        if overlap == 0:
            continue
        hits.append(
            {
                "id": str(uuid4())[:8],
                "title": row["title"],
                "content": row["content"],
                "source": row["source"],
                "score": min(1.0, 0.35 + 0.12 * overlap),
                "tags": row["tags"].split(),
            }
        )
    if not hits:
        # Always return at least one grounded stub to avoid empty synthesis.
        row = _OFFLINE_CORPUS[0]
        hits.append(
            {
                "id": str(uuid4())[:8],
                "title": row["title"],
                "content": row["content"],
                "source": row["source"],
                "score": 0.4,
                "tags": row["tags"].split(),
            }
        )
    return hits


def gatherer_node(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    run_id = state.get("run_id") or str(uuid4())
    query = state.get("query") or ""
    tracer = TraceCollector(run_id)
    start = tracer.start("gatherer", "Gathering evidence for subtasks")

    subtasks = list(state.get("subtasks") or [])
    evidence: list[dict[str, Any]] = list(state.get("evidence") or [])
    store = MemoryStore(capacity=settings.memory_capacity)
    store.extend_from_dicts(list(state.get("memory_items") or []))

    for subtask in subtasks:
        if subtask.get("status") == "done":
            continue
        subtask["status"] = "gathering"
        hits = _match_corpus(str(subtask.get("question", query)))
        for hit in hits:
            hit["subtask_id"] = subtask.get("id", "")
            evidence.append(hit)
            store.add(
                f"{hit['title']}: {hit['content']}",
                agent_id="gatherer",
                priority=float(hit.get("score", 0.5)),
                context_tags=list(hit.get("tags") or []),
            )
        subtask["status"] = "done"

    ranked, meta = rank_evidence(
        evidence,
        query=query,
        recency_bias=settings.ranking_recency_bias,
    )
    end = tracer.end(
        "gatherer",
        f"Collected and ranked {len(ranked)} evidence items",
        evidence_count=len(ranked),
        ranking=meta,
    )
    return {
        "subtasks": subtasks,
        "evidence": ranked,
        "ranking_meta": meta,
        "memory_items": store.as_dicts(),
        "traces": [start, end],
        "metrics": {"evidence_count": len(ranked)},
    }
