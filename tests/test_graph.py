"""Core LangGraph orchestration tests (offline, no API key)."""

from research_agent.graph.builder import build_research_graph, run_research
from research_agent.memory.pruning import prune_memory
from research_agent.memory.store import MemoryStore
from research_agent.ranking.adaptive import rank_evidence


def test_task_decomposition_produces_subtasks():
    result = run_research("LangGraph memory pruning for research agents")
    assert len(result.subtasks) >= 2
    assert result.report
    assert result.traces
    assert result.mode == "offline"


def test_memory_pruning_removes_low_signal_items():
    store = MemoryStore(capacity=10)
    for i in range(8):
        store.add(
            f"noise item {i} unrelated fluff",
            agent_id="gatherer",
            priority=0.1,
        )
    store.add(
        "LangGraph checkpoint memory pruning reduces latency",
        agent_id="analyst",
        priority=0.95,
        context_tags=["memory", "langgraph"],
    )
    pruned, removed = prune_memory(
        store,
        query="LangGraph memory pruning latency",
        keep_ratio=0.4,
    )
    assert removed >= 3
    assert len(pruned.items) < len(store.items)
    assert any("LangGraph" in item.content for item in pruned.items)


def test_adaptive_ranking_orders_by_score():
    evidence = [
        {
            "id": "1",
            "subtask_id": "a",
            "title": "Unrelated notes",
            "content": "cooking recipes and travel tips",
            "source": "blog",
            "score": 0.9,
            "tags": ["food"],
        },
        {
            "id": "2",
            "subtask_id": "a",
            "title": "LangGraph refine loops",
            "content": "Bounded refine loops reduce hallucinations in research agents",
            "source": "docs.langchain.com",
            "score": 0.4,
            "tags": ["langgraph", "refinement"],
        },
    ]
    ranked, meta = rank_evidence(
        evidence,
        query="LangGraph refine loops hallucinations",
        analysis_keywords=["langgraph", "refinement"],
    )
    assert ranked[0]["id"] == "2"
    assert meta["count"] == 2


def test_graph_conditional_path_completes():
    graph = build_research_graph()
    final = graph.invoke(
        {
            "run_id": "test-run",
            "query": "adaptive ranking evidence quality research agent",
            "subtasks": [],
            "evidence": [],
            "analysis": {},
            "report": "",
            "confidence": 0.0,
            "refine_loops": 0,
            "max_refine_loops": 1,
            "memory_items": [],
            "memory_pruned": 0,
            "ranking_meta": {},
            "should_refine": False,
            "traces": [],
            "metrics": {},
            "mode": "offline",
        },
        config={"configurable": {"thread_id": "test-thread"}},
    )
    assert final.get("report")
    assert isinstance(final.get("traces"), list)
    nodes = {event.get("node") for event in final.get("traces") or []}
    assert "planner" in nodes
    assert "synthesizer" in nodes


def test_latency_reduction_metric_present():
    result = run_research("execution traces Next.js visualizer LangGraph")
    assert result.baseline_latency_ms >= result.latency_ms
    assert result.latency_reduction_pct >= 0
