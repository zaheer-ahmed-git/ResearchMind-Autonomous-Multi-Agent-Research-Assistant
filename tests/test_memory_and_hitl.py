"""Short-term memory, vector memory, and human checkpoint tests."""

from research_agent.hitl.checkpoint import HumanCheckpoint
from research_agent.memory.stm import ShortTermMemory
from research_agent.memory.vector_store import VectorStore


def test_short_term_memory_filters_by_category_and_priority():
    memory = ShortTermMemory(max_size=10)
    memory.add("langgraph checkpoint note", "research_agent", category="research", priority=8)
    memory.add("noise", "analysis_agent", category="analysis", priority=2)
    found = memory.get(category="research", min_priority=5)
    assert len(found) == 1
    assert found[0]["agent_id"] == "research_agent"
    assert memory.remove(found[0]["id"]) is True
    assert memory.get(category="research") == []


def test_vector_store_ranks_related_documents():
    store = VectorStore()
    store.add(
        "LangGraph human checkpoint interrupt resume",
        source="docs",
        doc_type="research",
        tags=["langgraph", "hitl"],
        confidence=0.9,
    )
    store.add(
        "cookie recipe with butter and sugar",
        source="blog",
        doc_type="research",
        tags=["food"],
        confidence=0.2,
    )
    hits = store.similarity_search("LangGraph checkpoint human review", k=2)
    assert hits
    assert hits[0]["doc_type"] == "research"
    assert "LangGraph" in hits[0]["content"]


def test_checkpoint_registry_submit():
    registry = HumanCheckpoint()
    registry.clear()
    checkpoint_id = registry.create(
        task_id="run-1",
        checkpoint_type="final_review",
        data={"query": "agents"},
        priority=2,
    )
    assert registry.pending()[0]["checkpoint_id"] == checkpoint_id
    saved = registry.submit(checkpoint_id, approved=True, feedback="ok", reviewer="zaheer")
    assert saved["status"] == "approved"
    assert registry.pending() == []
