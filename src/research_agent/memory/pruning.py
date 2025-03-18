"""Memory pruning to keep high-signal context and cut graph latency."""

from __future__ import annotations

from research_agent.memory.store import MemoryItem, MemoryStore


def _relevance(item: MemoryItem, query_tokens: set[str], now: float) -> float:
    text = item.content.lower()
    overlap = sum(1 for token in query_tokens if token in text)
    lexical = overlap / max(1, len(query_tokens))
    age_seconds = max(0.0, now - item.timestamp)
    recency = 1.0 / (1.0 + age_seconds / 60.0)
    return 0.55 * item.priority + 0.30 * lexical + 0.15 * recency


def prune_memory(
    store: MemoryStore,
    *,
    query: str,
    keep_ratio: float = 0.55,
    hard_cap: int | None = None,
) -> tuple[MemoryStore, int]:
    """Drop low-signal memories; returns pruned store and removed count.

    Keeping only high-priority, query-relevant items shrinks state size and
    reduces LangGraph checkpoint / node payload cost (portfolio target ~35%).
    """
    items = store.items
    if not items:
        return store, 0

    tokens = {t for t in query.lower().replace("?", " ").split() if len(t) > 2}
    now = max(item.timestamp for item in items)
    scored = sorted(
        ((_relevance(item, tokens, now), item) for item in items),
        key=lambda pair: pair[0],
        reverse=True,
    )

    target_cap = hard_cap or store.capacity
    keep_n = max(2, min(target_cap, int(len(scored) * keep_ratio + 0.999)))
    kept = [item for _, item in scored[:keep_n]]
    removed = len(items) - len(kept)

    pruned = MemoryStore(capacity=store.capacity)
    pruned.replace(kept)
    return pruned, removed
